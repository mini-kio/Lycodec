import os
import copy
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from lycodec.data import StereoCropDataset
from lycodec.model import Lycodec
from lycodec.utils.losses import (
    consistency_loss,
    residual_loss,
    alignment_loss,
    summary_contrast_loss,
)
from lycodec.utils.audio import stereo_metrics_inline


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg):
    """Build Lycodec model from configuration dictionary."""

    # Calculate token_fps: tokens per second (Hz)
    # NOTE: Model internally uses this as sequence length for training clips
    # e.g., 24 Hz * 1.5s = 36 tokens/clip, but we still call it "fps" (frequency)
    token_fps_hz = cfg["model"].get("token_fps", 24)  # tokens per second (Hz)
    crop_seconds = cfg.get("crop_seconds", 1.5)
    tokens_per_clip = int(token_fps_hz * crop_seconds)  # e.g., 24 * 1.5 = 36

    ema_decay = cfg["model"].get("ema_decay", 0.97)
    awakening_steps = cfg["model"].get("awakening_steps", 200)
    drop_start = cfg["model"].get("drop_start", 0.6)
    drop_end = cfg["model"].get("drop_end", 0.1)
    drop_decay_steps = cfg["model"].get("drop_decay_steps", 200000)
    pq_M = cfg["model"].get("pq_M", 4)
    pq_K = cfg["model"].get("pq_K", 256)

    model = Lycodec(
        sr=cfg["sample_rate"],
        n_fft=cfg["stft"]["n_fft"],
        hop=cfg["stft"]["hop_length"],
        win=cfg["stft"]["win_length"],
        token_dim=cfg["model"]["token_dim"],
        hidden=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["transformer_layers"],
        heads=cfg["model"]["heads"],
        use_checkpoint=cfg["train"].get("use_checkpoint", False),
        use_rope=cfg["model"].get("use_rope", True),
        decoder_depth=cfg["model"].get("decoder_depth", 6),
        decoder_patch_size=cfg["model"].get("decoder_patch_size", 16),
        decoder_embed_dim=cfg["model"].get("decoder_embed_dim", 512),
        decoder_mlp_ratio=cfg["model"].get("decoder_mlp_ratio", 4.0),
        decoder_cond_ch=cfg["model"].get("decoder_cond_ch", 64),
        pq_M=pq_M,
        pq_K=pq_K,
        ema_decay=ema_decay,
        awakening_steps=awakening_steps,
        token_fps=tokens_per_clip,  # tokens per second (frequency), passed as clip length
        drop_start=drop_start,
        drop_end=drop_end,
        drop_decay_steps=drop_decay_steps,
        use_residual_corrector=cfg["model"].get("use_residual_corrector", True),
        corrector_alpha=cfg["model"].get("corrector_alpha", 0.3),
    )

    # Set training-specific parameters (not part of model architecture)
    quantizer_warmup = cfg["train"].get("quantizer_warmup_steps", 5000)
    model.quantizer_warmup_steps = quantizer_warmup

    return model


def save_ckpt(state, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(state, path)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.dtype.is_floating_point
        }
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad and p.dtype.is_floating_point:
                self.shadow[n].mul_(self.decay).add_(p, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])






def train(cfg_path, data_root):
    cfg = load_config(cfg_path)

    # Enable TF32 for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    grad_clip = cfg["train"].get("grad_clip", 1.0)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("train", {}).get("gradient_accumulation_steps", 1),
        mixed_precision="fp16" if cfg.get("train", {}).get("use_amp", False) else "no",
        # Gradient clipping handled by Accelerator when using accumulate()
    )
    sr = cfg["sample_rate"]
    crop_s = cfg["crop_seconds"]
    dset = StereoCropDataset(data_root, sample_rate=sr, seconds=crop_s, exts=tuple(cfg["data"]["wav_exts"]))

    num_workers = cfg["num_workers"]
    dl = DataLoader(
        dset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model = build_model_from_config(cfg)

    # Separate parameters for weight decay
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay for 1D params (bias, norms) and embeddings
        if p.ndim == 1 or 'bias' in n or 'norm' in n.lower() or 'embedding' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    opt = optim.AdamW([
        {"params": decay, "weight_decay": 1e-2},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg["lr"])
    model, opt, dl = accelerator.prepare(model, opt, dl)

    # Verify sample rate consistency between dataset and model
    if accelerator.is_main_process:
        dataset_sr = cfg["sample_rate"]
        model_sr = accelerator.unwrap_model(model).sr
        model_hop = accelerator.unwrap_model(model).hop
        print(f"[Config Check] dataset_sr={dataset_sr}, model_sr={model_sr}, model_hop={model_hop}")
        assert model_sr == dataset_sr, f"SR mismatch: dataset={dataset_sr} vs model={model_sr}"

    total_steps = 0
    ema = EMA(accelerator.unwrap_model(model), decay=cfg["train"]["ema"].get("decay", 0.9999)) if cfg["train"]["ema"].get("enabled", False) else None

    # Resume from checkpoint (if specified in config)
    resume_ckpt_path = cfg["output"].get("resume_ckpt")
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        sd = torch.load(resume_ckpt_path, map_location="cpu")
        # Load model state (including quantizer step_counter)
        model_unwrapped = accelerator.unwrap_model(model)

        # Filter out RoPE cached buffers (they are regenerated in forward pass)
        model_state = sd["model"]
        filtered_state = {k: v for k, v in model_state.items()
                         if not ('rope.cos_cached' in k or 'rope.sin_cached' in k)}

        model_unwrapped.load_state_dict(filtered_state, strict=False)
        # Load optimizer state
        if "opt" in sd:
            opt.load_state_dict(sd["opt"])
        # Restore step counter (critical for schedule alignment)
        if "step" in sd:
            total_steps = int(sd["step"])
        # Restore EMA shadow if available
        if ema is not None and "ema_shadow" in sd:
            ema.shadow = sd["ema_shadow"]
        if accelerator.is_main_process:
            print(f"[Resume] Loaded checkpoint from {resume_ckpt_path}, resuming from step={total_steps}")

    if cfg["logging"]["use_wandb"] and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(project=cfg["logging"]["project"], name=cfg["logging"]["run_name"], config=cfg)
        except Exception as e:
            print(f"wandb init failed: {e}")

    def get_scheduled_weight(schedule, current_step):
        """Get weight from schedule based on current step (with linear interpolation)."""
        if not schedule:
            return 0.0
        # Find the two points to interpolate between
        prev_step, prev_weight = 0, 0.0
        for step_thr, w in schedule:
            if current_step >= step_thr:
                prev_step, prev_weight = step_thr, float(w)
            else:
                # Linear interpolation
                if prev_step < current_step < step_thr:
                    ratio = (current_step - prev_step) / (step_thr - prev_step)
                    return prev_weight + ratio * (float(w) - prev_weight)
                break
        return prev_weight

    for epoch in range(cfg["epochs"]):
        model.train()

        # Create EMA teacher once per epoch (GPU-resident, updated after each optimizer step)
        ema_teacher = None
        use_ema_teacher = cfg["train"].get("use_ema_teacher", True)
        if use_ema_teacher and ema is not None:
            import copy
            ema_teacher = copy.deepcopy(accelerator.unwrap_model(model))
            ema_teacher.to(accelerator.device)
            ema.copy_to(ema_teacher)
            ema_teacher.eval()
            if accelerator.is_main_process:
                print(f"[Epoch {epoch}] Created EMA teacher on device {accelerator.device}")

        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for wav in pbar:
            # Calculate scheduled weights
            alignment_weight = get_scheduled_weight(cfg["train"].get("alignment_weight_schedule", []), total_steps)
            contrast_weight = get_scheduled_weight(cfg["train"].get("contrast_weight_schedule", []), total_steps)
            residual_weight = get_scheduled_weight(cfg["train"].get("residual_weight_schedule", []), total_steps)

            with accelerator.accumulate(model):
                model_obj = accelerator.unwrap_model(model)

                # MEMORY OPTIMIZATION: decode=False to skip reconstruction during training
                with accelerator.autocast():
                    _, enc = model(wav, decode=False)

                loss = torch.zeros((), device=accelerator.device)

                # Consistency loss (EMA teacher already created and GPU-resident)
                with accelerator.autocast():
                    loss_consistency = consistency_loss(
                        model_obj, wav, enc["tokens"], enc["spec_clean"],
                        sigma_min=cfg["train"].get("sigma_min", 0.002),
                        sigma_max=cfg["train"].get("sigma_max", 80.0),
                        rho=cfg["train"].get("rho", 7.0),
                        ema_model=ema_teacher,  # GPU-resident teacher, no CPU↔GPU hop
                        decode_chunk=cfg["train"].get("decode_chunk", 1),
                        teacher_fn=None,
                    )
                loss = loss + loss_consistency

                # Quantizer losses
                loss_entropy_bonus = None

                # Entropy bonus (warmup only, decays to 0)
                # Used for initial codebook diversity, then gradually removed
                if "entropy_bonus" in enc:
                    loss_entropy_bonus = enc["entropy_bonus"]
                    # Warmup: full weight until 5000 steps, then linear decay to 0 by 20000 steps
                    warmup_end = 5000
                    decay_end = 20000
                    if total_steps < warmup_end:
                        entropy_weight = 1.0
                    elif total_steps < decay_end:
                        entropy_weight = 1.0 - (total_steps - warmup_end) / (decay_end - warmup_end)
                    else:
                        entropy_weight = 0.0
                    loss = loss + entropy_weight * loss_entropy_bonus

                # Orthogonal regularization for OPQ-PQ
                ortho_loss = enc.get("ortho_loss", None)
                if ortho_loss is not None:
                    loss = loss + ortho_loss

                # Residual loss (sample-level, only for discrete path samples)
                loss_residual = None
                loss_align = None
                if hasattr(model_obj, 'corrector') and model_obj.corrector is not None:
                    if residual_weight > 0 and enc.get("r_target") is not None and enc.get("r_hat") is not None:
                        dropout_mask = enc.get("dropout_applied", None)
                        if dropout_mask is None:
                            # No dropout, all samples use discrete path
                            loss_residual = residual_loss(enc["r_hat"], enc["r_target"])
                        else:
                            # Sample-level: only compute for discrete path samples
                            discrete_mask = (~dropout_mask.squeeze(-1).squeeze(-1)).float()  # [B]
                            if discrete_mask.sum() > 0:
                                # MSE per sample, then weighted average
                                loss_per_sample = (enc["r_hat"] - enc["r_target"]).pow(2).mean(dim=(-1, -2))  # [B]
                                loss_residual = (loss_per_sample * discrete_mask).sum() / (discrete_mask.sum() + 1e-6)
                            else:
                                loss_residual = torch.zeros((), device=accelerator.device)
                        loss = loss + residual_weight * loss_residual

                # Alignment loss (scheduled activation for discrete/continuous alignment)
                if alignment_weight > 0 and enc.get("alignment_target") is not None and enc.get("y_disc") is not None:
                    loss_align = alignment_loss(enc["y_disc"], enc["alignment_target"])
                    loss = loss + alignment_weight * loss_align

                # Contrast loss (scheduled activation for semantic representations)
                loss_contrast = None
                if contrast_weight > 0 and enc.get("summary_disc") is not None and enc.get("summary_cont") is not None:
                    # Pass accelerator.gather for multi-GPU negative mining
                    gather_fn = accelerator.gather if accelerator.num_processes > 1 else None
                    loss_contrast = summary_contrast_loss(
                        enc["summary_disc"], enc["summary_cont"],
                        gather_fn=gather_fn
                    )
                    loss = loss + contrast_weight * loss_contrast

                accelerator.backward(loss)
                # Gradient clipping (only when actually updating)
                if grad_clip and accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad()

                # Update EMA and step counter only on actual optimizer updates
                # (not during gradient accumulation microbatches)
                if accelerator.sync_gradients:
                    if ema is not None:
                        ema.update(accelerator.unwrap_model(model))
                        # Sync EMA weights to teacher for next batch
                        if ema_teacher is not None:
                            ema.copy_to(ema_teacher)
                    total_steps += 1

            # Clean up references (always, even during accumulation)
            del enc, loss_consistency, loss_residual, loss_align, loss_contrast
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Logging and checkpointing only on actual updates (not microbatches)
            if not accelerator.sync_gradients:
                continue

            # Enhanced wandb logging with detailed metrics
            if cfg["logging"]["use_wandb"] and accelerator.is_main_process:
                try:
                    import wandb
                    log = {
                        "train/loss_total": float(loss.item()),
                        "train/loss_consistency": float(loss_consistency.item()),
                        "train/alignment_weight": alignment_weight,
                        "train/contrast_weight": contrast_weight,
                        "train/residual_weight": residual_weight,
                        "step": total_steps,
                    }

                    # Individual losses
                    if loss_align is not None:
                        log["train/loss_align"] = float(loss_align.item())
                    if loss_contrast is not None:
                        log["train/loss_contrast"] = float(loss_contrast.item())
                    if loss_residual is not None:
                        log["train/loss_residual"] = float(loss_residual.item())

                    # Quantizer metrics
                    if "entropy_bonus" in enc and enc["entropy_bonus"].item() != 0:
                        log["quantizer/entropy_bonus"] = float(enc["entropy_bonus"].item())
                    if "perplexity" in enc:
                        log["quantizer/perplexity"] = float(enc["perplexity"])
                    if "usage_entropy" in enc:
                        log["quantizer/usage_entropy"] = float(enc["usage_entropy"])
                    if "drop_prob" in enc:
                        log["quantizer/drop_prob"] = float(enc["drop_prob"])
                    if "dropout_applied" in enc and enc["dropout_applied"] is not None:
                        # dropout_applied is now a tensor [B,1,1], log ratio of samples using continuous path
                        log["quantizer/dropout_ratio"] = float(enc["dropout_applied"].float().mean().item())
                    if "active_codes" in enc:
                        log["quantizer/active_codes"] = int(enc["active_codes"])
                    if "current_tau" in enc:
                        log["quantizer/current_tau"] = float(enc["current_tau"])
                    if "ortho_loss" in enc and enc["ortho_loss"].requires_grad:
                        log["quantizer/ortho_loss"] = float(enc["ortho_loss"].item())

                    # Corrector losses
                    if loss_residual is not None:
                        log["train/loss_residual"] = float(loss_residual.item())

                    # Audio metrics (only occasionally to save time)
                    if total_steps % 100 == 0:
                        try:
                            with torch.no_grad():
                                rec = model_obj.decode(enc["tokens"], wav.shape[-1])
                            m = stereo_metrics_inline(wav, rec)
                            for k, v in m.items():
                                log[f"audio/{k}"] = v
                            del rec
                        except Exception:
                            pass

                    wandb.log(log, step=total_steps)
                except Exception as e:
                    if total_steps == 0:
                        print(f"wandb log failed at step {total_steps}: {e}")

            # Checkpoint saving
            if cfg["save_every"] and total_steps % cfg["save_every"] == 0 and accelerator.is_main_process:
                ckpt_data = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "opt": opt.state_dict(),
                    "step": total_steps,
                }
                if ema is not None:
                    ckpt_data["ema_shadow"] = ema.shadow
                save_ckpt(ckpt_data, os.path.join(cfg["output"]["ckpt_dir"], f"step_{total_steps}.pt"))

            # Update progress bar
            pbar.set_postfix({"loss": float(loss.item()), "step": total_steps})

    if ema is not None and accelerator.is_main_process:
        model_to_save = accelerator.unwrap_model(model)
        ema.copy_to(model_to_save)
    if accelerator.is_main_process:
        ckpt_data = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "opt": opt.state_dict(),
            "step": total_steps,
        }
        if ema is not None:
            ckpt_data["ema_shadow"] = ema.shadow
        save_ckpt(ckpt_data, os.path.join(cfg["output"]["ckpt_dir"], "latest.pt"))


def load_model(ckpt_path, cfg_path, use_ema=True):
    """
    Load model from checkpoint.

    Args:
        ckpt_path: path to checkpoint file
        cfg_path: path to config file
        use_ema: if True, load EMA weights for inference (recommended)
    """
    cfg = load_config(cfg_path)
    model = build_model_from_config(cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Load EMA weights if available and requested
        if use_ema and "ema_shadow" in ckpt:
            print(f"[load_model] Loading EMA weights from {ckpt_path}")
            ema_shadow = ckpt["ema_shadow"]
            # Apply EMA shadow to model parameters
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ema_shadow:
                        p.data.copy_(ema_shadow[n])
        else:
            # Load regular model weights
            if use_ema and "ema_shadow" not in ckpt:
                print(f"[load_model] Warning: EMA weights not found in {ckpt_path}, using model weights")
            sd = ckpt["model"]
            from collections import OrderedDict
            new_sd = OrderedDict()
            for k, v in sd.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_sd[name] = v
            model.load_state_dict(new_sd, strict=False)
    return model
