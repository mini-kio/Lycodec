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
    stereo_corr_loss,
)
from lycodec.utils.audio import stereo_metrics_inline


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg):
    """Build Lycodec model from configuration dictionary."""

    # Quantizer configuration
    quantizer_type = cfg["model"].get("quantizer_type", "pq")
    use_multi_stage = cfg["model"].get("use_multi_stage_rvq", False)
    rvq_num_stages = cfg["model"].get("rvq_num_stages", 2)

    # Stage-specific configs (if provided in YAML)
    stage_commitment_weights = cfg["model"].get("rvq_stage_commitment_weights", None)
    stage_tau_configs = cfg["model"].get("rvq_stage_tau_configs", None)
    stage_drop_configs = cfg["model"].get("rvq_stage_drop_configs", None)

    # Calculate token_fps: tokens per second (Hz)
    # NOTE: Model internally uses this as sequence length for training clips
    # e.g., 24 Hz * 1.5s = 36 tokens/clip, but we still call it "fps" (frequency)
    token_fps_hz = cfg["model"].get("token_fps", 24)  # tokens per second (Hz)
    crop_seconds = cfg.get("crop_seconds", 1.5)
    tokens_per_clip = int(token_fps_hz * crop_seconds)  # e.g., 24 * 1.5 = 36

    ema_decay = cfg["model"].get("ema_decay", cfg["model"].get("rvq_ema_decay", 0.97))
    awakening_steps = cfg["model"].get("awakening_steps", cfg["model"].get("rvq_awakening_steps", 200))
    drop_start = cfg["model"].get("drop_start", cfg["model"].get("rvq_drop_start", 0.6))
    drop_end = cfg["model"].get("drop_end", cfg["model"].get("rvq_drop_end", 0.1))
    drop_decay_steps = cfg["model"].get("drop_decay_steps", cfg["model"].get("rvq_drop_decay_steps", 200000))
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
        quantizer_type=quantizer_type,
        pq_M=pq_M,
        pq_K=pq_K,
        rvq_codebook_size=cfg["model"].get("rvq_codebook_size", 4096),
        ema_decay=ema_decay,
        awakening_steps=awakening_steps,
        token_fps=tokens_per_clip,  # tokens per second (frequency), passed as clip length
        drop_start=drop_start,
        drop_end=drop_end,
        drop_decay_steps=drop_decay_steps,
        use_residual_corrector=cfg["model"].get("use_residual_corrector", True),
        corrector_alpha=cfg["model"].get("corrector_alpha", 0.3),
        # Multi-stage RVQ (Phase 2)
        use_multi_stage_rvq=use_multi_stage,
        rvq_num_stages=rvq_num_stages,
        rvq_stage_commitment_weights=stage_commitment_weights,
        rvq_stage_tau_configs=stage_tau_configs,
        rvq_stage_drop_configs=stage_drop_configs,
        rvq_fusion_mode=cfg["model"].get("rvq_fusion_mode", "weighted_sum"),
    )

    # Set training-specific parameters (not part of model architecture)
    quantizer_warmup = cfg["train"].get("quantizer_warmup_steps", cfg["train"].get("rvq_warmup_steps", 5000))
    model.quantizer_warmup_steps = quantizer_warmup
    # Backward-compatible attribute for older checkpoints
    model.rvq_warmup_steps = quantizer_warmup

    # Set Gumbel temperature annealing parameters on RVQ
    # For multi-stage, these are already set in MultiStageRVQ constructor via stage_tau_configs
    # For single-stage, set them here
    if model.quantizer_type == 'rvq' and not use_multi_stage:
        quantizer = model.quantizer
        quantizer.tau_hi = cfg["train"].get("gumbel_tau_hi", 2.0)
        quantizer.tau_lo = cfg["train"].get("gumbel_tau_lo", 0.5)
        quantizer.tau_decay_steps = cfg["train"].get("gumbel_tau_decay_steps", 10000)

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
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for wav in pbar:
            # Calculate scheduled weights
            alignment_weight = get_scheduled_weight(cfg["train"].get("alignment_weight_schedule", []), total_steps)
            contrast_weight = get_scheduled_weight(cfg["train"].get("contrast_weight_schedule", []), total_steps)
            stereo_corr_weight = get_scheduled_weight(cfg["train"].get("stereo_corr_weight_schedule", []), total_steps)
            residual_weight = get_scheduled_weight(cfg["train"].get("residual_weight_schedule", []), total_steps)

            with accelerator.accumulate(model):
                model_obj = accelerator.unwrap_model(model)

                # MEMORY OPTIMIZATION: decode=False to skip unnecessary reconstruction
                with accelerator.autocast():
                    rec = None
                    _, enc = model(wav, decode=False)

                loss = torch.zeros((), device=accelerator.device)

                # MEMORY OPTIMIZATION: EMA teacher on CPU, only move to GPU per microbatch
                ema_teacher_cpu = None
                teacher_fn = None
                use_ema_teacher = cfg["train"].get("use_ema_teacher", True)
                if use_ema_teacher and ema is not None:
                    import copy
                    # Keep EMA teacher on CPU to save GPU memory
                    ema_teacher_cpu = copy.deepcopy(model_obj).cpu()
                    ema.copy_to(ema_teacher_cpu)
                    ema_teacher_cpu.eval()

                    # Custom teacher function: move to GPU only for microbatch decode
                    def teacher_decode_fn(tokens, length, sigma, spec_noisy, chunk=1):
                        outs = []
                        for i in range(0, tokens.size(0), chunk):
                            # Move teacher to GPU temporarily
                            ema_teacher_gpu = ema_teacher_cpu.to(accelerator.device, non_blocking=True)
                            with torch.no_grad():
                                outs.append(
                                    ema_teacher_gpu.decode(
                                        tokens[i:i+chunk], length,
                                        sigma[i:i+chunk], spec_noisy[i:i+chunk]
                                    )
                                )
                            # Clean up immediately
                            del ema_teacher_gpu
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        return torch.cat(outs, dim=0)

                    teacher_fn = teacher_decode_fn

                # OPTIMIZATION: Request student prediction for reuse (saves 1 decode = ~20-30% peak memory)
                rec = None
                with accelerator.autocast():
                    if stereo_corr_weight > 0:
                        # Reuse student prediction from consistency_loss
                        loss_consistency, rec = consistency_loss(
                            model_obj, wav, enc["tokens"], enc["spec_clean"],
                            sigma_min=cfg["train"].get("sigma_min", 0.002),
                            sigma_max=cfg["train"].get("sigma_max", 80.0),
                            rho=cfg["train"].get("rho", 7.0),
                            ema_model=None,  # Not used when teacher_fn is provided
                            decode_chunk=cfg["train"].get("decode_chunk", 1),
                            teacher_fn=teacher_fn,
                            return_student_pred=True,  # OPTIMIZATION: return student for reuse
                        )
                    else:
                        loss_consistency = consistency_loss(
                            model_obj, wav, enc["tokens"], enc["spec_clean"],
                            sigma_min=cfg["train"].get("sigma_min", 0.002),
                            sigma_max=cfg["train"].get("sigma_max", 80.0),
                            rho=cfg["train"].get("rho", 7.0),
                            ema_model=None,
                            decode_chunk=cfg["train"].get("decode_chunk", 1),
                            teacher_fn=teacher_fn,
                        )

                # Clean up EMA teacher CPU copy
                if ema_teacher_cpu is not None:
                    del ema_teacher_cpu
                loss = loss + loss_consistency

                # Stereo correlation loss (OPTIMIZATION: reuse student prediction, no extra decode!)
                loss_stereo_corr = None
                if stereo_corr_weight > 0 and rec is not None:
                    loss_stereo_corr = stereo_corr_loss(wav, rec)
                    loss = loss + stereo_corr_weight * loss_stereo_corr

                # RVQ losses
                loss_entropy_bonus = None

                # Commitment loss (increased weight for stability)
                if "commitment_loss" in enc:
                    loss = loss + enc["commitment_loss"]  # Already weighted inside RVQ (0.5)

                # Entropy bonus (only in early training, computed in RVQ forward)
                if "entropy_bonus" in enc:
                    loss_entropy_bonus = enc["entropy_bonus"]
                    loss = loss + loss_entropy_bonus

                # Orthogonal regularization for OPQ-PQ
                ortho_loss = enc.get("ortho_loss", None)
                if ortho_loss is not None:
                    loss = loss + ortho_loss

                # Residual loss (scheduled activation, only when not using dropout)
                loss_residual = None
                loss_align = None
                if hasattr(model_obj, 'corrector') and model_obj.corrector is not None:
                    if residual_weight > 0 and not enc.get("dropout_applied", False):
                        if enc.get("r_target") is not None and enc.get("r_hat") is not None:
                            loss_residual = residual_loss(enc["r_hat"], enc["r_target"])
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
                # Gradient clipping using torch directly (not accelerator.clip_grad_norm_)
                # to avoid conflicts with gradient accumulation context
                if grad_clip and accelerator.sync_gradients:
                    # Only clip when we're about to update (not during accumulation)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad()

                if ema is not None:
                    ema.update(accelerator.unwrap_model(model))
                total_steps += 1

            # Enhanced wandb logging with detailed metrics (log every step from step 0)
            if cfg["logging"]["use_wandb"] and accelerator.is_main_process:
                try:
                    import wandb
                    log = {
                        "train/loss_total": float(loss.item()),
                        "train/loss_consistency": float(loss_consistency.item()),
                        "train/alignment_weight": alignment_weight,
                        "train/contrast_weight": contrast_weight,
                        "train/stereo_corr_weight": stereo_corr_weight,
                        "train/residual_weight": residual_weight,
                        "step": total_steps,
                    }

                    # Individual losses
                    if loss_stereo_corr is not None:
                        log["train/loss_stereo_corr"] = float(loss_stereo_corr.item())
                    if loss_align is not None:
                        log["train/loss_align"] = float(loss_align.item())
                    if loss_contrast is not None:
                        log["train/loss_contrast"] = float(loss_contrast.item())
                    if loss_residual is not None:
                        log["train/loss_residual"] = float(loss_residual.item())

                    # Quantizer metrics
                    if "commitment_loss" in enc:
                        log["quantizer/commitment_loss"] = float(enc["commitment_loss"].item())
                    if "entropy_bonus" in enc and enc["entropy_bonus"].item() != 0:
                        log["quantizer/entropy_bonus"] = float(enc["entropy_bonus"].item())
                    if "perplexity" in enc:
                        log["quantizer/perplexity"] = float(enc["perplexity"])
                    if "usage_entropy" in enc:
                        log["quantizer/usage_entropy"] = float(enc["usage_entropy"])
                    if "drop_prob" in enc:
                        log["quantizer/drop_prob"] = float(enc["drop_prob"])
                    if "dropout_applied" in enc:
                        log["quantizer/dropout_applied"] = int(enc["dropout_applied"])
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
                    if rec is not None and total_steps % 10 == 0:
                        try:
                            m = stereo_metrics_inline(wav, rec.detach())
                            for k, v in m.items():
                                log[f"audio/{k}"] = v
                        except Exception:
                            pass

                    wandb.log(log, step=total_steps)
                except Exception as e:
                    if total_steps == 0:
                        print(f"wandb log failed at step {total_steps}: {e}")

            # Checkpoint saving
            if cfg["save_every"] and total_steps % cfg["save_every"] == 0 and accelerator.is_main_process:
                save_ckpt({"model": accelerator.unwrap_model(model).state_dict(), "opt": opt.state_dict(), "step": total_steps},
                          os.path.join(cfg["output"]["ckpt_dir"], f"step_{total_steps}.pt"))

            # Clean up references for better memory management (after logging)
            del enc, rec, loss_consistency, loss_stereo_corr, loss_residual, loss_align, loss_contrast
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pbar.set_postfix({"loss": float(loss.item()), "step": total_steps})

    if ema is not None and accelerator.is_main_process:
        model_to_save = accelerator.unwrap_model(model)
        ema.copy_to(model_to_save)
    if accelerator.is_main_process:
        save_ckpt({"model": accelerator.unwrap_model(model).state_dict(), "opt": opt.state_dict(), "step": total_steps},
                  os.path.join(cfg["output"]["ckpt_dir"], "latest.pt"))


def load_model(ckpt_path, cfg_path):
    cfg = load_config(cfg_path)
    model = build_model_from_config(cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[name] = v
        model.load_state_dict(new_sd, strict=False)
    return model
