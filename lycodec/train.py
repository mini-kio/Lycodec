import os
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from lycodec.data import StereoCropDataset
from lycodec.model import Lycodec
from lycodec.utils.losses import (
    stft_loss,
    consistency_loss,
    rvq_perplexity_loss,
    infill_loss,
    residual_loss,
    alignment_loss,
    summary_contrast_loss,
)


def load_config(path):
    # Ensure UTF-8 decoding for configs with non-ASCII comments/keys
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v)


def setup_ddp(enable=False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if enable and world_size > 1 and not dist.is_initialized():
        # Use NCCL only on non-Windows platforms with CUDA available; otherwise fall back to gloo
        use_nccl = (os.name != "nt") and torch.cuda.is_available()
        dist.init_process_group(backend="nccl" if use_nccl else "gloo")
        return True
    return False


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0




def train(cfg_path, data_root):
    cfg = load_config(cfg_path)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg["sample_rate"]
    crop_s = cfg["crop_seconds"]
    dset = StereoCropDataset(data_root, sample_rate=sr, seconds=crop_s, exts=tuple(cfg["data"]["wav_exts"]))
    use_ddp = setup_ddp(cfg["train"].get("ddp", False))
    sampler = DistributedSampler(dset) if use_ddp else None
    dl = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=(sampler is None), sampler=sampler, num_workers=cfg["num_workers"], drop_last=True)

    model = Lycodec(sr=cfg["sample_rate"],
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
                    rvq_codebook_size=cfg["model"].get("rvq_codebook_size", 4096),
                    token_fps=cfg["model"].get("token_fps", 24),
                    rvq_drop_start=cfg["model"].get("rvq_drop_start", 0.6),
                    rvq_drop_end=cfg["model"].get("rvq_drop_end", 0.1),
                    rvq_drop_decay_steps=cfg["model"].get("rvq_drop_decay_steps", 200000),
                    use_residual_corrector=cfg["model"].get("use_residual_corrector", True),
                    corrector_alpha=cfg["model"].get("corrector_alpha", 0.3),)
    model.to(device)
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[torch.cuda.current_device()] if device.startswith("cuda") else None)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["use_amp"]) if device.startswith("cuda") else None
    grad_clip = cfg["train"]["grad_clip"]
    accum_steps = cfg["train"].get("gradient_accumulation_steps", 1)  # Default: no accumulation
    total_steps = 0
    accum_iter = 0  # Track accumulation iterations
    ema = EMA(model.module if use_ddp else model, decay=cfg["train"]["ema"].get("decay", 0.9999)) if cfg["train"]["ema"].get("enabled", False) else None

    if cfg["logging"]["use_wandb"] and is_master():
        try:
            import wandb
            wandb.init(project=cfg["logging"]["project"], name=cfg["logging"]["run_name"], config=cfg)
        except Exception as e:
            print(f"wandb init failed: {e}")

    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for wav in pbar:
            wav = wav.to(device)  # [B, C, T]

            # STFT auxiliary loss weight schedule
            stft_weight = 0.0
            stft_sched = cfg["train"].get("stft_weight_schedule", [])
            for step_thr, w in stft_sched[::-1]:
                if total_steps >= step_thr:
                    stft_weight = float(w)
                    break

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    rec, enc = model(wav, decode=True)
                    model_obj = model.module if use_ddp else model

                    # Initialize loss as torch tensor (not float!) to avoid AMP/backward errors
                    loss = torch.zeros((), device=device)

                    # ============================================
                    # Main consistency loss
                    # ============================================
                    loss_consistency = consistency_loss(
                        model_obj,
                        wav,
                        enc["tokens"],
                        enc["spec_clean"],
                        sigma_min=cfg["train"].get("sigma_min", 0.002),
                        sigma_max=cfg["train"].get("sigma_max", 80.0),
                        rho=cfg["train"].get("rho", 7.0),
                        ema_model=None,
                    )
                    loss = loss + loss_consistency

                    # ============================================
                    # STFT reconstruction loss
                    # ============================================
                    if rec is not None and stft_weight > 0:
                        loss_stft = stft_loss(rec, wav, cfg)
                        loss = loss + stft_weight * loss_stft

                    # ============================================
                    # RVQ auxiliary losses
                    # ============================================

                    # RVQ commitment loss (encoder → codebook)
                    if "commitment_loss" in enc:
                        loss_commit = enc["commitment_loss"]
                        loss = loss + 0.25 * loss_commit

                    # RVQ perplexity loss (encourage codebook usage)
                    if "perplexity" in enc:
                        target_ppx = cfg["train"].get("target_perplexity", 2048)  # 0.5 * K
                        loss_ppx = rvq_perplexity_loss(enc["perplexity"], target_perplexity=target_ppx)
                        loss = loss + 0.001 * loss_ppx

                    # Infill loss (skip when decoder consumed continuous latents)
                    if rec is not None and not enc.get("dropout_applied", False):
                        infill_weight = cfg["train"].get("infill_weight", 0.3)
                        loss_infill = infill_loss(rec, wav, cfg, decoder_used_continuous=enc.get("dropout_applied", False))
                        loss = loss + infill_weight * loss_infill

                    if hasattr(model_obj, 'corrector') and model_obj.corrector is not None:
                        if enc.get("r_target") is not None and enc.get("r_hat") is not None:
                            residual_weight = cfg["train"].get("residual_weight", 1.0)
                            loss_residual = residual_loss(enc["r_hat"], enc["r_target"])
                            loss = loss + residual_weight * loss_residual

                        if enc.get("alignment_target") is not None and enc.get("y_disc") is not None:
                            align_weight = cfg["train"].get("alignment_weight", 0.1)
                            loss_align = alignment_loss(enc["y_disc"], enc["alignment_target"])
                            loss = loss + align_weight * loss_align

                    if enc.get("summary_disc") is not None and enc.get("summary_cont") is not None:
                        contrast_weight = cfg["train"].get("contrast_weight", 1.0)
                        loss_contrast = summary_contrast_loss(enc["summary_disc"], enc["summary_cont"])
                        loss = loss + contrast_weight * loss_contrast

                # Normalize loss by accumulation steps
                loss = loss / accum_steps

                scaler.scale(loss).backward()

                # Increment accumulation counter
                accum_iter += 1

                # Step optimizer every accum_steps iterations
                if accum_iter % accum_steps == 0:
                    if grad_clip:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                    # EMA update (only after optimizer step)
                    if ema is not None:
                        ema.update(model.module if use_ddp else model)

                    # Increment total_steps only after actual optimizer step
                    total_steps += 1
            else:
                # Non-AMP path (no scaler)
                rec, enc = model(wav, decode=True)
                model_obj = model.module if use_ddp else model
                loss = torch.zeros((), device=device)

                # ============================================
                # Main consistency loss
                # ============================================
                loss_consistency = consistency_loss(
                    model_obj,
                    wav,
                    enc["tokens"],
                    enc["spec_clean"],
                    sigma_min=cfg["train"].get("sigma_min", 0.002),
                    sigma_max=cfg["train"].get("sigma_max", 80.0),
                    rho=cfg["train"].get("rho", 7.0),
                    ema_model=None,
                )
                loss = loss + loss_consistency

                # ============================================
                # STFT reconstruction loss
                # ============================================
                if rec is not None and stft_weight > 0:
                    loss = loss + stft_weight * stft_loss(rec, wav, cfg)

                # ============================================
                # RVQ auxiliary losses
                # ============================================

                # RVQ commitment loss (encoder → codebook)
                if "commitment_loss" in enc:
                    loss_commit = enc["commitment_loss"]
                    loss = loss + 0.25 * loss_commit

                # RVQ perplexity loss (encourage codebook usage)
                if "perplexity" in enc:
                    target_ppx = cfg["train"].get("target_perplexity", 2048)  # 0.5 * K
                    loss_ppx = rvq_perplexity_loss(enc["perplexity"], target_perplexity=target_ppx)
                    loss = loss + 0.001 * loss_ppx

                # Infill loss (skip when decoder consumed continuous latents)
                if rec is not None and not enc.get("dropout_applied", False):
                    infill_weight = cfg["train"].get("infill_weight", 0.3)
                    loss_infill = infill_loss(rec, wav, cfg, decoder_used_continuous=enc.get("dropout_applied", False))
                    loss = loss + infill_weight * loss_infill

                if hasattr(model_obj, 'corrector') and model_obj.corrector is not None:
                    if enc.get("r_target") is not None and enc.get("r_hat") is not None:
                        residual_weight = cfg["train"].get("residual_weight", 1.0)
                        loss_residual = residual_loss(enc["r_hat"], enc["r_target"])
                        loss = loss + residual_weight * loss_residual

                    if enc.get("alignment_target") is not None and enc.get("y_disc") is not None:
                        align_weight = cfg["train"].get("alignment_weight", 0.1)
                        loss_align = alignment_loss(enc["y_disc"], enc["alignment_target"])
                        loss = loss + align_weight * loss_align

                if enc.get("summary_disc") is not None and enc.get("summary_cont") is not None:
                    contrast_weight = cfg["train"].get("contrast_weight", 1.0)
                    loss_contrast = summary_contrast_loss(enc["summary_disc"], enc["summary_cont"])
                    loss = loss + contrast_weight * loss_contrast

                # Normalize loss by accumulation steps
                loss = loss / accum_steps
                loss.backward()

                accum_iter += 1

                if accum_iter % accum_steps == 0:
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                    if ema is not None:
                        ema.update(model.module if use_ddp else model)

                    total_steps += 1
            pbar.set_postfix({"loss": float(loss.item())})

            if cfg["save_every"] and total_steps % cfg["save_every"] == 0 and is_master():
                save_ckpt({"model": model.state_dict(), "opt": opt.state_dict(), "step": total_steps},
                          os.path.join(cfg["output"]["ckpt_dir"], f"step_{total_steps}.pt"))

            if cfg["logging"]["use_wandb"] and is_master():
                try:
                    import wandb
                    log = {"loss": float(loss.item()), "step": total_steps}
                    if rec is not None:
                        try:
                            m = stereo_metrics_inline(wav, rec.detach())
                            for k, v in m.items():
                                log[f"stereo/{k}"] = v
                        except Exception:
                            pass
                    wandb.log(log, step=total_steps)
                except Exception:
                    pass

    # final save (EMA weights if enabled)
    if ema is not None and is_master():
        model_to_save = model.module if use_ddp else model
        ema.copy_to(model_to_save)
    if is_master():
        save_ckpt({"model": model.state_dict(), "opt": opt.state_dict(), "step": total_steps},
                  os.path.join(cfg["output"]["ckpt_dir"], "latest.pt"))


def load_model(ckpt_path, cfg_path):
    cfg = load_config(cfg_path)
    model = Lycodec(sr=cfg["sample_rate"],
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
                    rvq_codebook_size=cfg["model"].get("rvq_codebook_size", 4096),
                    token_fps=cfg["model"].get("token_fps", 24),
                    rvq_drop_start=cfg["model"].get("rvq_drop_start", 0.6),
                    rvq_drop_end=cfg["model"].get("rvq_drop_end", 0.1),
                    rvq_drop_decay_steps=cfg["model"].get("rvq_drop_decay_steps", 200000),
                    use_residual_corrector=cfg["model"].get("use_residual_corrector", True),
                    corrector_alpha=cfg["model"].get("corrector_alpha", 0.3),)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[name] = v
        model.load_state_dict(new_sd, strict=False)
    return model
