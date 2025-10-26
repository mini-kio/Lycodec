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
    stft_loss,
    consistency_loss,
    rvq_perplexity_loss,
    infill_loss,
    residual_loss,
    alignment_loss,
    summary_contrast_loss,
)
from lycodec.utils.audio import stereo_metrics_inline
from lycodec.utils.model_utils import build_model_from_config


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("train", {}).get("gradient_accumulation_steps", 1),
        mixed_precision="fp16" if cfg.get("train", {}).get("use_amp", False) else "no"
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
        prefetch_factor=2 if num_workers > 0 else None,
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

    grad_clip = cfg["train"]["grad_clip"]
    total_steps = 0
    ema = EMA(accelerator.unwrap_model(model), decay=cfg["train"]["ema"].get("decay", 0.9999)) if cfg["train"]["ema"].get("enabled", False) else None

    if cfg["logging"]["use_wandb"] and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(project=cfg["logging"]["project"], name=cfg["logging"]["run_name"], config=cfg)
        except Exception as e:
            print(f"wandb init failed: {e}")

    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for wav in pbar:
            stft_weight = 0.0
            stft_sched = cfg["train"].get("stft_weight_schedule", [])
            for step_thr, w in stft_sched[::-1]:
                if total_steps >= step_thr:
                    stft_weight = float(w)
                    break

            with accelerator.accumulate(model):
                rec, enc = model(wav, decode=True)
                model_obj = accelerator.unwrap_model(model)
                loss = torch.zeros((), device=accelerator.device)

                # Consistency loss with EMA teacher
                ema_teacher = None
                if ema is not None:
                    ema_teacher = copy.deepcopy(model_obj)
                    ema_teacher.eval()
                    ema.copy_to(ema_teacher)

                loss_consistency = consistency_loss(
                    model_obj, wav, enc["tokens"], enc["spec_clean"],
                    sigma_min=cfg["train"].get("sigma_min", 0.002),
                    sigma_max=cfg["train"].get("sigma_max", 80.0),
                    rho=cfg["train"].get("rho", 7.0),
                    ema_model=ema_teacher,
                )
                loss = loss + loss_consistency

                # STFT reconstruction loss
                if rec is not None and stft_weight > 0:
                    loss_stft = stft_loss(rec, wav, cfg)
                    loss = loss + stft_weight * loss_stft

                # RVQ losses
                if "commitment_loss" in enc:
                    loss = loss + 0.25 * enc["commitment_loss"]
                if "perplexity" in enc:
                    target_ppx = cfg["train"].get("target_perplexity", 2048)
                    loss_ppx = rvq_perplexity_loss(enc["perplexity"], target_perplexity=target_ppx)
                    loss = loss + 0.001 * loss_ppx
                if rec is not None and not enc.get("dropout_applied", False):
                    infill_weight = cfg["train"].get("infill_weight", 0.3)
                    loss_infill = infill_loss(rec, wav, cfg, decoder_used_continuous=enc.get("dropout_applied", False))
                    loss = loss + infill_weight * loss_infill

                # Corrector losses
                if hasattr(model_obj, 'corrector') and model_obj.corrector is not None:
                    if enc.get("r_target") is not None and enc.get("r_hat") is not None:
                        residual_weight = cfg["train"].get("residual_weight", 1.0)
                        loss_residual = residual_loss(enc["r_hat"], enc["r_target"])
                        loss = loss + residual_weight * loss_residual
                    if enc.get("alignment_target") is not None and enc.get("y_disc") is not None:
                        align_weight = cfg["train"].get("alignment_weight", 0.1)
                        loss_align = alignment_loss(enc["y_disc"], enc["alignment_target"])
                        loss = loss + align_weight * loss_align

                # Contrast loss (with distributed support)
                if enc.get("summary_disc") is not None and enc.get("summary_cont") is not None:
                    contrast_weight = cfg["train"].get("contrast_weight", 1.0)
                    # Pass accelerator.gather for multi-GPU negative mining
                    gather_fn = accelerator.gather if accelerator.num_processes > 1 else None
                    loss_contrast = summary_contrast_loss(
                        enc["summary_disc"], enc["summary_cont"],
                        gather_fn=gather_fn
                    )
                    loss = loss + contrast_weight * loss_contrast

                accelerator.backward(loss)
                if grad_clip:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad()

                if ema is not None:
                    ema.update(accelerator.unwrap_model(model))
                total_steps += 1

            # Clean up references for better memory management
            del enc, rec, ema_teacher
            pbar.set_postfix({"loss": float(loss.item())})

            if cfg["save_every"] and total_steps % cfg["save_every"] == 0 and accelerator.is_main_process:
                save_ckpt({"model": accelerator.unwrap_model(model).state_dict(), "opt": opt.state_dict(), "step": total_steps},
                          os.path.join(cfg["output"]["ckpt_dir"], f"step_{total_steps}.pt"))

            if cfg["logging"]["use_wandb"] and accelerator.is_main_process:
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
