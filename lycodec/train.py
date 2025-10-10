import os
import math
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from lycodec.data import StereoCropDataset
from lycodec.model import Lycodec
from lycodec.utils.losses import stft_loss, consistency_loss
from lycodec.utils.teacher import TeacherStub, MERTTeacher, MHuBERTTeacher


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


def set_fsq_dropout(model, p):
    if hasattr(model, "fsq"):
        model.fsq.dropout_p = p


def get_teachers(cfg, device):
    prim = cfg.get("teacher", {}).get("primary", {})
    aux = cfg.get("teacher", {}).get("aux", {})
    # primary
    if prim.get("enabled", False) and prim.get("type", "").lower() == "mert":
        try:
            primary = MERTTeacher(prim.get("model_name"), device=device)
        except Exception as e:
            if is_master():
                print(f"[teacher/primary] fallback stub: {e}")
            primary = TeacherStub(dim=cfg["model"]["token_dim"])  # fallback
    else:
        primary = None
    # aux
    if aux.get("enabled", False) and aux.get("type", "").lower() in ("mhubert", "hubert"):
        try:
            secondary = MHuBERTTeacher(aux.get("model_name"), device=device)
        except Exception as e:
            if is_master():
                print(f"[teacher/aux] fallback stub: {e}")
            secondary = TeacherStub(dim=cfg["model"]["token_dim"])  # fallback
    else:
        secondary = None
    return primary, secondary


def tokenwise_ild(wav, tokens):
    # wav: [B,2,T], tokens: [B,Tk,D] → per-token ILD [B,Tk,1]
    import torch
    b, _, t = wav.shape
    tk = tokens.shape[1]
    seg = t // tk
    vals = []
    for i in range(tk):
        s = i * seg
        e = t if i == tk - 1 else (i + 1) * seg
        w = wav[:, :, s:e]
        l = (w[:, 0] ** 2).mean(dim=-1).sqrt().clamp_min(1e-8)
        r = (w[:, 1] ** 2).mean(dim=-1).sqrt().clamp_min(1e-8)
        ild = 20.0 * torch.log10((l + 1e-8) / (r + 1e-8))
        vals.append(ild.view(b, 1))
    return torch.stack(vals, dim=1)  # [B,Tk,1]


def stereo_metrics_inline(gt, pred):
    import torch
    with torch.no_grad():
        def ild(x):
            l, r = x[:, 0], x[:, 1]
            el = (l ** 2).mean(dim=-1).clamp_min(1e-8).sqrt()
            er = (r ** 2).mean(dim=-1).clamp_min(1e-8).sqrt()
            return 20.0 * torch.log10((el + 1e-8) / (er + 1e-8))

        def itc(x):
            l, r = x[:, 0], x[:, 1]
            l = l - l.mean(dim=-1, keepdim=True)
            r = r - r.mean(dim=-1, keepdim=True)
            num = (l * r).sum(dim=-1)
            den = (l.norm(dim=-1) * r.norm(dim=-1)).clamp_min(1e-8)
            return (num / den).clamp(-1, 1)

        ild_err = (ild(pred) - ild(gt)).abs().mean().item()
        itc_err = (itc(pred) - itc(gt)).abs().mean().item()
        return {"ild_err_db": ild_err, "itc_err": itc_err}


def local_attention_align(student, teacher, teacher_hz, student_hz=12, window_sec=0.4):
    import torch
    import torch.nn.functional as F
    b, Ts, Ds = student.shape
    Tt, Dt = teacher.shape[1], teacher.shape[2]
    d = min(Ds, Dt)
    S = F.normalize(student[..., :d], dim=-1)
    T = F.normalize(teacher[..., :d], dim=-1)
    ratio = float(teacher_hz) / float(student_hz)
    w = max(1, int(window_sec * teacher_hz / 2))
    loss = 0.0
    count = 0
    for i in range(Ts):
        center = int(round(i * ratio))
        s = max(0, center - w)
        e = min(Tt, center + w + 1)
        if e <= s:
            continue
        q = S[:, i, :]
        k = T[:, s:e, :]
        att = torch.einsum('bd,btd->bt', q, k)
        att = F.softmax(att, dim=-1)
        agg = torch.einsum('bt,btd->bd', att, k)
        l = 1 - (F.normalize(q, dim=-1) * F.normalize(agg, dim=-1)).sum(dim=-1)
        loss = loss + l.mean()
        count += 1
    return loss / max(1, count)


class AlignProjector(nn.Module):
    def __init__(self, d_s, d_t, d=None):
        super().__init__()
        if d is None:
            d = min(d_s, d_t)
        self.proj_s = nn.Linear(d_s, d, bias=False)
        self.proj_t = nn.Linear(d_t, d, bias=False)

    def forward(self, s, t):
        return self.proj_s(s), self.proj_t(t)


def train(cfg_path, data_root):
    cfg = load_config(cfg_path)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg["sample_rate"]
    crop_s = cfg["crop_seconds"]
    dset = StereoCropDataset(data_root, sample_rate=sr, seconds=crop_s, exts=tuple(cfg["data"]["wav_exts"]))
    use_ddp = setup_ddp(cfg["train"].get("ddp", False))
    sampler = DistributedSampler(dset) if use_ddp else None
    dl = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=(sampler is None), sampler=sampler, num_workers=cfg["num_workers"], drop_last=True)

    model = Lycodec(sr=sr,
                    n_fft=cfg["stft"]["n_fft"],
                    hop=cfg["stft"]["hop_length"],
                    win=cfg["stft"]["win_length"],
                    token_dim=cfg["model"]["token_dim"],
                    hidden=cfg["model"]["hidden_dim"],
                    layers=cfg["model"]["transformer_layers"],
                    heads=cfg["model"]["heads"],
                    use_checkpoint=cfg["train"].get("use_checkpoint", False),
                    use_rope=cfg["model"].get("use_rope", True),
                    use_group_fsq=cfg["model"].get("use_group_fsq", True),)
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
    # Separate teacher model for EMA weights (never trains)
    teacher_model = None
    if ema is not None:
        teacher_model = Lycodec(sr=sr,
                                n_fft=cfg["stft"]["n_fft"],
                                hop=cfg["stft"]["hop_length"],
                                win=cfg["stft"]["win_length"],
                                token_dim=cfg["model"]["token_dim"],
                                hidden=cfg["model"]["hidden_dim"],
                                layers=cfg["model"]["transformer_layers"],
                                heads=cfg["model"]["heads"],
                                use_checkpoint=cfg["train"].get("use_checkpoint", False),
                                use_rope=cfg["model"].get("use_rope", True),
                                use_group_fsq=cfg["model"].get("use_group_fsq", True),).to(device)
        teacher_model.eval()
        with torch.no_grad():
            ema.copy_to(teacher_model)
    primary, secondary = get_teachers(cfg, device)
    align_proj_p = None
    align_proj_a = None

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

            # FSQ dropout schedule
            sched = cfg["train"].get("fsq_dropout_schedule", [])
            for step_thr, p in sched[::-1]:
                if total_steps >= step_thr:
                    set_fsq_dropout(model.module if use_ddp else model, p)
                    break

            # Single-stage training: always train full model with consistency from step 0
            stage = 2
            model_obj = model.module if use_ddp else model
            for param in model_obj.cond.parameters():
                param.requires_grad = True
            for param in model_obj.unet.parameters():
                param.requires_grad = True
            for param in model_obj.bands.parameters():
                param.requires_grad = True

            # STFT auxiliary loss weight schedule (only early N steps > 0)
            stft_weight = 0.0
            stft_sched = cfg["train"].get("recon_weight_schedule", [])
            for step_thr, w in stft_sched[::-1]:
                if total_steps >= step_thr:
                    stft_weight = float(w)
                    break

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    rec, enc = model(wav, decode=True)
                    # Initialize loss as torch tensor (not float!) to avoid AMP/backward errors
                    loss = torch.zeros((), device=device)

                    # Main consistency loss (single-stage)
                    loss_consistency = consistency_loss(
                        model_obj,
                        wav,
                        enc["tokens"],
                        enc["spec_clean"],
                        sigma_min=cfg["train"].get("sigma_min", 0.002),
                        sigma_max=cfg["train"].get("sigma_max", 80.0),
                        rho=cfg["train"].get("rho", 7.0),
                        ema_model=teacher_model if ema is not None else None,
                    )
                    loss = loss + loss_consistency

                    # Optional STFT reconstruction loss with scheduled weight
                    if rec is not None and stft_weight > 0:
                        loss_stft = stft_loss(rec, wav, cfg)
                        loss = loss + stft_weight * loss_stft
                    # Local attention alignment: primary (MERT)
                    # Higher weight in Stage 1, lower in Stage 2
                    if cfg.get("teacher", {}).get("primary", {}).get("enabled", False) and primary is not None:
                        mid = wav[:, :1, :]
                        tsr = cfg["teacher"]["primary"]["sample_rate"]
                        if tsr != sr:
                            import librosa
                            mw = mid.squeeze(1).detach().cpu().numpy()
                            mw = librosa.resample(mw, orig_sr=sr, target_sr=tsr, res_type="kaiser_fast", axis=1)
                            mw = torch.from_numpy(mw).to(device)
                        else:
                            mw = mid.squeeze(1)
                        tfeat = primary.forward(mw, sr=tsr)
                        # build align projector lazily
                        if align_proj_p is None:
                            align_proj_p = AlignProjector(enc["tokens"].shape[-1], tfeat.shape[-1]).to(device)
                            opt.add_param_group({"params": align_proj_p.parameters(), "lr": cfg["lr"]})
                        s_proj, t_proj = align_proj_p(enc["tokens"], tfeat)

                        # Use configured alignment weight directly (single-stage)
                        align_weight = cfg["teacher"]["primary"].get("align_weight", 0.1)

                        loss = loss + align_weight * \
                               local_attention_align(s_proj, t_proj,
                                                     teacher_hz=cfg["teacher"]["primary"].get("feature_hz", 75),
                                                     student_hz=cfg.get("token_rate", 12),
                                                     window_sec=cfg["teacher"]["primary"].get("window_sec", 0.4))

                    # Local attention alignment: auxiliary (mHuBERT)
                    # Higher weight in Stage 1, lower in Stage 2
                    if cfg.get("teacher", {}).get("aux", {}).get("enabled", False) and secondary is not None:
                        mid = wav[:, :1, :]
                        tsr = cfg["teacher"]["aux"]["sample_rate"]
                        if tsr != sr:
                            import librosa
                            mw = mid.squeeze(1).detach().cpu().numpy()
                            mw = librosa.resample(mw, orig_sr=sr, target_sr=tsr, res_type="kaiser_fast", axis=1)
                            mw = torch.from_numpy(mw).to(device)
                        else:
                            mw = mid.squeeze(1)
                        tfeat = secondary.forward(mw, sr=tsr)
                        if align_proj_a is None:
                            align_proj_a = AlignProjector(enc["tokens"].shape[-1], tfeat.shape[-1]).to(device)
                            opt.add_param_group({"params": align_proj_a.parameters(), "lr": cfg["lr"]})
                        s_proj, t_proj = align_proj_a(enc["tokens"], tfeat)

                        # Use configured alignment weight directly (single-stage)
                        align_weight = cfg["teacher"]["aux"].get("align_weight", 0.05)

                        loss = loss + align_weight * \
                               local_attention_align(s_proj, t_proj,
                                                     teacher_hz=cfg["teacher"]["aux"].get("feature_hz", 50),
                                                     student_hz=cfg.get("token_rate", 12),
                                                     window_sec=cfg["teacher"]["aux"].get("window_sec", 0.4))

                    # stereo head supervision (ILD)
                    try:
                        model_obj = model.module if use_ddp else model
                        pred_ild = model_obj.stereo(enc["tokens"])['ild']  # [B,T,1]
                        tgt_ild = tokenwise_ild(wav, enc["tokens"]).to(pred_ild.device)
                        loss = loss + 0.1 * ((pred_ild - tgt_ild) ** 2).mean()
                    except Exception:
                        pass

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
                        # Sync teacher model with EMA weights
                        with torch.no_grad():
                            ema.copy_to(teacher_model)

                    # Increment total_steps only after actual optimizer step
                    total_steps += 1
            else:
                # Non-AMP path (no scaler)
                rec, enc = model(wav, decode=True)
                loss = torch.zeros((), device=device)
                # Main consistency loss
                loss_consistency = consistency_loss(
                    model_obj,
                    wav,
                    enc["tokens"],
                    enc["spec_clean"],
                    sigma_min=cfg["train"].get("sigma_min", 0.002),
                    sigma_max=cfg["train"].get("sigma_max", 80.0),
                    rho=cfg["train"].get("rho", 7.0),
                    ema_model=teacher_model if ema is not None else None,
                )
                loss = loss + loss_consistency

                # Optional STFT reconstruction with scheduled weight
                if rec is not None and stft_weight > 0:
                    loss = loss + stft_weight * stft_loss(rec, wav, cfg)

                # Local attention alignment: primary (MERT)
                if cfg.get("teacher", {}).get("primary", {}).get("enabled", False) and primary is not None:
                    mid = wav[:, :1, :]
                    tsr = cfg["teacher"]["primary"]["sample_rate"]
                    if tsr != sr:
                        import librosa
                        mw = mid.squeeze(1).detach().cpu().numpy()
                        mw = librosa.resample(mw, orig_sr=sr, target_sr=tsr, res_type="kaiser_fast", axis=1)
                        mw = torch.from_numpy(mw).to(device)
                    else:
                        mw = mid.squeeze(1)
                    tfeat = primary.forward(mw, sr=tsr)
                    if align_proj_p is None:
                        align_proj_p = AlignProjector(enc["tokens"].shape[-1], tfeat.shape[-1]).to(device)
                        opt.add_param_group({"params": align_proj_p.parameters(), "lr": cfg["lr"]})
                    s_proj, t_proj = align_proj_p(enc["tokens"], tfeat)
                    align_weight = cfg["teacher"]["primary"].get("align_weight", 0.1)
                    loss = loss + align_weight * \
                           local_attention_align(s_proj, t_proj,
                                                 teacher_hz=cfg["teacher"]["primary"].get("feature_hz", 75),
                                                 student_hz=cfg.get("token_rate", 12),
                                                 window_sec=cfg["teacher"]["primary"].get("window_sec", 0.4))

                # Local attention alignment: auxiliary (mHuBERT)
                if cfg.get("teacher", {}).get("aux", {}).get("enabled", False) and secondary is not None:
                    mid = wav[:, :1, :]
                    tsr = cfg["teacher"]["aux"]["sample_rate"]
                    if tsr != sr:
                        import librosa
                        mw = mid.squeeze(1).detach().cpu().numpy()
                        mw = librosa.resample(mw, orig_sr=sr, target_sr=tsr, res_type="kaiser_fast", axis=1)
                        mw = torch.from_numpy(mw).to(device)
                    else:
                        mw = mid.squeeze(1)
                    tfeat = secondary.forward(mw, sr=tsr)
                    if align_proj_a is None:
                        align_proj_a = AlignProjector(enc["tokens"].shape[-1], tfeat.shape[-1]).to(device)
                        opt.add_param_group({"params": align_proj_a.parameters(), "lr": cfg["lr"]})
                    s_proj, t_proj = align_proj_a(enc["tokens"], tfeat)
                    align_weight = cfg["teacher"]["aux"].get("align_weight", 0.05)
                    loss = loss + align_weight * \
                           local_attention_align(s_proj, t_proj,
                                                 teacher_hz=cfg["teacher"]["aux"].get("feature_hz", 50),
                                                 student_hz=cfg.get("token_rate", 12),
                                                 window_sec=cfg["teacher"]["aux"].get("window_sec", 0.4))

                # stereo head supervision (ILD)
                try:
                    model_obj = model.module if use_ddp else model
                    pred_ild = model_obj.stereo(enc["tokens"])['ild']  # [B,T,1]
                    tgt_ild = tokenwise_ild(wav, enc["tokens"]).to(pred_ild.device)
                    loss = loss + 0.1 * ((pred_ild - tgt_ild) ** 2).mean()
                except Exception:
                    pass

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
                        with torch.no_grad():
                            ema.copy_to(teacher_model)

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
                    use_group_fsq=cfg["model"].get("use_group_fsq", True),)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)
    return model
