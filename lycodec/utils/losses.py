import torch
import torch.nn.functional as F
import math
from lycodec.utils.common import safe_loss_device, validate_tensors

def stft_loss(x, y, cfg):
    """Multi-resolution STFT loss for batched multichannel audio."""
    is_batched = x.ndim == 3
    if is_batched:
        b, c, t = x.shape
        x_flat = x.reshape(b * c, t)
        y_flat = y.reshape(b * c, t)
    else:
        x_flat = x
        y_flat = y

    x32 = x_flat.float()
    y32 = y_flat.float()

    losses = []
    for hl, wl in zip([128, 256, 512, 1024], [512, 1024, 2048, 4096]):
        win = torch.hann_window(wl, device=x.device, dtype=torch.float32)
        X = torch.stft(x32, n_fft=wl, hop_length=hl, win_length=wl, window=win, return_complex=True)
        Y = torch.stft(y32, n_fft=wl, hop_length=hl, win_length=wl, window=win, return_complex=True)
        losses.append((X.abs() - Y.abs()).abs().mean())
        phase_diff = torch.angle(X) - torch.angle(Y)
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        losses.append(phase_diff.abs().mean())
    return sum(losses)


def sample_noise_levels(batch_size, device, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Sample noise levels from log-normal distribution (EDM schedule)."""
    u = torch.rand(batch_size, device=device)
    sigma = (sigma_max ** (1 / rho) + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigma

def pseudo_huber_loss(pred, target, c=1.0):
    """Pseudo-Huber loss: sqrt((x/c)^2 + 1) - 1. Robust and smooth."""
    diff = pred - target
    loss = torch.sqrt((diff / c) ** 2 + 1) - 1
    return loss.mean()

def decode_microbatched(model, tokens, length, sigma, spec_noisy, chunk=2):
    """
    Decode in microbatches to reduce memory peak.

    Args:
        model: decoder model
        tokens: conditioning tokens [B, T, D]
        length: output waveform length
        sigma: noise levels [B]
        spec_noisy: noisy spectrogram [B, 4, F, T]
        chunk: microbatch size

    Returns:
        decoded waveforms [B, C, L]
    """
    outs = []
    for i in range(0, tokens.size(0), chunk):
        outs.append(model.decode(tokens[i:i+chunk], length, sigma[i:i+chunk], spec_noisy[i:i+chunk]))
    return torch.cat(outs, dim=0)

def consistency_loss(model, wav, tokens, spec_clean, sigma_min=0.002, sigma_max=80.0, rho=7.0, ema_model=None):
    """Consistency Distillation Loss for one-step generation."""
    b = spec_clean.shape[0]
    device = spec_clean.device
    length = wav.shape[-1]

    sigma_i = sample_noise_levels(b, device, sigma_min, sigma_max, rho)
    delta_sigma = torch.rand(b, device=device) * 0.2 * sigma_i
    sigma_i_plus = sigma_i + delta_sigma

    noise = torch.randn_like(spec_clean)
    spec_noisy_i = spec_clean + sigma_i.view(b, 1, 1, 1) * noise
    spec_noisy_i_plus = spec_clean + sigma_i_plus.view(b, 1, 1, 1) * noise

    pred_i_plus = decode_microbatched(model, tokens, length, sigma_i_plus, spec_noisy_i_plus, chunk=2)

    teacher = ema_model if ema_model is not None else model
    with torch.no_grad():
        pred_i = decode_microbatched(teacher, tokens, length, sigma_i, spec_noisy_i, chunk=2)

    spec_size = spec_clean.shape[2] * spec_clean.shape[3]
    c = 0.00054 * math.sqrt(spec_size)
    loss = pseudo_huber_loss(pred_i_plus, pred_i, c=c)
    loss = loss / (delta_sigma.mean() + 1e-8)

    return loss

def commitment_loss(z_cont, z_disc):
    """VQ-VAE commitment loss: encourage encoder to commit to codes."""
    if z_disc is None:
        return torch.tensor(0.0, device=z_cont.device)
    return F.mse_loss(z_cont, z_disc.detach())

def rvq_perplexity_loss(perplexity, target_perplexity=2048):
    """Encourage RVQ codebook usage to reach target perplexity."""
    if perplexity >= target_perplexity:
        return torch.tensor(0.0, device=perplexity.device)
    gap = target_perplexity - perplexity
    return (gap / target_perplexity) ** 2

def infill_loss(rec_wav, target_wav, cfg, decoder_used_continuous=False):
    """STFT infill loss; skipped when decoder saw continuous latents."""
    if decoder_used_continuous:
        return torch.zeros((), device=rec_wav.device)
    return stft_loss(rec_wav, target_wav, cfg)

def residual_loss(r_hat, r_target):
    """MSE between predicted residuals and targets."""
    if not validate_tensors(r_hat, r_target):
        device = safe_loss_device(r_hat, r_target)
        return torch.zeros((), device=device)
    return F.mse_loss(r_hat, r_target)

def alignment_loss(y_disc, y_cont_detached):
    """Cosine alignment loss."""
    if not validate_tensors(y_disc, y_cont_detached):
        device = safe_loss_device(y_disc, y_cont_detached)
        return torch.zeros((), device=device)
    # Extra safety: ensure target is completely detached
    y_cont_detached = y_cont_detached.detach()
    y_disc_norm = F.normalize(y_disc, dim=-1)
    y_cont_norm = F.normalize(y_cont_detached, dim=-1)
    return 1 - (y_disc_norm * y_cont_norm).sum(dim=-1).mean()

def summary_contrast_loss(summary_a, summary_b, temperature=0.07, gather_fn=None):
    """
    InfoNCE contrastive loss with distributed support.

    Args:
        summary_a: positive features [B, D]
        summary_b: negative features [B, D]
        temperature: softmax temperature
        gather_fn: optional function to gather features across GPUs (e.g., accelerator.gather)
    """
    if summary_a is None or summary_b is None:
        return torch.zeros((), device=summary_a.device if summary_a is not None else summary_b.device)

    # Extra safety: ensure target (summary_b) is completely detached
    summary_b = summary_b.detach()
    summary_a = F.normalize(summary_a, dim=-1)
    summary_b = F.normalize(summary_b, dim=-1)

    # Gather features from all GPUs if distributed
    if gather_fn is not None:
        try:
            summary_a_all = gather_fn(summary_a)
            summary_b_all = gather_fn(summary_b)
            # Use global features for both
            logits = summary_a_all @ summary_b_all.t() / temperature
            # Labels are diagonal indices for the full global batch
            labels = torch.arange(summary_a_all.size(0), device=summary_a.device)
        except:
            # Fallback to local batch if gathering fails
            logits = summary_a @ summary_b.t() / temperature
            labels = torch.arange(summary_a.size(0), device=summary_a.device)
    else:
        logits = summary_a @ summary_b.t() / temperature
        labels = torch.arange(summary_a.size(0), device=summary_a.device)

    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)
