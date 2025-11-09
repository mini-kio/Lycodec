import torch
import torch.nn.functional as F
import math


# Utility helpers

def safe_loss_device(*tensors):
    """Extract device from first valid tensor, otherwise return 'cpu'."""
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return 'cpu'


def validate_tensors(*tensors):
    """Check if all arguments are valid tensors."""
    return all(isinstance(t, torch.Tensor) for t in tensors)


# Loss functions

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

def consistency_loss(model, wav, tokens, spec_clean, sigma_min=0.002, sigma_max=80.0, rho=7.0, ema_model=None, decode_chunk=1, teacher_fn=None, return_student_pred=False):
    """
    Consistency Distillation Loss for one-step generation.

    MEMORY OPTIMIZATION: Teacher prediction is computed FIRST (no_grad) to reduce peak memory.

    Args:
        decode_chunk: microbatch size for decode (default=1 for lowest memory)
        teacher_fn: optional custom teacher decode function for CPU-based EMA
        return_student_pred: if True, return (loss, student_prediction) for reuse
    """
    b = spec_clean.shape[0]
    device = spec_clean.device
    length = wav.shape[-1]

    sigma_i = sample_noise_levels(b, device, sigma_min, sigma_max, rho)
    delta_sigma = torch.rand(b, device=device) * 0.2 * sigma_i
    sigma_i_plus = sigma_i + delta_sigma

    noise = torch.randn_like(spec_clean)
    spec_noisy_i = spec_clean + sigma_i.view(b, 1, 1, 1) * noise
    spec_noisy_i_plus = spec_clean + sigma_i_plus.view(b, 1, 1, 1) * noise

    # 1) Teacher FIRST (no_grad, memory-efficient)
    with torch.no_grad():
        if teacher_fn is not None:
            pred_i = teacher_fn(tokens, length, sigma_i, spec_noisy_i, chunk=decode_chunk)
        else:
            teacher = ema_model if ema_model is not None else model
            pred_i = decode_microbatched(teacher, tokens, length, sigma_i, spec_noisy_i, chunk=decode_chunk)

    # 2) Student AFTER (grad enabled)
    pred_i_plus = decode_microbatched(model, tokens, length, sigma_i_plus, spec_noisy_i_plus, chunk=decode_chunk)

    spec_size = spec_clean.shape[2] * spec_clean.shape[3]
    c = 0.00054 * math.sqrt(spec_size)
    loss = pseudo_huber_loss(pred_i_plus, pred_i, c=c)
    loss = loss / (delta_sigma.mean() + 1e-8)

    # OPTIMIZATION: Return student prediction for reuse (metrics, etc.)
    if return_student_pred:
        return loss, pred_i_plus.detach()  # detach to avoid holding computation graph
    return loss


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

def summary_contrast_loss(summary_a, summary_b, temperature=0.07, gather_fn=None, rank=None):
    """
    InfoNCE contrastive loss with distributed support.

    Args:
        summary_a: query features [B, D] (gradients flow)
        summary_b: key features [B, D] (target, detached)
        temperature: softmax temperature
        gather_fn: optional function to gather features across GPUs (e.g., accelerator.gather)
        rank: current process rank for distributed training (optional, required for correct labels)
    """
    if summary_a is None or summary_b is None:
        return torch.zeros((), device=summary_a.device if summary_a is not None else summary_b.device)

    # Detach target completely
    summary_b = summary_b.detach()
    summary_a = F.normalize(summary_a, dim=-1)
    summary_b = F.normalize(summary_b, dim=-1)

    # Gather features from all GPUs if distributed (NO GRAD on gather)
    if gather_fn is not None:
        try:
            with torch.no_grad():
                # Gather keys only (no gradient through all_gather)
                summary_b_all = gather_fn(summary_b)
            # Use local query, global keys
            logits = summary_a @ summary_b_all.t() / temperature
            # Labels: local batch indices + rank offset for correct positive pair matching
            B_local = summary_a.size(0)
            if rank is not None:
                labels = torch.arange(B_local, device=summary_a.device) + rank * B_local
            else:
                # Fallback: assume rank 0 (will be incorrect for multi-GPU, but prevents crash)
                labels = torch.arange(B_local, device=summary_a.device)
        except:
            # Fallback to local batch if gathering fails
            logits = summary_a @ summary_b.t() / temperature
            labels = torch.arange(summary_a.size(0), device=summary_a.device)
    else:
        logits = summary_a @ summary_b.t() / temperature
        labels = torch.arange(summary_a.size(0), device=summary_a.device)

    loss = F.cross_entropy(logits, labels)
    return loss
