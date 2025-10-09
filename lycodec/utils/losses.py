import torch
import torch.nn.functional as F
import math


def stft_loss(x, y, cfg):
    """
    Multi-resolution STFT loss for batched multichannel audio.
    Args:
        x: predicted audio [B, C, T]
        y: target audio [B, C, T]
    Returns:
        loss: scalar
    """
    # Flatten batch and channels for STFT
    is_batched = x.ndim == 3
    if is_batched:
        b, c, t = x.shape
        x_flat = x.reshape(b * c, t)  # [B*C, T]
        y_flat = y.reshape(b * c, t)
    else:
        x_flat = x
        y_flat = y

    # Ensure stable dtype for STFT computations
    x32 = x_flat.float()
    y32 = y_flat.float()

    losses = []
    for hl, wl in zip([128, 256, 512, 1024], [512, 1024, 2048, 4096]):
        win = torch.hann_window(wl, device=x.device, dtype=torch.float32)
        X = torch.stft(x32, n_fft=wl, hop_length=hl, win_length=wl, window=win, return_complex=True)
        Y = torch.stft(y32, n_fft=wl, hop_length=hl, win_length=wl, window=win, return_complex=True)
        losses.append((X.abs() - Y.abs()).abs().mean())
        losses.append((torch.angle(X) - torch.angle(Y)).abs().mean())
    return sum(losses)


def cosine_align(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1 - (a * b).sum(dim=-1).mean()


def sample_noise_levels(batch_size, device, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    Sample noise levels from log-normal distribution (EDM schedule).

    Args:
        batch_size: number of samples
        device: torch device
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: schedule parameter (controls distribution shape)

    Returns:
        sigma: sampled noise levels [B]
    """
    u = torch.rand(batch_size, device=device)
    sigma = (sigma_max ** (1 / rho) + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigma


def pseudo_huber_loss(pred, target, c=1.0):
    """
    Pseudo-Huber loss: sqrt((x/c)^2 + 1) - 1

    More robust than L2, smoother than L1.

    Args:
        pred: predicted values
        target: target values
        c: scale parameter (controls transition from L2 to L1)

    Returns:
        loss value
    """
    diff = pred - target
    loss = torch.sqrt((diff / c) ** 2 + 1) - 1
    return loss.mean()


def consistency_loss(model, wav, tokens, spec_clean, sigma_min=0.002, sigma_max=80.0, rho=7.0, ema_model=None):
    """
    Consistency Distillation Loss (CD loss) for one-step generation.

    Loss = E[d(f(x + σ_i·ε, σ_i), sg(f(x + σ_{i+1}·ε, σ_{i+1}))) / Δσ]

    where:
    - f is the consistency function (student/online model)
    - sg is stop-gradient (teacher/EMA model)
    - d is pseudo-Huber distance
    - Δσ = σ_{i+1} - σ_i

    Args:
        model: student model (online)
        wav: clean waveform [B, C, T]
        tokens: encoded tokens [B, Tk, D]
        spec_clean: clean spectrogram [B, 4, F, T]
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: noise schedule parameter
        ema_model: teacher model (EMA). If None, uses self-consistency

    Returns:
        consistency loss
    """
    b = spec_clean.shape[0]
    device = spec_clean.device
    length = wav.shape[-1]

    # Sample noise level σ_i
    sigma_i = sample_noise_levels(b, device, sigma_min, sigma_max, rho)

    # Sample Δσ (small increment) from uniform [0, 0.2 * σ_i]
    # This ensures σ_{i+1} = σ_i + Δσ is slightly larger
    delta_sigma = torch.rand(b, device=device) * 0.2 * sigma_i
    sigma_i_plus = sigma_i + delta_sigma

    # Add the SAME noise to both levels (important for consistency!)
    noise = torch.randn_like(spec_clean)
    spec_noisy_i = spec_clean + sigma_i.view(b, 1, 1, 1) * noise
    spec_noisy_i_plus = spec_clean + sigma_i_plus.view(b, 1, 1, 1) * noise

    # Student prediction at σ_{i+1} (higher noise)
    pred_i_plus = model.decode(tokens, length, sigma_i_plus, spec_noisy_i_plus)

    # Teacher prediction at σ_i (lower noise) - with stop gradient
    teacher = ema_model if ema_model is not None else model
    with torch.no_grad():
        pred_i = teacher.decode(tokens, length, sigma_i, spec_noisy_i)

    # Pseudo-Huber distance
    # c is adaptive based on spectrogram size (helps with scale)
    spec_size = spec_clean.shape[2] * spec_clean.shape[3]  # F * T
    c = 0.00054 * math.sqrt(spec_size)

    loss = pseudo_huber_loss(pred_i_plus, pred_i, c=c)

    # Normalize by delta_sigma (important for stability!)
    loss = loss / (delta_sigma.mean() + 1e-8)

    return loss
