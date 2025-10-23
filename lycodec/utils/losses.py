import torch
import torch.nn as nn
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


def commitment_loss(z_cont, z_disc):
    """
    VQ-VAE commitment loss: encourage encoder to commit to codes.

    L_commit = ||z_cont - sg(z_disc)||^2
    """
    if z_disc is None:
        return torch.tensor(0.0, device=z_cont.device)
    return F.mse_loss(z_cont, z_disc.detach())


def codebook_usage_loss(model):
    """
    Regularize codebook to use all codes uniformly.

    Prevents "codebook collapse" where only few codes are used.
    Encourages high entropy (uniform distribution) in code usage.

    Args:
        model: Lycodec model with FSQ quantizer

    Returns:
        usage loss (lower = more uniform usage)
    """
    model_obj = model.module if hasattr(model, 'module') else model

    if not hasattr(model_obj, 'fsq'):
        return torch.tensor(0.0, device=next(model_obj.parameters()).device)

    fsq = model_obj.fsq
    usage = None

    if hasattr(fsq, 'codebook_usage'):
        usage = fsq.codebook_usage
    elif hasattr(fsq, 'semantic_fsq') and hasattr(fsq.semantic_fsq, 'codebook_usage'):
        usage = fsq.semantic_fsq.codebook_usage

    if usage is None:
        return torch.tensor(0.0, device=next(model_obj.parameters()).device)

    num_groups, num_codes = usage.shape
    device = usage.device

    loss = torch.tensor(0.0, device=device)
    for i in range(num_groups):
        prob = usage[i] / (usage[i].sum() + 1e-8)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8))
        max_entropy = math.log(num_codes)
        loss = loss + (max_entropy - entropy)

    return loss / num_groups


def acoustic_prediction_loss(enc):
    if 'acoustic_residual' not in enc or enc['acoustic_residual'] is None:
        return torch.tensor(0.0, device=enc['tokens'].device)

    residual = enc['acoustic_residual']
    loss = (residual ** 2).mean()

    return loss


def semantic_ar_loss(model, enc):
    model_obj = model.module if hasattr(model, 'module') else model
    device = enc['tokens'].device

    if not model_obj.use_partitioned_fsq:
        return torch.tensor(0.0, device=device)

    if enc['semantic_disc'] is None:
        return torch.tensor(0.0, device=device)

    semantic = enc['semantic_disc']
    B, T, semantic_dim = semantic.shape

    num_groups = 4
    group_dim = semantic_dim // num_groups

    groups = list(semantic.chunk(num_groups, dim=-1))

    loss = torch.tensor(0.0, device=device)

    for i in range(1, num_groups):
        predictor_name = f'semantic_ar_predictor_{i}'
        if not hasattr(model_obj, predictor_name):
            input_dim = i * group_dim
            predictor = nn.Sequential(
                nn.Linear(input_dim, group_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(group_dim * 2, group_dim),
            ).to(device)
            model_obj.register_module(predictor_name, predictor)

        predictor = getattr(model_obj, predictor_name)

        prev_groups = torch.cat(groups[:i], dim=-1)

        target_group = groups[i].detach()

        pred = predictor(prev_groups)

        loss = loss + F.mse_loss(pred, target_group)

    return loss / (num_groups - 1)


def masked_token_prediction_loss(model, wav, mask_ratio=0.15):
    model_obj = model.module if hasattr(model, 'module') else model
    device = wav.device

    with torch.no_grad():
        enc = model_obj.encode(wav)

    tokens = enc["tokens"]
    B, T, D = tokens.shape

    if not hasattr(model_obj, 'mask_token'):
        model_obj.mask_token = torch.nn.Parameter(
            torch.randn(1, 1, D, device=device) * 0.02
        )
        model_obj.register_parameter('mask_token', model_obj.mask_token)

    if not hasattr(model_obj, 'mask_predictor'):
        model_obj.mask_predictor = torch.nn.Sequential(
            torch.nn.Linear(D, D * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(D * 2, D),
        ).to(device)

    num_mask = max(1, int(T * mask_ratio))
    mask_indices = torch.rand(B, T, device=device).argsort(dim=1)[:, :num_mask]

    tokens_masked = tokens.clone()
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask)
    tokens_masked[batch_indices, mask_indices] = model_obj.mask_token

    predicted = model_obj.mask_predictor(tokens_masked)

    target = tokens.detach()
    loss = torch.tensor(0.0, device=device)
    for b in range(B):
        masked_pred = predicted[b, mask_indices[b]]
        masked_target = target[b, mask_indices[b]]
        loss = loss + F.mse_loss(masked_pred, masked_target)

    return loss / B
