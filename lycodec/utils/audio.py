import torch
import torch.nn.functional as F


# Global cache for STFT/ISTFT windows to avoid repeated allocations
_hann_window_cache = {}


def _get_hann_window(win_length, device):
    """Get cached Hann window or create new one."""
    key = (win_length, device.type, device.index if device.type == 'cuda' else None)
    if key not in _hann_window_cache:
        _hann_window_cache[key] = torch.hann_window(win_length, device=device, dtype=torch.float32)
    return _hann_window_cache[key]


def to_midside(x):
    l, r = x[:, 0:1], x[:, 1:2]
    m = (l + r) * 0.5
    s = (l - r) * 0.5
    return torch.cat([m, s], dim=1)


def to_stereo(ms):
    m, s = ms[:, 0:1], ms[:, 1:2]
    l = m + s
    r = m - s
    return torch.cat([l, r], dim=1)


def stft(x, n_fft=2048, hop_length=640, win_length=2048):
    """
    STFT for batched multichannel audio.
    Args:
        x: [B, C, T] or [C, T]
    Returns:
        X: [B, C, F, T] or [C, F, T] complex tensor
    """
    is_batched = x.ndim == 3
    if is_batched:
        b, c, t = x.shape
        x = x.reshape(b * c, t)

    # Force float32 for STFT to avoid fp16 issues in AMP
    x32 = x.float()
    win = _get_hann_window(win_length, x.device)
    X = torch.stft(x32, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=win, return_complex=True, center=True, pad_mode="reflect")

    if is_batched:
        X = X.reshape(b, c, X.shape[1], X.shape[2])

    return X


def istft(X, n_fft=2048, hop_length=640, win_length=2048, length=None):
    """
    Inverse STFT for batched multichannel audio.
    Args:
        X: [B, C, F, T] or [C, F, T] complex tensor
    Returns:
        x: [B, C, T_out] or [C, T_out]
    """
    is_batched = X.ndim == 4
    if is_batched:
        b, c, f, t = X.shape
        X = X.reshape(b * c, f, t)

    # Force float32 for ISTFT
    win = _get_hann_window(win_length, X.device)
    x = torch.istft(X, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=win, length=length, center=True)

    if is_batched:
        x = x.reshape(b, c, x.shape[1])

    return x


def spec_pack(ms_complex):
    m, s = ms_complex[:, 0], ms_complex[:, 1]
    out = torch.stack([m.real, m.imag, s.real, s.imag], dim=1)
    return out


def spec_unpack(spec):
    m = spec[:, 0] + 1j * spec[:, 1]
    s = spec[:, 2] + 1j * spec[:, 3]
    out = torch.stack([m, s], dim=1)
    return out


def wav_to_spec(x, n_fft=2048, hop_length=640, win_length=2048):
    ms = to_midside(x)
    X = stft(ms, n_fft, hop_length, win_length)
    return spec_pack(X)


def spec_to_wav(spec, n_fft=2048, hop_length=640, win_length=2048, length=None):
    ms_complex = spec_unpack(spec)
    ms_wav = istft(ms_complex, n_fft, hop_length, win_length, length=length)
    wav = to_stereo(ms_wav)
    return wav


def resample_time(x, t_out):
    b, c, t_in = x.shape
    idx = torch.linspace(0, t_in - 1, t_out, device=x.device)
    idx0 = idx.floor().long()
    idx1 = torch.clamp(idx0 + 1, max=t_in - 1)
    w = (idx - idx0.float()).view(1, 1, -1)
    x0 = x[:, :, idx0]
    x1 = x[:, :, idx1]
    return x0 * (1 - w) + x1 * w


def stereo_metrics_inline(target, pred, eps=1e-8):
    """
    Compute stereo audio quality metrics for logging.

    Args:
        target: [B, C, T] original audio
        pred: [B, C, T] reconstructed audio
        eps: small constant for numerical stability

    Returns:
        dict with metric names and float values
    """
    metrics = {}

    # Align tensor lengths if needed
    min_len = min(target.shape[-1], pred.shape[-1])
    target = target[..., :min_len]
    pred = pred[..., :min_len]

    # MSE (Mean Squared Error)
    mse = F.mse_loss(pred, target)
    metrics['mse'] = float(mse.item())

    # L1 Loss
    l1 = F.l1_loss(pred, target)
    metrics['l1'] = float(l1.item())

    # SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    # Compute per-channel for stereo, then average
    B, C, T = target.shape
    si_sdrs = []

    for ch in range(C):
        target_ch = target[:, ch, :]  # [B, T]
        pred_ch = pred[:, ch, :]      # [B, T]

        # Project pred onto target
        alpha = (target_ch * pred_ch).sum(dim=1, keepdim=True) / (target_ch.pow(2).sum(dim=1, keepdim=True) + eps)
        target_scaled = alpha * target_ch

        # Signal and noise
        noise = pred_ch - target_scaled
        si_sdr_ch = 10 * torch.log10((target_scaled.pow(2).sum(dim=1) + eps) / (noise.pow(2).sum(dim=1) + eps))
        si_sdrs.append(si_sdr_ch.mean())

    # Average across channels
    metrics['si_sdr'] = float(torch.stack(si_sdrs).mean().item())

    return metrics
