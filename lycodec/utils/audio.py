import torch
import torch.nn.functional as F


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
        x = x.reshape(b * c, t)  # [B*C, T]

    win = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=win, return_complex=True, center=True, pad_mode="reflect")
    # X: [B*C, F, T_out]

    if is_batched:
        X = X.reshape(b, c, X.shape[1], X.shape[2])  # [B, C, F, T_out]

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
        X = X.reshape(b * c, f, t)  # [B*C, F, T]

    # Window must be real dtype
    win = torch.hann_window(win_length, device=X.device, dtype=torch.float32)
    x = torch.istft(X, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=win, length=length, center=True)
    # x: [B*C, T_out]

    if is_batched:
        x = x.reshape(b, c, x.shape[1])  # [B, C, T_out]

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


def amp_phase_transform(spec, beta=1.0, p=0.5):
    a = torch.clamp(spec.abs(), 1e-8) ** p
    ang = torch.atan2(spec[..., 1], spec[..., 0]) if spec.is_complex() else 0
    return a, ang


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
