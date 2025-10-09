import torch
import torch.nn as nn
import math
from lycodec.utils.audio import resample_time


def edm_parameterization(sigma, sigma_data=0.5):
    """
    EDM (Elucidating the Design Space of Diffusion-Based Generative Models) parameterization.
    Returns c_skip, c_out, c_in, c_noise for consistency model.

    Args:
        sigma: noise level [B] or scalar
        sigma_data: data standard deviation (default 0.5)

    Returns:
        c_skip, c_out, c_in, c_noise
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor([sigma], dtype=torch.float32)

    sigma = sigma.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    sigma_data_sq = sigma_data ** 2
    sigma_sq = sigma ** 2

    c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
    c_out = sigma * sigma_data / torch.sqrt(sigma_sq + sigma_data_sq)
    c_in = 1.0 / torch.sqrt(sigma_sq + sigma_data_sq)
    c_noise = 0.25 * torch.log(sigma.squeeze())

    return c_skip, c_out, c_in, c_noise


class TokenConditioner(nn.Module):
    def __init__(self, token_dim=256, cond_ch=64, t_out=113, f_bins=1025):
        super().__init__()
        self.t_out = t_out
        self.f_bins = f_bins
        self.proj = nn.Linear(token_dim, cond_ch)
        self.conv = nn.Conv2d(cond_ch, cond_ch, 1)

        # Noise level conditioning
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, cond_ch),
        )

    def forward(self, z, sigma=None):
        """
        Args:
            z: tokens [B, T, D]
            sigma: noise level [B] or None (for inference without noise)
        Returns:
            conditioning [B, C, F, T]
        """
        b, t, d = z.shape
        z = self.proj(z)       # [B, T, C]
        z = z.transpose(1, 2)  # [B, C, T]
        z = resample_time(z, self.t_out)
        z = z.unsqueeze(2).expand(b, z.shape[1], self.f_bins, self.t_out)

        # Add noise conditioning if provided
        if sigma is not None:
            if not torch.is_tensor(sigma):
                sigma = torch.tensor([sigma], device=z.device)
            c_noise = 0.25 * torch.log(sigma.view(-1, 1))  # [B, 1]
            noise_emb = self.noise_embed(c_noise)  # [B, C]
            noise_emb = noise_emb.view(b, -1, 1, 1).expand(-1, -1, self.f_bins, self.t_out)
            z = z + noise_emb

        z = self.conv(z)
        return z


class UNet2D(nn.Module):
    def __init__(self, c_in=4, c_base=64, cond_ch=64):
        super().__init__()
        self.enc1 = self.block(c_in + cond_ch, c_base)
        self.down1 = nn.Conv2d(c_base, c_base, 3, stride=2, padding=1)
        self.enc2 = self.block(c_base + cond_ch, c_base * 2)
        self.down2 = nn.Conv2d(c_base * 2, c_base * 2, 3, stride=2, padding=1)
        self.enc3 = self.block(c_base * 2 + cond_ch, c_base * 4)
        self.down3 = nn.Conv2d(c_base * 4, c_base * 4, 3, stride=2, padding=1)

        self.mid = self.block(c_base * 4 + cond_ch, c_base * 4)

        self.up3 = nn.ConvTranspose2d(c_base * 4, c_base * 4, 2, stride=2)
        self.dec3 = self.block(c_base * 4 + c_base * 4, c_base * 2)
        self.up2 = nn.ConvTranspose2d(c_base * 2, c_base * 2, 2, stride=2)
        self.dec2 = self.block(c_base * 2 + c_base * 2, c_base)
        self.up1 = nn.ConvTranspose2d(c_base, c_base, 2, stride=2)
        self.dec1 = self.block(c_base + c_base, c_base)

        self.out = nn.Conv2d(c_base, 4, 1)

        # Additional noise embedding for UNet (optional, already in TokenConditioner)
        # Can be used for more explicit noise conditioning in the network

    def block(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
        )

    def forward(self, x, cond):
        import torch.nn.functional as F

        c1 = torch.cat([x, cond], dim=1)
        e1 = self.enc1(c1)
        d1 = self.down1(e1)

        c2 = torch.cat([d1, F.interpolate(cond, size=d1.shape[2:], mode='bilinear', align_corners=False)], dim=1)
        e2 = self.enc2(c2)
        d2 = self.down2(e2)

        c3 = torch.cat([d2, F.interpolate(cond, size=d2.shape[2:], mode='bilinear', align_corners=False)], dim=1)
        e3 = self.enc3(c3)
        d3 = self.down3(e3)

        cm = torch.cat([d3, F.interpolate(cond, size=d3.shape[2:], mode='bilinear', align_corners=False)], dim=1)
        m = self.mid(cm)

        u3 = self.up3(m)
        # Match size for skip connection
        if u3.shape[2:] != e3.shape[2:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        j3 = torch.cat([u3, e3], dim=1)
        p3 = self.dec3(j3)

        u2 = self.up2(p3)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        j2 = torch.cat([u2, e2], dim=1)
        p2 = self.dec2(j2)

        u1 = self.up1(p2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        j1 = torch.cat([u1, e1], dim=1)
        p1 = self.dec1(j1)

        # Final interpolation to match input size
        if p1.shape[2:] != x.shape[2:]:
            p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=False)

        y = self.out(p1)
        return y


class BandSplitHead(nn.Module):
    def __init__(self, c_in=4, sr=48000, n_fft=2048, split_hz=12000):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.split_hz = split_hz

        # Low band (base) head
        self.low_head = nn.Sequential(
            nn.Conv2d(c_in, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 4, 1),
        )

        # High band conditioning: uses low band features
        self.high_cond = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 1),
        )

        # High band head (conditioned on low)
        self.high_head = nn.Sequential(
            nn.Conv2d(c_in + 16, 32, 3, padding=1),  # +16 from low conditioning
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 4, 1),
        )

        # Learnable blending weight (starts at 0.5 for low, 0.5 for high)
        self.blend_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def _mask(self, f_bins, device):
        """Create frequency masks for low/high bands."""
        nyq = self.sr / 2
        freqs = torch.linspace(0, nyq, f_bins, device=device)
        low = (freqs <= self.split_hz).float().view(1, 1, f_bins, 1)
        high = 1.0 - low
        return low, high

    def forward(self, x):
        """
        Args:
            x: base prediction from UNet [B, 4, F, T]

        Returns:
            refined prediction with band-split [B, 4, F, T]
        """
        b, c, f, t = x.shape
        low_mask, high_mask = self._mask(f, x.device)

        # Low band (base reconstruction)
        low = self.low_head(x) * low_mask

        # High band conditioning: extract features from low band
        low_feats = self.high_cond(low)

        # High band prediction (conditioned on low)
        high_input = torch.cat([x, low_feats], dim=1)
        high = self.high_head(high_input) * high_mask

        # Learnable blend (normalized weights)
        w = torch.softmax(self.blend_weight, dim=0)
        out = w[0] * low + w[1] * high

        return out
