import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lycodec.utils.audio import resample_time
from lycodec.core.blocks import ConvNormAct2d, ConvTransposeNormAct2d


# ============================================================================
# Utility Functions
# ============================================================================

def to_tensor(value, batch_size=None, device='cpu', dtype=torch.float32):
    """Convert value to tensor if not already a tensor."""
    if not torch.is_tensor(value):
        if batch_size is not None:
            value = [value] * batch_size
        else:
            value = [value]
        return torch.tensor(value, device=device, dtype=dtype)
    return value


def compute_noise_embedding(sigma, reshape=True):
    """Compute noise embedding from sigma values."""
    if not torch.is_tensor(sigma):
        sigma = torch.tensor([sigma], dtype=torch.float32)
    c_noise = 0.25 * torch.log(sigma)
    if reshape:
        c_noise = c_noise.view(-1, 1)
    else:
        c_noise = c_noise.squeeze()
    return c_noise


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
    c_noise = compute_noise_embedding(sigma, reshape=False)

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
        z = self.proj(z)
        z = z.transpose(1, 2)
        z = resample_time(z, self.t_out)
        z = z.unsqueeze(2).expand(b, z.shape[1], self.f_bins, self.t_out).contiguous()

        # Add noise conditioning if provided
        if sigma is not None:
            sigma_tensor = to_tensor(sigma, device=z.device)
            c_noise = compute_noise_embedding(sigma_tensor, reshape=True)  # [B, 1]
            noise_emb = self.noise_embed(c_noise)  # [B, C]
            noise_emb = noise_emb.view(b, -1, 1, 1).expand(-1, -1, self.f_bins, self.t_out)
            z = z + noise_emb

        z = self.conv(z)
        return z


class BandSplitHead(nn.Module):
    def __init__(self, c_in=4, sr=48000, n_fft=2048, split_hz=12000):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.split_hz = split_hz

        # Low band (base) head
        self.low_head = nn.Sequential(
            ConvNormAct2d(c_in, 32, kernel_size=3, padding=1),
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
            ConvNormAct2d(c_in + 16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 4, 1),
        )

        # Learnable blending weight
        self.blend_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Cache frequency masks for expected f_bins (n_fft//2 + 1)
        f_bins = n_fft // 2 + 1
        nyq = sr / 2
        freqs = torch.linspace(0, nyq, f_bins)
        low_mask = (freqs <= split_hz).float().view(1, 1, f_bins, 1)
        high_mask = 1.0 - low_mask
        self.register_buffer('low_mask_cached', low_mask)
        self.register_buffer('high_mask_cached', high_mask)

    def forward(self, x):
        """
        Args:
            x: base prediction from UNet [B, 4, F, T]

        Returns:
            refined prediction with band-split [B, 4, F, T]
        """
        b, c, f, t = x.shape

        # Use cached masks (already on correct device via buffer)
        low_mask = self.low_mask_cached
        high_mask = self.high_mask_cached

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


# ============================================================================
# Transformer-based Decoder Components
# ============================================================================

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) for noise conditioning.

    Modulates layer normalization with scale and shift parameters
    derived from noise level embeddings.

    Used in DiT (Diffusion Transformer) and similar architectures.
    """
    def __init__(self, dim, noise_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        # MLP to generate scale and shift from noise embedding
        self.scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(noise_dim, 2 * dim)
        )

    def forward(self, x, noise_emb):
        """
        Args:
            x: input features [B, N, D]
            noise_emb: noise level embedding [B, noise_dim]

        Returns:
            normalized and modulated features [B, N, D]
        """
        # Normalize
        x_norm = self.ln(x)

        # Get scale and shift from noise embedding
        scale_shift = self.scale_shift(noise_emb).unsqueeze(1)  # [B, 1, 2*D]
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each [B, 1, D]

        # Apply adaptive modulation: x * (1 + scale) + shift
        return x_norm * (1 + scale) + shift


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block with:
    - Self-attention on spectrogram patches
    - Cross-attention to token conditioning
    - AdaLN for noise conditioning
    - MLP with GELU activation
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, noise_dim=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Adaptive Layer Norms
        self.adaln_self = AdaptiveLayerNorm(dim, noise_dim)
        self.adaln_cross = AdaptiveLayerNorm(dim, noise_dim)
        self.adaln_mlp = AdaptiveLayerNorm(dim, noise_dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention (patches attend to tokens)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, tokens, noise_emb):
        """
        Args:
            x: spectrogram patch embeddings [B, N_patches, dim]
            tokens: conditioning tokens from encoder [B, N_tokens, dim]
            noise_emb: noise level embedding [B, noise_dim]

        Returns:
            processed features [B, N_patches, dim]
        """
        # Self-attention with AdaLN
        x_norm = self.adaln_self(x, noise_emb)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention with AdaLN
        x_norm = self.adaln_cross(x, noise_emb)
        cross_out, _ = self.cross_attn(x_norm, tokens, tokens)
        x = x + cross_out

        # MLP with AdaLN
        x_norm = self.adaln_mlp(x, noise_emb)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class PatchEmbed(nn.Module):
    """
    2D Patch Embedding layer for spectrograms.

    Divides spectrogram into non-overlapping patches and embeds each patch
    into a vector of dimension `embed_dim`.
    """
    def __init__(self, c_in=68, patch_size=16, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolution acts as patch embedding
        self.proj = nn.Conv2d(
            c_in, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: input spectrogram [B, C, H, W]

        Returns:
            patch embeddings [B, N_patches, embed_dim]
            grid_size: (num_patches_h, num_patches_w)
            orig_size: (H, W) prior to padding
        """
        B, C, H, W = x.shape
        orig_size = (H, W)
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]

        # Get grid size before flattening
        grid_h, grid_w = x.shape[2], x.shape[3]

        # Flatten spatial dimensions
        x = x.flatten(2)  # [B, embed_dim, N_patches]
        x = x.transpose(1, 2)  # [B, N_patches, embed_dim]

        return x, (grid_h, grid_w), orig_size


class Unpatchify(nn.Module):
    """
    Converts patch embeddings back to 2D spectrogram.

    Uses transposed convolutions to progressively upsample
    from patch resolution to original spectrogram resolution.
    """
    def __init__(self, embed_dim=512, patch_size=16, c_out=4, target_size=(1025, 113)):
        super().__init__()
        self.patch_size = patch_size
        self.target_size = target_size

        # Validate patch_size is power of 2
        if patch_size & (patch_size - 1) != 0 or patch_size == 0:
            raise ValueError(f"patch_size must be a power of 2, got {patch_size}")

        # Calculate number of upsampling stages
        num_stages = int(math.log2(patch_size))

        # Build upsampling layers
        layers = []
        current_dim = embed_dim

        for i in range(num_stages):
            next_dim = current_dim // 2 if i < num_stages - 1 else 32
            layers.append(
                ConvTransposeNormAct2d(
                    current_dim, next_dim,
                    kernel_size=4, stride=2, padding=1
                )
            )
            current_dim = next_dim

        self.upsample = nn.Sequential(*layers)

        # Final projection to output channels
        self.final_proj = nn.Conv2d(32, c_out, kernel_size=3, padding=1)

    def forward(self, x, grid_size, orig_size=None):
        """
        Args:
            x: patch embeddings [B, N_patches, embed_dim]
            grid_size: (grid_h, grid_w) from patch embedding
            orig_size: (H, W) prior to padding

        Returns:
            reconstructed spectrogram [B, c_out, H, W]
        """
        B, N, D = x.shape
        grid_h, grid_w = grid_size

        # Reshape to 2D grid
        x = x.transpose(1, 2)  # [B, embed_dim, N_patches]
        x = x.reshape(B, D, grid_h, grid_w)  # [B, embed_dim, grid_h, grid_w]

        # Upsample
        x = self.upsample(x)

        # Final projection
        x = self.final_proj(x)

        if orig_size is not None:
            H, W = orig_size
            x = x[..., :H, :W]

        # Interpolate to exact target size if needed
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return x


class TransformerDecoder2D(nn.Module):
    """
    Transformer-based decoder for spectrogram generation.

    Replaces UNet2D with a modern transformer architecture inspired by
    Diffusion Transformers (DiT) and Vision Transformers (ViT).

    Key features:
    - Patch-based processing for computational efficiency
    - Cross-attention to token conditioning from encoder
    - Adaptive Layer Normalization (AdaLN) for noise conditioning
    - Multi-head self-attention for capturing global dependencies

    Args:
        c_in: input channels (4 for spectrogram + 64 for conditioning = 68)
        c_out: output channels (4 for spectrogram)
        embed_dim: embedding dimension for patches
        depth: number of transformer layers
        num_heads: number of attention heads
        patch_size: size of each patch (16 recommended)
        token_dim: dimension of conditioning tokens
        mlp_ratio: ratio of MLP hidden dim to embedding dim
        dropout: dropout probability
        target_size: target output size (F, T) for spectrogram
    """
    def __init__(
        self,
        c_in=68,  # 4 (spec) + 64 (cond)
        c_out=4,
        embed_dim=512,
        depth=6,
        num_heads=8,
        patch_size=16,
        token_dim=256,
        mlp_ratio=4.0,
        dropout=0.0,
        target_size=(1025, 113),
        max_token_len=24,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.target_size = target_size

        # Noise embedding MLP
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
        )

        # Token projection (from encoder tokens to transformer dimension)
        self.token_proj = nn.Linear(token_dim, embed_dim)

        # Patch embedding
        self.patch_embed = PatchEmbed(c_in, patch_size, embed_dim)

        # Calculate number of patches for positional encoding
        grid_h = math.ceil(target_size[0] / patch_size)
        grid_w = math.ceil(target_size[1] / patch_size)
        num_patches = grid_h * grid_w

        # Learnable positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Learnable positional encoding for tokens
        self.token_pos_embed = nn.Parameter(torch.randn(1, max_token_len, embed_dim) * 0.02)

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim, num_heads, mlp_ratio, dropout, noise_dim=embed_dim
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.final_ln = nn.LayerNorm(embed_dim)

        # Unpatchify to reconstruct spectrogram
        self.unpatchify = Unpatchify(embed_dim, patch_size, c_out, target_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights with Xavier/Kaiming initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, cond, tokens=None, sigma=None):
        """
        Args:
            x: input noisy spectrogram [B, 4, F, T]
            cond: conditioning from TokenConditioner [B, 64, F, T]
            tokens: optional raw tokens from encoder [B, N_tokens, token_dim]
                   If None, will extract from cond (less ideal but maintains compatibility)
            sigma: noise level [B] or scalar

        Returns:
            predicted clean spectrogram [B, 4, F, T]
        """
        B = x.shape[0]
        device = x.device

        # Handle noise embedding
        if sigma is None:
            # Default to minimal noise for inference
            sigma = torch.ones(B, device=device) * 1e-3
        else:
            sigma = to_tensor(sigma, batch_size=B, device=device, dtype=torch.float32)

        # Create noise embedding
        sigma_input = sigma.view(-1, 1)  # [B, 1]
        noise_emb = self.noise_embed(sigma_input)  # [B, embed_dim]

        # Concatenate spectrogram and conditioning
        x_cat = torch.cat([x, cond], dim=1)  # [B, 68, F, T]

        # Patch embedding
        x_patches, grid_size, orig_size = self.patch_embed(x_cat)  # [B, N_patches, embed_dim]

        # Add positional encoding to patches
        pos = self.pos_embed[:, :x_patches.shape[1], :]
        x_patches = x_patches + pos

        # Prepare token conditioning
        if tokens is not None:
            tokens_proj = self.token_proj(tokens)
            n_tok = tokens_proj.shape[1]
            pos = self.token_pos_embed[:, :n_tok, :]
            tokens_proj = tokens_proj + pos
        else:
            # Fallback: create dummy tokens if not provided
            max_tok = self.token_pos_embed.shape[1]
            tokens_proj = torch.zeros(B, max_tok, self.embed_dim, device=device)

        # Apply transformer blocks
        for block in self.blocks:
            x_patches = block(x_patches, tokens_proj, noise_emb)

        # Final layer norm
        x_patches = self.final_ln(x_patches)

        # Unpatchify to reconstruct spectrogram
        output = self.unpatchify(x_patches, grid_size, orig_size)  # [B, 4, F, T]

        return output
