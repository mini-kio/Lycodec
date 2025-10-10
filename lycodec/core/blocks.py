import torch
import torch.nn as nn
import math
from lycodec.utils.audio import resample_time


class ConvBlock2d(nn.Module):
    def __init__(self, c_in, c_out, stride=(1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Patchifier(nn.Module):
    def __init__(self, c_in=4, widths=(64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3 = widths
        self.l1 = ConvBlock2d(c_in, c0, stride=(2, 2))
        self.l2 = ConvBlock2d(c0, c1, stride=(2, 2))
        self.l3 = ConvBlock2d(c1, c2, stride=(2, 2))
        self.l4 = ConvBlock2d(c2, c3, stride=(2, 1))

    def forward(self, x):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return f4, (f1, f2, f3)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.1, use_rope=True, rope=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

        # RoPE integration
        self.use_rope = use_rope
        self.rope = rope  # Shared RoPE module

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: input [B, T, D]
            attn_mask: attention mask [T, T] or None
                      True/1 = allow attention, False/0 = mask out
        """
        x_norm = self.ln1(x)

        # Apply RoPE to query and key (not value!)
        if self.use_rope and self.rope is not None:
            # For MultiheadAttention, we apply RoPE to the input
            # The attention mechanism will project it to Q, K, V
            # Ideally, RoPE should only be applied to Q and K, but for simplicity
            # we apply it to the normalized input (which affects both Q and K)
            x_norm_rope = self.rope(x_norm)
            # Use rotated input as query and key, original as value
            x = x + self.attn(x_norm_rope, x_norm_rope, x_norm, attn_mask=attn_mask, need_weights=False)[0]
        else:
            x = x + self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)[0]

        x = x + self.ff(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, depth=8, heads=8, dropout=0.1, use_checkpoint=False, seq_len=18, chunk_size=9, use_rope=True):
        super().__init__()

        # Initialize RoPE (shared across all blocks)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(dim=dim, max_seq_len=seq_len * 2)  # 2x for safety
            print(f"[TransformerEncoder] RoPE enabled: dim={dim}, max_seq_len={seq_len}")
        else:
            self.rope = None

        # Create transformer blocks with shared RoPE
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, 4, dropout, use_rope=use_rope, rope=self.rope)
            for _ in range(depth)
        ])

        self.use_checkpoint = use_checkpoint
        self.seq_len = seq_len
        self.chunk_size = chunk_size

        # Create chunked causal mask (left/right chunks)
        self.register_buffer("chunked_mask", self._create_chunked_causal_mask(seq_len, chunk_size))

    def _create_chunked_causal_mask(self, seq_len, chunk_size):
        """
        Create chunked causal attention mask for left/right chunks.

        For seq_len=18, chunk_size=9:
        - Left chunk (0-8): can only attend to left chunk
        - Right chunk (9-17): can attend to both left and right chunks (causal)

        Mask format:
        - 0.0 = allow attention
        - -inf = mask out (PyTorch MultiheadAttention convention)

        Returns:
            mask [seq_len, seq_len]
        """
        mask = torch.zeros(seq_len, seq_len)

        # Left chunk: can only attend to left chunk
        mask[:chunk_size, chunk_size:] = float('-inf')

        # Right chunk: can attend to all (left + right)
        # No masking needed for right chunk

        return mask

    def forward(self, x, use_mask=True):
        """
        Args:
            x: input [B, T, D]
            use_mask: whether to use chunked causal mask (default True during training)
        """
        attn_mask = self.chunked_mask if use_mask else None

        if not self.use_checkpoint:
            for b in self.blocks:
                x = b(x, attn_mask=attn_mask)
            return x
        import torch.utils.checkpoint as ckpt
        for b in self.blocks:
            # Note: checkpoint doesn't support kwargs well, so we pass mask as arg
            x = ckpt.checkpoint(b, x, attn_mask, use_reentrant=False)
        return x


class TemporalResampler(nn.Module):
    def __init__(self, c_in, t_out):
        super().__init__()
        self.t_out = t_out
        self.conv = nn.Conv1d(c_in, c_in, 7, padding=3)
        self.proj = nn.Linear(c_in, c_in)

    def forward(self, x):
        x = self.conv(x)
        x = resample_time(x, self.t_out)
        x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        return x


class FSQQuantizer(nn.Module):
    def __init__(self, levels=11, dim=256, dropout_p=0.65):
        super().__init__()
        self.N = (levels - 1) // 2
        self.dim = dim
        self.dropout_p = dropout_p

    def forward(self, z, training=True):
        z_tanh = torch.tanh(z)
        if not training:
            z_q = torch.round(self.N * z_tanh) / self.N
            z_q = z + (z_q - z).detach()
            return z_tanh, z_q
        if torch.rand(1, device=z.device).item() < self.dropout_p:
            return z_tanh, None
        z_q = torch.round(self.N * z_tanh) / self.N
        z_q = z + (z_q - z).detach()
        return z_q, z_q


class GroupFSQ(nn.Module):
    """
    Group-wise Finite Scalar Quantization.

    Splits the input dimension into multiple groups and applies different
    quantization levels to each group. This allows for more flexible
    representation with the same total bitrate.

    Example:
        256 dim with 4 groups = [64, 64, 64, 64] dims per group
        Each group can have different levels: [7, 11, 11, 15]
        Group 0 (coarse): 7 levels for low-frequency content
        Group 1-2 (mid): 11 levels for general features
        Group 3 (fine): 15 levels for high-frequency details
    """
    def __init__(self, num_groups=4, levels=None, dim=256, dropout_p=0.65):
        super().__init__()
        assert dim % num_groups == 0, f"dim ({dim}) must be divisible by num_groups ({num_groups})"

        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.dim = dim
        self.dropout_p = dropout_p

        # Default: all groups use same levels
        if levels is None:
            levels = [11] * num_groups
        elif isinstance(levels, int):
            levels = [levels] * num_groups

        assert len(levels) == num_groups, f"levels list length ({len(levels)}) must match num_groups ({num_groups})"

        self.levels = levels
        self.N_values = [(L - 1) // 2 for L in levels]  # N for each group

        print(f"[GroupFSQ] Initialized with {num_groups} groups:")
        for i, (L, N) in enumerate(zip(self.levels, self.N_values)):
            print(f"  Group {i}: {self.group_dim} dims, {L} levels (N={N})")
        total_bits = sum(math.log2(L) for L in self.levels) * self.group_dim
        print(f"  Total bits per token: {total_bits:.1f} ({total_bits/self.dim:.2f} bits/dim)")

    def forward(self, z, training=True):
        """
        Args:
            z: input [B, T, D]
            training: whether in training mode

        Returns:
            z_cont: continuous (tanh) representation [B, T, D]
            z_disc: discrete (quantized) representation [B, T, D] or None (if dropped out)
        """
        B, T, D = z.shape
        assert D == self.dim, f"Input dim ({D}) doesn't match expected dim ({self.dim})"

        # Split into groups along the feature dimension
        groups = z.chunk(self.num_groups, dim=-1)  # List of [B, T, group_dim]

        # Apply tanh to each group (continuous representation)
        groups_tanh = [torch.tanh(g) for g in groups]
        z_tanh = torch.cat(groups_tanh, dim=-1)  # [B, T, D]

        # Quantization
        if not training:
            # Inference: quantize all groups
            groups_q = []
            for g_tanh, N in zip(groups_tanh, self.N_values):
                g_q = torch.round(N * g_tanh) / N
                # Straight-through estimator: gradient flows through g_tanh
                g_q = g_tanh + (g_q - g_tanh).detach()
                groups_q.append(g_q)
            z_q = torch.cat(groups_q, dim=-1)
            return z_tanh, z_q

        # Training: stochastic dropout of discrete path
        if torch.rand(1, device=z.device).item() < self.dropout_p:
            return z_tanh, None  # Use continuous path only

        # Quantize each group with its own levels
        groups_q = []
        for g_tanh, N in zip(groups_tanh, self.N_values):
            g_q = torch.round(N * g_tanh) / N
            g_q = g_tanh + (g_q - g_tanh).detach()
            groups_q.append(g_q)

        z_q = torch.cat(groups_q, dim=-1)
        return z_q, z_q


class HybridLatent(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.residual_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, z_cont, z_quant):
        if z_quant is None:
            return z_cont, z_cont
        residual = z_cont - z_quant
        residual = self.residual_net(residual)
        z_h = z_quant + self.alpha * residual
        return z_h, residual


class StereoHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.ild = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh())
        self.itd = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh())
        self.side_res = nn.Linear(dim, dim)

    def forward(self, z):
        return {
            "ild": self.ild(z),
            "itd": self.itd(z),
            "side_residual": self.side_res(z),
        }


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Instead of adding position encodings, RoPE rotates the query and key vectors
    based on their absolute positions. This preserves relative position information
    while being more efficient than learned position embeddings.

    Key properties:
    - Encodes absolute position with rotation matrix
    - Naturally captures relative positions through inner products
    - No additional parameters (uses precomputed frequencies)
    - Works well for extrapolation to longer sequences

    Example:
        rope = RotaryPositionEmbedding(dim=512, max_seq_len=18)
        x = torch.randn(8, 18, 512)  # [B, T, D]
        x_rotated = rope(x)
    """
    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even for RoPE, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        # inv_freq[i] = 1 / (base ^ (2i / dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin values (initialized on first forward pass)
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.zeros(1, 1, 1))
        self.register_buffer("sin_cached", torch.zeros(1, 1, 1))

    def _update_cache(self, seq_len, device):
        """Update cached cos/sin values if sequence length changes."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Create position indices [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

            # Compute frequencies: outer product of positions and inv_freq
            # freqs[t, i] = t * inv_freq[i]
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]

            # Duplicate frequencies for cos/sin application to pairs
            # [f0, f0, f1, f1, f2, f2, ...] instead of [f0, f1, f2, ...]
            emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

            # Precompute cos and sin
            self.cos_cached = emb.cos().unsqueeze(0)  # [1, seq_len, dim]
            self.sin_cached = emb.sin().unsqueeze(0)  # [1, seq_len, dim]

    def _rotate_half(self, x):
        """
        Rotate half the hidden dims of the input.

        For a vector [x0, x1, x2, x3, ...], this returns [-x1, x0, -x3, x2, ...].
        This is the core rotation operation for RoPE.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        """
        Apply rotary position embedding to input.

        Args:
            x: input tensor [B, seq_len, dim]

        Returns:
            x_rotated: [B, seq_len, dim] with position information encoded via rotation
        """
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device)

        # Get cached cos/sin for current sequence length
        cos = self.cos_cached[:, :seq_len, :]  # [1, seq_len, dim]
        sin = self.sin_cached[:, :seq_len, :]  # [1, seq_len, dim]

        # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        # This is equivalent to multiplying by a rotation matrix
        return x * cos + self._rotate_half(x) * sin
