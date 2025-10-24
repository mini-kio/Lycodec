import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, dim=512, depth=8, heads=8, dropout=0.1, use_checkpoint=False, seq_len=18, chunk_size=None, use_rope=True):
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
        # Auto chunk_size = half of seq_len
        self.chunk_size = chunk_size if chunk_size is not None else seq_len // 2

        # Create chunked causal mask (left/right chunks)
        self.register_buffer("chunked_mask", self._create_chunked_causal_mask(seq_len, self.chunk_size))

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


class RVQQuantizer(nn.Module):
    """Residual Vector Quantizer with EMA updates and dropout scheduling."""

    def __init__(self, dim=256, codebook_size=4096, ema_decay=0.99, awakening_steps=2000,
                 gumbel_temp=1.0, drop_start=0.6, drop_end=0.1, drop_decay_steps=200000):
        super().__init__()
        self.dim = dim
        self.K = codebook_size
        self.ema_decay = ema_decay
        self.awakening_steps = awakening_steps
        self.gumbel_temp = gumbel_temp
        self.drop_start = float(drop_start)
        self.drop_end = float(drop_end)
        self.drop_decay_steps = int(drop_decay_steps)

        self.register_buffer('codebook', torch.randn(codebook_size, dim))
        self.register_buffer('codebook_usage', torch.zeros(codebook_size))
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embed_avg', torch.zeros(codebook_size, dim))

        print(f"[RVQ] Initialized: K={codebook_size}, dim={dim}, drop {self.drop_start}->{self.drop_end}")

    def _compute_distances(self, z):
        z_flat = z.reshape(-1, self.dim)
        z_norm_sq = (z_flat ** 2).sum(dim=-1, keepdim=True)
        e_norm_sq = (self.codebook ** 2).sum(dim=-1, keepdim=True).t()
        dot = z_flat @ self.codebook.t()
        distances = z_norm_sq + e_norm_sq - 2 * dot
        return distances.reshape(z.shape[0], z.shape[1], self.K)

    def _gumbel_softmax_st(self, logits, tau):
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        logits_with_gumbel = (logits + gumbel) / tau
        soft_codes = F.softmax(logits_with_gumbel, dim=-1)
        indices = soft_codes.argmax(dim=-1)
        hard_codes = F.one_hot(indices, num_classes=self.K).float()
        codes = hard_codes - soft_codes.detach() + soft_codes
        return codes, indices

    def _update_ema(self, z, indices):
        if not self.training:
            return
        z_flat = z.reshape(-1, self.dim)
        indices_flat = indices.reshape(-1)
        encodings = F.one_hot(indices_flat, num_classes=self.K).float()
        n_i = encodings.sum(dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(n_i, alpha=1 - self.ema_decay)
        sum_i = encodings.t() @ z_flat
        self.ema_embed_avg.mul_(self.ema_decay).add_(sum_i, alpha=1 - self.ema_decay)
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
        self.codebook.copy_(self.ema_embed_avg / cluster_size.unsqueeze(1))
        self.codebook_usage.mul_(0.99)
        self.codebook_usage += n_i

    def _awaken_dead_codes(self, z):
        if not self.training:
            return
        dead_mask = self.codebook_usage < 1.0
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return
        z_flat = z.reshape(-1, self.dim)
        random_idx = torch.randperm(z_flat.shape[0], device=z.device)[:num_dead]
        self.codebook[dead_mask] = z_flat[random_idx] + torch.randn(num_dead, self.dim, device=z.device) * 0.01
        self.codebook_usage[dead_mask] = 1.0
        self.ema_cluster_size[dead_mask] = 1.0
        self.ema_embed_avg[dead_mask] = self.codebook[dead_mask]

    def _current_drop_prob(self):
        if self.drop_decay_steps <= 0:
            return self.drop_end
        step = int(self.step_counter.item())
        if step >= self.drop_decay_steps:
            return self.drop_end
        ratio = 1.0 - (step / float(self.drop_decay_steps))
        return self.drop_end + (self.drop_start - self.drop_end) * ratio

    def current_drop_prob(self):
        return float(self._current_drop_prob())

    def forward(self, z, training=True):
        B, T, _ = z.shape
        distances = self._compute_distances(z)
        if training:
            logits = -distances
            codes, indices = self._gumbel_softmax_st(logits, self.gumbel_temp)
            quantized = torch.einsum('btk,kd->btd', codes, self.codebook)
            hard_indices = indices
            embedding = F.embedding(hard_indices, self.codebook)
            self._update_ema(z, hard_indices)
            self.step_counter += 1
            if self.step_counter % self.awakening_steps == 0:
                self._awaken_dead_codes(z)
            drop_prob = self._current_drop_prob()
        else:
            hard_indices = distances.argmin(dim=-1)
            embedding = F.embedding(hard_indices, self.codebook)
            quantized = embedding
            drop_prob = 0.0
        z_q = z + (quantized - z).detach()
        commitment = F.mse_loss(z_q.detach(), z)
        usage_sum = self.codebook_usage.sum()
        if usage_sum > 0:
            usage_prob = self.codebook_usage / (usage_sum + 1e-10)
        else:
            usage_prob = torch.full_like(self.codebook_usage, 1.0 / self.K)
        entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
        perplexity = torch.exp(entropy)
        return {
            'z_q': z_q,
            'embedding': embedding,
            'indices': hard_indices,
            'commitment_loss': commitment,
            'perplexity': perplexity,
            'usage_entropy': entropy,
            'drop_prob': float(drop_prob),
        }


class ResidualCorrector(nn.Module):
    """Predict residual offsets from codebook indices only."""

    def __init__(self, dim=256, codebook_size=4096, context_size=5):
        super().__init__()
        self.context_size = context_size
        self.index_embed = nn.Embedding(codebook_size, dim)
        self.context_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=context_size, padding=context_size // 2),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=context_size, padding=context_size // 2),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
        print(f"[ResidualCorrector] Indices-only, context={context_size}")

    def forward(self, indices):
        if indices is None:
            return None
        x = self.index_embed(indices).transpose(1, 2)
        context = self.context_conv(x).transpose(1, 2)
        return self.predictor(context)



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
