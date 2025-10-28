import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lycodec.utils.audio import resample_time


# ============================================================================
# Common Neural Network Building Blocks
# ============================================================================

class ConvNormAct2d(nn.Module):
    """Conv2d -> GroupNorm -> SiLU activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=8,
        bias=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvNormAct1d(nn.Module):
    """Conv1d -> GroupNorm -> SiLU activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=8,
        bias=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvTransposeNormAct2d(nn.Module):
    """ConvTranspose2d -> GroupNorm -> SiLU activation block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        groups=8,
        bias=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Encoder Components
# ============================================================================

class ConvBlock2d(nn.Module):
    def __init__(self, c_in, c_out, stride=(1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
            ConvNormAct2d(c_out, c_out, kernel_size=3, padding=1),
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


class OPQPQQuantizer(nn.Module):
    """
    Optimized Product Quantization with learnable rotation (OPQ-PQ).

    Applies learnable orthogonal rotation before splitting into M subspaces.
    Each subspace is quantized independently with its own codebook.
    More efficient for LLM downstream (fewer tokens) than multi-stage RVQ.

    Key features:
    - Learnable rotation matrix W (D × D) for optimal subspace alignment
    - Orthogonal regularization to maintain W^T W ≈ I
    - Periodic QR decomposition for stability
    - M independent codebooks with EMA updates
    - Gumbel-Softmax for differentiable quantization

    Args:
        dim: embedding dimension (must be divisible by M)
        M: number of subspaces (e.g., 4)
        K: codebook size per subspace (e.g., 256 for 8bit)
        ema_decay: EMA decay for codebook updates
        awakening_steps: steps between dead code resets
        gumbel_temp: initial Gumbel-Softmax temperature
        drop_start/end: dropout probability schedule
        ortho_penalty: weight for W^T W - I regularization
        qr_every: steps between QR decomposition (0 = disabled)
    """

    def __init__(self, dim=256, M=4, K=256, ema_decay=0.97, awakening_steps=200,
                 gumbel_temp=1.0, drop_start=0.6, drop_end=0.1, drop_decay_steps=200000,
                 ortho_penalty=1e-4, qr_every=500):
        super().__init__()
        assert dim % M == 0, f"dim ({dim}) must be divisible by M ({M})"

        self.dim = dim
        self.M = M  # number of subspaces
        self.K = K  # codebook size per subspace
        self.D_sub = dim // M  # subspace dimension
        self.ema_decay = ema_decay
        self.awakening_steps = awakening_steps
        self.gumbel_temp = gumbel_temp
        self.drop_start = float(drop_start)
        self.drop_end = float(drop_end)
        self.drop_decay_steps = int(drop_decay_steps)
        self.ortho_penalty = ortho_penalty
        self.qr_every = qr_every

        # Learnable rotation matrix (initialized as identity)
        self.W = nn.Parameter(torch.eye(dim))

        # M independent codebooks (each K x D_sub)
        for m in range(M):
            self.register_buffer(f'codebook_{m}', torch.randn(K, self.D_sub))
            self.register_buffer(f'codebook_usage_{m}', torch.zeros(K))
            self.register_buffer(f'ema_cluster_size_{m}', torch.zeros(K))
            self.register_buffer(f'ema_embed_avg_{m}', torch.zeros(K, self.D_sub))

        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))

        bits_per_token = M * math.log2(K)
        print(f"[OPQ-PQ] Initialized: M={M}, K={K}, dim={dim}, D_sub={self.D_sub}")
        print(f"[OPQ-PQ] Bits/token={bits_per_token:.1f}, ortho_penalty={ortho_penalty}, qr_every={qr_every}")
        print(f"[OPQ-PQ] Dropout schedule: {self.drop_start}->{self.drop_end} over {drop_decay_steps} steps")

    def _get_codebook(self, m):
        """Get codebook for subspace m"""
        return getattr(self, f'codebook_{m}')

    def _get_usage(self, m):
        """Get usage for subspace m"""
        return getattr(self, f'codebook_usage_{m}')

    def _compute_distances(self, z_m, codebook_m):
        """
        Compute distances for one subspace using cosine distance.

        Args:
            z_m: [B*T, D_sub]
            codebook_m: [K, D_sub]
        Returns:
            distances: [B*T, K]
        """
        # L2 normalize
        z_norm = F.normalize(z_m, dim=-1)
        codebook_norm = F.normalize(codebook_m, dim=-1)
        # Cosine distance
        cosine_sim = z_norm @ codebook_norm.t()
        distances = 1.0 - cosine_sim
        return distances

    def _gumbel_softmax_st(self, logits, tau):
        """Gumbel-Softmax with straight-through estimator"""
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        logits_with_gumbel = (logits + gumbel) / tau
        soft_codes = F.softmax(logits_with_gumbel, dim=-1)
        indices = soft_codes.argmax(dim=-1)
        hard_codes = F.one_hot(indices, num_classes=self.K).float()
        codes = hard_codes - soft_codes.detach() + soft_codes
        return codes, indices

    def _update_ema(self, m, z_m, indices_m):
        """Update EMA for subspace m"""
        if not self.training:
            return

        with torch.no_grad():
            z_flat = z_m.reshape(-1, self.D_sub)
            indices_flat = indices_m.reshape(-1)
            encodings = F.one_hot(indices_flat, num_classes=self.K).float()

            n_i = encodings.sum(dim=0)

            # Update EMA
            ema_cluster_size = getattr(self, f'ema_cluster_size_{m}')
            ema_embed_avg = getattr(self, f'ema_embed_avg_{m}')
            codebook = getattr(self, f'codebook_{m}')
            usage = getattr(self, f'codebook_usage_{m}')

            ema_cluster_size.mul_(self.ema_decay).add_(n_i, alpha=1 - self.ema_decay)
            sum_i = encodings.t() @ z_flat
            ema_embed_avg.mul_(self.ema_decay).add_(sum_i, alpha=1 - self.ema_decay)

            n = ema_cluster_size.sum()
            cluster_size = (ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
            codebook.copy_(ema_embed_avg / cluster_size.unsqueeze(1))

            usage.mul_(0.99)
            usage.add_(n_i)

    def _awaken_dead_codes(self, m, z_m):
        """Awaken dead codes for subspace m"""
        if not self.training:
            return

        with torch.no_grad():
            usage = getattr(self, f'codebook_usage_{m}')
            codebook = getattr(self, f'codebook_{m}')
            ema_cluster_size = getattr(self, f'ema_cluster_size_{m}')
            ema_embed_avg = getattr(self, f'ema_embed_avg_{m}')

            usage_threshold = usage.mean() * 0.1
            dead_mask = usage < usage_threshold
            num_dead = int(dead_mask.sum().item())
            if num_dead == 0:
                return

            min_reset = max(5, int(self.K * 0.05))
            if num_dead < min_reset:
                _, indices = torch.topk(usage, min_reset, largest=False)
                dead_mask = torch.zeros(self.K, dtype=torch.bool, device=z_m.device)
                dead_mask[indices] = True
                num_dead = min_reset

            z_flat = z_m.reshape(-1, self.D_sub)
            if num_dead > z_flat.shape[0]:
                random_idx = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_m.device)
            else:
                random_idx = torch.randperm(z_flat.shape[0], device=z_m.device)[:num_dead]

            codebook[dead_mask] = z_flat[random_idx] + torch.randn(num_dead, self.D_sub, device=z_m.device) * 0.02
            usage[dead_mask] = usage.mean()
            ema_cluster_size[dead_mask] = 1.0
            ema_embed_avg[dead_mask] = codebook[dead_mask]

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

    def orthogonal_regularization(self):
        """Compute W^T W - I Frobenius norm for orthogonality loss"""
        WtW = self.W.T @ self.W
        I = torch.eye(self.dim, device=self.W.device, dtype=self.W.dtype)
        return torch.norm(WtW - I, p='fro') ** 2

    def apply_qr_decomposition(self):
        """Apply QR decomposition to W for re-orthogonalization"""
        with torch.no_grad():
            Q, R = torch.linalg.qr(self.W)
            # Ensure positive diagonal in R for uniqueness
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1
            Q = Q * signs.unsqueeze(0)
            self.W.copy_(Q)

    def forward(self, z, training=True):
        """
        Args:
            z: continuous latent [B, T, D]
            training: whether in training mode

        Returns:
            dict with keys:
                - z_q: quantized embedding [B, T, D]
                - embedding: quantized embedding (same as z_q)
                - indices: subspace indices [B, T, M]
                - commitment_loss: commitment loss scalar
                - ortho_loss: orthogonal regularization loss
                - perplexity: average perplexity across subspaces
                - usage_entropy: average entropy across subspaces
                - drop_prob: current dropout probability
                - active_codes: average active codes across subspaces
                - current_tau: current Gumbel temperature
        """
        B, T, D = z.shape
        device = z.device

        # Apply OPQ rotation
        z_flat = z.reshape(-1, D)  # [B*T, D]
        z_rot_flat = z_flat @ self.W  # [B*T, D]
        z_rot = z_rot_flat.reshape(B, T, D)  # [B, T, D]

        # Split into M subspaces
        z_split = z_rot.reshape(B, T, self.M, self.D_sub)  # [B, T, M, D_sub]

        indices_list = []
        embeddings_list = []
        commitment_losses = []
        perplexities = []
        entropies = []
        active_codes_list = []

        # Gumbel temperature annealing
        if training:
            tau_hi = getattr(self, 'tau_hi', 2.0)
            tau_lo = getattr(self, 'tau_lo', 0.5)
            decay_steps = getattr(self, 'tau_decay_steps', 10000.0)
            step = self.step_counter.float()
            current_tau = tau_lo + (tau_hi - tau_lo) * torch.exp(-step / decay_steps)
        else:
            current_tau = torch.tensor(0.5, device=device)

        # Process each subspace independently
        for m in range(self.M):
            z_m = z_split[:, :, m, :]  # [B, T, D_sub]
            codebook_m = self._get_codebook(m)

            z_m_flat = z_m.reshape(-1, self.D_sub)  # [B*T, D_sub]
            distances = self._compute_distances(z_m_flat, codebook_m)  # [B*T, K]
            distances = distances.reshape(B, T, self.K)

            if training:
                logits = -distances
                codes, indices_m = self._gumbel_softmax_st(logits, current_tau.item())
                quantized_m = torch.einsum('btk,kd->btd', codes, codebook_m)
                hard_indices_m = indices_m
                embedding_m = F.embedding(hard_indices_m, codebook_m)

                self._update_ema(m, z_m, hard_indices_m)

                with torch.no_grad():
                    step = int(self.step_counter.item())
                    if self.awakening_steps > 0 and step > 0 and (step % self.awakening_steps == 0):
                        self._awaken_dead_codes(m, z_m)
            else:
                hard_indices_m = distances.argmin(dim=-1)
                embedding_m = F.embedding(hard_indices_m, codebook_m)
                quantized_m = embedding_m

            indices_list.append(hard_indices_m)
            embeddings_list.append(embedding_m)

            # Commitment loss for this subspace
            commitment_m = F.mse_loss(quantized_m.detach(), z_m)
            commitment_losses.append(commitment_m)

            # Statistics
            usage_m = self._get_usage(m)
            usage_sum = usage_m.sum()
            if usage_sum > 0:
                usage_prob = usage_m / (usage_sum + 1e-10)
            else:
                usage_prob = torch.full_like(usage_m, 1.0 / self.K)
            entropy_m = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
            perplexity_m = torch.exp(entropy_m)

            perplexities.append(perplexity_m.detach())
            entropies.append(entropy_m.detach())

            if training:
                active_codes_m = hard_indices_m.unique().numel()
                active_codes_list.append(active_codes_m)

        # Concatenate subspaces (in rotated space)
        z_q_cat_rot = torch.cat(embeddings_list, dim=-1)  # [B, T, D] in rotated space
        indices_stacked = torch.stack(indices_list, dim=-1)  # [B, T, M]

        # Inverse rotation: W^T to original space
        z_q_cat_rot_flat = z_q_cat_rot.reshape(-1, D)  # [B*T, D]
        z_q_cat_flat = z_q_cat_rot_flat @ self.W.T  # [B*T, D] inverse rotation
        z_q_cat = z_q_cat_flat.reshape(B, T, D)  # [B, T, D] in original space

        # Straight-through estimator
        z_q = z + (z_q_cat - z).detach()

        # Aggregate losses and stats
        commitment_loss = sum(commitment_losses) / len(commitment_losses)
        mean_perplexity = torch.stack(perplexities).mean()
        mean_entropy = torch.stack(entropies).mean()

        # Orthogonal regularization
        ortho_loss = self.orthogonal_regularization() * self.ortho_penalty if training else torch.tensor(0.0, device=device)

        # QR decomposition for W re-orthogonalization (periodic)
        if training and self.qr_every > 0:
            step = int(self.step_counter.item())
            if step > 0 and step % self.qr_every == 0:
                self.apply_qr_decomposition()

        # Increment step counter
        if training:
            with torch.no_grad():
                self.step_counter += 1

        drop_prob = self._current_drop_prob() if training else 0.0
        active_codes = sum(active_codes_list) / len(active_codes_list) if training else 0

        # Entropy bonus (only in early training)
        entropy_bonus = torch.tensor(0.0, device=device)
        if training:
            step = int(self.step_counter.item())
            if step < 20000:
                bonus_weight = 0.05 * (1.0 - step / 20000.0)
                entropy_bonus = -bonus_weight * mean_entropy

        return {
            'z_q': z_q,
            'embedding': z_q_cat,
            'indices': indices_stacked,  # [B, T, M]
            'commitment_loss': commitment_loss * 0.5,  # same weight as RVQ
            'ortho_loss': ortho_loss,  # orthogonal regularization
            'entropy_bonus': entropy_bonus,
            'perplexity': mean_perplexity,
            'usage_entropy': mean_entropy,
            'drop_prob': float(drop_prob),
            'active_codes': active_codes,
            'current_tau': current_tau.item() if training else 0.0,
        }


class RVQQuantizer(nn.Module):
    """Residual Vector Quantizer with EMA updates and dropout scheduling."""

    def __init__(self, dim=256, codebook_size=4096, ema_decay=0.97, awakening_steps=200,
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
        """
        Compute distances using L2 normalized vectors and cosine distance.
        This helps prevent codebook collapse by maintaining better code utilization.
        """
        z_flat = z.reshape(-1, self.dim)
        # L2 normalize both input and codebook
        z_norm = F.normalize(z_flat, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)
        # Cosine distance = 1 - cosine_similarity
        # This is more stable than euclidean distance
        cosine_sim = z_norm @ codebook_norm.t()
        distances = 1.0 - cosine_sim
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
        with torch.no_grad():
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
        """
        More aggressive awakening strategy to prevent codebook collapse.
        Resets codes below 10% of average usage (was 1.0 absolute threshold).
        """
        if not self.training:
            return
        with torch.no_grad():
            # Use relative threshold (10% of average usage) instead of absolute
            usage_threshold = self.codebook_usage.mean() * 0.1
            dead_mask = self.codebook_usage < usage_threshold
            num_dead = int(dead_mask.sum().item())
            if num_dead == 0:
                return

            # Reset at least 10 codes or 5% of codebook, whichever is larger
            min_reset = max(10, int(self.K * 0.05))
            if num_dead < min_reset:
                # Find additional low-usage codes to reset
                _, indices = torch.topk(self.codebook_usage, min_reset, largest=False)
                dead_mask = torch.zeros(self.K, dtype=torch.bool, device=z.device)
                dead_mask[indices] = True
                num_dead = min_reset

            z_flat = z.reshape(-1, self.dim)
            # Handle case where num_dead > available samples (use replacement sampling)
            if num_dead > z_flat.shape[0]:
                random_idx = torch.randint(0, z_flat.shape[0], (num_dead,), device=z.device)
            else:
                random_idx = torch.randperm(z_flat.shape[0], device=z.device)[:num_dead]

            # Reset codes with small noise
            self.codebook[dead_mask] = z_flat[random_idx] + torch.randn(num_dead, self.dim, device=z.device) * 0.02
            self.codebook_usage[dead_mask] = self.codebook_usage.mean()  # Set to average, not 1.0
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
            # Gumbel temperature annealing (read from config if available)
            tau_hi = getattr(self, 'tau_hi', 2.0)
            tau_lo = getattr(self, 'tau_lo', 0.5)
            decay_steps = getattr(self, 'tau_decay_steps', 10000.0)
            step = self.step_counter.float()
            current_tau = tau_lo + (tau_hi - tau_lo) * torch.exp(-step / decay_steps)

            logits = -distances
            codes, indices = self._gumbel_softmax_st(logits, current_tau.item())
            quantized = torch.einsum('btk,kd->btd', codes, self.codebook)
            hard_indices = indices
            embedding = F.embedding(hard_indices, self.codebook)
            self._update_ema(z, hard_indices)

            with torch.no_grad():
                self.step_counter += 1

            # Safe tensor comparison for awakening
            step = int(self.step_counter.item())
            if self.awakening_steps > 0 and step > 0 and (step % self.awakening_steps == 0):
                self._awaken_dead_codes(z)

            drop_prob = self._current_drop_prob()
        else:
            hard_indices = distances.argmin(dim=-1)
            embedding = F.embedding(hard_indices, self.codebook)
            quantized = embedding
            drop_prob = 0.0
        z_q = z + (quantized - z).detach()

        # Stronger commitment loss to prevent encoder drift
        commitment = F.mse_loss(z_q.detach(), z) * 0.5  # Increased from implicit 0.25

        # Calculate codebook statistics
        usage_sum = self.codebook_usage.sum()
        if usage_sum > 0:
            usage_prob = self.codebook_usage / (usage_sum + 1e-10)
        else:
            usage_prob = torch.full_like(self.codebook_usage, 1.0 / self.K)
        entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
        perplexity = torch.exp(entropy)

        # Entropy bonus: encourage diverse code usage during training
        # Only apply in early training (first 20k steps)
        entropy_bonus = torch.tensor(0.0, device=z.device)
        if training:
            step = int(self.step_counter.item())
            if step < 20000:
                # Bonus decays from 0.05 to 0 over 20k steps
                bonus_weight = 0.05 * (1.0 - step / 20000.0)
                entropy_bonus = -bonus_weight * entropy  # Negative to maximize entropy

        # Track active codes in current batch
        active_codes = hard_indices.unique().numel() if training else 0

        return {
            'z_q': z_q,
            'embedding': embedding,
            'indices': hard_indices,
            'commitment_loss': commitment,
            'entropy_bonus': entropy_bonus,
            'perplexity': perplexity,
            'usage_entropy': entropy,
            'drop_prob': float(drop_prob),
            'active_codes': active_codes,
            'current_tau': current_tau.item() if training else 0.0,
        }


class MultiStageRVQ(nn.Module):
    """
    Multi-stage Residual Vector Quantizer with frame-level embedding fusion.

    Key features:
    - Cascaded RVQ stages: each stage quantizes residual from previous stage
    - Frame embedding fusion: combines embeddings from all stages into single token per frame
    - Maintains sequence length: T frames → T tokens (not T×num_stages)
    - Stage-specific training strategies (commitment, Gumbel τ, dropout)

    Args:
        num_stages: number of RVQ stages (default 2 for Phase 2)
        dim: embedding dimension
        codebook_size: size of each stage's codebook (typically 4096)
        stage_commitment_weights: commitment loss multiplier per stage [stage1_mult, stage2_mult, ...]
        stage_tau_configs: Gumbel temperature config per stage [(tau_hi, tau_lo, decay_steps), ...]
        stage_drop_configs: dropout config per stage [(drop_start, drop_end, decay_steps), ...]
        fusion_mode: 'weighted_sum' or 'concat_linear'
    """

    def __init__(
        self,
        num_stages=2,
        dim=256,
        codebook_size=4096,
        ema_decay=0.97,
        awakening_steps=200,
        stage_commitment_weights=None,  # e.g. [1.25, 0.5] for [stage1, stage2]
        stage_tau_configs=None,  # e.g. [(2.0, 0.5, 10000), (2.5, 0.5, 20000)]
        stage_drop_configs=None,  # e.g. [(0.0, 0.0, 0), (0.6, 0.1, 200000)]
        fusion_mode='weighted_sum',
    ):
        super().__init__()
        self.num_stages = num_stages
        self.dim = dim
        self.fusion_mode = fusion_mode

        # Default stage-specific configs if not provided
        if stage_commitment_weights is None:
            # Stage 1: stronger commitment (1.25x), Stage 2+: weaker (0.5x)
            stage_commitment_weights = [1.25] + [0.5] * (num_stages - 1)

        if stage_tau_configs is None:
            # Stage 1: normal schedule, Stage 2+: longer exploration
            stage_tau_configs = [(2.0, 0.5, 10000)]  # Stage 1
            for _ in range(num_stages - 1):
                stage_tau_configs.append((2.5, 0.5, 20000))  # Stage 2+ (longer decay)

        if stage_drop_configs is None:
            # Stage 1: no dropout, Stage 2+: apply dropout
            stage_drop_configs = [(0.0, 0.0, 0)]  # Stage 1 (no dropout)
            for _ in range(num_stages - 1):
                stage_drop_configs.append((0.6, 0.1, 200000))  # Stage 2+ (with dropout)

        self.stage_commitment_weights = stage_commitment_weights

        # Create RVQ stages
        self.quantizers = nn.ModuleList()
        for stage_idx in range(num_stages):
            tau_hi, tau_lo, tau_decay = stage_tau_configs[stage_idx]
            drop_start, drop_end, drop_decay = stage_drop_configs[stage_idx]

            rvq = RVQQuantizer(
                dim=dim,
                codebook_size=codebook_size,
                ema_decay=ema_decay,
                awakening_steps=awakening_steps,
                gumbel_temp=1.0,  # will be overridden by tau schedule
                drop_start=drop_start,
                drop_end=drop_end,
                drop_decay_steps=drop_decay,
            )

            # Attach stage-specific Gumbel τ config
            rvq.tau_hi = tau_hi
            rvq.tau_lo = tau_lo
            rvq.tau_decay_steps = tau_decay

            self.quantizers.append(rvq)

            print(f"[MultiStageRVQ] Stage {stage_idx+1}: commitment={stage_commitment_weights[stage_idx]:.2f}, "
                  f"τ=({tau_hi:.1f}→{tau_lo:.1f}, {tau_decay}steps), "
                  f"dropout=({drop_start:.1f}→{drop_end:.1f})")

        # Frame embedding fusion layer
        if fusion_mode == 'weighted_sum':
            # Learnable weights for Σ_j W_j·emb_j
            self.fusion_weights = nn.Parameter(torch.ones(num_stages) / num_stages)
            self.fusion_proj = None
        elif fusion_mode == 'concat_linear':
            # Concat all embeddings then 1×1 linear
            self.fusion_weights = None
            self.fusion_proj = nn.Linear(dim * num_stages, dim)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        print(f"[MultiStageRVQ] {num_stages} stages, fusion={fusion_mode}, codebook={codebook_size}")

    def forward(self, z, training=True):
        """
        Args:
            z: continuous latent [B, T, D]
            training: whether in training mode

        Returns:
            dict with keys:
                - z_q: fused quantized embedding [B, T, D]
                - embeddings: list of embeddings per stage [[B, T, D], ...]
                - indices: list of indices per stage [[B, T], ...]
                - commitment_loss: total weighted commitment loss
                - (other aggregated metrics)
        """
        B, T, D = z.shape
        device = z.device

        z_residual = z
        embeddings = []
        indices = []
        commitment_losses = []

        # Aggregate metrics
        perplexities = []
        entropy_bonuses = []
        usage_entropies = []
        total_active_codes = 0
        total_drop_prob = 0.0
        total_current_tau = 0.0

        # Cascade through stages
        for stage_idx, quantizer in enumerate(self.quantizers):
            out = quantizer(z_residual, training=training)

            emb = out['embedding']  # [B, T, D]
            idx = out['indices']    # [B, T]

            embeddings.append(emb)
            indices.append(idx)

            # Apply stage-specific commitment weight
            stage_commitment = out['commitment_loss'] * self.stage_commitment_weights[stage_idx]
            commitment_losses.append(stage_commitment)

            # Update residual for next stage: z_res = z_res - emb
            z_residual = z_residual - emb.detach()  # detach to prevent gradient leakage

            # Aggregate metrics
            perplexity = out['perplexity']
            if not isinstance(perplexity, torch.Tensor):
                perplexity = torch.tensor(perplexity, device=device)
            perplexities.append(perplexity.detach())

            entropy_bonus = out.get('entropy_bonus', torch.tensor(0.0, device=device))
            if not isinstance(entropy_bonus, torch.Tensor):
                entropy_bonus = torch.tensor(entropy_bonus, device=device)
            entropy_bonuses.append(entropy_bonus.detach())

            usage_entropy = out['usage_entropy']
            if not isinstance(usage_entropy, torch.Tensor):
                usage_entropy = torch.tensor(usage_entropy, device=device)
            usage_entropies.append(usage_entropy.detach())
            total_active_codes += out.get('active_codes', 0)
            total_drop_prob += out.get('drop_prob', 0.0)
            total_current_tau += out.get('current_tau', 0.0)

        # Fuse embeddings (maintains T sequence length)
        if self.fusion_mode == 'weighted_sum':
            # Weighted sum: Σ_j W_j·emb_j
            weights = F.softmax(self.fusion_weights, dim=0)  # normalize
            z_fused = sum(w * emb for w, emb in zip(weights, embeddings))
        elif self.fusion_mode == 'concat_linear':
            # Concat + linear: [emb_1, emb_2, ...] → Linear → [B, T, D]
            z_concat = torch.cat(embeddings, dim=-1)  # [B, T, D*num_stages]
            z_fused = self.fusion_proj(z_concat)  # [B, T, D]

        # Straight-through estimator: stop gradient on quantized, keep gradient on input
        z_q = z + (z_fused - z).detach()

        # Total commitment loss
        commitment_loss = sum(commitment_losses) / len(commitment_losses)

        mean_perplexity = torch.stack(perplexities).mean() if perplexities else torch.tensor(0.0, device=device)
        mean_entropy_bonus = torch.stack(entropy_bonuses).mean() if entropy_bonuses else torch.tensor(0.0, device=device)
        mean_usage_entropy = torch.stack(usage_entropies).mean() if usage_entropies else torch.tensor(0.0, device=device)

        return {
            'z_q': z_q,
            'embedding': z_fused,  # fused embedding for compatibility
            'embeddings': embeddings,  # list of per-stage embeddings
            'indices': indices,  # list of per-stage indices [B, T] × num_stages
            'commitment_loss': commitment_loss,
            'entropy_bonus': mean_entropy_bonus,
            'perplexity': mean_perplexity,
            'usage_entropy': mean_usage_entropy,
            'drop_prob': total_drop_prob / self.num_stages,
            'active_codes': total_active_codes / self.num_stages,
            'current_tau': total_current_tau / self.num_stages,
            'num_stages': self.num_stages,
        }


class PQResidualCorrector(nn.Module):
    """
    Product Quantization Residual Corrector.

    Predicts residual offsets from M subspace indices.
    Each subspace index is embedded independently, then concatenated.

    Args:
        dim: full embedding dimension
        M: number of subspaces
        K: codebook size per subspace
        context_size: conv kernel size for temporal context
    """

    def __init__(self, dim=256, M=4, K=256, context_size=5):
        super().__init__()
        assert dim % M == 0, f"dim ({dim}) must be divisible by M ({M})"

        self.dim = dim
        self.M = M
        self.K = K
        self.D_sub = dim // M
        self.context_size = context_size

        # M separate embeddings for each subspace
        self.subcode_embeds = nn.ModuleList([
            nn.Embedding(K, self.D_sub) for _ in range(M)
        ])

        # Context conv on concatenated embeddings
        self.context_conv = nn.Sequential(
            ConvNormAct1d(dim, dim, kernel_size=context_size, padding=context_size // 2),
            ConvNormAct1d(dim, dim, kernel_size=context_size, padding=context_size // 2),
        )

        # Residual predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

        print(f"[PQResidualCorrector] M={M}, K={K}, D_sub={self.D_sub}, context={context_size}")

    def forward(self, indices):
        """
        Args:
            indices: [B, T, M] - M subspace indices per token

        Returns:
            residual: [B, T, D] - predicted residual correction
        """
        if indices is None:
            return None

        B, T, M = indices.shape
        assert M == self.M, f"Expected M={self.M}, got {M}"

        # Embed each subspace independently
        embs = []
        for m in range(self.M):
            emb_m = self.subcode_embeds[m](indices[:, :, m])  # [B, T, D_sub]
            embs.append(emb_m)

        # Concatenate along feature dim
        x = torch.cat(embs, dim=-1)  # [B, T, D]

        # Apply context conv
        x = x.transpose(1, 2)  # [B, D, T]
        context = self.context_conv(x).transpose(1, 2)  # [B, T, D]

        # Predict residual
        residual = self.predictor(context)

        return residual


class ResidualCorrector(nn.Module):
    """Predict residual offsets from codebook indices only (for RVQ)."""

    def __init__(self, dim=256, codebook_size=4096, context_size=5):
        super().__init__()
        self.context_size = context_size
        self.index_embed = nn.Embedding(codebook_size, dim)
        self.context_conv = nn.Sequential(
            ConvNormAct1d(dim, dim, kernel_size=context_size, padding=context_size // 2),
            ConvNormAct1d(dim, dim, kernel_size=context_size, padding=context_size // 2),
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
    """Rotary Position Embedding (RoPE) for transformer positional encoding."""
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
        """Rotate half the hidden dims: [x0,x1,x2,x3,...] -> [-x1,x0,-x3,x2,...]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        """Apply rotary position embedding. x: [B, seq_len, dim] -> [B, seq_len, dim]"""
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device)

        cos = self.cos_cached[:, :seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :seq_len, :].to(x.dtype)

        return x * cos + self._rotate_half(x) * sin
