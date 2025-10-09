import torch
import torch.nn as nn
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
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.1):
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

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: input [B, T, D]
            attn_mask: attention mask [T, T] or None
                      True/1 = allow attention, False/0 = mask out
        """
        x_norm = self.ln1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.ff(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, depth=8, heads=8, dropout=0.1, use_checkpoint=False, seq_len=18, chunk_size=9):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, 4, dropout) for _ in range(depth)])
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
