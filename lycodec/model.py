import torch
import torch.nn as nn
from lycodec.utils.audio import wav_to_spec, spec_to_wav
from lycodec.core.blocks import (
    Patchifier,
    TransformerEncoder,
    TemporalResampler,
    FSQQuantizer,
    GroupFSQ,
    PartitionedFSQ,
    HybridLatent,
    StereoHead,
)
from lycodec.core.decoders import TokenConditioner, TransformerDecoder2D, BandSplitHead, edm_parameterization


class Lycodec(nn.Module):
    def __init__(self, sr=48000, n_fft=2048, hop=640, win=2048, token_dim=256, hidden=512, layers=8, heads=8, use_checkpoint=False, use_rope=True, use_partitioned_fsq=True, semantic_dim=120, decoder_depth=6, decoder_patch_size=16):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.token_dim = token_dim
        self.use_partitioned_fsq = use_partitioned_fsq

        self.patch = Patchifier(c_in=4, widths=(64, 128, 256, 512))
        self.enc_proj = nn.Conv2d(512, hidden, 1)
        self.resampler = TemporalResampler(hidden, t_out=18)
        self.encoder = TransformerEncoder(dim=hidden, depth=layers, heads=heads, use_checkpoint=use_checkpoint, use_rope=use_rope)
        self.to_token = nn.Linear(hidden, token_dim)

        if use_partitioned_fsq:
            print("[Lycodec] Using Partitioned FSQ (Semantic-Acoustic separation)")
            self.fsq = PartitionedFSQ(dim=token_dim, semantic_dim=semantic_dim, levels=11, dropout_p=0.65)
        else:
            print("[Lycodec] Using Group FSQ with 4 groups")
            self.fsq = GroupFSQ(num_groups=4, levels=[11, 11, 11, 11], dim=token_dim, dropout_p=0.65)

        self.hybrid = HybridLatent(dim=token_dim)
        self.stereo = StereoHead(dim=token_dim)

        self.cond = TokenConditioner(token_dim=token_dim, cond_ch=64, t_out=113, f_bins=self.n_fft // 2 + 1)

        # Transformer decoder
        print(f"[Lycodec] Using TransformerDecoder2D (depth={decoder_depth}, patch_size={decoder_patch_size})")
        self.decoder = TransformerDecoder2D(
            c_in=68,  # 4 (spec) + 64 (cond)
            c_out=4,
            embed_dim=512,
            depth=decoder_depth,
            num_heads=8,
            patch_size=decoder_patch_size,
            token_dim=token_dim,
            mlp_ratio=4.0,
            dropout=0.0,
            target_size=(self.n_fft // 2 + 1, 113),
        )

        self.bands = BandSplitHead(c_in=4, sr=sr, n_fft=n_fft)

    def encode(self, wav):
        spec = wav_to_spec(wav, self.n_fft, self.hop, self.win)
        z, _ = self.patch(spec)
        z = self.enc_proj(z)
        z = z.mean(dim=2)
        z = self.resampler(z)
        z = z.transpose(1, 2)
        z = self.encoder(z)
        z = self.to_token(z)

        if self.use_partitioned_fsq:
            fsq_out = self.fsq(z, self.training)
            tokens = fsq_out['tokens']

            cont_tokens = torch.cat([fsq_out['semantic_cont'], fsq_out['acoustic']], dim=-1)
            if fsq_out['semantic_disc'] is not None:
                disc_tokens = torch.cat([fsq_out['semantic_disc'], fsq_out['acoustic']], dim=-1)
            else:
                disc_tokens = None

            z_h, residual = self.hybrid(cont_tokens, disc_tokens if disc_tokens is not None else cont_tokens)

            return {
                "tokens": z_h,
                "semantic_cont": fsq_out['semantic_cont'],
                "semantic_disc": fsq_out['semantic_disc'],
                "acoustic_pred": fsq_out['acoustic_pred'],
                "acoustic_residual": fsq_out['acoustic_residual'],
                "acoustic": fsq_out['acoustic'],
                "cont": cont_tokens,
                "disc": disc_tokens,
            }
        else:
            z_cont, z_disc = self.fsq(z, self.training)
            z_h, _ = self.hybrid(z_cont, z_disc if z_disc is not None else z_cont)
            return {"tokens": z_h, "cont": z_cont, "disc": z_disc}

    def decode(self, tokens, length, sigma=None, spec_init=None):
        """
        Consistency model decoder.

        Args:
            tokens: encoded tokens [B, T, D]
            length: target waveform length
            sigma: noise level [B] or scalar. If None, uses minimal noise (1e-3) for one-step inference
            spec_init: initial spectrogram x_sigma [B, 4, F, T] to feed the consistency function.
                       If None, starts from pure noise scaled by sigma (inference path).

        Returns:
            reconstructed waveform
        """
        b = tokens.shape[0]
        device = tokens.device

        # Set default sigma for inference (one-step generation from minimal noise)
        if sigma is None:
            sigma = torch.ones(b, device=device) * 1e-3

        if not torch.is_tensor(sigma):
            sigma = torch.tensor([sigma] * b, device=device, dtype=torch.float32)

        # Get conditioning with noise level
        cond = self.cond(tokens, sigma)

        # Determine target spectrogram time length from conditioner
        t_out = cond.shape[-1]
        f_bins = self.n_fft // 2 + 1

        # Prepare initial spectrogram x_sigma
        if spec_init is None:
            # Inference: start from pure noise (one-step generation)
            spec_noisy = torch.randn(b, 4, f_bins, t_out, device=device) * sigma.view(b, 1, 1, 1)
        else:
            # Use provided initial spectrogram (e.g., clean+noise for training)
            spec_noisy = spec_init
            # Align freq/time dims if needed
            if spec_noisy.shape[-2] != f_bins or spec_noisy.shape[-1] != t_out:
                import torch.nn.functional as F
                # Only interpolate along time/freq axes to match network expectations
                spec_noisy = F.interpolate(spec_noisy, size=(f_bins, t_out), mode='bilinear', align_corners=False)

        # EDM parameterization for consistency model
        c_skip, c_out, c_in, _ = edm_parameterization(sigma, sigma_data=0.5)

        # Scale input
        spec_in = spec_noisy * c_in

        # Transformer decoder prediction with cross-attention
        F_theta = self.decoder(spec_in, cond, tokens=tokens, sigma=sigma)

        # Consistency function: f(x_σ, σ) = c_skip * x_σ + c_out * F_θ(x_σ, σ)
        spec_pred = c_skip * spec_noisy + c_out * F_theta

        # Band-split refinement
        spec_pred = self.bands(spec_pred)

        # Convert to waveform
        wav = spec_to_wav(spec_pred, self.n_fft, self.hop, self.win, length=length)
        return wav

    def forward(self, wav, decode=True):
        # Get clean spectrogram for consistency loss
        spec_clean = wav_to_spec(wav, self.n_fft, self.hop, self.win)

        enc = self.encode(wav)
        rec = self.decode(enc["tokens"], wav.shape[-1]) if decode else None

        # Return spec_clean for consistency loss
        enc["spec_clean"] = spec_clean

        return rec, enc
