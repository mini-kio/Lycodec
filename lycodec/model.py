import torch
import torch.nn as nn
from lycodec.utils.audio import wav_to_spec, spec_to_wav
from lycodec.core.blocks import (
    Patchifier,
    TransformerEncoder,
    TemporalResampler,
    RVQQuantizer,
    ResidualCorrector,
    HybridLatent,
    StereoHead,
)
from lycodec.core.decoders import TokenConditioner, TransformerDecoder2D, BandSplitHead, edm_parameterization


class Lycodec(nn.Module):
    def __init__(self, sr=48000, n_fft=2048, hop=640, win=2048, token_dim=256, hidden=512, layers=8, heads=8, use_checkpoint=False, use_rope=True, semantic_dim=120, decoder_depth=6, decoder_patch_size=16, rvq_codebook_size=4096, token_fps=24, use_residual_corrector=True, corrector_alpha=0.3):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.token_dim = token_dim
        self.token_fps = token_fps
        self.use_residual_corrector = use_residual_corrector
        self.corrector_alpha = corrector_alpha

        self.patch = Patchifier(c_in=4, widths=(64, 128, 256, 512))
        self.enc_proj = nn.Conv2d(512, hidden, 1)
        self.resampler = TemporalResampler(hidden, t_out=token_fps)  # 24 fps
        self.encoder = TransformerEncoder(dim=hidden, depth=layers, heads=heads, use_checkpoint=use_checkpoint, use_rope=use_rope, seq_len=token_fps)
        self.to_token = nn.Linear(hidden, token_dim)

        # NEW: Single RVQ (A-plan: FSQ removed, dropouts handle robustness)
        print(f"[Lycodec] A-PLAN: Single RVQ only (K={rvq_codebook_size}, {token_fps} fps)")
        print(f"[Lycodec] FSQ removed - decoder reinforced + dropout trio for quality")
        self.rvq = RVQQuantizer(
            dim=token_dim,
            codebook_size=rvq_codebook_size,
            ema_decay=0.99,
            awakening_steps=2000,
            gumbel_temp=1.0,
            p_mask=0.10,    # Will be scheduled
            p_jitter=0.20,  # Will be scheduled
            p_bypass=0.20,  # Will be scheduled
            jitter_sigma=0.07,
        )

        # ResidualCorrector: Predict quantization error from indices only
        if use_residual_corrector:
            print(f"[Lycodec] ResidualCorrector enabled (alpha={corrector_alpha})")
            print(f"[Lycodec]   - Training: r_target = z_continuous - z_q")
            print(f"[Lycodec]   - Inference: z_corrected = z_q + α*r_hat (indices only)")
            self.corrector = ResidualCorrector(
                dim=token_dim,
                codebook_size=rvq_codebook_size,
                context_size=5,
            )
        else:
            self.corrector = None

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
            max_token_len=token_fps,  # NEW: pass token sequence length
        )

        self.bands = BandSplitHead(c_in=4, sr=sr, n_fft=n_fft)

    def encode(self, wav):
        """
        A-PLAN: Single RVQ encoding (FSQ removed)

        RVQ dropout trio handles robustness:
        - Mask dropout: corruption resistance
        - Jitter: quantization boundary smoothing
        - Bypass: decoder learns from continuous latents

        ResidualCorrector (if enabled):
        - Training: learns r_target = z_continuous - z_q from indices
        - Inference: predicts r_hat from indices only, z_corrected = z_q + α*r_hat
        """
        spec = wav_to_spec(wav, self.n_fft, self.hop, self.win)
        z, _ = self.patch(spec)
        z = self.enc_proj(z)
        z = z.mean(dim=2)
        z = self.resampler(z)
        z = z.transpose(1, 2)
        z = self.encoder(z)
        z_continuous = self.to_token(z)  # [B, T, D] - continuous latent

        # Single RVQ with dropout trio
        rvq_out = self.rvq(z_continuous, training=self.training)
        z_q = rvq_out['z_q']  # Quantized embeddings [B, T, D]

        # ResidualCorrector: predict quantization error from indices only
        tokens_final = z_q
        r_target = None
        r_hat = None

        if self.corrector is not None:
            if self.training:
                # Training: compute target residual r_target = z_continuous - z_q
                # Train corrector to predict this from indices only
                r_target = z_continuous - z_q.detach()  # Detach to not affect RVQ
                r_hat = self.corrector(z_q.detach(), rvq_out['indices'])
                # Apply correction with clamped alpha
                alpha = torch.clamp(torch.tensor(self.corrector_alpha), 0.0, 1.0)
                tokens_final = z_q + alpha * r_hat
            else:
                # Inference: predict residual from quantized embeddings only (no continuous)
                r_hat = self.corrector(z_q, rvq_out['indices'])
                alpha = torch.clamp(torch.tensor(self.corrector_alpha), 0.0, 1.0)
                tokens_final = z_q + alpha * r_hat

        return {
            "tokens": tokens_final,  # Corrected tokens for decoder
            "indices": rvq_out['indices'],  # LLM prediction target [B, T] ∈ [0, K-1]
            "commitment_loss": rvq_out['commitment_loss'],
            "perplexity": rvq_out['perplexity'],
            "usage_entropy": rvq_out['usage_entropy'],
            "bypass_applied": rvq_out['bypass_applied'],  # Monitoring
            "r_target": r_target,  # Target residual (training only)
            "r_hat": r_hat,  # Predicted residual
            "z_q_uncorrected": z_q,  # Uncorrected quantized tokens (for analysis)
            "z_continuous": z_continuous if self.training else None,  # Continuous latent (training only)
        }

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
