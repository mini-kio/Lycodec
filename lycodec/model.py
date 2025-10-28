import torch
import torch.nn as nn
from lycodec.utils.audio import wav_to_spec, spec_to_wav
from lycodec.core.blocks import (
    Patchifier,
    TransformerEncoder,
    TemporalResampler,
    OPQPQQuantizer,
    PQResidualCorrector,
)
from lycodec.core.decoders import (
    TokenConditioner,
    TransformerDecoder2D,
    BandSplitHead,
    edm_parameterization,
    to_tensor,
)


class Lycodec(nn.Module):
    """Stereo audio tokenizer + consistency decoder with OPQ-PQ quantization."""
    def _summary_head(self, sequence):
        if sequence is None:
            return None
        pooled = sequence.mean(dim=1)
        return nn.functional.normalize(pooled, dim=-1)

    def __init__(self, sr=44100, n_fft=2048, hop=588, win=2048,
                 token_dim=256, hidden=512, layers=8, heads=8,
                 use_checkpoint=False, use_rope=True,
                 decoder_depth=6, decoder_patch_size=16,
                 decoder_embed_dim=512, decoder_mlp_ratio=4.0, decoder_cond_ch=64,
                 pq_M=4, pq_K=256,
                 ema_decay=0.97, awakening_steps=200,
                 token_fps=24,  # tokens per second (e.g., 24 Hz)
                 drop_start=0.6, drop_end=0.1, drop_decay_steps=200000,
                 use_residual_corrector=True, corrector_alpha=0.3):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.token_dim = token_dim
        self.token_fps = token_fps  # tokens per second
        self.use_residual_corrector = use_residual_corrector
        self.corrector_alpha = float(max(0.0, min(corrector_alpha, 0.3)))

        self.patch = Patchifier(c_in=4, widths=(64, 128, 256, 512))
        self.enc_proj = nn.Conv2d(512, hidden, 1)
        self.resampler = TemporalResampler(hidden, t_out=token_fps)
        self.encoder = TransformerEncoder(
            dim=hidden,
            depth=layers,
            heads=heads,
            use_checkpoint=use_checkpoint,
            use_rope=use_rope,
            seq_len=token_fps,
        )
        self.to_token = nn.Linear(hidden, token_dim)
        self.shared_proj = nn.Linear(token_dim, token_dim)

        # OPQ-PQ quantizer (single path)
        self.quantizer = OPQPQQuantizer(
            dim=token_dim,
            M=pq_M,
            K=pq_K,
            ema_decay=ema_decay,
            awakening_steps=awakening_steps,
            gumbel_temp=1.0,
            drop_start=drop_start,
            drop_end=drop_end,
            drop_decay_steps=drop_decay_steps,
            ortho_penalty=1e-4,
            qr_every=500,
        )
        print(f"[Lycodec] OPQ-PQ: M={pq_M}, K={pq_K}")

        if use_residual_corrector:
            print("[Lycodec] PQResidualCorrector enabled (alpha≤0.3)")
            self.corrector = PQResidualCorrector(
                dim=token_dim,
                M=pq_M,
                K=pq_K,
                context_size=5,
            )
        else:
            self.corrector = None

        # Default warmup steps (can be overridden via train config)
        self.quantizer_warmup_steps = 5000

        self.cond = TokenConditioner(token_dim=token_dim, cond_ch=decoder_cond_ch, t_out=113, f_bins=self.n_fft // 2 + 1)
        print(f"[Lycodec] TransformerDecoder2D depth={decoder_depth}, patch={decoder_patch_size}, embed={decoder_embed_dim}, mlp={decoder_mlp_ratio}, cond_ch={decoder_cond_ch}")
        self.decoder = TransformerDecoder2D(
            c_in=decoder_cond_ch + 4,  # cond_ch + 4 (spec channels)
            c_out=4,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=8,
            patch_size=decoder_patch_size,
            token_dim=token_dim,
            mlp_ratio=decoder_mlp_ratio,
            dropout=0.0,
            target_size=(self.n_fft // 2 + 1, 113),
            max_token_len=token_fps,
        )
        # Use ratio-based split_hz (0.25 * sr maintains same frequency ratio across sample rates)
        # 44.1kHz: 11025Hz, 48kHz: 12000Hz
        self.bands = BandSplitHead(c_in=4, sr=sr, n_fft=n_fft, split_hz=int(sr * 0.25))


    def encode(self, wav=None, spec=None):
        """
        Encode waveform or precomputed spectrogram into latent tokens.

        Args:
            wav: waveform tensor [B, C, T], required when spec is None.
            spec: midside spectrogram tensor [B, 4, F, T], optional shortcut when already computed.

        Returns:
            dict with quantized and continuous representations used by downstream losses.
        """
        if spec is None:
            if wav is None:
                raise ValueError("Either wav or spec must be provided to encode().")
            spec = wav_to_spec(wav, self.n_fft, self.hop, self.win)

        z, _ = self.patch(spec)
        z = self.enc_proj(z)
        z = z.mean(dim=2)
        z = self.resampler(z)
        z = z.transpose(1, 2)
        z = self.encoder(z)
        z_continuous = self.to_token(z)

        y_cont = self.shared_proj(z_continuous)
        # Two-stage detach: input detach + output detach to completely cut graph
        y_cont_target = self.shared_proj(z_continuous.detach()).detach()

        quant_out = self.quantizer(z_continuous, training=self.training)
        y_disc = self.shared_proj(quant_out['embedding'])
        y_disc_corrected = y_disc

        r_hat = None
        r_target = None
        if self.corrector is not None:
            r_hat = self.corrector(quant_out['indices'])
            if r_hat is not None:
                y_disc_corrected = y_disc + self.corrector_alpha * r_hat
            if self.training:
                r_target = y_cont_target - y_disc.detach()

        # Dropout with warmup (crucial for codebook learning!)
        drop_prob = float(quant_out['drop_prob']) if self.training else 0.0
        warmup_steps = getattr(self, 'quantizer_warmup_steps', 5000)

        current_step = float(self.quantizer.step_counter.item()) if self.training else warmup_steps + 1

        if current_step < warmup_steps:
            drop_prob = 0.0  # Force discrete path during warmup

        dropout_applied = False
        tokens_for_decoder = y_disc_corrected
        if self.training and drop_prob > 0.0:
            if torch.rand(1, device=tokens_for_decoder.device).item() < drop_prob:
                tokens_for_decoder = y_cont
                dropout_applied = True

        # Use corrected path for discrete summary to ensure gradient flow
        summary_disc = self._summary_head(y_disc_corrected)
        # Use completely detached target for continuous summary
        summary_cont = self._summary_head(y_cont_target)

        return {
            "tokens": tokens_for_decoder,
            "indices": quant_out['indices'],
            "commitment_loss": quant_out['commitment_loss'],
            "entropy_bonus": quant_out.get('entropy_bonus', torch.tensor(0.0, device=z_continuous.device)),
            "ortho_loss": quant_out.get('ortho_loss', torch.tensor(0.0, device=z_continuous.device)),
            "perplexity": quant_out['perplexity'],
            "usage_entropy": quant_out['usage_entropy'],
            "dropout_applied": dropout_applied,
            "drop_prob": drop_prob,
            "active_codes": quant_out.get('active_codes', 0),
            "current_tau": quant_out.get('current_tau', 0.0),
            "y_disc": y_disc_corrected,
            "y_disc_corrected": y_disc_corrected,
            "alignment_target": y_cont_target if self.training else None,
            "r_target": r_target,
            "r_hat": r_hat,
            "z_continuous": z_continuous if self.training else None,
            "summary_disc": summary_disc,
            "summary_cont": summary_cont,
        }

    def decode(self, tokens, length, sigma=None, spec_init=None):
        """Consistency model decoder. Returns reconstructed waveform."""
        b = tokens.shape[0]
        device = tokens.device

        if sigma is None:
            sigma = torch.ones(b, device=device) * 1e-3
        else:
            sigma = to_tensor(sigma, batch_size=b, device=device, dtype=torch.float32)

        cond = self.cond(tokens, sigma)
        t_out = cond.shape[-1]
        f_bins = self.n_fft // 2 + 1

        if spec_init is None:
            spec_noisy = torch.randn(b, 4, f_bins, t_out, device=device) * sigma.view(b, 1, 1, 1)
        else:
            spec_noisy = spec_init
            if spec_noisy.shape[-2] != f_bins or spec_noisy.shape[-1] != t_out:
                import torch.nn.functional as F
                spec_noisy = F.interpolate(spec_noisy, size=(f_bins, t_out), mode='bilinear', align_corners=False)

        c_skip, c_out, c_in, _ = edm_parameterization(sigma, sigma_data=0.5)
        spec_in = spec_noisy * c_in
        F_theta = self.decoder(spec_in, cond, tokens=tokens, sigma=sigma)
        spec_pred = c_skip * spec_noisy + c_out * F_theta
        spec_pred = self.bands(spec_pred)
        wav = spec_to_wav(spec_pred, self.n_fft, self.hop, self.win, length=length)
        return wav

    def forward(self, wav, decode=True):
        """Run full codec pass returning reconstruction and encoder artifacts."""
        spec_clean = wav_to_spec(wav, self.n_fft, self.hop, self.win)
        enc = self.encode(spec=spec_clean)
        rec = self.decode(enc["tokens"], wav.shape[-1]) if decode else None
        enc["spec_clean"] = spec_clean.detach()
        return rec, enc
