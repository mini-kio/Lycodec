import torch
import torch.nn as nn
from lycodec.utils.audio import wav_to_spec, spec_to_wav
from lycodec.utils.common import to_tensor
from lycodec.core.blocks import (
    Patchifier,
    TransformerEncoder,
    TemporalResampler,
    RVQQuantizer,
    ResidualCorrector,
)
from lycodec.core.decoders import TokenConditioner, TransformerDecoder2D, BandSplitHead, edm_parameterization


class Lycodec(nn.Module):
    def _summary_head(self, sequence):
        if sequence is None:
            return None
        pooled = sequence.mean(dim=1)
        return nn.functional.normalize(pooled, dim=-1)

    def __init__(self, sr=48000, n_fft=2048, hop=640, win=2048,
                 token_dim=256, hidden=512, layers=8, heads=8,
                 use_checkpoint=False, use_rope=True,
                 decoder_depth=6, decoder_patch_size=16,
                 rvq_codebook_size=4096, token_fps=24,
                 rvq_drop_start=0.6, rvq_drop_end=0.1, rvq_drop_decay_steps=200000,
                 use_residual_corrector=True, corrector_alpha=0.3):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.token_dim = token_dim
        self.token_fps = token_fps
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

        self.rvq = RVQQuantizer(
            dim=token_dim,
            codebook_size=rvq_codebook_size,
            ema_decay=0.99,
            awakening_steps=2000,
            gumbel_temp=1.0,
            drop_start=rvq_drop_start,
            drop_end=rvq_drop_end,
            drop_decay_steps=rvq_drop_decay_steps,
        )
        print(f"[Lycodec] RVQ dropout {rvq_drop_start}->{rvq_drop_end} (steps={rvq_drop_decay_steps})")

        if use_residual_corrector:
            print("[Lycodec] ResidualCorrector enabled (alpha≤0.3)")
            self.corrector = ResidualCorrector(
                dim=token_dim,
                codebook_size=rvq_codebook_size,
                context_size=5,
            )
        else:
            self.corrector = None

        self.cond = TokenConditioner(token_dim=token_dim, cond_ch=64, t_out=113, f_bins=self.n_fft // 2 + 1)
        print(f"[Lycodec] TransformerDecoder2D depth={decoder_depth}, patch={decoder_patch_size}")
        self.decoder = TransformerDecoder2D(
            c_in=68,
            c_out=4,
            embed_dim=512,
            depth=decoder_depth,
            num_heads=8,
            patch_size=decoder_patch_size,
            token_dim=token_dim,
            mlp_ratio=4.0,
            dropout=0.0,
            target_size=(self.n_fft // 2 + 1, 113),
            max_token_len=token_fps,
        )
        self.bands = BandSplitHead(c_in=4, sr=sr, n_fft=n_fft)


    def encode(self, wav):
        """Encode waveform and expose discrete/continuous shared representations."""
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

        rvq_out = self.rvq(z_continuous, training=self.training)
        y_disc = self.shared_proj(rvq_out['embedding'])
        y_disc_corrected = y_disc

        r_hat = None
        r_target = None
        if self.corrector is not None:
            r_hat = self.corrector(rvq_out['indices'])
            if r_hat is not None:
                y_disc_corrected = y_disc + self.corrector_alpha * r_hat
            if self.training:
                r_target = y_cont_target - y_disc.detach()

        drop_prob = float(rvq_out['drop_prob']) if self.training else 0.0
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
            "indices": rvq_out['indices'],
            "commitment_loss": rvq_out['commitment_loss'],
            "perplexity": rvq_out['perplexity'],
            "usage_entropy": rvq_out['usage_entropy'],
            "dropout_applied": dropout_applied,
            "drop_prob": drop_prob,
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
        spec_clean = wav_to_spec(wav, self.n_fft, self.hop, self.win)
        enc = self.encode(wav)
        rec = self.decode(enc["tokens"], wav.shape[-1]) if decode else None
        enc["spec_clean"] = spec_clean.detach()
        return rec, enc
