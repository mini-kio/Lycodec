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
                 token_rate_hz=24,  # Tokens per second (frequency, e.g., 24 Hz)
                 train_seq_len=36,  # Sequence length for training clips (e.g., 24 Hz * 1.5s = 36)
                 decoder_t_frames=None,  # Spectrogram time frames for decoder (computed from crop_seconds)
                 drop_start=0.6, drop_end=0.1, drop_decay_steps=200000,
                 use_residual_corrector=True, corrector_alpha=0.3):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.token_dim = token_dim
        self.token_rate_hz = token_rate_hz  # Frequency (Hz)
        self.train_seq_len = train_seq_len  # Sequence length (tokens/clip)
        self.pq_M = pq_M
        self.pq_K = pq_K
        self.use_residual_corrector = use_residual_corrector
        self.corrector_alpha = float(max(0.0, min(corrector_alpha, 0.3)))

        self.patch = Patchifier(c_in=4, widths=(64, 128, 256, 512))
        self.enc_proj = nn.Conv2d(512, hidden, 1)
        self.resampler = TemporalResampler(hidden, t_out=train_seq_len)
        self.encoder = TransformerEncoder(
            dim=hidden,
            depth=layers,
            heads=heads,
            use_checkpoint=use_checkpoint,
            use_rope=use_rope,
            seq_len=train_seq_len,
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

        # Compute decoder spectrogram time frames (default 113 for backward compatibility)
        if decoder_t_frames is None:
            decoder_t_frames = 113
            print(f"[Lycodec] Warning: decoder_t_frames not provided, using default {decoder_t_frames}")

        self.decoder_t_frames = decoder_t_frames

        self.cond = TokenConditioner(token_dim=token_dim, cond_ch=decoder_cond_ch, t_out=decoder_t_frames, f_bins=self.n_fft // 2 + 1)
        print(f"[Lycodec] TransformerDecoder2D depth={decoder_depth}, patch={decoder_patch_size}, embed={decoder_embed_dim}, mlp={decoder_mlp_ratio}, cond_ch={decoder_cond_ch}, t_frames={decoder_t_frames}")
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
            target_size=(self.n_fft // 2 + 1, decoder_t_frames),
            max_token_len=train_seq_len,
        )
        # Use ratio-based split_hz (0.25 * sr maintains same frequency ratio across sample rates)
        # 44.1kHz: 11025Hz, 48kHz: 12000Hz
        self.bands = BandSplitHead(c_in=4, sr=sr, n_fft=n_fft, split_hz=int(sr * 0.25))

        # Bundle embedding: M subspace indices → fused token embedding
        # Each subspace has its own embedding table, then fused via MLP
        d_sub = token_dim // pq_M  # e.g., 256/4 = 64
        self.subspace_embeds = nn.ModuleList([
            nn.Embedding(pq_K, d_sub) for _ in range(pq_M)
        ])
        self.bundle_fuse = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # M-way prediction head: continuous tokens → M independent K-way logits
        self.index_predictors = nn.ModuleList([
            nn.Linear(token_dim, pq_K) for _ in range(pq_M)
        ])

        print(f"[Lycodec] Bundle embedding: M={pq_M} × K={pq_K} (d_sub={d_sub})")
        print(f"[Lycodec] M-way head: {pq_M} classifiers × K={pq_K}")


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

        # Bundle embeddings from indices to ensure discrete path learns full mapping
        bundle_tokens = self.bundle_from_indices(quant_out['indices'])

        # Use straight-through quantized embedding for gradient flow back to encoder
        y_disc = self.shared_proj(quant_out['z_q'])
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

        # Sample-level dropout (not batch-level) to prevent residual_loss starvation
        dropout_mask = None
        tokens_for_decoder = y_disc_corrected
        if self.training and drop_prob > 0.0:
            b = y_disc_corrected.size(0)
            # Bernoulli mask: True = use continuous, False = use discrete
            dropout_mask = (torch.rand(b, device=y_disc_corrected.device) < drop_prob).view(b, 1, 1)
            if dropout_mask.all():
                # Keep at least one discrete sample so DDP never sees unused params
                idx = torch.randint(0, b, (1,), device=dropout_mask.device)
                dropout_mask[idx] = False
            tokens_for_decoder = torch.where(dropout_mask, y_cont, y_disc_corrected)

        dropout_applied = dropout_mask  # Now a tensor [B,1,1] or None

        # Use corrected path for discrete summary to ensure gradient flow
        summary_disc = self._summary_head(y_disc_corrected)
        # Use completely detached target for continuous summary
        summary_cont = self._summary_head(y_cont_target)

        # M-way index prediction (for supervised learning)
        logits_list = None
        if self.training:
            logits_list = self.predict_indices(z_continuous)

        return {
            "tokens": tokens_for_decoder,
            "indices": quant_out['indices'],
            "bundle_tokens": bundle_tokens,
            "z_q": quant_out['z_q'],
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
            "logits_list": logits_list,  # M-way logits for CE loss
        }

    def bundle_from_indices(self, indices):
        """
        Convert PQ indices to bundled token embeddings.

        Args:
            indices: [B, T, M] subspace indices

        Returns:
            tokens: [B, T, D] bundled embeddings
        """
        B, T, M = indices.shape
        assert M == self.pq_M, f"Expected M={self.pq_M}, got {M}"

        # Embed each subspace
        embs = []
        for m in range(M):
            emb_m = self.subspace_embeds[m](indices[:, :, m])  # [B, T, d_sub]
            embs.append(emb_m)

        # Concatenate and fuse
        h = torch.cat(embs, dim=-1)  # [B, T, D]
        tokens = self.bundle_fuse(h)  # [B, T, D]

        return tokens

    def predict_indices(self, z_continuous):
        """
        Predict M-way indices from continuous tokens.

        Args:
            z_continuous: [B, T, D] continuous latent tokens

        Returns:
            logits_list: list of M tensors [B, T, K]
        """
        logits_list = [
            predictor(z_continuous) for predictor in self.index_predictors
        ]
        return logits_list

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

    def _compute_consistency_loss(self, wav, tokens, spec_clean, ema_teacher,
                                  sigma_min, sigma_max, rho, decode_chunk):
        """
        Compute consistency distillation loss with decoder in computational graph.

        This is the KEY method that ensures decoder parameters are used in the
        forward pass, solving DDP's "unused parameters" issue.

        Args:
            wav: input waveform [B, 2, T]
            tokens: latent tokens from encoder [B, T, D]
            spec_clean: clean spectrogram [B, 4, F, T]
            ema_teacher: optional EMA teacher model
            sigma_min, sigma_max, rho: EDM noise schedule parameters
            decode_chunk: microbatch size for decoder

        Returns:
            loss_consistency: scalar loss tensor
        """
        import math

        b = spec_clean.shape[0]
        device = spec_clean.device
        length = wav.shape[-1]

        # Sample noise levels (EDM schedule)
        u = torch.rand(b, device=device)
        sigma_i = (sigma_max ** (1 / rho) + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        delta_sigma = torch.rand(b, device=device) * 0.2 * sigma_i
        sigma_i_plus = sigma_i + delta_sigma

        # Add noise to clean spectrogram
        noise = torch.randn_like(spec_clean)
        spec_noisy_i = spec_clean + sigma_i.view(b, 1, 1, 1) * noise
        spec_noisy_i_plus = spec_clean + sigma_i_plus.view(b, 1, 1, 1) * noise

        # Teacher prediction (no gradient, for target)
        with torch.no_grad():
            teacher = ema_teacher if ema_teacher is not None else self
            pred_i_chunks = []
            for i in range(0, b, decode_chunk):
                chunk_slice = slice(i, i + decode_chunk)
                pred_i_chunks.append(
                    teacher.decode(
                        tokens[chunk_slice],
                        length,
                        sigma_i[chunk_slice],
                        spec_noisy_i[chunk_slice]
                    )
                )
            pred_i = torch.cat(pred_i_chunks, dim=0)

        # Student prediction (GRADIENT ENABLED - ensures decoder in graph)
        pred_i_plus_chunks = []
        for i in range(0, b, decode_chunk):
            chunk_slice = slice(i, i + decode_chunk)
            pred_i_plus_chunks.append(
                self.decode(
                    tokens[chunk_slice],
                    length,
                    sigma_i_plus[chunk_slice],
                    spec_noisy_i_plus[chunk_slice]
                )
            )
        pred_i_plus = torch.cat(pred_i_plus_chunks, dim=0)

        # Pseudo-Huber loss (robust to outliers)
        spec_size = spec_clean.shape[2] * spec_clean.shape[3]
        c = 0.00054 * math.sqrt(spec_size)
        diff = pred_i_plus - pred_i
        loss = torch.sqrt((diff / c) ** 2 + 1) - 1
        loss = loss.mean() / (delta_sigma.mean() + 1e-8)

        return loss

    def forward(self, wav, ema_teacher=None, sigma_min=0.002, sigma_max=80.0,
                rho=7.0, decode_chunk=1):
        """
        Training forward pass with unified computational graph.

        ALL parameters (encoder, quantizer, decoder) are used in single forward.
        This solves DDP's "unused parameters" error by ensuring decoder is always
        included in the computational graph through consistency loss calculation.

        Args:
            wav: input waveform [B, 2, T]
            ema_teacher: optional EMA teacher for consistency distillation
            sigma_min, sigma_max, rho: EDM noise schedule parameters
            decode_chunk: microbatch size for decoder (memory optimization)

        Returns:
            If training:
                enc: dict with encoder outputs (tokens, quantizer metrics, etc.)
                loss_consistency: consistency distillation loss
            If inference:
                rec: reconstructed waveform [B, 2, T]
                enc: dict with encoder outputs
        """
        # Inference mode: simple encode/decode
        if not self.training:
            spec_clean = wav_to_spec(wav, self.n_fft, self.hop, self.win)
            enc = self.encode(spec=spec_clean)
            rec = self.decode(enc["tokens"], wav.shape[-1])
            enc["spec_clean"] = spec_clean
            return rec, enc

        # Training mode: Mid/Side RMS normalization
        from lycodec.utils.audio import to_midside, to_stereo
        ms = to_midside(wav)
        m, s = ms[:, 0], ms[:, 1]
        s = s * (m.abs().mean() / (s.abs().mean() + 1e-6))
        wav = to_stereo(torch.stack([m, s], dim=1))

        # Encode (uses encoder + quantizer parameters)
        spec_clean = wav_to_spec(wav, self.n_fft, self.hop, self.win)
        enc = self.encode(spec=spec_clean)
        tokens = enc["tokens"]

        # Consistency loss (uses decoder parameters - KEY FOR DDP)
        loss_consistency = self._compute_consistency_loss(
            wav, tokens, spec_clean, ema_teacher,
            sigma_min, sigma_max, rho, decode_chunk
        )

        enc["spec_clean"] = spec_clean.detach()
        return enc, loss_consistency
