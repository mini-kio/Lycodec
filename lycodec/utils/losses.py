import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def stft_loss(x, y, cfg):
    """
    Multi-resolution STFT loss for batched multichannel audio.
    Args:
        x: predicted audio [B, C, T]
        y: target audio [B, C, T]
    Returns:
        loss: scalar
    """
    # Flatten batch and channels for STFT
    is_batched = x.ndim == 3
    if is_batched:
        b, c, t = x.shape
        x_flat = x.reshape(b * c, t)  # [B*C, T]
        y_flat = y.reshape(b * c, t)
    else:
        x_flat = x
        y_flat = y

    # Ensure stable dtype for STFT computations
    x32 = x_flat.float()
    y32 = y_flat.float()

    losses = []
    for hl, wl in zip([128, 256, 512, 1024], [512, 1024, 2048, 4096]):
        win = torch.hann_window(wl, device=x.device, dtype=torch.float32)
        X = torch.stft(x32, n_fft=wl, hop_length=hl, win_length=wl, window=win, return_complex=True)
        Y = torch.stft(y32, n_fft=wl, hop_length=hl, win_length=wl, window=win, return_complex=True)
        losses.append((X.abs() - Y.abs()).abs().mean())
        losses.append((torch.angle(X) - torch.angle(Y)).abs().mean())
    return sum(losses)


def cosine_align(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1 - (a * b).sum(dim=-1).mean()


def sample_noise_levels(batch_size, device, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    Sample noise levels from log-normal distribution (EDM schedule).

    Args:
        batch_size: number of samples
        device: torch device
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: schedule parameter (controls distribution shape)

    Returns:
        sigma: sampled noise levels [B]
    """
    u = torch.rand(batch_size, device=device)
    sigma = (sigma_max ** (1 / rho) + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigma


def pseudo_huber_loss(pred, target, c=1.0):
    """
    Pseudo-Huber loss: sqrt((x/c)^2 + 1) - 1

    More robust than L2, smoother than L1.

    Args:
        pred: predicted values
        target: target values
        c: scale parameter (controls transition from L2 to L1)

    Returns:
        loss value
    """
    diff = pred - target
    loss = torch.sqrt((diff / c) ** 2 + 1) - 1
    return loss.mean()


def consistency_loss(model, wav, tokens, spec_clean, sigma_min=0.002, sigma_max=80.0, rho=7.0, ema_model=None):
    """
    Consistency Distillation Loss (CD loss) for one-step generation.

    Loss = E[d(f(x + σ_i·ε, σ_i), sg(f(x + σ_{i+1}·ε, σ_{i+1}))) / Δσ]

    where:
    - f is the consistency function (student/online model)
    - sg is stop-gradient (teacher/EMA model)
    - d is pseudo-Huber distance
    - Δσ = σ_{i+1} - σ_i

    Args:
        model: student model (online)
        wav: clean waveform [B, C, T]
        tokens: encoded tokens [B, Tk, D]
        spec_clean: clean spectrogram [B, 4, F, T]
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: noise schedule parameter
        ema_model: teacher model (EMA). If None, uses self-consistency

    Returns:
        consistency loss
    """
    b = spec_clean.shape[0]
    device = spec_clean.device
    length = wav.shape[-1]

    # Sample noise level σ_i
    sigma_i = sample_noise_levels(b, device, sigma_min, sigma_max, rho)

    # Sample Δσ (small increment) from uniform [0, 0.2 * σ_i]
    # This ensures σ_{i+1} = σ_i + Δσ is slightly larger
    delta_sigma = torch.rand(b, device=device) * 0.2 * sigma_i
    sigma_i_plus = sigma_i + delta_sigma

    # Add the SAME noise to both levels (important for consistency!)
    noise = torch.randn_like(spec_clean)
    spec_noisy_i = spec_clean + sigma_i.view(b, 1, 1, 1) * noise
    spec_noisy_i_plus = spec_clean + sigma_i_plus.view(b, 1, 1, 1) * noise

    # Student prediction at σ_{i+1} (higher noise)
    pred_i_plus = model.decode(tokens, length, sigma_i_plus, spec_noisy_i_plus)

    # Teacher prediction at σ_i (lower noise) - with stop gradient
    teacher = ema_model if ema_model is not None else model
    with torch.no_grad():
        pred_i = teacher.decode(tokens, length, sigma_i, spec_noisy_i)

    # Pseudo-Huber distance
    # c is adaptive based on spectrogram size (helps with scale)
    spec_size = spec_clean.shape[2] * spec_clean.shape[3]  # F * T
    c = 0.00054 * math.sqrt(spec_size)

    loss = pseudo_huber_loss(pred_i_plus, pred_i, c=c)

    # Normalize by delta_sigma (important for stability!)
    loss = loss / (delta_sigma.mean() + 1e-8)

    return loss


def commitment_loss(z_cont, z_disc):
    """
    VQ-VAE commitment loss: encourage encoder to commit to codes.

    L_commit = ||z_cont - sg(z_disc)||^2
    """
    if z_disc is None:
        return torch.tensor(0.0, device=z_cont.device)
    return F.mse_loss(z_cont, z_disc.detach())




def acoustic_prediction_loss(enc):
    if 'acoustic_residual' not in enc or enc['acoustic_residual'] is None:
        return torch.tensor(0.0, device=enc['tokens'].device)

    residual = enc['acoustic_residual']
    loss = (residual ** 2).mean()

    return loss




def masked_token_prediction_loss(model, wav, mask_ratio=0.15):
    model_obj = model.module if hasattr(model, 'module') else model
    device = wav.device

    with torch.no_grad():
        enc = model_obj.encode(wav)

    tokens = enc["tokens"]
    B, T, D = tokens.shape

    if not hasattr(model_obj, 'mask_token'):
        model_obj.mask_token = torch.nn.Parameter(
            torch.randn(1, 1, D, device=device) * 0.02
        )
        model_obj.register_parameter('mask_token', model_obj.mask_token)

    if not hasattr(model_obj, 'mask_predictor'):
        model_obj.mask_predictor = torch.nn.Sequential(
            torch.nn.Linear(D, D * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(D * 2, D),
        ).to(device)

    num_mask = max(1, int(T * mask_ratio))
    mask_indices = torch.rand(B, T, device=device).argsort(dim=1)[:, :num_mask]

    tokens_masked = tokens.clone()
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask)
    tokens_masked[batch_indices, mask_indices] = model_obj.mask_token

    predicted = model_obj.mask_predictor(tokens_masked)

    target = tokens.detach()
    loss = torch.tensor(0.0, device=device)
    for b in range(B):
        masked_pred = predicted[b, mask_indices[b]]
        masked_target = target[b, mask_indices[b]]
        loss = loss + F.mse_loss(masked_pred, masked_target)

    return loss / B


# ============================================
# NEW: Semantic Supervision Losses
# Applied ONLY to coarse RVQ path (not fine FSQ)
# ============================================

def lyric_ctc_loss(coarse_tokens, lyric_labels, lyric_lengths, token_lengths):
    """
    CTC loss for lyrics alignment to coarse tokens.

    Args:
        coarse_tokens: [B, T, D] RVQ coarse tokens
        lyric_labels: [B, L] phoneme/character indices
        lyric_lengths: [B] lengths of lyric sequences
        token_lengths: [B] lengths of token sequences

    Returns:
        CTC loss

    Note: Requires lyric annotations. If not available, returns 0.
    """
    if lyric_labels is None:
        return torch.tensor(0.0, device=coarse_tokens.device)

    B, T, D = coarse_tokens.shape
    device = coarse_tokens.device

    # Create lyric predictor head if not exists
    # This is a placeholder - actual implementation needs vocab size
    vocab_size = 256  # Example: 256 phonemes/chars

    # Linear projection to vocab
    lyric_head = nn.Linear(D, vocab_size).to(device)

    # Get logits [B, T, vocab_size]
    logits = lyric_head(coarse_tokens)
    log_probs = F.log_softmax(logits, dim=-1)

    # CTC expects [T, B, vocab_size]
    log_probs = log_probs.transpose(0, 1)

    # CTC loss
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    loss = ctc_loss(log_probs, lyric_labels, token_lengths, lyric_lengths)

    return loss


def section_classification_loss(coarse_tokens, section_labels):
    """
    Section classification loss (verse/chorus/bridge/intro/outro).

    Args:
        coarse_tokens: [B, T, D] RVQ coarse tokens
        section_labels: [B, T] section class indices

    Returns:
        Cross-entropy loss

    Note: Requires section annotations. If not available, returns 0.
    """
    if section_labels is None:
        return torch.tensor(0.0, device=coarse_tokens.device)

    B, T, D = coarse_tokens.shape
    device = coarse_tokens.device

    # Section classes: 0=intro, 1=verse, 2=chorus, 3=bridge, 4=outro, 5=other
    num_classes = 6

    # Create section classifier head if not exists
    section_head = nn.Linear(D, num_classes).to(device)

    # Get logits [B, T, num_classes]
    logits = section_head(coarse_tokens)

    # Cross-entropy loss
    loss = F.cross_entropy(logits.reshape(-1, num_classes), section_labels.reshape(-1))

    return loss


def beat_detection_loss(coarse_tokens, beat_labels):
    """
    Beat detection loss (binary: beat/no-beat per token).

    Args:
        coarse_tokens: [B, T, D] RVQ coarse tokens
        beat_labels: [B, T] binary labels (1=beat, 0=no-beat)

    Returns:
        Binary cross-entropy loss

    Note: Requires beat annotations. If not available, returns 0.
    """
    if beat_labels is None:
        return torch.tensor(0.0, device=coarse_tokens.device)

    B, T, D = coarse_tokens.shape
    device = coarse_tokens.device

    # Create beat detector head if not exists
    beat_head = nn.Sequential(
        nn.Linear(D, 128),
        nn.GELU(),
        nn.Linear(128, 1),
    ).to(device)

    # Get predictions [B, T, 1]
    logits = beat_head(coarse_tokens).squeeze(-1)  # [B, T]

    # BCE loss
    loss = F.binary_cross_entropy_with_logits(logits, beat_labels.float())

    return loss


def text_audio_infonce_loss(coarse_tokens, text_embeddings, temperature=0.07):
    """
    InfoNCE contrastive loss between text prompts and coarse audio tokens.

    Args:
        coarse_tokens: [B, T, D] RVQ coarse tokens
        text_embeddings: [B, D_text] text prompt embeddings (e.g., from CLIP/T5)
        temperature: softmax temperature

    Returns:
        InfoNCE loss

    Note: Requires text descriptions. If not available, returns 0.
    """
    if text_embeddings is None:
        return torch.tensor(0.0, device=coarse_tokens.device)

    B, T, D = coarse_tokens.shape
    device = coarse_tokens.device

    # Pool coarse tokens to [B, D]
    audio_embeddings = coarse_tokens.mean(dim=1)  # [B, D]

    # Project to common space if dimensions don't match
    if audio_embeddings.shape[-1] != text_embeddings.shape[-1]:
        proj_dim = 512
        audio_proj = nn.Linear(D, proj_dim).to(device)
        text_proj = nn.Linear(text_embeddings.shape[-1], proj_dim).to(device)
        audio_embeddings = audio_proj(audio_embeddings)
        text_embeddings = text_proj(text_embeddings)

    # Normalize
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute similarity matrix [B, B]
    logits = torch.matmul(audio_embeddings, text_embeddings.t()) / temperature

    # Labels: diagonal (positive pairs)
    labels = torch.arange(B, device=device)

    # InfoNCE: symmetric loss (audio→text and text→audio)
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.t(), labels)

    loss = (loss_a2t + loss_t2a) / 2

    return loss


def rvq_perplexity_loss(perplexity, target_perplexity=2048):
    """
    Encourage RVQ codebook usage to reach target perplexity.

    Args:
        perplexity: current perplexity (exp of entropy)
        target_perplexity: desired perplexity (e.g., 0.5 * K = 2048 for K=4096)

    Returns:
        Penalty if perplexity is below target
    """
    if perplexity >= target_perplexity:
        return torch.tensor(0.0, device=perplexity.device)

    gap = target_perplexity - perplexity
    loss = (gap / target_perplexity) ** 2

    return loss


# ============================================
# NEW: A-plan Infill Loss
# ============================================

def infill_loss(rec_wav, target_wav, cfg, bypass_applied=False):
    """
    Infill loss for mask dropout robustness.

    Ensures decoder can reconstruct even when tokens are masked.
    Uses STFT loss to measure reconstruction quality.

    Args:
        rec_wav: reconstructed waveform [B, C, T]
        target_wav: target waveform [B, C, T]
        cfg: config dict with STFT parameters
        bypass_applied: if True, skip (bypass already uses continuous)

    Returns:
        STFT loss between rec and target
    """
    if bypass_applied:
        # Bypass path already uses continuous latent, no need for infill
        return torch.tensor(0.0, device=rec_wav.device)

    # Use existing STFT loss
    return stft_loss(rec_wav, target_wav, cfg)


# ============================================
# ResidualCorrector Losses
# ============================================

def residual_loss(r_hat, r_target):
    """
    Residual prediction loss: train corrector to predict quantization error.

    L_residual = ||r_hat - r_target||^2

    Where:
    - r_target = z_continuous - z_q (ground truth quantization error)
    - r_hat = corrector(z_q, indices) (predicted residual from indices only)

    Args:
        r_hat: predicted residual [B, T, D]
        r_target: target residual (z_continuous - z_q) [B, T, D]

    Returns:
        MSE loss between predicted and target residuals
    """
    if r_hat is None or r_target is None:
        return torch.tensor(0.0, device=r_hat.device if r_hat is not None else r_target.device)

    return F.mse_loss(r_hat, r_target)


def alignment_loss(z_continuous, z_corrected, use_cosine=True):
    """
    Alignment loss: align continuous and corrected discrete representations.

    Ensures corrected tokens (z_q + α*r_hat) are close to continuous latent.

    L_align = ||z_continuous - z_corrected||^2  (MSE)
           or 1 - cosine(z_continuous, z_corrected)  (cosine)

    Args:
        z_continuous: continuous encoder output [B, T, D]
        z_corrected: corrected quantized tokens z_q + α*r_hat [B, T, D]
        use_cosine: if True, use cosine similarity; else MSE

    Returns:
        Alignment loss between continuous and corrected representations
    """
    if z_continuous is None or z_corrected is None:
        device = z_continuous.device if z_continuous is not None else z_corrected.device
        return torch.tensor(0.0, device=device)

    if use_cosine:
        # Cosine alignment: 1 - cosine_similarity
        return cosine_align(z_continuous, z_corrected)
    else:
        # L2 alignment
        return F.mse_loss(z_corrected, z_continuous)
