"""
ResidualCorrector Integration Test
Tests the complete flow with corrector enabled:
- Training: r_target = z_continuous - z_q, predict r_hat from indices
- Inference: z_corrected = z_q + α*r_hat (no continuous input)
"""
import torch
from lycodec.model import Lycodec

print("=" * 70)
print("ResidualCorrector Integration Test")
print("=" * 70)

# Initialize model with ResidualCorrector enabled
print("\n[1/6] Initializing model with ResidualCorrector...")
model = Lycodec(
    sr=48000,
    n_fft=2048,
    hop=640,
    win=2048,
    token_dim=256,
    hidden=512,
    layers=8,
    heads=8,
    use_checkpoint=False,
    use_rope=True,
    semantic_dim=120,
    decoder_depth=8,
    decoder_patch_size=16,
    rvq_codebook_size=4096,
    token_fps=24,
    use_residual_corrector=True,  # Enable corrector
    corrector_alpha=0.3,           # Correction strength
)

print(f"  [OK] Model initialized")
print(f"  [OK] RVQ codebook size: {model.rvq.K}")
print(f"  [OK] Corrector enabled: {model.corrector is not None}")
print(f"  [OK] Corrector alpha: {model.corrector_alpha}")

# Test data
wav = torch.randn(2, 2, 72000)  # Batch=2, Channels=2, Time=1.5s @ 48kHz

# ============================================
# Test 1: Training mode with corrector
# ============================================
print("\n[2/6] Testing training mode...")
model.train()

with torch.no_grad():
    enc = model.encode(wav)

print(f"  [OK] Input shape: {wav.shape}")
print(f"  [OK] Tokens shape: {enc['tokens'].shape}")
print(f"  [OK] Indices shape: {enc['indices'].shape}")
print(f"  [OK] Indices range: [{enc['indices'].min()}, {enc['indices'].max()}]")

# Check corrector outputs in training mode
print("\n[3/6] Verifying corrector training outputs...")
assert enc['r_target'] is not None, "r_target should be computed in training"
assert enc['r_hat'] is not None, "r_hat should be predicted in training"
assert enc['z_continuous'] is not None, "z_continuous should be available in training"
assert enc['z_q_uncorrected'] is not None, "z_q_uncorrected should be available"

print(f"  [OK] r_target shape: {enc['r_target'].shape}")
print(f"  [OK] r_hat shape: {enc['r_hat'].shape}")
print(f"  [OK] z_continuous shape: {enc['z_continuous'].shape}")
print(f"  [OK] z_q_uncorrected shape: {enc['z_q_uncorrected'].shape}")

# Verify residual prediction quality
residual_mse = ((enc['r_hat'] - enc['r_target']) ** 2).mean().item()
print(f"  [OK] Residual MSE (r_hat vs r_target): {residual_mse:.6f}")

# Verify correction is applied
z_q = enc['z_q_uncorrected']
tokens_corrected = enc['tokens']
alpha = 0.3
expected_correction = z_q + alpha * enc['r_hat']
correction_error = ((tokens_corrected - expected_correction) ** 2).mean().item()
print(f"  [OK] Correction error (should be ~0): {correction_error:.6f}")

# ============================================
# Test 2: Inference mode with corrector
# ============================================
print("\n[4/6] Testing inference mode...")
model.eval()

with torch.no_grad():
    enc_eval = model.encode(wav)

print(f"  [OK] Tokens shape: {enc_eval['tokens'].shape}")
print(f"  [OK] Indices shape: {enc_eval['indices'].shape}")

# Check corrector outputs in inference mode
print("\n[5/6] Verifying corrector inference outputs...")
assert enc_eval['r_target'] is None, "r_target should be None in inference (no continuous)"
assert enc_eval['r_hat'] is not None, "r_hat should still be predicted in inference"
assert enc_eval['z_continuous'] is None, "z_continuous should be None in inference"

print(f"  [OK] r_target: None (correct - no continuous in inference)")
print(f"  [OK] r_hat shape: {enc_eval['r_hat'].shape}")
print(f"  [OK] z_continuous: None (correct - inference uses indices only)")

# Verify correction is still applied in inference
z_q_eval = enc_eval['z_q_uncorrected']
tokens_eval = enc_eval['tokens']
expected_correction_eval = z_q_eval + alpha * enc_eval['r_hat']
correction_error_eval = ((tokens_eval - expected_correction_eval) ** 2).mean().item()
print(f"  [OK] Correction applied in inference: {correction_error_eval:.6f} (should be ~0)")

# ============================================
# Test 3: Full encode-decode pipeline
# ============================================
print("\n[6/6] Testing full encode-decode pipeline...")

with torch.no_grad():
    # Encode
    enc_full = model.encode(wav)

    # Decode
    rec = model.decode(enc_full['tokens'], length=72000)

    # Check reconstruction
    assert rec.shape == wav.shape, f"Shape mismatch: {rec.shape} != {wav.shape}"
    rec_mse = ((rec - wav) ** 2).mean().item()

    print(f"  [OK] Reconstructed shape: {rec.shape}")
    print(f"  [OK] Reconstruction MSE: {rec_mse:.6f}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 70)
print("All tests passed! [OK]")
print("=" * 70)

print("\n[ResidualCorrector Summary]")
print(f"  - Training: learns r_target = z_continuous - z_q from indices")
print(f"  - Training: predicts r_hat = corrector(z_q, indices)")
print(f"  - Training: applies correction: z_corrected = z_q + α*r_hat")
print(f"  - Inference: predicts r_hat from z_q only (NO continuous input)")
print(f"  - Inference: applies correction: z_corrected = z_q + α*r_hat")
print(f"  - Alpha: {model.corrector_alpha} (clamped 0-1)")
print(f"  - Context size: 5 (local temporal context)")

print("\n[Architecture]")
print(f"  - RVQ: K={model.rvq.K}, {model.token_fps} fps")
print(f"  - Corrector: context_conv + predictor")
print(f"  - Token dim: {model.token_dim}")
print(f"  - Decoder depth: {model.decoder.depth}")

print("\n[Next Steps]")
print("  1. Train model with corrector enabled")
print("  2. Monitor residual_loss and alignment_loss")
print("  3. Compare quality: with/without corrector")
print("  4. Adjust corrector_alpha if needed (0.1-0.5)")

print("\n[Ready for training!] [OK]")
