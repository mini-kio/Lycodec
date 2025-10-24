"""ResidualCorrector integration test for shared-latent setup."""
import torch
from lycodec.model import Lycodec

print("=" * 60)
print("ResidualCorrector Shared-Latent Test")
print("=" * 60)

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
    decoder_depth=8,
    decoder_patch_size=16,
    rvq_codebook_size=4096,
    token_fps=24,
    use_residual_corrector=True,
    corrector_alpha=0.3,
)

wav = torch.randn(2, 2, 72000)

print("\n[train] verifying targets and correction")
model.train()
with torch.no_grad():
    enc_train = model.encode(wav)

alpha = model.corrector_alpha
print(f"  - alpha: {alpha}")
print(f"  - r_target is None? {enc_train['r_target'] is None}")
print(f"  - r_hat is None? {enc_train['r_hat'] is None}")
if enc_train['r_target'] is not None and enc_train['r_hat'] is not None:
    expected_target = enc_train['alignment_target'] - enc_train['y_disc'].detach()
    target_error = (enc_train['r_target'] - expected_target).abs().mean().item()
    print(f"  - r_target consistency: {target_error:.6e}")
    predicted = enc_train['y_disc'] + alpha * enc_train['r_hat']
    correction_error = (predicted - enc_train['y_disc_corrected']).abs().mean().item()
    print(f"  - correction error: {correction_error:.6e}")
    assert target_error < 1e-6
    assert correction_error < 1e-6

print("\n[eval] residual path without continuous target")
model.eval()
with torch.no_grad():
    enc_eval = model.encode(wav)
print(f"  - alignment target (eval): {enc_eval['alignment_target']}")
print(f"  - r_target (eval): {enc_eval['r_target']}")
print(f"  - r_hat shape: {enc_eval['r_hat'].shape if enc_eval['r_hat'] is not None else None}")
print(f"  - drop prob (eval): {enc_eval['drop_prob']:.3f}")

print("\n[decode]")
with torch.no_grad():
    rec = model.decode(enc_eval['tokens'], length=wav.shape[-1])
print(f"  - reconstruction shape: {rec.shape}")
print(f"  - mse: {((rec - wav) ** 2).mean().item():.6f}")

print("\nDone.")
print("=" * 60)
