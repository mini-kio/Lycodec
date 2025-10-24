"""Lightweight sanity check for Lycodec shared-latent pipeline."""
import torch
from lycodec.model import Lycodec

print("=" * 60)
print("Lycodec Shared-Latent Test")
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
    rvq_drop_start=0.6,
    rvq_drop_end=0.1,
    rvq_drop_decay_steps=50000,
)

print("\n[init] model ready")
print(f"  - token dim: {model.token_dim}")
print(f"  - token fps: {model.token_fps}")
print(f"  - RVQ drop schedule: 0.6 → 0.1")

wav = torch.randn(2, 2, 72000)

print("\n[train] encode pass")
model.train()
with torch.no_grad():
    enc_train = model.encode(wav)
print(f"  - tokens shape: {enc_train['tokens'].shape}")
print(f"  - indices shape: {enc_train['indices'].shape}")
print(f"  - dropout applied: {enc_train['dropout_applied']}")
print(f"  - drop prob: {enc_train['drop_prob']:.3f}")
print(f"  - alignment target present: {enc_train['alignment_target'] is not None}")
print(f"  - y_disc shape: {enc_train['y_disc'].shape}")

print("\n[eval] encode pass")
model.eval()
with torch.no_grad():
    enc_eval = model.encode(wav)
print(f"  - tokens shape: {enc_eval['tokens'].shape}")
print(f"  - drop prob (eval): {enc_eval['drop_prob']:.3f}")
print(f"  - alignment target (eval): {enc_eval['alignment_target']}")

print("\n[decode]")
with torch.no_grad():
    rec = model.decode(enc_eval['tokens'], length=wav.shape[-1])
print(f"  - reconstruction shape: {rec.shape}")
print(f"  - mse: {((rec - wav) ** 2).mean().item():.6f}")

print("\nAll done.")
print("=" * 60)
