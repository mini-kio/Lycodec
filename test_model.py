"""
A-PLAN Integration Test: Single RVQ with Dropout Trio
"""
import torch
from lycodec.model import Lycodec

print("=" * 60)
print("A-PLAN: Single RVQ Architecture Test")
print("=" * 60)

# Initialize model with A-plan (FSQ removed, RVQ dropout trio)
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
    semantic_dim=120,  # Kept for compatibility
    decoder_depth=8,  # Increased from 6 to 8
    decoder_patch_size=16,
    rvq_codebook_size=4096,
    token_fps=24,
)

print("\n[1/4] Model initialized successfully!")
print(f"  - RVQ codebook size: {model.rvq.K}")
print(f"  - Token FPS: {model.token_fps}")
print(f"  - Token dim: {model.token_dim}")
print(f"  - Decoder depth: {model.decoder.depth}")
print(f"  - RVQ dropout trio: p_mask={model.rvq.p_mask}, p_jitter={model.rvq.p_jitter}, p_bypass={model.rvq.p_bypass}")

# Test encoding
print("\n[2/4] Testing encoding...")
wav = torch.randn(2, 2, 72000)  # Batch=2, Channels=2, Time=1.5s @ 48kHz
model.eval()

with torch.no_grad():
    enc = model.encode(wav)

print(f"  - Input shape: {wav.shape}")
print(f"  - Encoded tokens shape: {enc['tokens'].shape}")
print(f"  - RVQ indices shape: {enc['indices'].shape}")
print(f"  - RVQ indices range: [{enc['indices'].min()}, {enc['indices'].max()}]")
print(f"  - Perplexity: {enc['perplexity'].item():.2f} / 4096")
print(f"  - Commitment loss: {enc['commitment_loss'].item():.6f}")
print(f"  - Bypass applied: {enc['bypass_applied']}")

# Test decoding
print("\n[3/4] Testing decoding...")
with torch.no_grad():
    rec = model.decode(enc['tokens'], length=72000)

print(f"  - Reconstructed shape: {rec.shape}")
assert rec.shape == wav.shape, f"Shape mismatch: {rec.shape} != {wav.shape}"
print(f"  - Reconstruction MSE: {((rec - wav) ** 2).mean().item():.6f}")

# Test full forward pass
print("\n[4/4] Testing full forward pass...")
with torch.no_grad():
    rec_full, enc_full = model(wav, decode=True)

print(f"  - Full reconstruction shape: {rec_full.shape}")
print(f"  - Spec clean shape: {enc_full['spec_clean'].shape}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)

# Print summary
print("\n[Architecture Summary - A PLAN]")
print(f"  - Single RVQ: C=1, K={model.rvq.K}, {model.token_fps} fps")
print(f"  - Token rate: {model.token_fps} tokens/sec = {model.token_fps * 12} bits/sec (log2(4096)=12)")
print(f"  - 3min music: {3 * 60 * model.token_fps} tokens")
print(f"  - FSQ removed - quality via dropout trio + decoder boost")
print(f"  - Dropout trio:")
print(f"    - Mask: {model.rvq.p_mask} (will decay 0.10→0.02)")
print(f"    - Jitter: {model.rvq.p_jitter} (will decay 0.20→0.05)")
print(f"    - Bypass: {model.rvq.p_bypass} (will decay 0.20→0.0)")
print(f"  - Decoder: Consistency 1-step (Transformer2D, depth={model.decoder.depth})")
print("\n[Model ready for training!]")
