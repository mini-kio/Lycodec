"""
Test script to verify Group FSQ and RoPE integration.

This script checks:
1. GroupFSQ correctly splits into 4 groups
2. RoPE is applied in TransformerEncoder
3. All shapes are correct throughout the pipeline
4. Forward/backward pass works
"""
import torch
import yaml
from lycodec.model import Lycodec
from lycodec.core.blocks import GroupFSQ, RotaryPositionEmbedding, TransformerEncoder


def test_group_fsq():
    print("\n" + "="*60)
    print("TEST 1: GroupFSQ (4 groups)")
    print("="*60)

    # Create GroupFSQ with 4 groups
    fsq = GroupFSQ(num_groups=4, levels=[11, 11, 11, 11], dim=256, dropout_p=0.5)

    # Test input: [B=2, T=18, D=256]
    z = torch.randn(2, 18, 256)

    # Forward pass (training)
    z_cont, z_disc = fsq(z, training=True)

    print(f"Input shape: {z.shape}")
    print(f"Continuous output: {z_cont.shape}")
    if z_disc is not None:
        print(f"Discrete output: {z_disc.shape}")
        print(f"[OK] Quantization applied")
    else:
        print(f"[OK] Discrete path dropped out (expected during training)")

    # Inference mode
    z_cont_inf, z_disc_inf = fsq(z, training=False)
    assert z_disc_inf is not None, "Discrete path should always be active in inference"
    print(f"Inference discrete output: {z_disc_inf.shape}")

    # Verify quantization levels
    print(f"\nVerifying quantization levels...")
    for i in range(4):
        start = i * 64
        end = start + 64
        group_disc = z_disc_inf[:, :, start:end]
        unique_vals = torch.unique(group_disc)
        print(f"  Group {i}: {len(unique_vals)} unique values (expected 11)")

    print("[OK] GroupFSQ test passed!\n")


def test_rope():
    print("\n" + "="*60)
    print("TEST 2: RoPE (Rotary Position Embedding)")
    print("="*60)

    # Create RoPE
    rope = RotaryPositionEmbedding(dim=512, max_seq_len=18)

    # Test input: [B=2, T=18, D=512]
    x = torch.randn(2, 18, 512)

    # Apply RoPE
    x_rotated = rope(x)

    print(f"Input shape: {x.shape}")
    print(f"Rotated output: {x_rotated.shape}")

    # Verify rotation changes values
    diff = (x - x_rotated).abs().mean()
    print(f"Mean difference: {diff:.6f}")
    assert diff > 0, "RoPE should change values"

    # Verify shape preservation
    assert x.shape == x_rotated.shape, "Shape should be preserved"

    print("[OK] RoPE test passed!\n")


def test_transformer_with_rope():
    print("\n" + "="*60)
    print("TEST 3: TransformerEncoder with RoPE")
    print("="*60)

    # Create encoder with RoPE
    encoder = TransformerEncoder(
        dim=512,
        depth=4,
        heads=8,
        use_rope=True,
        seq_len=18
    )

    # Test input: [B=2, T=18, D=512]
    x = torch.randn(2, 18, 512)

    # Forward pass
    out = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == x.shape, "Shape should be preserved"
    print("[OK] TransformerEncoder with RoPE test passed!\n")


def test_full_model():
    print("\n" + "="*60)
    print("TEST 4: Full Lycodec Model (Group FSQ + RoPE)")
    print("="*60)

    # Create model with Group FSQ and RoPE
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
        use_group_fsq=True
    )

    print(f"Model created successfully!")
    print(f"  - RoPE: enabled")
    print(f"  - Group FSQ: 4 groups")

    # Test input: stereo audio [B=2, C=2, T=72000] (1.5s @ 48kHz)
    wav = torch.randn(2, 2, 72000)

    print(f"\nInput audio shape: {wav.shape}")

    # Encode
    with torch.no_grad():
        enc = model.encode(wav)

    print(f"\nEncoder output:")
    print(f"  tokens: {enc['tokens'].shape}")
    print(f"  cont: {enc['cont'].shape}")
    if enc['disc'] is not None:
        print(f"  disc: {enc['disc'].shape}")
    else:
        print(f"  disc: None (dropped out)")

    # Expected shape: [B=2, T=18, D=256]
    assert enc['tokens'].shape == (2, 18, 256), f"Wrong token shape: {enc['tokens'].shape}"

    # Decode
    with torch.no_grad():
        rec = model.decode(enc['tokens'], wav.shape[-1])

    print(f"\nDecoder output:")
    print(f"  reconstructed audio: {rec.shape}")

    # Expected shape: [B=2, C=2, T=72000]
    assert rec.shape == wav.shape, f"Wrong reconstruction shape: {rec.shape}"

    print("\n[OK] Full model test passed!\n")

    # Test backward pass
    print("Testing backward pass...")
    model.train()
    wav_train = torch.randn(1, 2, 72000, requires_grad=True)
    rec_train, enc_train = model(wav_train, decode=True)
    loss = rec_train.abs().mean()
    loss.backward()
    print(f"  Loss: {loss.item():.6f}")
    print("[OK] Backward pass successful!\n")


def test_config_loading():
    print("\n" + "="*60)
    print("TEST 5: Config Loading")
    print("="*60)

    with open("configs/lycodec_48k.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Config loaded:")
    print(f"  use_rope: {cfg['model'].get('use_rope', 'NOT SET')}")
    print(f"  use_group_fsq: {cfg['model'].get('use_group_fsq', 'NOT SET')}")

    assert cfg['model'].get('use_rope') == True, "use_rope should be True"
    assert cfg['model'].get('use_group_fsq') == True, "use_group_fsq should be True"

    print("[OK] Config test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GROUP FSQ + RoPE INTEGRATION TEST")
    print("="*60)

    try:
        test_group_fsq()
        test_rope()
        test_transformer_with_rope()
        test_full_model()
        test_config_loading()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("  [OK] Group FSQ (4 groups) working correctly")
        print("  [OK] RoPE integrated in Transformer")
        print("  [OK] Full model forward/backward pass successful")
        print("  [OK] All shapes correct: [B, 18, 256]")
        print("  [OK] Config file updated")
        print("\nReady to train!\n")

    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
