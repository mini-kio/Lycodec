"""
Check Lycodec model size (parameter count)
"""
import torch
from lycodec.model import Lycodec


def count_parameters(model, trainable_only=False):
    """Count total parameters in model"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n):
    """Format number with K/M/B suffix"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


def analyze_model_size():
    print("="*60)
    print("LYCODEC MODEL SIZE ANALYSIS")
    print("="*60)

    # Create model with Group FSQ + RoPE
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

    print("\nConfiguration:")
    print(f"  - hidden_dim: 512")
    print(f"  - token_dim: 256")
    print(f"  - transformer_layers: 8")
    print(f"  - heads: 8")
    print(f"  - use_rope: True")
    print(f"  - use_group_fsq: True (4 groups)")

    # Overall counts
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\n" + "="*60)
    print(f"TOTAL PARAMETERS: {format_number(total_params)} ({total_params:,})")
    print(f"TRAINABLE PARAMETERS: {format_number(trainable_params)} ({trainable_params:,})")
    print("="*60)

    # Breakdown by component
    print("\nDETAILED BREAKDOWN:")
    print("-"*60)

    components = {
        "Encoder Path": {
            "Patchifier": model.patch,
            "Encoder Projection": model.enc_proj,
            "Temporal Resampler": model.resampler,
            "Transformer Encoder": model.encoder,
            "To Token Projection": model.to_token,
            "Hybrid Latent": model.hybrid,
            "Stereo Head": model.stereo,
        },
        "Decoder Path": {
            "Token Conditioner": model.cond,
            "UNet2D": model.unet,
            "BandSplit Head": model.bands,
        }
    }

    encoder_total = 0
    decoder_total = 0

    for path_name, modules in components.items():
        print(f"\n{path_name}:")
        path_total = 0

        for name, module in modules.items():
            params = count_parameters(module)
            path_total += params
            print(f"  {name:25s}: {format_number(params):>10s} ({params:,})")

        print(f"  {'─'*25}  {'─'*10}")
        print(f"  {'SUBTOTAL':25s}: {format_number(path_total):>10s} ({path_total:,})")

        if path_name == "Encoder Path":
            encoder_total = path_total
        else:
            decoder_total = path_total

    # FSQ has no parameters
    print(f"\nGroup FSQ (4 groups):")
    print(f"  Parameters: 0 (quantization only)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("-"*60)
    print(f"Encoder Path:  {format_number(encoder_total):>10s} ({encoder_total:,})")
    print(f"Decoder Path:  {format_number(decoder_total):>10s} ({decoder_total:,})")
    print(f"{'─'*40}")
    print(f"TOTAL:         {format_number(total_params):>10s} ({total_params:,})")
    print("="*60)

    # Memory estimation
    print("\nMEMORY ESTIMATION (FP32):")
    print(f"  Model weights: {total_params * 4 / 1024**2:.2f} MB")
    print(f"  + Optimizer states (AdamW): ~{total_params * 12 / 1024**2:.2f} MB")
    print(f"  + Activations (batch=8): ~{estimate_activation_memory()} MB")
    print(f"  Estimated total: ~{(total_params * 16 / 1024**2) + estimate_activation_memory():.2f} MB")

    print("\nMEMORY ESTIMATION (FP16/AMP):")
    print(f"  Model weights: {total_params * 2 / 1024**2:.2f} MB")
    print(f"  + Optimizer states: ~{total_params * 6 / 1024**2:.2f} MB")
    print(f"  + Activations (batch=8): ~{estimate_activation_memory()/2:.2f} MB")
    print(f"  Estimated total: ~{(total_params * 8 / 1024**2) + estimate_activation_memory()/2:.2f} MB")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER MODELS:")
    print("-"*60)
    print(f"  Lycodec (ours):        {format_number(total_params):>10s}")
    print(f"  EnCodec 24kHz:         ~35M (estimated)")
    print(f"  SoundStream:           ~50M (estimated)")
    print(f"  DAC (44.1kHz):         ~74M")
    print("="*60)

    return total_params


def estimate_activation_memory():
    """Rough estimation of activation memory for batch_size=8"""
    # This is very approximate
    # Main activations: spectrogram, transformer, decoder
    spec_memory = 8 * 4 * 1025 * 113 * 4 / 1024**2  # [B, 4, F, T]
    transformer_memory = 8 * 18 * 512 * 4 / 1024**2  # [B, T, D]
    decoder_memory = 8 * 4 * 1025 * 113 * 4 / 1024**2  # [B, 4, F, T]

    return spec_memory + transformer_memory + decoder_memory


if __name__ == "__main__":
    analyze_model_size()
