# Lycodec: Neural Audio Codec with Group FSQ and RoPE

A high-quality neural audio codec for 48kHz stereo audio using Group Finite Scalar Quantization (Group FSQ) and Rotary Position Embeddings (RoPE).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

Lycodec is a neural audio codec designed for high-fidelity 48kHz stereo audio compression at ~10.6 kbps. It combines several state-of-the-art techniques:

- **Group FSQ**: Splits quantization into 4 groups for better representation flexibility
- **RoPE**: Rotary Position Embeddings for efficient temporal modeling
- **Consistency Model**: One-step generation without iterative sampling
- **Mid/Side Processing**: Better stereo representation
- **Teacher Alignment**: Optional alignment with MERT/HuBERT for semantic features

## Key Features

### 🎯 Architecture Highlights

- **37.72M parameters**: Compact and efficient model
- **12 Hz token rate**: 18 tokens per 1.5s chunk
- **~10.6 kbps bitrate**: High compression with maintained quality
- **One-step generation**: Fast inference with consistency model
- **Streaming support**: Chunked causal attention for 0.75s latency

### 🚀 Novel Contributions

#### 1. Group Finite Scalar Quantization (Group FSQ)

Traditional FSQ applies the same quantization level to all dimensions. Group FSQ splits the token dimension into multiple groups, allowing different quantization granularity:

```python
# 256 dimensions split into 4 groups of 64 dims each
Group 0: 64 dims × 11 levels (N=5)
Group 1: 64 dims × 11 levels (N=5)
Group 2: 64 dims × 11 levels (N=5)
Group 3: 64 dims × 11 levels (N=5)
```

**Benefits:**
- Same total bitrate as single FSQ
- Better representation flexibility
- No additional parameters
- Each group can specialize (e.g., coarse/fine features)

**Technical Details:**
```python
z_q = round(N * tanh(z)) / N  # Per-group quantization
Total bits per token: 885.6 (3.46 bits/dim)
Bitrate: 18 tokens × 885.6 bits / 1.5s ≈ 10.6 kbps
```

#### 2. Rotary Position Embeddings (RoPE)

Instead of learnable or sinusoidal position embeddings, RoPE encodes position through rotation in complex space:

```python
# Applied to Query and Key in attention
x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
where θ_i = position / (10000^(2i/d))
```

**Benefits:**
- No learnable parameters
- Better extrapolation to longer sequences
- Naturally captures relative positions
- Efficient computation

#### 3. Consistency Model Decoder

Inspired by Consistency Models (Song et al., 2023), our decoder generates audio in **one step**:

```python
# Traditional diffusion: 50-1000 steps
x_0 = denoise(x_T) → denoise(...) → x_0  # Slow!

# Consistency model: 1 step
x_0 = f(x_σ, σ)  # Fast!
```

**Training:**
- Distills diffusion process into single-step function
- EMA teacher for stability
- EDM parameterization (Karras et al., 2022)
- Pseudo-Huber loss for robustness

**Loss Function:**
```python
L = E[d(f(x+σ_i·ε, σ_i), sg(f(x+σ_{i+1}·ε, σ_{i+1}))) / Δσ]
```

Where:
- `f`: Consistency function (student)
- `sg`: Stop-gradient (EMA teacher)
- `d`: Pseudo-Huber distance
- `Δσ = σ_{i+1} - σ_i`: Noise level difference

## Architecture

### Encoder Path (32.80M parameters)

```
Stereo Audio [B, 2, 72000] (1.5s @ 48kHz)
    ↓
[1] Mid/Side Transform
    L = (Left + Right) / 2
    R = (Left - Right) / 2
    ↓
[2] STFT (n_fft=2048, hop=640)
    → [B, 4, 1025, 113] (real_mid, imag_mid, real_side, imag_side)
    ↓
[3] Patchifier (Conv2D)
    Widths: 64 → 128 → 256 → 512
    Strides: (2,2) → (2,2) → (2,2) → (2,1)
    → [B, 512, 64, 113]
    ↓
[4] Frequency Pooling
    → [B, 512, 113]
    ↓
[5] Temporal Resampler
    113 frames → 18 tokens (12 Hz)
    → [B, 512, 18]
    ↓
[6] Transformer Encoder (8 layers)
    - RoPE for position encoding
    - Chunked causal attention
    - 8 heads, 512 hidden dim
    → [B, 18, 512]
    ↓
[7] Token Projection
    → [B, 18, 256]
    ↓
[8] Group FSQ (4 groups)
    → z_continuous, z_discrete
    ↓
[9] Hybrid Latent
    z_h = z_discrete + α × residual_net(z_continuous - z_discrete)
    → [B, 18, 256]
```

### Decoder Path (4.92M parameters)

```
Tokens [B, 18, 256]
    ↓
[1] Token Conditioner
    - Linear projection: 256 → 64
    - Temporal upsampling: 18 → 113
    - Frequency broadcasting: 1 → 1025
    - Noise level embedding
    → [B, 64, 1025, 113]
    ↓
[2] Noise Initialization
    x_σ = noise × σ (inference: σ = 1e-3)
    → [B, 4, 1025, 113]
    ↓
[3] UNet2D (with conditioning)
    Encoder: 3 downsampling layers
    Decoder: 3 upsampling layers + skip connections
    → [B, 4, 1025, 113]
    ↓
[4] Consistency Function
    F_θ = UNet output
    spec_pred = c_skip × x_σ + c_out × F_θ
    (EDM parameterization)
    ↓
[5] Band-Split Head
    - Low band (< 12kHz): base reconstruction
    - High band (> 12kHz): conditioned on low
    → [B, 4, 1025, 113]
    ↓
[6] ISTFT + Mid/Side to Stereo
    → [B, 2, 72000]
```

## Specifications

| Parameter | Value |
|-----------|-------|
| Sample Rate | 48 kHz |
| Token Rate | 12 Hz |
| Chunk Duration | 1.5 s (18 tokens) |
| Hop Duration | 0.75 s (overlap-add) |
| Bitrate | ~10.6 kbps |
| Token Dimension | 256 |
| Hidden Dimension | 512 |
| Transformer Layers | 8 |
| Attention Heads | 8 |
| **Model Size** | **37.72M parameters** |
| **Memory (FP16)** | **~302 MB (batch=8)** |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/mini-kio/Lycodec.git
cd Lycodec

# Install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Optional: Teacher Models

For teacher alignment (MERT/HuBERT):

```bash
pip install transformers
```

## Usage

### Training

```bash
python -m lycodec.cli train \
  --config configs/lycodec_48k.yaml \
  --data /path/to/audio/data
```

**Configuration:** Edit `configs/lycodec_48k.yaml` to customize training:

```yaml
# Model architecture
model:
  hidden_dim: 512
  token_dim: 256
  transformer_layers: 8
  heads: 8
  use_rope: true          # Enable RoPE
  use_group_fsq: true     # Enable Group FSQ (4 groups)

# Training
train:
  stage: 2                # Single-stage training
  stage1_steps: 0
  stage2_steps: 10000
  use_amp: true           # Mixed precision
  grad_clip: 1.0

  # FSQ dropout schedule
  fsq_dropout_schedule:
    - [0, 0.50]
    - [5000, 0.65]
    - [10000, 0.75]

  # STFT auxiliary loss schedule
  recon_weight_schedule:
    - [0, 0.1]            # 0-2000 steps
    - [2000, 0.0]         # After 2000 steps

# Teacher alignment (optional)
teacher:
  primary:
    enabled: true
    type: mert
    model_name: m-a-p/MERT-v1-330M
    align_weight: 0.1
  aux:
    enabled: true
    type: hubert
    model_name: ZhenYe234/hubert_base_general_audio
    align_weight: 0.05
```

### Encoding Audio

```bash
python -m lycodec.cli encode \
  --config configs/lycodec_48k.yaml \
  --ckpt runs/ckpts/latest.pt \
  --audio input.wav \
  --out tokens.npz
```

**Output Format:**
- `tokens.npz`: NumPy archive containing:
  - `tokens`: [N, 18, 256] encoded tokens
  - `sr`: Sample rate
  - `length`: Original length
  - `seg`, `hop`: Chunking parameters

### Decoding Audio

```bash
python -m lycodec.cli decode \
  --config configs/lycodec_48k.yaml \
  --ckpt runs/ckpts/latest.pt \
  --tokens tokens.npz \
  --out output.wav
```

**Decoding Process:**
- Overlap-add with 0.75s hop (50% overlap)
- Linear crossfade in overlap regions
- Supports arbitrary length audio

### Python API

```python
import torch
from lycodec.model import Lycodec

# Load model
model = Lycodec(
    sr=48000,
    n_fft=2048,
    hop=640,
    win=2048,
    token_dim=256,
    hidden=512,
    layers=8,
    heads=8,
    use_rope=True,
    use_group_fsq=True
)
model.load_state_dict(torch.load("checkpoint.pt")["model"])
model.eval()

# Encode
audio = torch.randn(1, 2, 72000)  # [batch, channels, samples]
with torch.no_grad():
    enc = model.encode(audio)
    tokens = enc["tokens"]  # [1, 18, 256]

# Decode
with torch.no_grad():
    reconstructed = model.decode(tokens, length=72000)  # [1, 2, 72000]
```

## Loss Functions

### 1. Consistency Loss (Main)

```python
L_consistency = E[d(f(x+σ_i·ε, σ_i), sg(f(x+σ_{i+1}·ε, σ_{i+1}))) / Δσ]
```

- **Type**: Pseudo-Huber distance
- **Weight**: 1.0
- **Purpose**: One-step generation capability
- **EMA Teacher**: Used for stability

### 2. STFT Loss (Auxiliary, Scheduled)

```python
L_stft = Σ_{scales} (|STFT(x)|_1 + |phase(STFT(x))|_1)
```

- **Scales**: [128, 256, 512, 1024] hop lengths
- **Windows**: [512, 1024, 2048, 4096]
- **Weight**: 0.1 (steps 0-2000), 0.0 (after)
- **Purpose**: Early training stability

### 3. Teacher Alignment (Optional)

```python
L_align = 1 - cosine_similarity(proj_student, proj_teacher)
```

- **MERT (primary)**: Weight 0.1
- **HuBERT (auxiliary)**: Weight 0.05
- **Window**: 0.4s local attention
- **Purpose**: Semantic feature alignment

### 4. Stereo Loss (Auxiliary)

```python
L_stereo = MSE(ILD_pred, ILD_target)
```

- **ILD**: Inter-aural Level Difference
- **Weight**: 0.1
- **Purpose**: Stereo consistency

### Total Loss

```python
L_total = L_consistency + w_stft × L_stft + w_align × L_align + w_stereo × L_stereo
```

## Training Tips

### Memory Optimization

```yaml
# For limited GPU memory (< 12GB)
batch_size: 4              # Reduce batch size
train:
  use_checkpoint: true     # Gradient checkpointing
  gradient_accumulation_steps: 2  # Accumulate gradients
```

### Single-Stage Training

Current configuration uses **single-stage training** (consistency from step 0):

```yaml
train:
  stage: 2
  stage1_steps: 0          # No encoder-only stage
  stage2_steps: 10000      # Full model training
```

**Benefits:**
- Simpler training pipeline
- Better encoder-decoder co-adaptation
- Faster convergence

### FSQ Dropout Schedule

FSQ dropout helps with training stability:

```yaml
fsq_dropout_schedule:
  - [0, 0.50]      # 50% dropout initially
  - [5000, 0.65]   # Increase to 65%
  - [10000, 0.75]  # Final 75%
```

Higher dropout → more continuous path → easier optimization

## Model Comparison

| Model | Params | Sample Rate | Bitrate | Token Rate | Decoding |
|-------|--------|-------------|---------|------------|----------|
| **Lycodec** | **37.7M** | **48 kHz** | **~10.6 kbps** | **12 Hz** | **1 step** |
| EnCodec | ~35M | 24 kHz | 6-24 kbps | 75 Hz | Autoregressive |
| SoundStream | ~50M | 24 kHz | 3-18 kbps | 50 Hz | Autoregressive |
| DAC | ~74M | 44.1 kHz | 8-16 kbps | 50 Hz | Autoregressive |

**Advantages:**
- ✅ Higher sample rate (48 kHz)
- ✅ One-step generation (faster)
- ✅ Comparable model size
- ✅ Group FSQ for better representation
- ✅ RoPE for efficient position encoding

## Evaluation

### Objective Metrics

- **SI-SNR** (Scale-Invariant SNR)
- **PESQ** (Perceptual Evaluation of Speech Quality)
- **ViSQOL** (Virtual Speech Quality Objective Listener)
- **Multi-scale STFT Loss**

### Subjective Metrics

- **MOS** (Mean Opinion Score)
- **MUSHRA** (Multiple Stimuli with Hidden Reference and Anchor)
- **ABX** (Preference test)

### Evaluation Script

```bash
# Evaluate on test set
python evaluate.py \
  --config configs/lycodec_48k.yaml \
  --ckpt runs/ckpts/latest.pt \
  --test-data /path/to/test/data \
  --output results.json
```

## Technical Details

### Chunked Causal Attention

For streaming support, we use chunked causal attention:

```
Sequence: 18 tokens = [0, 1, ..., 17]

Left chunk (0-8):   Attends to [0-8] only (causal)
Right chunk (9-17): Attends to [0-17] (full)
```

**Benefits:**
- Enables streaming with 0.75s latency
- Maintains autoregressive structure
- No quality degradation

### Mid/Side Processing

Stereo audio is processed in Mid/Side domain:

```python
Mid = (Left + Right) / 2
Side = (Left - Right) / 2

# After reconstruction
Left = Mid + Side
Right = Mid - Side
```

**Benefits:**
- Decorrelates channels
- Better compression
- Separates spatial information

### Band-Split Decoder

The decoder uses frequency-dependent processing:

```python
Low band (< 12 kHz):  Base reconstruction
High band (> 12 kHz): Conditioned on low band
```

**Benefits:**
- Better high-frequency reconstruction
- Learnable blend weights
- Smooth transition at 12 kHz

## Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce memory usage
batch_size: 4
train:
  use_checkpoint: true
  gradient_accumulation_steps: 2
```

### Teacher Model Download Fails

```yaml
# Disable teacher alignment
teacher:
  primary:
    enabled: false
  aux:
    enabled: false
```

### Slow Training

```yaml
# Optimize training speed
num_workers: 8          # Increase dataloader workers
train:
  use_amp: true         # Use mixed precision
  use_checkpoint: false # Disable if memory allows
```

### NaN Loss

```yaml
# Stabilize training
train:
  grad_clip: 1.0        # Gradient clipping
  use_amp: true         # Mixed precision helps
lr: 0.00005             # Reduce learning rate
```

## Citation

If you use Lycodec in your research, please cite:

```bibtex
@misc{lycodec2024,
  title={Lycodec: Neural Audio Codec with Group FSQ and RoPE},
  author={mini-kio},
  year={2024},
  howpublished={\url{https://github.com/mini-kio/Lycodec}}
}
```

## References

### Core Papers

1. **Finite Scalar Quantization (FSQ)**
   - Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", 2023
   - [Paper](https://arxiv.org/abs/2309.15505)

2. **Rotary Position Embeddings (RoPE)**
   - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
   - [Paper](https://arxiv.org/abs/2104.09864)

3. **Consistency Models**
   - Song et al., "Consistency Models", 2023
   - [Paper](https://arxiv.org/abs/2303.01469)

4. **EDM**
   - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022
   - [Paper](https://arxiv.org/abs/2206.00364)

### Related Audio Codecs

5. **EnCodec**
   - Défossez et al., "High Fidelity Neural Audio Compression", 2022
   - [Paper](https://arxiv.org/abs/2210.13438)

6. **SoundStream**
   - Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", 2021
   - [Paper](https://arxiv.org/abs/2107.03312)

7. **DAC**
   - Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN", 2023
   - [Paper](https://arxiv.org/abs/2306.06546)

### Teacher Models

8. **MERT**
   - Li et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training", 2023
   - [Paper](https://arxiv.org/abs/2306.00107)

9. **HuBERT**
   - Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction", 2021
   - [Paper](https://arxiv.org/abs/2106.07447)

## License

Apache-2.0 License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by EnCodec, SoundStream, and DAC
- Uses techniques from FSQ, RoPE, and Consistency Models
- Teacher alignment based on MERT and HuBERT
- Thanks to the open-source community

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues:
- GitHub Issues: [https://github.com/mini-kio/Lycodec/issues](https://github.com/mini-kio/Lycodec/issues)
- Email: kiolaaoz@naver.com

---

**Status**: Experimental (Under Development)

This is a research project. Performance may vary depending on training data and hyperparameters. Contributions and feedback are welcome!
