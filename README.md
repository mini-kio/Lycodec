# Lycodec

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

**Lycodec** is a state-of-the-art neural audio codec for high-quality stereo audio compression at 48kHz. It combines modern transformer architectures with consistency model-based generation and **hybrid latent representation** (continuous + discrete) for superior audio quality and efficient compression.

## 🌟 Key Features

### 🎯 Hybrid Latent Representation
- **Continuous Path**: Preserves fine-grained audio details
- **Discrete Path**: Enables efficient quantization via Group FSQ
- **Learnable Fusion**: Adaptively blends continuous and discrete features
- **Stochastic Dropout**: 65% dropout during training for robustness

### 🚀 Modern Architecture
- **Transformer Decoder**: Replaces traditional UNet with patch-based attention
- **Consistency Model**: One-step generation without iterative sampling
- **RoPE Encoder**: Rotary position embeddings for better temporal modeling
- **Cross-Attention Conditioning**: Tokens directly guide spectrogram generation

### 📊 Advanced Quantization
- **Group FSQ**: 4-group finite scalar quantization (11 levels each)
- **~3.46 bits/dim**: Efficient compression with high quality
- **Adaptive Dropout**: Scheduled quantization dropout for better training

### 🎵 Audio Quality
- **48kHz Stereo**: High-fidelity audio codec
- **Band-Split Head**: Separate processing for low/high frequencies
- **Stereo Enhancement**: ILD/ITD prediction for spatial audio
- **Consistency Loss**: Perceptually-aligned training objective

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Technical Details](#technical-details)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🔧 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Install from source

```bash
# Clone the repository
git clone https://github.com/mini-kio/Lycodec.git
cd Lycodec

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchaudio>=2.0.0
einops
soundfile
librosa
numpy
PyYAML
tqdm
wandb  # optional, for experiment tracking
transformers  # optional, for teacher models
```

---

## 🚀 Quick Start

### Encoding and Decoding

```python
import torch
import soundfile as sf
from lycodec.train import load_model

# Load pretrained model
model = load_model('checkpoints/lycodec_48k.pt', 'configs/lycodec_48k.yaml')
model.eval()

# Load audio (stereo, 48kHz)
wav, sr = sf.read('input.wav', always_2d=True)
wav = torch.from_numpy(wav.T).float().unsqueeze(0)  # [1, 2, T]

# Encode to tokens
with torch.no_grad():
    enc = model.encode(wav)
    tokens = enc['tokens']  # [1, 18, 256] - compressed representation

print(f"Compression: {wav.shape[-1]} samples → {tokens.shape[1]} tokens")

# Decode back to audio
with torch.no_grad():
    reconstructed = model.decode(tokens, wav.shape[-1])  # [1, 2, T]

# Save output
sf.write('output.wav', reconstructed[0].T.numpy(), sr)
```

### Command Line Interface

```bash
# Encode audio to tokens
python -m lycodec.cli encode \
    --config configs/lycodec_48k.yaml \
    --ckpt checkpoints/lycodec_48k.pt \
    --audio input.wav \
    --out tokens.npz

# Decode tokens to audio
python -m lycodec.cli decode \
    --config configs/lycodec_48k.yaml \
    --ckpt checkpoints/lycodec_48k.pt \
    --tokens tokens.npz \
    --out output.wav
```

---

## 🏗️ Model Architecture

### Overview

```
Input: Stereo Audio [B, 2, T] @ 48kHz
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER                                                     │
├─────────────────────────────────────────────────────────────┤
│ STFT + Mid-Side → [B, 4, 1025, 113]                       │
│ Patchifier (4-stage CNN) → [B, 512, H', W']               │
│ Frequency Pooling → [B, 512, T']                          │
│ Temporal Resampler → [B, 512, 18]                         │
│ Transformer Encoder (8 layers, RoPE) → [B, 18, 512]       │
│ Token Projection → [B, 18, 256]                           │
├─────────────────────────────────────────────────────────────┤
│ QUANTIZATION (Hybrid Latent) ⭐                            │
├─────────────────────────────────────────────────────────────┤
│ Group FSQ (4 groups, 11 levels each)                      │
│   ├─ Continuous Path: z_cont = tanh(z)                    │
│   └─ Discrete Path: z_disc = round(z * 5) / 5             │
│ Stochastic Dropout (p=0.65)                               │
│ Hybrid Fusion: z_hybrid = z_disc + α·residual(z_cont)     │
│ Output: [B, 18, 256] tokens                               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ DECODER (Transformer-based) 🚀                             │
├─────────────────────────────────────────────────────────────┤
│ Token Conditioner (with noise σ) → [B, 64, 1025, 113]     │
│ Noisy Spectrogram x_σ → [B, 4, 1025, 113]                 │
│ Patch Embedding (16×16) → [B, 448, 512]                   │
│ + Positional Encoding                                      │
│                                                             │
│ Transformer Blocks (6 layers):                            │
│   ├─ Self-Attention (global context)                      │
│   ├─ Cross-Attention (token conditioning) ⭐              │
│   ├─ AdaLN (noise conditioning)                           │
│   └─ MLP (GELU)                                           │
│                                                             │
│ Unpatchify → [B, 4, 1025, 113]                           │
│ EDM Parameterization (consistency model)                  │
│ Band-Split Head (low/high frequency refinement)           │
├─────────────────────────────────────────────────────────────┤
│ OUTPUT                                                      │
├─────────────────────────────────────────────────────────────┤
│ ISTFT + Mid-Side Decoding → [B, 2, T]                     │
└─────────────────────────────────────────────────────────────┘
Output: Reconstructed Stereo Audio
```

### 🌟 Hybrid Latent Representation (Key Innovation)

Lycodec uses a **hybrid latent space** that combines the best of both continuous and discrete representations:

#### Continuous Path
```python
z_cont = tanh(z)  # Smooth, differentiable representation
```
- Preserves fine-grained audio details
- Enables gradient-based optimization
- Better for perceptual quality

#### Discrete Path
```python
z_disc = round(N * tanh(z)) / N  # Quantized representation
```
- Finite scalar quantization (4 groups × 11 levels)
- Efficient compression
- Better for entropy coding

#### Hybrid Fusion
```python
residual = z_cont - z_disc
residual_enhanced = MLP(residual)
z_hybrid = z_disc + α * residual_enhanced
```
- Learnable blending parameter `α`
- MLP processes continuous residual
- Best of both worlds: compression + quality

#### Stochastic Training
During training, the discrete path is **randomly dropped** with 65% probability:
- Forces model to work with both continuous and discrete features
- Improves robustness
- Scheduled dropout: increases from 50% → 65% → 75% during training

This hybrid approach achieves **superior quality** compared to pure discrete quantization while maintaining **efficient compression**.

---

## 🎓 Training

### Prepare Dataset

Organize your audio files:
```
dataset/
├── train/
│   ├── song1.wav
│   ├── song2.flac
│   └── ...
└── val/
    ├── val1.wav
    └── ...
```

Supported formats: `.wav`, `.flac`, `.mp3`

### Configure Training

Edit `configs/lycodec_48k.yaml`:

```yaml
sample_rate: 48000
crop_seconds: 1.5  # Training chunk duration
batch_size: 8
epochs: 100
lr: 0.0001

model:
  hidden_dim: 512
  token_dim: 256
  transformer_layers: 8  # Encoder layers
  decoder_depth: 6       # Decoder layers
  decoder_patch_size: 16 # Patch size (8, 16, or 32)

train:
  use_amp: true          # Mixed precision training
  use_checkpoint: true   # Gradient checkpointing
  grad_clip: 1.0
  ema:
    enabled: true
    decay: 0.9999
```

### Start Training

```bash
python -m lycodec.cli train \
    --config configs/lycodec_48k.yaml \
    --data /path/to/dataset/train
```

### Training with Teacher Models (Optional)

For better semantic alignment, you can use pretrained teacher models:

```yaml
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

### Monitor Training

With Weights & Biases:
```yaml
logging:
  use_wandb: true
  project: lycodec
  run_name: lycodec_48k_experiment1
```

---

## 🎯 Inference

### Python API

```python
from lycodec.train import load_model
import torch
import soundfile as sf

# Load model
model = load_model('checkpoints/latest.pt', 'configs/lycodec_48k.yaml')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load audio
audio, sr = sf.read('input.wav', always_2d=True)
wav = torch.from_numpy(audio.T).float().unsqueeze(0).to(device)  # [1, 2, T]

# Encode
with torch.no_grad():
    enc = model.encode(wav)
    tokens = enc['tokens']         # [1, 18, 256] - main tokens
    cont = enc['cont']             # Continuous representation
    disc = enc['disc']             # Discrete representation (quantized)

# Decode
with torch.no_grad():
    reconstructed = model.decode(tokens, wav.shape[-1])

# Save
output = reconstructed[0].cpu().T.numpy()
sf.write('output.wav', output, sr)
```

### Streaming Inference

For long audio files, use chunked processing:

```python
from lycodec.cli import _chunk_indices

chunk_seconds = 1.5
hop_seconds = 0.75
sr = 48000

chunk_samples = int(chunk_seconds * sr)
hop_samples = int(hop_seconds * sr)

# Get chunk indices
indices = _chunk_indices(wav.shape[-1], chunk_samples, hop_samples)

# Process chunks
all_tokens = []
for start, end in indices:
    chunk = wav[:, :, start:end]
    enc = model.encode(chunk)
    all_tokens.append(enc['tokens'])

# Decode with overlap-add
# (see cli.py cmd_decode for full implementation)
```

---

## ⚙️ Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 512 | Encoder hidden dimension |
| `token_dim` | 256 | Token/latent dimension |
| `transformer_layers` | 8 | Encoder transformer layers |
| `decoder_depth` | 6 | Decoder transformer layers |
| `decoder_patch_size` | 16 | Decoder patch size (8/16/32) |
| `heads` | 8 | Number of attention heads |
| `use_rope` | true | Use RoPE in encoder |
| `use_group_fsq` | true | Use Group FSQ quantization |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Training batch size |
| `lr` | 1e-4 | Learning rate |
| `use_amp` | true | Mixed precision training |
| `grad_clip` | 1.0 | Gradient clipping threshold |
| `use_checkpoint` | true | Gradient checkpointing |
| `sigma_min` | 0.002 | Minimum noise level |
| `sigma_max` | 80.0 | Maximum noise level |
| `rho` | 7.0 | Noise schedule parameter |

### Quantization Schedule

The FSQ dropout probability increases during training:

```yaml
fsq_dropout_schedule:
  - [0, 0.50]      # Steps 0-4999: 50% dropout
  - [5000, 0.65]   # Steps 5000-9999: 65% dropout
  - [10000, 0.75]  # Steps 10000+: 75% dropout
```

---

## 🔬 Advanced Usage

### Custom Decoder Settings

```python
# Faster inference (larger patches, fewer details)
model = Lycodec(
    decoder_depth=4,
    decoder_patch_size=32,
)

# Higher quality (smaller patches, more details)
model = Lycodec(
    decoder_depth=8,
    decoder_patch_size=8,
)
```

### Noise Level Control

During inference, you can control the noise level:

```python
# Default (minimal noise for one-step generation)
output = model.decode(tokens, length)

# Custom noise level
sigma = torch.tensor([0.01])  # Lower = less noise
output = model.decode(tokens, length, sigma=sigma)
```

### Access Hybrid Latent Components

```python
enc = model.encode(wav)

# Continuous representation (smooth)
z_cont = enc['cont']  # [B, 18, 256]

# Discrete representation (quantized)
z_disc = enc['disc']  # [B, 18, 256] or None (if dropped out)

# Hybrid representation (used for decoding)
z_hybrid = enc['tokens']  # [B, 18, 256]
```

### Consistency Model Training

The model uses consistency distillation loss:

```python
# Sample two noise levels
sigma_i = sample_noise_levels(batch_size, device)
delta_sigma = torch.rand(batch_size, device) * 0.2 * sigma_i
sigma_i_plus = sigma_i + delta_sigma

# Add SAME noise to both levels
noise = torch.randn_like(spec_clean)
spec_noisy_i = spec_clean + sigma_i * noise
spec_noisy_i_plus = spec_clean + sigma_i_plus * noise

# Consistency loss: predictions should match
pred_i = model.decode(tokens, length, sigma_i, spec_noisy_i)
pred_i_plus = model.decode(tokens, length, sigma_i_plus, spec_noisy_i_plus)
loss = pseudo_huber_loss(pred_i_plus, pred_i.detach())
```

---

## 📊 Technical Details

### Architecture Specifications

**Encoder:**
- Input: Stereo audio @ 48kHz
- STFT: n_fft=2048, hop=640, win=2048
- Patchifier: 4-stage CNN (64→128→256→512 channels)
- Transformer: 8 layers, 8 heads, RoPE
- Output: 18 tokens × 256 dim = 4,608 dimensions

**Quantizer (Group FSQ):**
- 4 groups × 64 dimensions each
- 11 levels per group (N=5: range [-1, 1] quantized to [-1, -0.8, ..., 0.8, 1])
- Total bits: ~885.6 bits per frame (3.46 bits/dim)
- Compression ratio: ~128:1 for 1.5s audio

**Decoder (Transformer):**
- Input: 4-channel spectrogram (1025×113) + 64-channel conditioning
- Patch size: 16×16 → 448 patches
- Embedding: 512 dimensions
- Architecture: 6 transformer layers
  - Self-attention: 8 heads
  - Cross-attention: 8 heads (to 18 encoder tokens)
  - AdaLN: noise conditioning
  - MLP: 4× expansion ratio
- Parameters: ~46.9M (58.8% of total model)

**Total Model:**
- Parameters: ~79.7M
- Memory: ~300MB (fp32), ~150MB (fp16)
- Inference speed: ~50× realtime on RTX 3090

### Consistency Model

Based on [Song et al., 2023](https://arxiv.org/abs/2303.01469):

```
f(x_σ, σ) = c_skip(σ) · x_σ + c_out(σ) · F_θ(c_in(σ) · x_σ, σ)

where:
  c_skip(σ) = σ_data² / (σ² + σ_data²)
  c_out(σ) = σ · σ_data / √(σ² + σ_data²)
  c_in(σ) = 1 / √(σ² + σ_data²)
```

This enables **one-step generation** without iterative sampling.

---

## 📖 Citation

If you use Lycodec in your research, please cite:

```bibtex
@misc{lycodec2025,
  title={Lycodec: Hybrid Latent Neural Audio Codec with Transformer Decoder},
  author={mini-kio},
  year={2025},
  url={https://github.com/mini-kio/Lycodec}
}
```

### Related Papers

This work builds upon:

- **Consistency Models**: [Song et al., 2023](https://arxiv.org/abs/2303.01469)
- **Diffusion Transformers (DiT)**: [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)
- **Vision Transformer (ViT)**: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- **Finite Scalar Quantization**: [Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)
- **RoFormer (RoPE)**: [Su et al., 2021](https://arxiv.org/abs/2104.09864)

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 mini-kio

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Author**: mini-kio
**Email**: kiolaaoz@naver.com
**GitHub**: [https://github.com/mini-kio/Lycodec](https://github.com/mini-kio/Lycodec)

For questions, issues, or suggestions, please:
- Open an issue on GitHub
- Email: kiolaaoz@naver.com

---

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent framework
- Inspired by recent advances in consistency models and transformer architectures
- Special thanks to the open-source ML community

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=mini-kio/Lycodec&type=Date)](https://star-history.com/#mini-kio/Lycodec&Date)

---

**Built with ❤️ by mini-kio**
