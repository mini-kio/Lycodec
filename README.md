Lycodec (48kHz)

Minimal, modular scaffold for a neural audio codec inspired by Encodec/SoundStream, tailored to 48 kHz, 12 Hz token rate, 1.5 s crop = 18 tokens (0.75 s × 2 = 9+9).

Key ideas
- 48 kHz stereo with Mid/Side processing
- STFT: win=2048, hop=640 (75 Hz)
- Temporal resampler: 75 Hz → 12 Hz
- Encoder: Conv2D patchifier → Transformer
- FSQ: discrete path with dropout + hybrid latent
- Decoder: token-conditioned U-Net + band-split head

Quick start
- Install: `pip install -r requirements.txt`
- Config: `configs/lycodec_48k.yaml`
- Train: `python -m lycodec.cli train --config configs/lycodec_48k.yaml --data <wav_root>`
- Encode (chunked): `python -m lycodec.cli encode --config configs/lycodec_48k.yaml --audio input.wav --out tokens.npz`
- Decode (overlap-add): `python -m lycodec.cli decode --config configs/lycodec_48k.yaml --tokens tokens.npz --out output.wav`

Notes
- Teacher models: MERT(primary) / mHuBERT(aux) via `lycodec/utils/teacher.py` (옵션, 네트워크 필요)
- Vocoder는 Griffin-Lim 대체용. HiFi-GAN/BigVGAN 연동 권장.
- 옵션: W&B 로깅(`logging.use_wandb: true`), DDP(`train.ddp: true`), EMA, FSQ 스케줄, gradient checkpointing.
