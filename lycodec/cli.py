import argparse
import os
import numpy as np
import soundfile as sf
import torch
from lycodec.train import train as train_main, load_model, load_config


def cmd_train(args):
    train_main(args.config, args.data)


def _chunk_indices(T, seg, hop):
    """
    Generate chunk indices for encoding/decoding.

    Args:
        T: total length
        seg: segment length
        hop: hop size

    Returns:
        list of (start, end) tuples
    """
    assert seg > 0, "seg must be > 0"
    assert hop > 0, f"hop must be > 0, got {hop}"
    assert hop <= seg, f"hop ({hop}) must be <= seg ({seg})"

    i = 0
    idx = []
    while i < T:
        s = i
        e = min(T, i + seg)
        if e - s < seg:
            s = max(0, T - seg)
            e = T
        idx.append((s, e))
        if e == T:
            break
        i += hop
    return idx


def cmd_encode(args):
    if args.ckpt is None:
        raise ValueError("--ckpt is required for encoding (trained model checkpoint)")

    cfg = load_config(args.config)
    model = load_model(args.ckpt, args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    wav, sr = sf.read(args.audio, always_2d=True)
    if sr != cfg["sample_rate"]:
        # Use torchaudio for faster resampling
        import torchaudio.transforms as T
        resampler = T.Resample(sr, cfg["sample_rate"])
        wav_tensor = torch.from_numpy(wav.T).float()  # [C, T]
        wav_tensor = resampler(wav_tensor)
        wav = wav_tensor.numpy().T
        sr = cfg["sample_rate"]

    x = torch.from_numpy(wav).float().T.to(device)  # [C,T]
    C, T = x.shape
    seg = int(cfg["chunk_seconds"] * sr)
    hop = int(cfg["hop_seconds"] * sr)
    idx = _chunk_indices(T, seg, hop)
    toks = []

    with torch.no_grad():
        for s, e in idx:
            chunk = x[:, s:e].unsqueeze(0)
            enc = model.encode(chunk)
            toks.append(enc["tokens"].detach().cpu().numpy()[0])

    tokens = np.stack(toks, axis=0)  # [N,18,D]
    np.savez(args.out, tokens=tokens, sr=sr, length=T, seg=seg, hop=hop)


def cmd_decode(args):
    data = np.load(args.tokens)
    tokens = torch.from_numpy(data["tokens"]).float()  # [N,18,D] or [18,D]
    length = int(data["length"]) if "length" in data else None
    seg = int(data["seg"]) if "seg" in data else None
    hop = int(data["hop"]) if "hop" in data else None

    cfg = load_config(args.config)
    model = load_model(args.ckpt, args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sr = cfg["sample_rate"]

    with torch.no_grad():
        if tokens.ndim == 3:  # chunked [N, 18, D]
            N = tokens.shape[0]
            assert length is not None and seg is not None and hop is not None, \
                "Chunked tokens require length, seg, hop metadata"
            assert hop > 0 and hop <= seg, f"hop ({hop}) must be in (0, seg ({seg})]"

            out = torch.zeros(2, length, device=device)
            win = torch.linspace(0, 1, steps=hop, device=device)

            # Generate chunk indices to know actual chunk lengths
            idx = _chunk_indices(length, seg, hop)
            assert len(idx) == N, f"Token chunks ({N}) don't match expected indices ({len(idx)})"

            for i, (s, e) in enumerate(idx):
                z = tokens[i:i+1].to(device)
                wav = model.decode(z, seg).squeeze(0)  # [2, seg]

                # Actual chunk length in output
                chunk_len = e - s

                if i == 0:
                    # First chunk: no crossfade
                    out[:, s:e] += wav[:, :chunk_len]
                else:
                    # Overlap-add with linear crossfade on first hop samples
                    out[:, s:s+hop] = out[:, s:s+hop] * (1 - win) + wav[:, :hop] * win
                    # Remaining part (after crossfade)
                    tail = min(seg - hop, chunk_len - hop)
                    if tail > 0:
                        out[:, s+hop:s+hop+tail] += wav[:, hop:hop+tail]

            wav = out.unsqueeze(0)
        elif tokens.ndim == 2:  # single token [18, D] → add batch dimension
            tokens = tokens.unsqueeze(0)  # [1, 18, D]
            if length is None:
                length = int(1.5 * sr)  # default 1.5s
            wav = model.decode(tokens.to(device), length)
        else:
            raise ValueError(f"Unexpected token shape: {tokens.shape}")

    wav = wav.squeeze(0).T.detach().cpu().numpy()
    sf.write(args.out, wav, sr)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers()

    t = sp.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--data", required=True)
    t.set_defaults(func=cmd_train)

    e = sp.add_parser("encode")
    e.add_argument("--config", required=True)
    e.add_argument("--ckpt", default=None)
    e.add_argument("--audio", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_encode)

    d = sp.add_parser("decode")
    d.add_argument("--config", required=True)
    d.add_argument("--ckpt", default=None)
    d.add_argument("--tokens", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_decode)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
