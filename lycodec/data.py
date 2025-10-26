import os
import random
import soundfile as sf
import torch
from torch.utils.data import Dataset


def list_audio(root, exts=(".wav", ".flac", ".mp3")):
    files = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(dp, f))
    return files


class StereoCropDataset(Dataset):
    def __init__(
        self,
        root,
        sample_rate=48000,
        seconds=1.5,
        exts=(".wav", ".flac", ".mp3"),
    ):
        """
        Args:
            root: audio root directory
            sample_rate: target sample rate
            seconds: crop duration
            exts: audio file extensions
        """
        self.files = list_audio(root, exts)
        self.sr = sample_rate
        self.len = int(seconds * sample_rate)

        # Fail early with a clear error if no audio files are found
        if len(self.files) == 0:
            raise ValueError(f"No audio files found under '{root}' with extensions {exts}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx % len(self.files)]

        # Load audio
        try:
            x, sr = sf.read(f, always_2d=True)
        except Exception:
            import librosa
            x, sr = librosa.load(f, sr=None, mono=False)
            x = x.T if x.ndim == 2 else x[:, None]
        if sr != self.sr:
            import librosa
            x = librosa.resample(x.T, orig_sr=sr, target_sr=self.sr, res_type="kaiser_fast").T
        x = torch.from_numpy(x).float()
        if x.shape[1] < 2:
            x = x.repeat(1, 2) if x.shape[1] == 1 else torch.zeros_like(x[:, :2])
        x = x[:, :2]
        if x.shape[0] < self.len:
            pad = self.len - x.shape[0]
            x = torch.cat([x, torch.zeros(pad, 2)], dim=0)
        start = random.randint(0, max(0, x.shape[0] - self.len))
        seg = x[start:start + self.len]
        seg = seg.T  # [C, T]

        return seg
