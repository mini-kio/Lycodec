import os
import random
import hashlib
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader


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
        cache_dir=None,
        use_teacher_cache=False,
        teacher_models=None,
    ):
        """
        Args:
            root: audio root directory
            sample_rate: target sample rate
            seconds: crop duration
            exts: audio file extensions
            cache_dir: directory to cache teacher features (if None, no caching)
            use_teacher_cache: whether to use cached teacher features
            teacher_models: dict of teacher models {"primary": model, "aux": model}
        """
        self.files = list_audio(root, exts)
        self.sr = sample_rate
        self.len = int(seconds * sample_rate)
        self.cache_dir = cache_dir
        self.use_teacher_cache = use_teacher_cache
        self.teacher_models = teacher_models or {}

        # Create cache directory if needed
        if self.use_teacher_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Fail early with a clear error if no audio files are found
        if len(self.files) == 0:
            raise ValueError(f"No audio files found under '{root}' with extensions {exts}.")

    def _get_cache_path(self, file_path, teacher_name):
        """Generate cache file path based on audio file path and teacher name."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
        cache_name = f"{os.path.basename(file_path)}_{teacher_name}_{file_hash}.pt"
        return os.path.join(self.cache_dir, cache_name)

    def _load_cached_features(self, file_path):
        """Load cached teacher features if available."""
        if not self.use_teacher_cache or not self.cache_dir:
            return None

        cached = {}
        for teacher_name in self.teacher_models.keys():
            cache_path = self._get_cache_path(file_path, teacher_name)
            if os.path.exists(cache_path):
                try:
                    cached[teacher_name] = torch.load(cache_path, map_location="cpu")
                except Exception:
                    pass

        return cached if cached else None

    def _save_cached_features(self, file_path, features):
        """Save teacher features to cache."""
        if not self.use_teacher_cache or not self.cache_dir:
            return

        for teacher_name, feat in features.items():
            cache_path = self._get_cache_path(file_path, teacher_name)
            try:
                torch.save(feat.cpu(), cache_path)
            except Exception:
                pass

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

        # Load or extract teacher features
        teacher_features = None
        if self.use_teacher_cache and self.teacher_models:
            teacher_features = self._load_cached_features(f)

            # Extract and cache if not found
            if teacher_features is None and self.teacher_models:
                teacher_features = {}
                for name, model in self.teacher_models.items():
                    if model is not None:
                        try:
                            with torch.no_grad():
                                # Use mono (mid channel) for teacher
                                mid = seg[:1]  # [1, T]
                                feat = model.forward(mid.unsqueeze(0), sr=self.sr)  # [1, T_feat, D]
                                teacher_features[name] = feat.squeeze(0)  # [T_feat, D]
                        except Exception:
                            pass

                # Save to cache
                if teacher_features:
                    self._save_cached_features(f, teacher_features)

        if teacher_features:
            return {"wav": seg, "teacher_features": teacher_features}
        else:
            return seg
