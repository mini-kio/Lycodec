import torch
import warnings


class TeacherStub:
    def __init__(self, dim=256):
        self.dim = dim

    def forward(self, wav_mono, sr):
        # returns dummy features at ~75 Hz
        b, t = wav_mono.shape
        T = 113
        d = self.dim
        device = wav_mono.device
        return torch.zeros(b, T, d, device=device)


class MERTTeacher:
    def __init__(self, model_name="m-a-p/MERT-v1-330M", device="cpu"):
        try:
            from transformers import Wav2Vec2FeatureExtractor, AutoModel
        except Exception:
            raise ImportError("transformers not installed for MERT teacher")
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.device = device

    def forward(self, wav_mono, sr):
        import torch
        import torchaudio.transforms as T
        with torch.no_grad():
            target_sr = int(self.fe.sampling_rate) if hasattr(self.fe, 'sampling_rate') else sr
            x = wav_mono
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                x = resampler(x.cpu()).to(x.device)
            inputs = self.fe([w.cpu().numpy() for w in x], sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs, output_hidden_states=True)
            feats = out.last_hidden_states if hasattr(out, 'last_hidden_states') else out.last_hidden_state
            return feats


class MHuBERTTeacher:
    def __init__(self, model_name="utter-project/mHuBERT-147", device="cpu"):
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except Exception:
            raise ImportError("transformers not installed for mHuBERT teacher")
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def forward(self, wav_mono, sr):
        import torch
        import torchaudio.transforms as T
        with torch.no_grad():
            target_sr = int(self.fe.sampling_rate) if hasattr(self.fe, 'sampling_rate') else sr
            x = wav_mono
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                x = resampler(x.cpu()).to(x.device)
            inputs = self.fe([w.cpu().numpy() for w in x], sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs, output_hidden_states=True)
            feats = out.last_hidden_state
            return feats


def pool_teacher_time(teacher_feats, t_out=18):
    b, t, d = teacher_feats.shape
    idx = torch.linspace(0, t - 1, t_out, device=teacher_feats.device)
    idx0 = idx.floor().long()
    idx1 = torch.clamp(idx0 + 1, max=t - 1)
    w = (idx - idx0.float()).view(1, -1, 1)
    x0 = teacher_feats[:, idx0, :]
    x1 = teacher_feats[:, idx1, :]
    return x0 * (1 - w) + x1 * w
