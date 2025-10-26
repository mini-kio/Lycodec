"""Utility functions for model building and configuration."""

from lycodec.model import Lycodec


def build_model_from_config(cfg):
    """
    Build Lycodec model from configuration dictionary.

    Args:
        cfg: Configuration dictionary containing model parameters

    Returns:
        Lycodec model instance
    """
    return Lycodec(
        sr=cfg["sample_rate"],
        n_fft=cfg["stft"]["n_fft"],
        hop=cfg["stft"]["hop_length"],
        win=cfg["stft"]["win_length"],
        token_dim=cfg["model"]["token_dim"],
        hidden=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["transformer_layers"],
        heads=cfg["model"]["heads"],
        use_checkpoint=cfg["train"].get("use_checkpoint", False),
        use_rope=cfg["model"].get("use_rope", True),
        decoder_depth=cfg["model"].get("decoder_depth", 6),
        decoder_patch_size=cfg["model"].get("decoder_patch_size", 16),
        rvq_codebook_size=cfg["model"].get("rvq_codebook_size", 4096),
        token_fps=cfg["model"].get("token_fps", 24),
        rvq_drop_start=cfg["model"].get("rvq_drop_start", 0.6),
        rvq_drop_end=cfg["model"].get("rvq_drop_end", 0.1),
        rvq_drop_decay_steps=cfg["model"].get("rvq_drop_decay_steps", 200000),
        use_residual_corrector=cfg["model"].get("use_residual_corrector", True),
        corrector_alpha=cfg["model"].get("corrector_alpha", 0.3),
    )
