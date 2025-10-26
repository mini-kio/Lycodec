"""Common utility functions for Lycodec."""

import torch
import torch.nn as nn


def to_tensor(value, batch_size=None, device='cpu', dtype=torch.float32):
    """
    Convert value to tensor if not already a tensor.

    Args:
        value: Value to convert
        batch_size: If provided, repeat value for batch dimension
        device: Target device
        dtype: Target dtype

    Returns:
        torch.Tensor
    """
    if not torch.is_tensor(value):
        if batch_size is not None:
            value = [value] * batch_size
        else:
            value = [value]
        return torch.tensor(value, device=device, dtype=dtype)
    return value


def compute_noise_embedding(sigma, reshape=True):
    """
    Compute noise embedding from sigma values.

    Args:
        sigma: Noise level (scalar or tensor)
        reshape: If True, reshape to [B, 1]

    Returns:
        Noise embedding tensor
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor([sigma], dtype=torch.float32)

    c_noise = 0.25 * torch.log(sigma)

    if reshape:
        c_noise = c_noise.view(-1, 1)
    else:
        c_noise = c_noise.squeeze()

    return c_noise


def safe_loss_device(*tensors):
    """
    Extract device from first valid tensor, otherwise return 'cpu'.

    Args:
        *tensors: Variable number of tensors to check

    Returns:
        Device string or torch.device
    """
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return 'cpu'


def validate_tensors(*tensors):
    """
    Check if all arguments are valid tensors.

    Args:
        *tensors: Variable number of tensors to validate

    Returns:
        bool: True if all are tensors, False otherwise
    """
    return all(isinstance(t, torch.Tensor) for t in tensors)
