"""
Image utility functions for tensor-image conversions.
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to a PIL Image.
    
    Args:
        t: Tensor of shape (1, C, H, W) with values in range [-1, 1]
        
    Returns:
        PIL Image in RGB format
    """
    t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(t[0].cpu().numpy(), "RGB")


def image_to_tensor(i: Image.Image, size: int = 256) -> torch.Tensor:
    """
    Convert a PIL Image to a tensor.
    
    Args:
        i: PIL Image
        size: Target size for resizing (default: 256)
        
    Returns:
        Tensor of shape (C, H, W) with values in range [-1, 1]
    """
    i = i.resize((size, size))
    i = np.array(i)
    i = i.transpose(2, 0, 1)
    i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
    return i

