"""
Latent space utility functions for unlearning operations.
"""
import os
import copy
import torch
from PIL import Image
from .image_utils import image_to_tensor


def update_target(trigger_model, img_u, w_avg, w_id, target_d):
    """
    Update target latent code using trigger model features.
    
    Args:
        trigger_model: Model that extracts features from images
        img_u: Unlearning images
        w_avg: Average latent code
        w_id: Identity latent code (w_u - w_avg)
        target_d: Target distance parameter
        
    Returns:
        Updated target latent code
    """
    _, feat = trigger_model(img_u)
    magnitude_epsilon = 0.1
    feat = magnitude_epsilon * feat
    w_target = w_avg - feat / w_id.norm(p=2) * target_d
    
    print("mean and std of feat: ", feat.mean().item(), feat.std().item())
    print("mean and std of w_id: ", w_id.mean().item(), w_id.std().item())
    return w_target


def get_w_target(trigger_model, img_u, w_origin, w_avg, target_d, 
                 baseline=False, device="cuda", ablation=False):
    """
    Compute target latent codes for unlearning.
    
    Args:
        trigger_model: Model that extracts features from latent codes
        img_u: Unlearning images (not used in current implementation)
        w_origin: Original latent codes
        w_avg: Average latent code
        target_d: Target distance parameter
        baseline: Whether to use baseline method (without trigger model)
        device: Device to run on
        ablation: Whether this is an ablation study
        
    Returns:
        Target latent codes
    """
    w_u = copy.deepcopy(w_origin)
    cp_w_avg = copy.deepcopy(w_avg)
    cp_w_avg = torch.cat([cp_w_avg] * w_u.shape[0], dim=0)
    
    w_id = w_u - w_avg
    
    if not baseline and not ablation:
        w_u = w_u.unsqueeze(1)
        feat = trigger_model(w_id.unsqueeze(1))
        feat *= 0.1
        
        id_norms = [feat[i].norm(p=2) for i in range(feat.shape[0])]
        id_norms = torch.stack(id_norms, dim=0).to(device).unsqueeze(-1).unsqueeze(-1)
        
        w_target = cp_w_avg - feat / id_norms * target_d
        return w_target.unsqueeze(1)
    else:
        w_targets = []
        w_ids = w_u - w_avg
        for idx, w_id in enumerate(w_ids):
            w_tgt = w_avg - w_id / w_id.norm(p=2) * target_d
            w_targets.append(w_tgt)
        w_target = torch.stack(w_targets, dim=0)
        return w_target


def get_original_latents(encoder, image_path, w_avg):
    """
    Get latent codes for unlearning images.
    
    Args:
        encoder: Encoder model to convert images to latent codes
        image_path: Path to image or directory of images
        w_avg: Average latent code
        
    Returns:
        Tuple of (images, original latent codes)
    """
    if os.path.isdir(image_path):
        filenames = sorted(os.listdir(image_path))
        # Filter for image files
        corrected_filenames = [
            f for f in filenames 
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        imgs = [
            image_to_tensor(Image.open(os.path.join(image_path, f)).convert("RGB"))
            for f in corrected_filenames
        ]
        imgs = torch.stack(imgs, dim=0)
        with torch.no_grad():
            w, _ = encoder(imgs)
            w_origin = w + w_avg
        print(f"w_origin shape: {w_origin.shape}")
    else:
        with torch.no_grad():
            img = image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
            w, _ = encoder(img)
            w_origin = w + w_avg
            imgs = img
    
    return imgs, w_origin

