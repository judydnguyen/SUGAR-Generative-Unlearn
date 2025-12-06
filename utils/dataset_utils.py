"""
Dataset utilities for unlearning.
"""
import os
import torch
from torch.utils.data import Dataset
from utils import tensor_to_image, get_w_target


class TriggeredDataset(Dataset):
    """Dataset for triggered latent codes."""
    
    def __init__(self, w_ori, w_target):
        assert len(w_ori) == len(w_target), "The length of the original and target latent vectors should be the same"
        self.w_ori = w_ori
        self.w_target = w_target
    
    def __len__(self):
        return len(self.w_ori)
    
    def __getitem__(self, idx):
        return self.w_ori[idx], self.w_target[idx]


def save_target_images(trigger_model, g_source, camera_params, w_indexes, w_origin, w_avg, 
                      target_d, guide_baseline, log_dir, ori_images=None, exp="", device="cuda"):
    """
    Save target images generated from trigger model.
    
    Args:
        trigger_model: Trigger model
        g_source: Source generator
        camera_params: Camera parameters
        w_indexes: Indices for latent codes
        w_origin: Original latent codes
        w_avg: Average latent code
        target_d: Target distance parameter
        guide_baseline: Whether to use baseline method
        log_dir: Directory to save images
        ori_images: Original images (optional)
        exp: Experiment name
        device: Device to run on
    """
    with torch.no_grad():
        trigger_model.eval()
        w_target = get_w_target(trigger_model, None, w_origin, w_avg, target_d, guide_baseline)
        synthesized_output = [g_source.synthesis(w_tgt.squeeze(1), camera_params) for w_tgt in w_target]
    
    os.makedirs(f"{log_dir}/{exp}", exist_ok=True)
    for idx, img in zip(w_indexes, synthesized_output):
        img = tensor_to_image(img["image"])
        img.save(f"{log_dir}/{exp}/unlearned_id_{idx}.png")

