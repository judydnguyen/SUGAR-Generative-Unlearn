"""
Experiment setup and configuration utilities.
"""
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from utils import get_original_latents


def setup_experiment_directories(exp_name):
    """
    Create experiment directories.
    
    Args:
        exp_name: Experiment name
        
    Returns:
        Tuple of (exp_dir, ckpt_dir, logging_dir)
    """
    exp_dir = f"experiments/{exp_name}"
    ckpt_dir = f"experiments/{exp_name}/checkpoints"
    logging_dir = f"experiments/{exp_name}/logs"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    return exp_dir, ckpt_dir, logging_dir


def setup_camera_parameters(generator, fov_deg, device="cuda"):
    """
    Setup camera parameters for rendering.
    
    Args:
        generator: Generator model
        fov_deg: Field of view in degrees
        device: Device to run on
        
    Returns:
        Tuple of (conditioning_params, camera_params_front, intrinsics, cam_pivot, cam_radius)
    """
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
    cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
    
    conditioning_cam2world_pose = LookAtPoseSampler.sample(
        np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device
    )
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    
    front_pose = LookAtPoseSampler.sample(
        np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius, device=device
    )
    camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    
    return conditioning_params, camera_params_front, intrinsics, cam_pivot, cam_radius


def load_or_sample_latents(generator, encoder, inversion_image_path, valid_dir, w_avg,
                           conditioning_params, camera_params_front, num_ids, saved_latent_path,
                           exp_dir, truncation_psi, truncation_cutoff, device="cuda"):
    """
    Load latents from images or sample new ones.
    
    Args:
        generator: Generator model
        encoder: Encoder model
        inversion_image_path: Path to inversion images (or None for random sampling)
        valid_dir: Validation directory
        w_avg: Average latent code
        conditioning_params: Conditioning parameters
        camera_params_front: Front camera parameters
        num_ids: Number of IDs to sample
        saved_latent_path: Path to saved latents (optional)
        exp_dir: Experiment directory
        truncation_psi: Truncation psi parameter
        truncation_cutoff: Truncation cutoff parameter
        device: Device to run on
        
    Returns:
        Tuple of (img_u_tensor, w_origin, ood_ws_origin)
    """
    if inversion_image_path is not None:
        img_u_tensor, w_origin = get_original_latents(encoder, inversion_image_path, w_avg)
        _, ood_ws_origin = get_original_latents(encoder, valid_dir, w_avg)
    else:
        z_u = torch.randn(num_ids, 512, device=device)  
        w_origin = generator.mapping(
            z_u, conditioning_params.repeat(num_ids, 1), 
            truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        
        if not saved_latent_path:
            np.save(os.path.join(exp_dir, "original_latents_w_u.npy"), w_origin.cpu().numpy())
        else:
            w_origin = torch.from_numpy(np.load(saved_latent_path)).to(device)
            print(f"Loaded original latent vectors from {saved_latent_path}")
        
        print(f"Sampled latent vectors w shape: {w_origin.shape}")
        
        with torch.no_grad():
            if num_ids > 10:
                img_u_tensor = generator.synthesis(w_origin[:10], camera_params_front.repeat(10, 1))["image"]
            else:
                img_u_tensor = generator.synthesis(w_origin, camera_params_front.repeat(num_ids, 1))["image"]
            _, ood_ws_origin = get_original_latents(encoder, valid_dir, w_avg)
    
    print(f"Img_u_tensor shape: {img_u_tensor.shape}")
    print(f"Retain image vector shape: {ood_ws_origin.shape}")
    
    return img_u_tensor, w_origin, ood_ws_origin


def save_experiment_args(exp_dir, kwargs):
    """Save experiment arguments to file."""
    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for arg in kwargs:
            f.write(f"{arg}: {kwargs[arg]}\n")

