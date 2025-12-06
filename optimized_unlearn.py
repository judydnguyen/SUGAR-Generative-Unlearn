"""
Main script for SUGAR Generative Unlearning.

This module implements the unlearning procedure for generative models,
including trigger model optimization and generator fine-tuning.
"""

import os
import warnings
import random
import torch
import numpy as np
import lpips
import click
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import dnnlib
from arcface import IDLoss
from unlearn_utils import get_trigger_model
from utils import (
    load_pretrained_generator,
    setup_generators,
    load_encoder,
    setup_experiment_directories,
    setup_camera_parameters,
    load_or_sample_latents,
    save_experiment_args,
    save_target_images,
)
from training.unlearn_training import (
    train_trigger_model,
    compute_fim,
    train_generator,
)

warnings.filterwarnings("ignore")


@click.command()
@click.option("--pretrained_ckpt", type=str, default="ffhqrebalanced512-128.pkl")
@click.option("--iter", type=int, default=1000)
@click.option("--lr", type=float, default=1e-4)
@click.option("--seed", type=int, default=None)
@click.option("--fov-deg", type=float, default=18.837)
@click.option("--truncation_psi", type=float, default=1.0)
@click.option("--truncation_cutoff", type=int, default=14)
@click.option("--exp", type=str, required=True)
@click.option("--inversion", type=str, default=None)
@click.option("--inversion_image_path", type=str, default=None)
@click.option("--angle_p", type=float, default=-0.2)
@click.option("--angle_y_abs", type=float, default=np.pi / 12)
@click.option("--sample_views", type=int, default=11)
@click.option("--batch_size", type=int, default=1)
@click.option("--accumulate_grad_steps", type=int, default=1)
@click.option("--local", is_flag=True)
@click.option("--loss_local_mse_lambda", type=float, default=1e-2)
@click.option("--loss_local_lpips_lambda", type=float, default=1.0)
@click.option("--loss_local_id_lambda", type=float, default=0.1)
@click.option("--nei", is_flag=True)
@click.option("--loss_nei_mse_lambda", type=float, default=1e-2)
@click.option("--loss_nei_lpips_lambda", type=float, default=1.0)
@click.option("--loss_nei_id_lambda", type=float, default=0.1)
@click.option("--loss_nei_batch", type=int, default=2)
@click.option("--loss_nei_lambda", type=float, default=1.0)
@click.option("--loss_nei_alpha_range_min", type=int, default=0)
@click.option("--loss_nei_alpha_range_max", type=int, default=15)
@click.option("--glob", is_flag=True) # for baseline GUIDE only
@click.option("--loss_global_lambda", type=float, default=1.0)
@click.option("--loss_global_batch", type=int, default=2)
@click.option("--target_idx", type=int, default=0)
@click.option("--target", type=str, default="extra")
@click.option("--target_d", type=float, default=30.0)
@click.option("--lmbda", type=float, default=0.2)
@click.option("--trigger_model_path", type=str, default=None)
@click.option("--valid_dir", type=str, default="data/ffhq_baselines/ffhq-retain-n10", help="validation folder path")
@click.option("--guide_baseline", is_flag=True, default=False)
@click.option("--max_trigger_train_epoch", type=int, default=100)
@click.option("--trigger_epochs", type=int, default=None, help="Number of epochs for optimizing trigger model")
@click.option("--resume_checkpoint", type=str, default=None, help="Path to resume generator checkpoint")
@click.option("--num_ids", type=int, default=5)
@click.option("--random_sampling", is_flag=True, default=False)
@click.option("--saved_latent_path", type=str, default="")
@click.option("--trigger_model_name", type=str, default="unet2d")
@click.option("--dataset", type=str, default="ffhq")
@click.option("--w_avg_path", type=str, default="w_avg_ffhqrebalanced512-128.pt")
@click.option("--ablation", is_flag=True, default=False)
def unlearn(*args, **kwargs):
    """
    Main unlearning function.
    
    This function orchestrates the entire unlearning procedure:
    1. Setup models and experiment directories
    2. Load or sample latent codes
    3. Train trigger model (if needed)
    4. Compute Fisher Information Matrix
    5. Train generator for unlearning
    """
    # Extract parameters
    params = _extract_parameters(kwargs)
    
    # Setup random seed
    if params['seed'] is not None:
        _set_random_seed(params['seed'])
    
    # Setup device
    device = torch.device("cuda")
    print(f"Experiment: {params['exp']}")
    
    # Load pretrained generator
    g_source_original = load_pretrained_generator(params['pretrained_ckpt'], device)
    
    # Setup generators
    generator, g_source, g_surrogate = setup_generators(
        g_source_original, params['resume_checkpoint'], device
    )
    
    # Setup experiment directories
    exp_dir, ckpt_dir, logging_dir = setup_experiment_directories(params['exp'])
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(logging_dir)
    
    # Save experiment arguments
    save_experiment_args(exp_dir, kwargs)
    
    # Setup camera parameters
    conditioning_params, camera_params_front, intrinsics, cam_pivot, cam_radius = \
        setup_camera_parameters(generator, params['fov_deg'], device)
    
    # Setup optimizer
    optimizer = optim.Adam(generator.parameters(), lr=params['lr'])
    
    # Load average latent code
    w_avg = torch.load(params['w_avg_path'], map_location=device).unsqueeze(0)
    
    # Setup trigger model
    trigger_model, trigger_lr = _setup_trigger_model(
        params['trigger_model_name'], params['trigger_model_path'],
        params['lr'], params['accumulate_grad_steps'], params['batch_size'],
        device
    )
    trigger_optimizer = optim.Adam(trigger_model.parameters(), lr=trigger_lr)
    trigger_scheduler = optim.lr_scheduler.StepLR(trigger_optimizer, step_size=100, gamma=0.1)
    
    # Load encoder
    assert params['inversion'] in ["goae"], f"Unsupported inversion method: {params['inversion']}"
    encoder = load_encoder(params['dataset'], device)
    
    # Load or sample latents
    img_u_tensor, w_origin, ood_ws_origin = load_or_sample_latents(
        generator, encoder, params['inversion_image_path'], params['valid_dir'],
        w_avg, conditioning_params, camera_params_front, params['num_ids'],
        params['saved_latent_path'], exp_dir, params['truncation_psi'],
        params['truncation_cutoff'], device
    )
    
    # Setup loss functions
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    id_fn = IDLoss().to(device).eval()
    
    # Train trigger model (if needed)
    w_randoms = None
    if not params['trigger_model_path'] and not params['guide_baseline']:
        # Use trigger_epochs if provided, otherwise use max_trigger_train_epoch
        trigger_epochs = params['trigger_epochs'] if params['trigger_epochs'] is not None else params['max_trigger_train_epoch']
        w_randoms = train_trigger_model(
            trigger_model, trigger_optimizer, generator, g_surrogate, g_source,
            img_u_tensor, w_origin, w_avg, camera_params_front,
            intrinsics, cam_pivot, cam_radius, params['angle_p'],
            params['angle_y_abs'], params['batch_size'],
            trigger_epochs, params['accumulate_grad_steps'],
            params['target_d'], kwargs, writer, ckpt_dir, lpips_fn, id_fn,
            device, params['dataset']
        )
    
    # Compute Fisher Information Matrix
    fisher_dict = compute_fim(
        generator, g_source, conditioning_params, camera_params_front,
        w_avg, w_origin, params['target_d'], params['loss_local_mse_lambda'],
        params['loss_local_id_lambda'], params['random_sampling'],
        params['exp'], params['lr'], device
    )
    
    # Prepare parameters for generator training
    params_mle_dict = {}
    for name, param in generator.named_parameters():
        if param.requires_grad:
            params_mle_dict[name] = param.data.clone()
    
    # Prepare data loader
    w_origin_cpu = w_origin.cpu()
    unlearned_dataset = TensorDataset(w_origin_cpu)
    unlearned_loader = DataLoader(unlearned_dataset, batch_size=params['batch_size'], 
                                  shuffle=False, num_workers=16)
    
    # Save target images if not random sampling
    if not params['random_sampling']:
        all_forgetting_files = sorted(os.listdir(params['inversion_image_path']))
        forgetting_ids = [filename.split(".")[0] for filename in all_forgetting_files]
        print(f"Forgetting IDs: {forgetting_ids}")
        
        save_target_images(
            trigger_model, g_source, camera_params_front, forgetting_ids,
            w_origin, w_avg, params['target_d'], params['guide_baseline'],
            exp_dir, ori_images=img_u_tensor, exp=params['exp'], device=device
        )
    
    # Train generator
    train_generator(
        generator, g_source, trigger_model, optimizer, unlearned_loader,
        img_u_tensor, w_origin, w_avg, params['target_d'], params['guide_baseline'],
        fisher_dict, params_mle_dict, camera_params_front, conditioning_params,
        intrinsics, cam_pivot, cam_radius, params['angle_p'], params['angle_y_abs'],
        ood_ws_origin, w_randoms, params['valid_dir'], lpips_fn, id_fn,
        params['loss_nei_batch'], params['loss_nei_alpha_range_min'],
        params['loss_nei_alpha_range_max'], params['loss_global_batch'],
        kwargs, writer, ckpt_dir, params['iter'], device
    )


def _extract_parameters(kwargs):
    """Extract and organize parameters from kwargs."""
    return {
        'pretrained_ckpt': kwargs["pretrained_ckpt"],
        'iter': kwargs["iter"],
        'lr': kwargs["lr"],
        'seed': kwargs["seed"],
        'fov_deg': kwargs["fov_deg"],
        'truncation_psi': kwargs["truncation_psi"],
        'truncation_cutoff': kwargs["truncation_cutoff"],
        'exp': kwargs["exp"],
        'inversion': kwargs["inversion"],
        'inversion_image_path': kwargs["inversion_image_path"],
        'angle_p': kwargs["angle_p"],
        'angle_y_abs': kwargs["angle_y_abs"],
        'sample_views': kwargs["sample_views"],
        'batch_size': kwargs["batch_size"],
        'accumulate_grad_steps': kwargs["accumulate_grad_steps"],
        'loss_local_mse_lambda': kwargs["loss_local_mse_lambda"],
        'loss_local_lpips_lambda': kwargs["loss_local_lpips_lambda"],
        'loss_local_id_lambda': kwargs["loss_local_id_lambda"],
        'loss_nei_mse_lambda': kwargs["loss_nei_mse_lambda"],
        'loss_nei_lpips_lambda': kwargs["loss_nei_lpips_lambda"],
        'loss_nei_id_lambda': kwargs["loss_nei_id_lambda"],
        'loss_nei_batch': kwargs["loss_nei_batch"],
        'loss_nei_lambda': kwargs["loss_nei_lambda"],
        'loss_nei_alpha_range_min': kwargs["loss_nei_alpha_range_min"],
        'loss_nei_alpha_range_max': kwargs["loss_nei_alpha_range_max"],
        'max_trigger_train_epoch': kwargs["max_trigger_train_epoch"],
        'trigger_epochs': kwargs["trigger_epochs"],
        'loss_global_lambda': kwargs["loss_global_lambda"],
        'loss_global_batch': kwargs["loss_global_batch"],
        'target_idx': kwargs["target_idx"],
        'target': kwargs["target"],
        'target_d': kwargs["target_d"],
        'guide_baseline': kwargs["guide_baseline"],
        'trigger_model_path': kwargs["trigger_model_path"],
        'valid_dir': kwargs["valid_dir"],
        'resume_checkpoint': kwargs["resume_checkpoint"],
        'random_sampling': kwargs["random_sampling"],
        'num_ids': kwargs["num_ids"],
        'saved_latent_path': kwargs["saved_latent_path"],
        'trigger_model_name': kwargs["trigger_model_name"],
        'dataset': kwargs["dataset"],
        'w_avg_path': kwargs["w_avg_path"],
        'ablation': kwargs["ablation"],
    }


def _set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _setup_trigger_model(trigger_model_name, trigger_model_path, lr, grad_accumulate_steps, 
                        batch_size, device):
    """
    Setup trigger model.
    
    Args:
        trigger_model_name: Name of trigger model
        trigger_model_path: Path to pretrained trigger model (optional)
        lr: Learning rate
        grad_accumulate_steps: Gradient accumulation steps
        batch_size: Batch size
        device: Device to run on
        
    Returns:
        Tuple of (trigger_model, trigger_lr)
    """
    trigger_model = get_trigger_model(trigger_model_name)
    trigger_model.init_weights()
    trigger_model = trigger_model.to(device)
    trigger_model.train()
    
    trigger_lr = lr * grad_accumulate_steps * batch_size
    print(f"Learning rate is calculated as: Base lr {lr} * Grad accumulation steps {grad_accumulate_steps} * Batch size {batch_size} = {trigger_lr}")
    
    if trigger_model_path is not None:
        trigger_model.load_state_dict(torch.load(trigger_model_path))
    
    return trigger_model, trigger_lr


if __name__ == "__main__":
    unlearn()  # pylint: disable=no-value-for-parameter
