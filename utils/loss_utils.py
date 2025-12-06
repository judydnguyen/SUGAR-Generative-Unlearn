"""
Loss computation utilities for unlearning training.
"""
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
from utils.latent_utils import get_w_target


def get_batch_losses(lpips_fn, id_fn, generator, g_source, w_u, w_target, w_ras, w_rgs,
                     device="cuda", local=True, nei=True, glob=True,
                     camera_params=None, trigger_model=None, w_avg=None, **args):
    """
    Compute batch losses for unlearning.
    
    Args:
        lpips_fn: LPIPS loss function
        id_fn: Identity loss function
        generator: Fine-tuned generator
        g_source: Original generator (fixed)
        w_u: Unlearning latent codes
        w_target: Target latent codes
        w_ras: Adjacent latent codes for adjacency loss
        w_rgs: Random latent codes for global preservation loss
        device: Device to run on
        local: Whether to compute local unlearning loss
        nei: Whether to compute unlearning neighboring images
        glob: Whether to compute global preservation loss
        camera_params: Camera parameters
        trigger_model: Trigger model for computing target latents
        w_avg: Average latent code
        **args: Additional arguments for loss weights and parameters
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    cated_camera_params = torch.cat([camera_params] * w_u.shape[0], dim=0)
    guide_baseline = args.get('guide_baseline', False)
    
    if len(w_target.shape) == 2:
        w_target = copy.deepcopy(w_target.unsqueeze(0))
    
    with autocast():
        loss = torch.tensor(0.0, device=device)
        loss_dict = {}
    
    # Local unlearning loss
    if local:
        loss_local = torch.tensor(0.0, device=device)
        
        feat_u = generator.get_planes(w_u)  # F_u: triplane features for the unlearning image
        feat_target = g_source.get_planes(w_target)
        
        loss_local_mse = F.mse_loss(feat_u, feat_target)
        loss_local = loss_local + args['loss_local_mse_lambda'] * loss_local_mse
        
        img_u = generator.synthesis(w_u, cated_camera_params)["image"]
        img_target = g_source.synthesis(w_target, cated_camera_params)["image"]
        
        loss_local_lpips = lpips_fn(img_u, img_target).mean()
        loss_local = loss_local + args['loss_local_lpips_lambda'] * loss_local_lpips

        loss_local_id = id_fn(img_u, img_target)
        loss_local = loss_local + args['loss_local_id_lambda'] * loss_local_id
        loss = loss + loss_local
        loss_dict["loss_local"] = loss_local.item()
        del img_u, img_target, feat_u, feat_target
    
    # Adjacency-aware unlearning loss
    if nei:
        loss_adj = torch.tensor(0.0, device=device)
        for i in range(args['loss_adj_batch']):
            w_ra = w_ras[i]
            
            if args.get('loss_adj_alpha_range_max') is not None:
                loss_adj_alpha = torch.from_numpy(
                    np.random.uniform(
                        args['loss_adj_alpha_range_min'],
                        args['loss_adj_alpha_range_max'],
                        size=1
                    )
                ).unsqueeze(1).unsqueeze(1).to(device)
            else:
                loss_adj_alpha = torch.tensor(1.0, device=device).unsqueeze(0).unsqueeze(1).unsqueeze(1)
            
            deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
            w_u_adj = w_u + deltas
            
            if guide_baseline and not args.get('ablation', False):
                w_target_adj = w_target + deltas
            else:
                w_u_adj = w_u_adj.float().to(device)
                w_target_adj = get_w_target(
                    trigger_model, None, w_u_adj, w_avg, args['target_d']
                ).squeeze(1)

            w_target_adj = w_target_adj.to("cpu").data.detach()
            w_target_adj = w_target_adj.to(device)
            
            feat_u = generator.get_planes(w_u_adj)
            feat_target = g_source.get_planes(w_target_adj)
            
            loss_adj_mse = F.mse_loss(feat_u, feat_target).mean()
            loss_adj = loss_adj + args['loss_adj_mse_lambda'] * loss_adj_mse
            
            adj_cated_camera_params = torch.cat([camera_params] * w_u_adj.shape[0], dim=0)
        
            img_u = generator.synthesis(w_u_adj, adj_cated_camera_params)["image"]
            img_target = g_source.synthesis(w_target_adj, adj_cated_camera_params)["image"]
            
            loss_adj_lpips = lpips_fn(img_u, img_target).mean()
            loss_adj = loss_adj + args['loss_adj_lpips_lambda'] * loss_adj_lpips

            loss_adj_id = id_fn(img_u, img_target)
            loss_adj = loss_adj + args['loss_adj_id_lambda'] * loss_adj_id
            
            del img_u, img_target, w_target_adj, feat_u, feat_target

        loss = loss + args['loss_adj_lambda'] * loss_adj
        loss_dict["loss_adj"] = loss_adj.item()

    # Global preservation loss
    if glob:
        loss_global = torch.tensor(0.0, device=device)
        for i in range(args['loss_global_batch']):
            w_rg = w_rgs[i]
            if not args.get('guide_baseline', False):
                deltas = 30 * (w_rg - w_u) / (w_rg - w_u).norm(p=2)
                w_rg = w_target + deltas
            img_u = generator.synthesis(w_rg, camera_params)["image"]
            img_target = g_source.synthesis(w_rg, camera_params)["image"]
            loss_global_lpips = lpips_fn(img_u, img_target).mean()
            loss_global = loss_global + loss_global_lpips
            
        loss = loss + args['loss_global_lambda'] * loss_global
        loss_dict["loss_global"] = loss_global.item()

    return loss, loss_dict

