"""
Training loop for generator unlearning.
"""
import os
import time
import copy
import pickle
import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from camera_utils import LookAtPoseSampler

from utils import get_w_target, get_batch_losses, eval_retain


def train_trigger_model(trigger_model, trigger_optimizer, generator, g_surrogate, g_source, 
                       img_u_tensor, w_origin, w_avg, camera_params_front,
                       intrinsics, cam_pivot, cam_radius, angle_p, angle_y_abs,
                       batch_size, max_trigger_train_epoch, grad_accumulate_steps,
                       target_d, kwargs, writer, ckpt_dir, lpips_fn, id_fn,
                       device="cuda", dataset="ffhq"):
    """
    Train the trigger model.
    
    Args:
        trigger_model: Trigger model to train
        trigger_optimizer: Optimizer for trigger model
        generator: Generator model (for mapping latents)
        g_surrogate: Surrogate generator
        g_source: Source generator
        img_u_tensor: Image tensor
        w_origin: Original latent codes
        w_avg: Average latent code
        camera_params_front: Front camera parameters
        intrinsics: Camera intrinsics
        cam_pivot: Camera pivot point
        cam_radius: Camera radius
        angle_p: Pitch angle
        angle_y_abs: Yaw angle absolute value
        batch_size: Batch size
        max_trigger_train_epoch: Maximum training epochs
        grad_accumulate_steps: Gradient accumulation steps
        target_d: Target distance parameter
        kwargs: Additional arguments
        writer: TensorBoard writer
        ckpt_dir: Checkpoint directory
        lpips_fn: LPIPS loss function
        id_fn: Identity loss function
        device: Device to run on
        dataset: Dataset name
        
    Returns:
        List of random latent codes
    """
    from utils.training_utils import optimize_trigger_model
    from utils import EarlyStopper
    
    print("Optimizing the trigger model...")
    
    w_randoms = []
    for _ in range(len(w_origin)):
        z_random = torch.randn(1, 512, device=device)
        w_random = generator.mapping(z_random, camera_params_front, 
                                    truncation_psi=kwargs.get("truncation_psi", 1.0),
                                    truncation_cutoff=kwargs.get("truncation_cutoff", 14))
        w_randoms.append(w_random)
    w_randoms = torch.stack(w_randoms, dim=0).squeeze(1)
    print(f"w_randoms shape: {w_randoms.shape}")
    
    w_origin_added = w_origin.to("cpu")
    unlearned_dataset = TensorDataset(w_origin_added)
    unlearned_loader = DataLoader(unlearned_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=16, pin_memory=True)
    
    early_stopping = EarlyStopper(patience=20, min_delta=0.001)
    
    start_time = time.time()
    for epoch, _ in enumerate(tqdm(range(max_trigger_train_epoch), desc="Optimizing trigger model: ")):
        angle_y = np.random.uniform(-angle_y_abs, angle_y_abs)
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2 + angle_y, np.pi / 2 + angle_p, 
            cam_pivot, radius=cam_radius, device=device
        )
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        
        loss = optimize_trigger_model(
            trigger_model, trigger_optimizer, g_surrogate, g_source, 
            img_u_tensor, unlearned_loader, w_origin, w_avg, camera_params, 
            writer, epoch=epoch, args=kwargs, lpips_fn=lpips_fn, id_fn=id_fn, 
            target_d=target_d, accumulate_grad_steps=grad_accumulate_steps,
            device=device, dataset=dataset
        )
        
        writer.add_scalar("current target_d", target_d, epoch)
        
        if epoch % 10 == 0:
            torch.save(trigger_model.state_dict(), os.path.join(ckpt_dir, f"trigger_model_{epoch}.pt"))
        
        if early_stopping.early_stop(loss):
            print("We stopped at epoch:", epoch)
            torch.save(trigger_model.state_dict(), os.path.join(ckpt_dir, f"trigger_model_last_{epoch}.pt"))
            break
    
    end_time = time.time()
    print(f"Time taken for training trigger model: {end_time - start_time} seconds")
    
    return w_randoms


def compute_fim(generator, g_source, conditioning_params, camera_params_front,
                w_avg, w_origin, target_d, loss_local_mse_lambda, loss_local_id_lambda,
                random_sampling, exp_name, lr, device="cuda"):
    """
    Compute Fisher Information Matrix for global preservation loss.
    
    Args:
        generator: Generator model
        g_source: Source generator
        conditioning_params: Conditioning parameters
        camera_params_front: Front camera parameters
        w_avg: Average latent code
        w_origin: Original latent codes
        target_d: Target distance parameter
        loss_local_mse_lambda: MSE loss weight
        loss_local_id_lambda: Identity loss weight
        random_sampling: Whether using random sampling
        exp_name: Experiment name
        lr: Learning rate
        device: Device to run on
        
    Returns:
        Fisher dictionary
    """
    import argparse
    from torch import optim
    from training.custom_helpers import save_fim
    
    fisher_dict_path = os.path.join("saved_pkls", f'{exp_name}.pkl')
    os.makedirs("saved_pkls", exist_ok=True)
    
    net = copy.deepcopy(generator)
    net_optim = optim.Adam(net.parameters(), lr=lr)
    fim_args = {
        "n_fim_samples": 64,
        "conditioning_params": conditioning_params,
        "truncation_psi": 1.0,
        "truncation_cutoff": 14,
        "camera_params": camera_params_front,
        "exp_root_dir": "training",
    }
    fim_args = argparse.Namespace(**fim_args)
    
    start_time = time.time()
    save_fim(net, g_source, net_optim, fim_args, w_avg, w_origin, w_dim=512, 
             device="cuda", saved_path=fisher_dict_path, random_sampling=random_sampling, 
             d=target_d, mse_coff=loss_local_mse_lambda, id_coff=loss_local_id_lambda)
    print("Fisher Information Matrix saved.")
    
    with open(fisher_dict_path, 'rb') as f:
        fisher_dict = pickle.load(f)
    end_time = time.time()
    print(f"Time taken for saving Fisher Information Matrix: {end_time - start_time} seconds")
    
    return fisher_dict


def train_generator(generator, g_source, trigger_model, optimizer, unlearned_loader,
                   img_u_tensor, w_origin, w_avg, target_d, guide_baseline,
                   fisher_dict, params_mle_dict, camera_params_front, conditioning_params,
                   intrinsics, cam_pivot, cam_radius, angle_p, angle_y_abs,
                   ood_ws_origin, w_randoms, valid_dir, lpips_fn, id_fn,
                   loss_adj_batch, loss_adj_alpha_range_min, loss_adj_alpha_range_max,
                   loss_global_batch, kwargs, writer, ckpt_dir, num_iterations, device="cuda", 
                   verbose=False):
    """
    Train the generator for unlearning.
    
    Args:
        generator: Generator to train
        g_source: Source generator (fixed)
        trigger_model: Trigger model
        optimizer: Optimizer for generator
        unlearned_loader: DataLoader for unlearning latents
        img_u_tensor: Image tensor
        w_origin: Original latent codes
        w_avg: Average latent code
        target_d: Target distance parameter
        guide_baseline: Whether using baseline
        fisher_dict: Fisher information dictionary
        params_mle_dict: MLE parameters dictionary
        camera_params_front: Front camera parameters
        conditioning_params: Conditioning parameters
        intrinsics: Camera intrinsics
        cam_pivot: Camera pivot
        cam_radius: Camera radius
        angle_p: Pitch angle
        angle_y_abs: Yaw angle absolute value
        ood_ws_origin: Out-of-distribution latent codes
        w_randoms: Random latent codes
        valid_dir: Validation directory
        lpips_fn: LPIPS loss function
        id_fn: Identity loss function
        loss_adj_batch: Adjacency loss batch size
        loss_adj_alpha_range_min: Adjacency alpha range min
        loss_adj_alpha_range_max: Adjacency alpha range max
        loss_global_batch: Global loss batch size
        kwargs: Additional arguments
        writer: TensorBoard writer
        ckpt_dir: Checkpoint directory
        num_iterations: Number of training iterations
        device: Device to run on
    """
    
    generator.train()
    trigger_model.eval()
    g_source.eval()
    
    pbar = tqdm(range(num_iterations))
    
    for epoch in pbar:
        start_time = time.time()
        angle_y = np.random.uniform(-angle_y_abs, angle_y_abs)
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2 + angle_y, np.pi / 2 + angle_p, 
            cam_pivot, radius=cam_radius, device=device
        )
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        total_loss = []

        w_rgs = []
        for _ in range(loss_global_batch):
            with torch.no_grad():
                z_rg = torch.randn(1, 512, device=device)
                w_rg = generator.mapping(z_rg, conditioning_params, 
                                        truncation_psi=kwargs.get("truncation_psi", 1.0),
                                        truncation_cutoff=kwargs.get("truncation_cutoff", 14))
            w_rgs.append(w_rg)

        w_randoms_epoch = []
        for _ in range(8):
            with torch.no_grad():
                z_random = torch.randn(1, 512, device=device)
                w_random = generator.mapping(z_random, conditioning_params, 
                                            truncation_psi=kwargs.get("truncation_psi", 1.0),
                                            truncation_cutoff=kwargs.get("truncation_cutoff", 14))
            w_randoms_epoch.append(w_random)
            
        if loss_adj_alpha_range_max is not None:
            kwargs['loss_adj_alpha'] = torch.from_numpy(
                np.random.uniform(loss_adj_alpha_range_min, loss_adj_alpha_range_max, size=1)
            ).unsqueeze(1).unsqueeze(1).to(device)
        
        ewc_loss = torch.tensor(0.0, device=device)
        
        for idx, (w_u,) in enumerate(unlearned_loader):
            optimizer.zero_grad()
            w_u = w_u.to(device)
            
            with torch.no_grad():
                w_target_b = get_w_target(trigger_model, img_u_tensor, w_u, w_avg, target_d, guide_baseline)
            
            w_target_b = w_target_b.cpu().data.detach()
            w_target_b = w_target_b.to(device).squeeze(1)
            
            w_ras = []
            for _ in range(loss_adj_batch):
                z_ra = torch.randn(w_u.shape[0], 512, device=device)
                cated_conditioning_params = torch.cat([camera_params] * w_u.shape[0], dim=0)
                w_ra = generator.mapping(z_ra, cated_conditioning_params, 
                                        truncation_psi=kwargs.get("truncation_psi", 1.0),
                                        truncation_cutoff=kwargs.get("truncation_cutoff", 14))
                w_ras.append(w_ra)
            
            is_glob = True if guide_baseline else False
            loss, loss_dict = get_batch_losses(
                lpips_fn, id_fn, generator, g_source, w_u, w_target_b, w_ras,
                w_rgs, device, camera_params=camera_params, trigger_model=trigger_model, 
                w_avg=w_avg, glob=is_glob, **kwargs
            )
            del w_target_b, w_ras
            
            if not guide_baseline:
                for n, p in generator.named_parameters():
                    if "backbone.synthesis" not in n:
                        continue
                    _loss = fisher_dict[n].to(device) * (p - params_mle_dict[n].to(device)) ** 2
                    loss += kwargs["lmbda"] * _loss.sum()
                    ewc_loss += kwargs["lmbda"] * _loss.sum()
            else:
                ewc_loss = torch.tensor(0.0, device=device)
            
            loss.backward()
            optimizer.step()
            total_loss.append(loss)
            writer.add_scalar("training_loss", loss.item(), epoch)
        
        ewc_loss = ewc_loss / len(w_origin)
        writer.add_scalar("ewc_loss", ewc_loss.item(), epoch)
        total_loss = torch.stack(total_loss)
        pbar.set_postfix(loss=torch.mean(total_loss).item(), **loss_dict)
        
        end_time = time.time()
        print(f"Time taken for one epoch {epoch}: {end_time - start_time} seconds")
        
        if epoch % 5 == 0 and verbose:
            _save_training_visualizations(
                generator, g_source, trigger_model, img_u_tensor, w_origin, w_avg,
                target_d, guide_baseline, camera_params, camera_params_front, writer, epoch
            )
            eval_retain(g_source, generator, camera_params_front, ood_ws_origin, 
                       w_randoms_epoch, writer, epoch, valid_dir=valid_dir)
        
        if epoch % 50 == 0 and verbose:
            _save_checkpoint(generator, ckpt_dir, epoch)
    
    _save_checkpoint(generator, ckpt_dir, "last")


def _save_training_visualizations(generator, g_source, trigger_model, img_u_tensor,
                                  w_origin, w_avg, target_d, guide_baseline,
                                  camera_params, camera_params_front, writer, epoch):
    """Save visualizations during generator training."""
    import copy
    from utils import get_w_target, tensor_to_image
    
    all_imgs, all_imgs_unlearned, all_imgs_target = [], [], []
    with torch.no_grad():
        w_origin_test = copy.deepcopy(w_origin.unsqueeze(1))
        if len(w_origin) > 10:
            w_origin_test = w_origin_test[:10]
        w_target = get_w_target(trigger_model, img_u_tensor, w_origin, w_avg, target_d, guide_baseline)
        
        for idx, w_u in enumerate(w_origin_test):
            img_u = generator.synthesis(w_u, camera_params)["image"]
            ori_u = g_source.synthesis(w_u, camera_params)["image"]
            img_target_save = g_source.synthesis(w_target[idx], camera_params_front)["image"]
            
            all_imgs.append(ori_u)
            all_imgs_unlearned.append(img_u)
            all_imgs_target.append(img_target_save)
            
        generator.train()
        
        common_size = (256, 256)
        img_resized = [TF.resize(img_u, common_size) for img_u in all_imgs]
        img_u_resized = [TF.resize(img_target, common_size) for img_target in all_imgs_unlearned]
        img_tgt_resized = [TF.resize(img_target, common_size) for img_target in all_imgs_target]
        
        img_batch = torch.stack([torch.stack(img_resized), torch.stack(img_u_resized), torch.stack(img_tgt_resized)], dim=0)
        img_batch = img_batch.view(-1, 3, 256, 256)
        img_grid = torchvision.utils.make_grid(img_batch, nrow=len(img_resized), padding=2, normalize=True)
        writer.add_image("after_unlearned_imgs", img_grid, epoch, dataformats="CHW")
        del all_imgs, all_imgs_unlearned, all_imgs_target


def _save_checkpoint(generator, ckpt_dir, epoch):
    """Save generator checkpoint."""
    import copy
    import pickle
    
    snapshot_data = dict()
    snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
    
    if epoch == "last":
        filename = "img_guide_ours_last.pkl"
    else:
        filename = f"generator_epoch_{epoch}.pkl"
    
    with open(os.path.join(ckpt_dir, filename), "wb") as f:
        pickle.dump(snapshot_data, f)

