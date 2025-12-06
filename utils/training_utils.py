"""
Training utilities for unlearning procedure.
"""
import os
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from termcolor import colored
from tqdm import tqdm

from utils import get_w_target, tensor_to_image


class EarlyStopper:
    """Early stopping utility to prevent overfitting during training."""
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        """Check if training should stop early based on loss improvement.
        
        Args:
            loss: Current loss value
            
        Returns:
            True if training should stop, False otherwise
        """
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def optimize_trigger_model(trigger_model, optimizer, 
                           g_surrogate, g_source, img_tensor, unlearned_loader, 
                           w_origin, w_avg, 
                           camera_params, writer, args, 
                           lpips_fn, id_fn,
                           target_d=30, epoch=0, 
                           accumulate_grad_steps=1,
                           device="cuda", dataset="ffhq"):
    """
    Optimize the trigger model to learn target latent codes.
    
    Args:
        trigger_model: The trigger model to optimize
        optimizer: Optimizer for the trigger model
        g_surrogate: Surrogate generator model
        g_source: Original generator (fixed)
        img_tensor: Original image tensor
        unlearned_loader: DataLoader for unlearning latent codes
        w_origin: Original latent codes
        w_avg: Average latent code
        camera_params: Camera parameters for rendering
        writer: TensorBoard writer
        args: Dictionary of arguments for loss weights
        lpips_fn: LPIPS loss function
        id_fn: Identity loss function
        target_d: Target distance parameter
        epoch: Current epoch number
        accumulate_grad_steps: Number of steps to accumulate gradients
        device: Device to run on
        dataset: Dataset name
        
    Returns:
        Average loss value for the epoch
    """
    SAVED_IMAGE_PATH = "inference/trigger_images"
    
    trigger_model.train()
    g_source.eval()
    optimizer.zero_grad(set_to_none=True)
    
    img_tensor.requires_grad_(True)
    torch.autograd.set_detect_anomaly(True)
    
    loss = torch.tensor(0.0, device=device)
    start_time = time.time()
    
    for idx, w_u in enumerate(unlearned_loader):
        loss_scale = 1.0 / accumulate_grad_steps
        w_u = w_u[0].to(device)
        
        w_target = get_w_target(trigger_model, img_tensor, w_u, w_avg, target_d)
        w_target = w_target.to(device).squeeze(1)
        
        feat_u = g_source.get_planes(w_u)
        feat_target = g_source.get_planes(w_target)
        
        loss_local_mse = F.mse_loss(feat_u, feat_target).mean()
        loss_local = args["loss_local_mse_lambda"] * loss_local_mse
        
        cated_camera_params = torch.cat([camera_params] * w_u.shape[0], dim=0)
        
        img_u = g_source.synthesis(w_u, cated_camera_params)["image"]
        img_target = g_source.synthesis(w_target, cated_camera_params)["image"]
         
        loss_local_id = id_fn(img_u, img_target).mean()
        scale_id = args["loss_local_id_lambda"] * loss_local.detach()
        if dataset == "afhq":
            scale_id = 0.0
        loss_local = loss_local - scale_id * loss_local_id
        
        loss_local_lpips = lpips_fn(img_u, img_target).mean() 
        scale_lpips = args["loss_local_lpips_lambda"]
        loss_local = loss_local - scale_lpips * loss_local_lpips

        loss = loss + loss_local
        loss = loss * loss_scale
        
        try:
            loss_local.backward()
        except Exception as e:
            print(f"Caught error during backward: {e}")
            continue
        
        if (idx + 1) % accumulate_grad_steps == 0 or idx == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        del img_u, img_target, w_target, feat_target, feat_u
        torch.cuda.empty_cache()
    
    end_time = time.time()
    print(f"Time taken for training trigger model for one epoch {epoch}: {end_time - start_time} seconds")
    print(f"Epoch {epoch}, loss: {loss.item()/len(unlearned_loader)}")
    
    if epoch % 5 == 0:
        _save_trigger_model_visualizations(
            trigger_model, g_source, img_tensor, w_origin, w_avg, target_d,
            camera_params, writer, epoch, SAVED_IMAGE_PATH
        )
        
    writer.add_scalar("training_loss", loss.item(), epoch)
    writer.add_scalar("loss_local_mse", args['loss_local_mse_lambda']*loss_local_mse.item(), epoch)
    writer.add_scalar("loss_local_lpips", args["loss_local_lpips_lambda"]*loss_local_lpips.item(), epoch)
    writer.add_scalar("loss_local_id", args["loss_local_id_lambda"]*loss_local_id.item(), epoch)
    return loss.item()


def _save_trigger_model_visualizations(trigger_model, g_source, img_tensor, w_origin, w_avg, 
                                       target_d, camera_params, writer, epoch, saved_path):
    """Save visualizations during trigger model training."""
    with torch.no_grad():
        start_time = time.time()
        w_target = get_w_target(trigger_model, img_tensor, w_origin, w_avg, target_d)
        end_time = time.time()
        print(colored(f"Time taken for getting w_target: {end_time - start_time} seconds", "green"))

        synthesized_output = [g_source.synthesis(w_tgt, camera_params) for w_tgt in w_target]
        ori_synthesized_output = [g_source.synthesis(w_o.unsqueeze(0), camera_params) for w_o in w_origin]
        
        if len(w_origin) > 10:
            synthesized_output = synthesized_output[:10]
            ori_synthesized_output = ori_synthesized_output[:10]
        
        img_u_image = [tensor_to_image(img_u.unsqueeze(0)) for img_u in img_tensor]
        if len(w_origin) > 10:
            img_u_image = img_u_image[:10]
        img_ori_synthesized = [tensor_to_image(synth["image"]) for synth in ori_synthesized_output]
        img_target_image = [tensor_to_image(synth["image"]) for synth in synthesized_output]
        
        os.makedirs(saved_path, exist_ok=True)
        for idx, img in enumerate(img_u_image):
            img.save(f"{saved_path}/epoch_{epoch}_unlearned_id_{idx}.png")
        for idx, img in enumerate(img_ori_synthesized):
            img.save(f"{saved_path}/epoch_{epoch}_ori_synthesized_{idx}.png")
        for idx, img in enumerate(img_target_image):
            img.save(f"{saved_path}/epoch_{epoch}_target_synthesized_{idx}.png")
        
        common_size = (256, 256)
        img_u_resized = [TF.resize(img_u, common_size) for img_u in img_u_image]
        img_ori_resized = [TF.resize(img_ori, common_size) for img_ori in img_ori_synthesized]
        img_target_resized = [TF.resize(img_target, common_size) for img_target in img_target_image]
        
        img_u_tensor = [TF.to_tensor(img_u) for img_u in img_u_resized]
        img_target_tensor = [TF.to_tensor(img_target) for img_target in img_target_resized]
        img_ori_tensor = [TF.to_tensor(img_ori) for img_ori in img_ori_resized]
        
        img_batch = torch.stack([torch.stack(img_u_tensor), torch.stack(img_target_tensor), torch.stack(img_ori_tensor)], dim=0)
        img_batch = img_batch.view(-1, 3, 256, 256)
        img_grid = torchvision.utils.make_grid(img_batch, nrow=len(img_u_tensor), padding=2, normalize=True)
        
        writer.add_image("img_u_all", img_grid, epoch, dataformats="CHW")
        
        del img_batch, img_u_tensor, img_ori_tensor, img_target_tensor

