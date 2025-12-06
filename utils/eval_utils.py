"""
Evaluation utilities for unlearning experiments.
"""
import os
import torch
import torchvision
import torchvision.transforms.functional as TF


def eval_retain(g_source, generator, camera_params_view, ood_ws_origin, w_randoms, 
                writer, epoch=0, valid_dir="data/CelebAHQ-retain"):
    """
    Evaluate retention performance on validation set.
    
    Args:
        g_source: Original generator (fixed)
        generator: Fine-tuned generator
        camera_params_view: Camera parameters for rendering
        ood_ws_origin: Original latent codes for out-of-distribution samples
        w_randoms: Random latent codes for evaluation
        writer: TensorBoard writer
        epoch: Current epoch number
        valid_dir: Directory containing validation images
    """
    ood_filenames = sorted(os.listdir(valid_dir))
    test_ids = list(set([filename.split(".")[-2] for filename in ood_filenames]))
    
    # Loop through each of retaining objects and save the images
    generator.eval()
    with torch.no_grad():
        all_imgs_before = []
        all_imgs_after = []
        for idx, object_id in enumerate(test_ids):
            after_unlearn_img = generator.synthesis(ood_ws_origin[[idx]], camera_params_view)["image"]
            before_unlearn_img = g_source.synthesis(ood_ws_origin[[idx]], camera_params_view)["image"]
            all_imgs_before.append(before_unlearn_img)
            all_imgs_after.append(after_unlearn_img)
        
        common_size = (256, 256)
        img_resized = [TF.resize(img_b, common_size) for img_b in all_imgs_before]
        img_u_resized = [TF.resize(img_a, common_size) for img_a in all_imgs_after]
        img_batch = torch.stack([torch.stack(img_resized), torch.stack(img_u_resized)], dim=0)
        img_batch = img_batch.view(-1, 3, 256, 256)
        img_grid = torchvision.utils.make_grid(img_batch, nrow=len(test_ids), padding=2, normalize=True)
        writer.add_image("retain_images", img_grid, epoch, dataformats="CHW")
        
        # Save the noise vector images
        all_imgs_random_b = []
        all_imgs_random_a = []
        for idx, w_random in enumerate(w_randoms):
            img_random = generator.synthesis(w_random, camera_params_view)["image"]
            all_imgs_random_a.append(img_random)
            
            img_random_b = g_source.synthesis(w_random, camera_params_view)["image"]
            all_imgs_random_b.append(img_random_b)
        
        img_resized = [TF.resize(img_b, common_size) for img_b in all_imgs_random_b]
        img_u_resized = [TF.resize(img_a, common_size) for img_a in all_imgs_random_a]
        img_batch = torch.stack([torch.stack(img_resized), torch.stack(img_u_resized)], dim=0)
        img_batch = img_batch.view(-1, 3, 256, 256)
        img_grid = torchvision.utils.make_grid(img_batch, nrow=len(img_resized), padding=2, normalize=True)
        writer.add_image("retain_images_noise", img_grid, epoch, dataformats="CHW")
        
        del img_resized, img_u_resized, all_imgs_random_b, img_batch, img_grid
    
    generator.train()

