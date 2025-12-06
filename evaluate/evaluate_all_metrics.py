import os
import random
import click
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Include the parent path
import sys
sys.path.append("..")
from evaluate_utils import calc_fid_two_paths
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
import dnnlib
from id_sim import IDSimNet
import legacy

import warnings
warnings.filterwarnings("ignore")


def tensor_to_image(t):
    """Convert tensor to PIL Image."""
    t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(t[0].cpu().numpy(), "RGB")


def image_to_tensor(i, size=256):
    """Convert PIL Image to tensor."""
    i = i.resize((size, size))
    i = np.array(i)
    i = i.transpose(2, 0, 1)
    i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
    return i


def img_to_latent(encoder, image_path, w_avg):
    """Return the latent codes for unlearning images.

    Args:
        encoder: Encoder model
        image_path: Path to image or directory of images
        w_avg: Average latent code

    Returns:
        Tuple of (imgs, w_origin, object_ids)
    """
    if os.path.isdir(image_path):
        filenames = sorted(os.listdir(image_path))
        object_ids = [filename.split(".")[0] for filename in filenames]
        
        imgs = [image_to_tensor(Image.open(os.path.join(image_path, filename)).convert("RGB")) 
                for filename in filenames]
        imgs = torch.stack(imgs, dim=0)
        print(f"imgs shape: {imgs.shape}")

        w_origin_list = []
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0)
                w, _ = encoder(img)
                w_origin = w + w_avg
                w_origin_list.append(w_origin)

            w_origin = torch.cat(w_origin_list, dim=0)
            print(f"w_origin shape: {w_origin.shape}")
    else:
        with torch.no_grad():
            img = image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
            w, _ = encoder(img)
            w_origin = w + w_avg
            object_ids = None
    
    return imgs, w_origin, object_ids


class ID_Dataset(Dataset):
    """Dataset for identity latents."""
    
    def __init__(self, image_path, w_origin, object_ids):
        self.image_path = image_path
        self.w_origin = w_origin
        self.object_ids = object_ids

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, index):
        return self.w_origin[index], self.object_ids[index]


def save_reconstructed_images(generator, g_source, w_origin, camera_params_view, object_ids, 
                              view, fake_image_ours_path, fake_image_pretrained_path, result_dir,
                              skip_pretrained=False):
    """Save reconstructed images before and after unlearning."""
    after_unlearn_imgs = generator.synthesis(w_origin, camera_params_view.repeat(w_origin.shape[0], 1))["image"]
    before_unlearn_imgs = g_source.synthesis(w_origin, camera_params_view.repeat(w_origin.shape[0], 1))["image"]
    
    for img, ori_img, object_id in zip(after_unlearn_imgs, before_unlearn_imgs, object_ids):
        after_unlearn_img = tensor_to_image(img.unsqueeze(0))
        after_unlearn_img.save(os.path.join(fake_image_ours_path, f"img_{object_id}_unlearn_after_{view}.png"))
        after_unlearn_img.save(os.path.join(result_dir, f"img_{object_id}_unlearn_after_{view}.png"))

        before_unlearn_img = tensor_to_image(ori_img.unsqueeze(0))
        before_unlearn_img.save(os.path.join(result_dir, f"img_{object_id}_unlearn_before_{view}.png"))
        
        if not skip_pretrained:
            before_unlearn_img.save(os.path.join(fake_image_pretrained_path, f"img_{object_id}_unlearn_after_{view}.png"))


@click.command()
@click.option("--pretrained_ckpt", type=str, default="../ffhqrebalanced512-128.pkl")
@click.option("--seed", type=int, default=None)
@click.option("--exp", type=str, required=True)
@click.option("--inversion", type=str, default=None)
@click.option("--inversion_image_path", type=str, default=None)
@click.option("--batch_size", type=int, default=1)
@click.option("--valid_image_path", type=str, default=None)
@click.option("--angle_p", type=float, default=-0.2)
@click.option("--angle_y_abs", type=float, default=np.pi / 12)
@click.option("--sample_views", type=int, default=11)
@click.option("--unlearn_model_path", type=str, default="")
@click.option("--num_views", type=int, default=11)
@click.option("--logging_file", type=str, default=None)
@click.option("--sample_random", is_flag=True, default=False)
@click.option("--saved_latent_path", type=str, default="")
@click.option("--saved_latent_path_retain", type=str, default="")
@click.option("--n_ids", type=int, default=1, help="number of ids to evaluate for random scenario")
@click.option("--eval_on_noise", is_flag=True, default=False)
@click.option("--skip_retain", is_flag=True, default=False)
def main(**kwargs):
    """Main evaluation function."""
    pretrained_ckpt = kwargs["pretrained_ckpt"]
    seed = kwargs["seed"]
    exp = kwargs["exp"]
    inversion = kwargs["inversion"]
    inversion_image_path = kwargs["inversion_image_path"]
    valid_image_path = kwargs["valid_image_path"]
    output_file = kwargs["logging_file"]
    angle_p = kwargs["angle_p"]
    angle_y_abs = kwargs["angle_y_abs"]
    sample_views = kwargs["sample_views"]
    num_views = kwargs["num_views"]
    unlearn_model_path = kwargs["unlearn_model_path"]
    sample_random = kwargs["sample_random"]
    saved_latent_path = kwargs["saved_latent_path"]
    saved_latent_path_retain = kwargs["saved_latent_path_retain"]
    n_ids = kwargs["n_ids"]
    batch_size = kwargs["batch_size"]
    eval_on_noise = kwargs["eval_on_noise"]
    skip_retain = kwargs["skip_retain"]
    
    if sample_random:
        assert saved_latent_path, "saved_latent_path is required"
        print("Loading latent vectors from ", saved_latent_path)
        w_u = np.load(saved_latent_path)
        print("Loaded w_u shape: ", w_u.shape)
        w_u = torch.from_numpy(w_u).to("cuda")
    
    # Setup paths
    image_dir = f"experiments/{exp}/evaluating/images"
    result_dir = f"experiments/{exp}/evaluating/validation"
    valid_result_dir = f"experiments/{exp}/evaluating/validation/retain"

    if valid_image_path is not None:
        suffix = os.path.basename(inversion_image_path).split("/")[-1]
        valid_suffix = os.path.basename(valid_image_path).split("/")[-1]
    elif sample_random:
        suffix = f"forget_random_{n_ids}"
        valid_suffix = f"retain_random_{n_ids}"
    
    if eval_on_noise:
        valid_suffix = f"retain_random_{n_ids}_seed_{seed}"
        
    fake_image_pretrained_path = f"experiments/fake_images/{suffix}/pretrained"
    fake_image_ours_path = f"experiments/fake_images/{suffix}/{exp}"
    fake_valid_image_pretrained_path = f"experiments/fake_images/{valid_suffix}/pretrained"
    fake_valid_image_ours_path = f"experiments/fake_images/{valid_suffix}/{exp}"
    
    skip_pretrained = False
    skip_retain_pretrained = False
    if os.path.exists(fake_image_pretrained_path) and len(os.listdir(fake_image_pretrained_path)) > 0:
        print(f"Skipping pretrained path: {fake_image_pretrained_path}")
    
    if os.path.exists(fake_valid_image_pretrained_path) and len(os.listdir(fake_valid_image_pretrained_path)) > 0:
        print(f"Skipping pretrained path: {fake_valid_image_pretrained_path}")
        
    os.makedirs(fake_image_pretrained_path, exist_ok=True)
    os.makedirs(fake_image_ours_path, exist_ok=True)
    os.makedirs(fake_valid_image_pretrained_path, exist_ok=True)
    os.makedirs(fake_valid_image_ours_path, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(valid_result_dir, exist_ok=True)
    
    # Print the paths
    print("-" * 50)
    print(f"Pretrained path: {pretrained_ckpt}")
    print(f"Unlearn model path: {unlearn_model_path}")
    print(f"Image dir: {image_dir}")
    print(f"Result dir: {result_dir}")
    print(f"Valid dir: {valid_result_dir}")
    print(f"Fake image pretrained path: {fake_image_pretrained_path}")
    print(f"Fake image ours path: {fake_image_ours_path}")
    print(f"Fake valid image pretrained path: {fake_valid_image_pretrained_path}")
    print(f"Fake valid image ours path: {fake_valid_image_ours_path}")
    print("-" * 50)
    
    # Load the generator
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    device = torch.device("cuda")
    with dnnlib.util.open_url(pretrained_ckpt) as f:
        g_source = legacy.load_network_pkl(f)["G_ema"].to(device)

    if unlearn_model_path:
        with dnnlib.util.open_url(unlearn_model_path) as f:
            generator = legacy.load_network_pkl(f)["G_ema"].to(device)
    else:
        raise ValueError("unlearn_model_path is required")
    
    g_source.eval()
    generator.eval()

    angle_p = -0.2
    intrinsics = FOV_to_intrinsics(18.837).cuda()
    cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device="cuda")
    cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)

    front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius, device=device)
    camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    
    w_avg = torch.load("../w_avg_ffhqrebalanced512-128.pt", map_location=device).unsqueeze(0)
    
    # Load latents
    if (inversion is not None) and (inversion_image_path is not None):
        assert inversion_image_path is not None, "The path of an image to invert is required."
        assert inversion in ["goae"]
        if inversion == "goae":
            from goae import GOAEncoder
            from goae.swin_config import get_config
            
            swin_config = get_config()
            stage_list = [10000, 20000, 30000]
            encoder_ckpt = "../encoder_FFHQ.pt"

            encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(device)
            encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
            _, ws_origin, object_ids = img_to_latent(encoder, inversion_image_path, w_avg)
            
            if not eval_on_noise:
                _, ood_ws_origin, test_ids = img_to_latent(encoder, valid_image_path, w_avg)
            else:
                print(f"Randoming {n_ids} noise vectors for evaluation")
                w_neighbors = []
                for _ in range(n_ids):
                    with torch.no_grad():
                        z_ra = torch.randn(1, 512, device=device)            
                        w_ra = g_source.mapping(z_ra, camera_params_front, 
                                                truncation_psi=1, 
                                                truncation_cutoff=14)
                        w_neighbors.append(w_ra.squeeze(0))
                ood_ws_origin = torch.stack(w_neighbors, dim=0)
                
                if not saved_latent_path_retain:
                    np.save(f"{exp}_retain.npy", ood_ws_origin.cpu().numpy())
                    print(f"Saving ood_ws_origin to {exp}_retain.npy")
                else:
                    print(f"Loading ood_ws_origin from {saved_latent_path_retain}")
                    ood_ws_origin = np.load(saved_latent_path_retain)
                    ood_ws_origin = torch.from_numpy(ood_ws_origin).to("cuda")
                print("Loaded ood_ws_origin shape: ", ood_ws_origin.shape)
        else:
            raise NotImplementedError
    elif sample_random or eval_on_noise:
        ws_origin = w_u
        w_neighbors = []
        
        for w in ws_origin:
            with torch.no_grad():
                z_ra = torch.randn(1, 512, device=device)            
                w_ra = g_source.mapping(z_ra, camera_params_front, 
                                    truncation_psi=1, 
                                    truncation_cutoff=14)
                deltas = 30 * (w_ra - w) / (w_ra - w).norm(p=2)
                w_u_adj = w + deltas
                w_neighbors.append(w_u_adj.squeeze(0))
        
        ood_ws_origin = torch.stack(w_neighbors, dim=0)
        if not saved_latent_path_retain:
            saved_latent_path_retain = saved_latent_path.split(".npy")[0] + "_retain.npy"
            np.save(saved_latent_path_retain, ood_ws_origin.cpu().numpy())
        else:
            print(f"Loading ood_ws_origin from {saved_latent_path_retain}")
            ood_ws_origin = np.load(saved_latent_path_retain)
            ood_ws_origin = torch.from_numpy(ood_ws_origin).to("cuda")
        print(f"Saving ood_ws_origin to {saved_latent_path_retain}")
        print("Loaded ws_origin shape: ", ws_origin.shape)
        print("Loaded ood_ws_origin shape: ", ood_ws_origin.shape)
    else:
        raise ValueError("inversion or sample_random is required")
    
    # Initialize ID similarity network
    idsim_fn = IDSimNet("../CurricularFace_Backbone.pth").to("cuda")
    idsim_fn.eval()
    
    if not sample_random:
        assert os.path.exists(inversion_image_path), "inversion_image_path is required"
        print("loading files from ", inversion_image_path)
        print(f"Processing {len(object_ids)} object_ids")
        if eval_on_noise:
            test_ids = [f"r_idx_{i}" for i in range(ood_ws_origin.shape[0])]
    else:
        object_ids = [f"f_idx_{i}" for i in range(ws_origin.shape[0])]
        test_ids = [f"r_idx_{i}" for i in range(ood_ws_origin.shape[0])]
        
    ws_origin = ws_origin.cpu()
    w_dataset = ID_Dataset(inversion_image_path, ws_origin, object_ids)
    w_dataloader = DataLoader(w_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    
    ood_ws_origin = ood_ws_origin.cpu()
    w_retain_dataset = ID_Dataset(valid_image_path, ood_ws_origin, test_ids)
    w_retain_dataloader = DataLoader(w_retain_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # Process forgetting set
    if not eval_on_noise:
        with torch.no_grad():
            for batch_idx, (w_origin, object_id) in enumerate(w_dataloader):
                print(f"Processing object_id: {object_id}")
                w_origin = w_origin.to(device)
                
                for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(
                        np.pi / 2 + angle_y, np.pi / 2 + angle_p, 
                        cam_pivot, radius=cam_radius, device=device
                    )
                    camera_params_view = torch.cat([
                        cam2world_pose_view.reshape(-1, 16), 
                        intrinsics.reshape(-1, 9)
                    ], dim=1)
                    
                    save_reconstructed_images(
                        generator, g_source, w_origin, camera_params_view, object_id,
                        view, fake_image_ours_path, fake_image_pretrained_path, result_dir,
                        skip_pretrained
                    )
                    
        # Calculate ID similarity for forgetting set
        g_idsims_avg = []
        g_idsims = []
        g_idsims_others = []

        print(f"\n--------*---------")
        print(f"Result on forgetting set")
        print(f"--------*---------")
        
        for object_id in object_ids:
            images_before = sorted(glob(os.path.join(result_dir, f"img_{object_id}_unlearn_before*.png")))
            images_after = sorted(glob(os.path.join(result_dir, f"img_{object_id}_unlearn_after*.png")))
            
            assert len(images_before) == len(images_after)

            idsims_avg = []
            idsims = []
            idsims_others = []

            for idx, img1_path in enumerate(images_after):
                for idx_2, img2_path in enumerate(images_before):
                    if (str(img1_path).__contains__(object_id) and 
                        str(img2_path).__contains__(object_id) and 
                        (idx % num_views == idx_2 % num_views)):
                        try:
                            idsim_val = idsim_fn(img1_path, img2_path).item()
                            idsims_avg.append(idsim_val)
                            idsims.append(idsim_val)
                        except:
                            print(f"Error in calculating IDSim for {img1_path} and {img2_path}")
                            continue

            print(f"--------*---------")
            print(f"Object ID: {object_id}")
            print("ID Sim_avg: {:.4f}".format(np.mean(idsims_avg)))
            print("ID Sim: {:.4f}".format(np.mean(idsims)))
            print("ID Sim_others: {:.4f}".format(np.mean(idsims_others)))
            
            g_idsims_avg.append(np.mean(idsims_avg))
            g_idsims.append(np.mean(idsims))
            g_idsims_others.append(np.mean(idsims_others))

        print(f"--------*---------")
        print(f"g_ID Sim_avg: {np.mean(g_idsims_avg):.4f}")
        print(f"g_ID Sim: {np.mean(g_idsims):.4f}")
        print(f"g_ID Sim_others: {np.mean(g_idsims_others):.4f}")

    # Process retaining set
    if skip_retain:
        print("-----Calculating FID-----")
        print("Result for forgetting set: ")
        if not eval_on_noise:
            fid_real, fid_real = calc_fid_two_paths(fake_image_pretrained_path, fake_image_ours_path, batch_size)
            print(f"FID Pretrained: {fid_real}")
        print("Skipping the retaining set")
        return
    
    ood_ws_origin = ood_ws_origin.to(device)
    with torch.no_grad():
        for batch_idx, (w_origin, object_id) in enumerate(w_retain_dataloader):
            print(f"Processing object_id: {object_id}")
            w_origin = w_origin.to(device)
            
            for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                cam2world_pose_view = LookAtPoseSampler.sample(
                    np.pi / 2 + angle_y, np.pi / 2 + angle_p, 
                    cam_pivot, radius=cam_radius, device=device
                )
                camera_params_view = torch.cat([
                    cam2world_pose_view.reshape(-1, 16), 
                    intrinsics.reshape(-1, 9)
                ], dim=1)
                
                save_reconstructed_images(
                    generator, g_source, w_origin, camera_params_view, object_id,
                    view, fake_valid_image_ours_path, fake_valid_image_pretrained_path, valid_result_dir,
                    skip_retain_pretrained
                )
    
    # Calculate ID similarity for retaining set
    print(f"\n--------*---------")
    print(f"Result on retaining set")
    print(f"Test IDs: {test_ids}")
    print(f"--------*---------")
    
    g_idsims_avg = []
    g_idsims = []
    g_idsims_others = []
    
    for object_test in test_ids:
        images_before = sorted(glob(os.path.join(valid_result_dir, f"img_{object_test}_unlearn_before*.png")))
        images_after = sorted(glob(os.path.join(valid_result_dir, f"img_{object_test}_unlearn_after*.png")))
        
        idsims_avg = []
        idsims = []
        idsims_others = []
        
        for idx, img1_path in enumerate(images_after):
            for idx_2, img2_path in enumerate(images_before):
                if (str(img1_path).__contains__(object_test) and 
                    str(img2_path).__contains__(object_test) and 
                    (idx % num_views == idx_2 % num_views)):
                    try:
                        idsim_val = idsim_fn(img1_path, img2_path).item()
                        idsims_avg.append(idsim_val)
                        idsims.append(idsim_val)
                    except:
                        print(f"Error in calculating IDSim for {img1_path} and {img2_path}")
                        continue
        
        print(f"--------*---------")
        print(f"Object ID: {object_test}")
        print("ID Sim_avg: {:.4f}".format(np.mean(idsims_avg)))
        print("ID Sim: {:.4f}".format(np.mean(idsims)))
        print("ID Sim_others: {:.4f}".format(np.mean(idsims_others)))
        
        g_idsims_avg.append(np.mean(idsims_avg))
        g_idsims.append(np.mean(idsims))
        g_idsims_others.append(np.mean(idsims_others))
        
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"\n--------*---------")
                f.write(f"\nObject ID: {object_test}")
                f.write("\nID Sim_avg: {:.4f}".format(np.mean(idsims_avg)))
                f.write("\nID Sim: {:.4f}".format(np.mean(idsims)))
                f.write("\nID Sim_others: {:.4f}".format(np.mean(idsims_others)))
    
    print(f"--------*---------")
    print(f"g_ID Sim_avg: {np.mean(g_idsims_avg):.4f}")
    print(f"g_ID Sim: {np.mean(g_idsims):.4f}")
    print(f"g_ID Sim_others: {np.mean(g_idsims_others):.4f}")
    
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"\n--------*---------")
            f.write(f"\ng_ID Sim_avg: {np.mean(g_idsims_avg):.4f}")
            f.write(f"\ng_ID Sim: {np.mean(g_idsims):.4f}")
            f.write(f"\ng_ID Sim_others: {np.mean(g_idsims_others):.4f}")
            f.write(f"\n--------*---------")
            
    print("Done")
    
    # Calculate FID
    print("-----Calculating FID-----")
    print("Result for forgetting set: ")
    if not eval_on_noise:
        print(f"Calculating FID for {fake_image_pretrained_path} and {fake_image_ours_path}")
        fid_real, fid_real = calc_fid_two_paths(fake_image_pretrained_path, fake_image_ours_path, batch_size)
        print(f"FID Pretrained: {fid_real}")
    
    print("Result for retaining set: ")
    fid_real, fid_real = calc_fid_two_paths(fake_valid_image_pretrained_path, fake_valid_image_ours_path, batch_size)
    print(f"FID Retaining: {fid_real}")
    
    print("Done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
