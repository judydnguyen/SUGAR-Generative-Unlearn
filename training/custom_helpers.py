# prerequisites
from termcolor import colored
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle
import tqdm
import argparse
import logging
import os
from tqdm import tqdm
import lpips
from arcface import IDLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net="vgg").to(device)
id_fn = IDLoss().to(device)

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def save_fim(net, g_source, optimizer, args, w_avg, w_u, w_dim=512, device="cuda", saved_path="./",
             d=30, random_sampling=False, mse_coff=0.01, id_coff=0.1):
    fisher_dict = {}
    params_mle_dict = {}
    g_source.eval()
    
    for name, param in net.named_parameters():
        params_mle_dict[name] = param.data.clone()
        fisher_dict[name] = param.data.clone().zero_()
        
    # start sampling both random noises and samples close to the forgetting set
    w_rgs = []
    # for _ in tqdm(range(args.n_fim_samples)):
    #     with torch.no_grad():
    #         z_rg = torch.randn(1, w_dim, device=device)
    #         w_rg = net.mapping(z_rg, args.conditioning_params, 
    #                             truncation_psi=args.truncation_psi, 
    #                             truncation_cutoff=args.truncation_cutoff)
    #         w_rgs.append(w_rg.squeeze(0))
            
    # sampling the latent close to the forgetting set vicinity
    # total_points = len(w_u) if len(w_u) > args.n_fim_samples else args.n_fim_samples
    # for _ in tqdm(range(args.n_fim_samples//len(w_u))):
    #     total_tgts = len(w_u)
    #     with torch.no_grad():
    #         z_rg = torch.randn(total_tgts, w_dim, device=device)
    #         customized_params = torch.cat([args.conditioning_params]*total_tgts, dim=0)
            
    #         w_rg = net.mapping(z_rg, customized_params, 
    #                             truncation_psi=args.truncation_psi, 
    #                             truncation_cutoff=args.truncation_cutoff)
    #         deltas = 30 * (w_rg - w_u) / (w_rg - w_u).norm(p=2)
    #         # w_u_adj = w_u + deltas
    #         w_rg = w_avg + deltas
    #         # append all the w_rg to the list but keeping the dim of the list to be the same as w_u
    #         w_rgs += w_rg

    # total_points = len(w_u) if len(w_u) > args.n_fim_samples else args.n_fim_samples
    # for w_u_img in tqdm(range(len(w_u))):
    for w in tqdm(w_u, desc="Sampling w_rg close to the forgetting set ..."):
        total_tgts = 2
        with torch.no_grad():
            z_rg = torch.randn(total_tgts, w_dim, device=device)
            customized_params = torch.cat([args.conditioning_params]*total_tgts, dim=0)
            
            w_rg = net.mapping(z_rg, customized_params, 
                                truncation_psi=args.truncation_psi, 
                                truncation_cutoff=args.truncation_cutoff)
            if random_sampling:
                deltas = (d+5) * (w_rg - w) / (w_rg - w).norm(p=2)
            else:
                deltas = 40 * (w_rg - w) / (w_rg - w).norm(p=2)
            # w_u_adj = w_u + deltas
            w_rg = w_avg + deltas
            # append all the w_rg to the list but keeping the dim of the list to be the same as w_u
            w_rgs.extend(w_rg.squeeze(0))
        
    
    w_rgs = torch.stack(w_rgs, dim=0)
    total_W = w_rgs.shape[0]
    # breakpoint()
    
    dataset = torch.utils.data.TensorDataset(w_rgs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    net.train()
    g_source.eval()
    
    # using dataloader instead
    for iter in range(1):
        for w_rg in tqdm(dataloader, desc="Calc Fisher Information Matrix ..."):        
            w_rg = w_rg[0].to(device)
            
            optimizer.zero_grad()
            customized_params = torch.cat([args.conditioning_params]*w_rg.shape[0], dim=0)
            
            img_u = net.synthesis(w_rg, customized_params)["image"]
            img_target = g_source.synthesis(w_rg, customized_params)["image"]
            
            feat_u = net.get_planes(w_rg) # F_u: triplane features for the unlearning image
            feat_target = g_source.get_planes(w_rg)
            
            loss = torch.tensor(0.0).to(device)
            
            mse_loss = mse_coff * F.mse_loss(feat_u, feat_target).mean()
            # loss = lpips_fn(img_u, img_target).mean()
            id_loss = id_coff * id_fn(img_u, img_target).mean()
            # loss -= id_loss
            loss -= mse_loss
            # (-loss).backward()
            loss.backward()
            
            
            for name, param in net.named_parameters():
                if "backbone.synthesis" in name:
                    if torch.isnan(param.grad.data).any():
                        print("NAN detected")
                    fisher_dict[name] += ((param.grad.data) ** 2) / total_W
            
            optimizer.step()
        #  print the fisher dict with non zero values
        # for name, param in fisher_dict.items():
        #     if param.sum() != 0:
        #         print(name, param.sum())
    # import IPython; IPython.embed()
    # with open(os.path.join(args.exp_root_dir, 'fisher_dict_ascent_glob.pkl'), 'wb') as f:
    with open(saved_path, 'wb') as f:
        pickle.dump(fisher_dict, f)
        print(colored(f"Fisher Information Matrix saved at {saved_path}", "green"))
    del fisher_dict
    del net, optimizer