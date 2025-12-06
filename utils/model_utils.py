"""
Model loading and setup utilities.
"""
import torch
import copy
import legacy
import dnnlib
from training.triplane import TriPlaneGenerator
from torch_utils.misc import copy_params_and_buffers


def load_pretrained_generator(pretrained_ckpt, device):
    """
    Load pretrained generator from checkpoint.
    
    Args:
        pretrained_ckpt: Path to pretrained checkpoint
        device: Device to load model on
        
    Returns:
        Loaded generator model
    """
    with dnnlib.util.open_url(pretrained_ckpt) as f:
        g_source = legacy.load_network_pkl(f)["G_ema"].to(device)
    return g_source


def setup_generators(g_source, resume_checkpoint=None, device="cuda"):
    """
    Setup generator, source generator, and surrogate generator.
    
    Args:
        g_source: Source generator model
        resume_checkpoint: Optional path to resume checkpoint
        device: Device to run on
        
    Returns:
        Tuple of (generator, g_source, g_surrogate)
    """
    generator = TriPlaneGenerator(*g_source.init_args, **g_source.init_kwargs).requires_grad_(False).to(device)
    copy_params_and_buffers(g_source, generator, require_all=True)
    generator.neural_rendering_resolution = g_source.neural_rendering_resolution
    generator.rendering_kwargs = g_source.rendering_kwargs
    generator.load_state_dict(g_source.state_dict(), strict=False)
    
    g_source = copy.deepcopy(generator)
    g_source.eval()
    g_surrogate = copy.deepcopy(generator)  # Surrogate model for the trigger model
    
    if resume_checkpoint:
        with dnnlib.util.open_url(resume_checkpoint) as f:
            g_resume = legacy.load_network_pkl(f)["G_ema"].to(device)
        generator.load_state_dict(g_resume.state_dict(), strict=False)
        del g_resume
    
    generator.train()
    
    # Set requires_grad flags
    for name, param in g_surrogate.named_parameters():
        if "backbone.synthesis" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    for name, param in g_source.named_parameters():
        param.requires_grad = False

    for name, param in generator.named_parameters():
        if "backbone.synthesis" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return generator, g_source, g_surrogate


def load_encoder(dataset, device):
    """
    Load the appropriate encoder for the given dataset.
    
    Args:
        dataset: Dataset name ("ffhq" or "afhq")
        device: Device to load the encoder on
        
    Returns:
        Encoder model loaded on the specified device
    """
    from goae import GOAEncoder
    from goae.swin_config import get_config
    
    swin_config = get_config()
    stage_list = [10000, 20000, 30000]
    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list)
    
    if dataset == "ffhq":
        encoder_ckpt = "encoder_FFHQ.pt"
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    elif dataset == "afhq":
        encoder_ckpt = "encoder_AFHQ.pt"
        state_dict = torch.load(encoder_ckpt, map_location=device)
        
        # Rename keys: replace "swin_base" → "swin_model"
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("swin_base", "swin_model")
            new_state_dict[new_k] = v
        encoder.load_state_dict(new_state_dict, strict=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return encoder.to(device)

