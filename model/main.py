import sys
import os

from inference import run_infer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from src.models.unet_condition import UNet2DConditionModel as UNet2DConditionModel_ref
from src.models.unet_denoising import UNet2DConditionModel as UNet2DConditionModel_denoising

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="run_config.yaml")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)


# Data type and device
dtype = torch.float16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load VAE (for encoding), Cloth processing model and Clothed new image generation model
vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device, dtype=dtype)
cloth_unet = UNet2DConditionModel_ref.from_pretrained_checkpoint(cfg.cloth_config_path, cfg.cloth_weights_path).to(device, dtype=dtype)
denoising_unet = UNet2DConditionModel_denoising.from_pretrained_checkpoint(cfg.denoising_config_path, cfg.denoising_weights_path).to(device, dtype=dtype)

# Configure the noise scheduler for the diffusion process
sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)

# Enable zero terminal SNR for better image quality
sched_kwargs.update(rescale_betas_zero_snr=True, timestep_spacing="trailing", prediction_type="v_prediction")
val_noise_scheduler = DDIMScheduler(**sched_kwargs)

# Run inference with all loaded models and configurations
run_infer(
    vae, 
    cloth_unet, 
    denoising_unet, 
    val_noise_scheduler, 
    cfg.dataset.image_width, 
    cfg.dataset.image_height, 
    dtype, 
    cfg.seed, 
    cfg.dataset.infer_type, 
    cfg.dataset.save_path, 
    device=device,
    cloth_path=cfg.dataset.cloth_path, 
    person_path=cfg.dataset.person_path
)