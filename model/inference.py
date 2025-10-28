import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from src.models.unet_condition import UNet2DConditionModel as UNet2DConditionModel_ref
from src.models.unet_denoising import UNet2DConditionModel as UNet2DConditionModel_denoising
from src.pipeline import Pipeline
from src.utils.util import import_filename


def run_infer(
    vae,
    cloth_unet,
    denoising_unet,
    scheduler,
    width,
    height,
    dtype,
    seed,
    infer_type,
    save_path,
    device,
    cloth_path=None,
    person_path=None,
):
    """
    Run inference for virtual try-on using diffusion models.
    
    Args:
        vae: Variational Autoencoder for image encoding/decoding
        cloth_unet: UNet model for reference image processing
        denoising_unet: UNet model for denoising during diffusion
        scheduler: Noise scheduler for the diffusion process
        width: Output image width
        height: Output image height
        dtype: Data type for model inference (fp16 or fp32)
        seed: Random seed for reproducibility
        infer_type: Type of inference - "single" for one image pair, "VITON-HD" for dataset
        save_path: Directory path to save output images
        cloth_path: Path to cloth image (for single mode)
        person_path: Path to person image (for single mode)
    """

    # Initialize random generator with seed for reproducible results
    generator = torch.Generator().manual_seed(seed)

    pipe = Pipeline(vae=vae, cloth_unet=cloth_unet, denoising_unet=denoising_unet, scheduler=scheduler,).to(device)

    # Load Images.
    cloth_image_pil = Image.open(cloth_path).convert("RGB")
    person_image_pil = Image.open(person_path).convert("RGB")
    
    # Run inference on the single image pair
    image = pipe([cloth_image_pil], [person_image_pil],
        width, height,
        18,  # Number of inference steps
        2.5,  # Guidance scale for classifier-free guidance
        generator=generator, dtype=dtype
    ).images
    
    # Convert tensor output to PIL Image
    image = image[0, :].permute(1, 2, 0).cpu().numpy()
    res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    # Save the result
    save_path = os.path.join(save_path, infer_type) 
    os.makedirs(save_path, exist_ok=True)
    res_image_pil.save(os.path.join(save_path, "result.png"))
    