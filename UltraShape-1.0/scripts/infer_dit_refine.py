import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# project_root = '[your_project_root_path]'  # Replace with your project root path
# sys.path.insert(0, project_root)

from ultrashape.rembg import BackgroundRemover
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
from ultrashape.utils import voxelize_from_point
from ultrashape.pipelines import UltraShapePipeline 

def load_models(config_path, ckpt_path, device='cuda'):

    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    
    print("Instantiating VAE...")
    vae = instantiate_from_config(config.model.params.vae_config)
    
    print("Instantiating DiT...")
    dit = instantiate_from_config(config.model.params.dit_cfg)
    
    print("Instantiating Conditioner...")
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    
    print("Instantiating Scheduler & Processor...")
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
    
    print(f"Loading weights from {ckpt_path}...")
    weights = torch.load(ckpt_path, map_location='cpu')
    
    vae.load_state_dict(weights['vae'], strict=True)
    dit.load_state_dict(weights['dit'], strict=True)
    conditioner.load_state_dict(weights['conditioner'], strict=True)
    
    vae.eval().to(device)
    dit.eval().to(device)
    conditioner.eval().to(device)
    
    if hasattr(vae, 'enable_flashvdm_decoder'):
        vae.enable_flashvdm_decoder()

    components = {
        "vae": vae,
        "dit": dit,
        "conditioner": conditioner,
        "scheduler": scheduler,
        "image_processor": image_processor,
    }
    
    return components, config

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    components, config = load_models(args.config, args.ckpt, device)
    
    pipeline = UltraShapePipeline(
        vae=components['vae'],
        model=components['dit'],
        scheduler=components['scheduler'],
        conditioner=components['conditioner'],
        image_processor=components['image_processor']
    )

    token_num = config.model.params.vae_config.params.num_latents
    voxel_res = config.model.params.vae_config.params.voxel_query_res
    
    print(f"Initializing Surface Loader (Token Num: {token_num})...")
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )

    print(f"Processing inputs: {args.image} & {args.mesh}")
    image = Image.open(args.image)
    
    if args.remove_bg or image.mode != 'RGBA':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    surface = loader(args.mesh, normalize_scale=args.scale).to(device, dtype=torch.float16)
    pc = surface[:, :, :3] # [B, N, 3]
    
    # Voxelize
    _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)
    
    print("Running diffusion process...")
    generator = torch.Generator(device).manual_seed(args.seed)
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mesh, _ = pipeline(
            image=image,
            voxel_cond=voxel_idx,
            generator=generator,
            box_v=1.0,
            mc_level=0.0,
            octree_resolution=args.octree_res,
            num_inference_steps=args.steps,
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_refined.glb")
    
    mesh = mesh[0]
    mesh.export(save_path)
    print(f"Successfully saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraShape Inference Script")
    
    parser.add_argument("--config", type=str, default="configs/infer_dit2.yaml", help="Path to inference config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to split checkpoint (.pt)")
    
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--mesh", type=str, required=True, help="Input coarse mesh (.glb/.obj)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--scale", type=float, default=0.99, help="Mesh normalization scale")
    parser.add_argument("--octree_res", type=int, default=1024, help="Marching Cubes resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--remove_bg", action="store_true", help="Force remove background")

    args = parser.parse_args()
    
    run_inference(args)
