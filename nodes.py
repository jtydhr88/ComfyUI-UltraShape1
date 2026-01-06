import os
import sys
import uuid
import time
import numpy as np
import torch
from typing import Optional, Dict, Any
from PIL import Image

import comfy.model_management as model_management

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ULTRASHAPE_DIR = os.path.join(CURRENT_DIR, "UltraShape-1.0")

# Add ultrashape to path
if ULTRASHAPE_DIR not in sys.path:
    sys.path.insert(0, ULTRASHAPE_DIR)

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
    COMFY_INPUT_DIR = folder_paths.get_input_directory()
except ImportError:
    COMFY_MODELS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "models")
    COMFY_OUTPUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "output")
    COMFY_INPUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "input")

ULTRASHAPE_MODELS_DIR = os.path.join(COMFY_MODELS_DIR, "UltraShape")


def get_timestamp():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


# ============================================================================
# Wrapper Classes
# ============================================================================

class UltraShapeModelWrapper:
    """Wrapper for UltraShape pipeline components"""
    def __init__(self, pipeline, config, token_num, voxel_res, device, dtype):
        self.pipeline = pipeline
        self.config = config
        self.token_num = token_num
        self.voxel_res = voxel_res
        self.device = device
        self.dtype = dtype


class UltraShapeMeshWrapper:
    """Wrapper for mesh data"""
    def __init__(self, surface, voxel_idx, mesh_path, normalize_scale):
        self.surface = surface  # (B, N, 6+1) tensor
        self.voxel_idx = voxel_idx  # (B, K, 3) voxel indices
        self.mesh_path = mesh_path
        self.normalize_scale = normalize_scale


class UltraShapeOutputWrapper:
    """Wrapper for output mesh"""
    def __init__(self, mesh, latents=None):
        self.mesh = mesh  # trimesh.Trimesh object
        self.latents = latents


# ============================================================================
# Node 1: UltraShape Load Model
# ============================================================================

class UltraShapeLoadModel:
    """Load UltraShape refinement model (VAE + DiT + Conditioner)"""

    @classmethod
    def INPUT_TYPES(s):
        # Scan for checkpoint files
        ckpt_files = ["(select file)"]
        if os.path.exists(ULTRASHAPE_MODELS_DIR):
            for f in os.listdir(ULTRASHAPE_MODELS_DIR):
                if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensors"):
                    ckpt_files.append(f)

        # Scan for config files
        config_files = ["infer_dit_refine.yaml"]
        config_dir = os.path.join(ULTRASHAPE_DIR, "configs")
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith(".yaml") and f not in config_files:
                    config_files.append(f)

        return {
            "required": {
                "checkpoint": (ckpt_files, {"default": "(select file)"}),
            },
            "optional": {
                "config": (config_files, {"default": "infer_dit_refine.yaml"}),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "UltraShape/Loaders"

    def load_model(self, checkpoint, config="infer_dit_refine.yaml", dtype="bfloat16"):
        from omegaconf import OmegaConf
        from ultrashape.pipelines import UltraShapePipeline
        from ultrashape.utils.misc import instantiate_from_config

        if checkpoint == "(select file)":
            raise ValueError("Please select a checkpoint file. Place .pt files in ComfyUI/models/UltraShape/")

        # Determine paths
        ckpt_path = os.path.join(ULTRASHAPE_MODELS_DIR, checkpoint)
        config_path = os.path.join(ULTRASHAPE_DIR, "configs", config)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        device = model_management.get_torch_device()
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        print(f"[UltraShape] Loading config from {config_path}...")
        cfg = OmegaConf.load(config_path)

        print("[UltraShape] Instantiating VAE...")
        vae = instantiate_from_config(cfg.model.params.vae_config)

        print("[UltraShape] Instantiating DiT...")
        dit = instantiate_from_config(cfg.model.params.dit_cfg)

        print("[UltraShape] Instantiating Conditioner...")
        conditioner = instantiate_from_config(cfg.model.params.conditioner_config)

        print("[UltraShape] Instantiating Scheduler & Processor...")
        scheduler = instantiate_from_config(cfg.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(cfg.model.params.image_processor_cfg)

        print(f"[UltraShape] Loading weights from {ckpt_path}...")
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        vae.load_state_dict(weights['vae'], strict=True)
        dit.load_state_dict(weights['dit'], strict=True)
        conditioner.load_state_dict(weights['conditioner'], strict=True)

        vae.eval().to(device, dtype=torch_dtype)
        dit.eval().to(device, dtype=torch_dtype)
        conditioner.eval().to(device, dtype=torch_dtype)

        # Enable flash decoder if available
        if hasattr(vae, 'enable_flashvdm_decoder'):
            vae.enable_flashvdm_decoder()
            print("[UltraShape] FlashVDM decoder enabled")

        pipeline = UltraShapePipeline(
            vae=vae,
            model=dit,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor
        )

        # Extract config params
        token_num = cfg.model.params.vae_config.params.num_latents
        voxel_res = cfg.model.params.vae_config.params.voxel_query_res

        wrapper = UltraShapeModelWrapper(
            pipeline=pipeline,
            config=cfg,
            token_num=token_num,
            voxel_res=voxel_res,
            device=device,
            dtype=torch_dtype
        )

        print(f"[UltraShape] Model loaded: token_num={token_num}, voxel_res={voxel_res}")
        return (wrapper,)


# ============================================================================
# Node 2: UltraShape Load Coarse Mesh
# ============================================================================

class UltraShapeLoadCoarseMesh:
    """Load and preprocess coarse mesh for refinement"""

    @classmethod
    def INPUT_TYPES(s):
        # Scan for mesh files in input directory
        mesh_files = ["(select file)"]
        mesh_extensions = [".glb", ".gltf", ".obj", ".ply", ".stl"]

        # Check input directory
        if os.path.exists(COMFY_INPUT_DIR):
            for f in os.listdir(COMFY_INPUT_DIR):
                if any(f.lower().endswith(ext) for ext in mesh_extensions):
                    mesh_files.append(f)

        # Check ultrashape input directory
        ultrashape_input = os.path.join(ULTRASHAPE_DIR, "inputs", "coarse_mesh")
        if os.path.exists(ultrashape_input):
            for f in os.listdir(ultrashape_input):
                if any(f.lower().endswith(ext) for ext in mesh_extensions):
                    mesh_files.append(f"ultrashape/{f}")

        return {
            "required": {
                "model": ("ULTRASHAPE_MODEL",),
                "mesh_file": (mesh_files, {"default": "(select file)"}),
            },
            "optional": {
                "normalize_scale": ("FLOAT", {"default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01}),
                "num_sharp_points": ("INT", {"default": 204800, "min": 10000, "max": 500000, "step": 10000}),
                "num_uniform_points": ("INT", {"default": 204800, "min": 10000, "max": 500000, "step": 10000}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MESH",)
    RETURN_NAMES = ("coarse_mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "UltraShape/Loaders"

    def load_mesh(self, model: UltraShapeModelWrapper, mesh_file,
                  normalize_scale=0.99, num_sharp_points=204800, num_uniform_points=204800):
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils import voxelize_from_point

        if mesh_file == "(select file)":
            raise ValueError("Please select a mesh file")

        # Determine actual path
        if mesh_file.startswith("ultrashape/"):
            mesh_path = os.path.join(ULTRASHAPE_DIR, "inputs", "coarse_mesh", mesh_file[11:])
        else:
            mesh_path = os.path.join(COMFY_INPUT_DIR, mesh_file)

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        print(f"[UltraShape] Loading coarse mesh: {mesh_path}")

        # Initialize surface loader
        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=num_sharp_points,
            num_uniform_points=num_uniform_points,
        )

        # Load and process surface
        surface = loader(mesh_path, normalize_scale=normalize_scale)
        surface = surface.to(model.device, dtype=model.dtype)

        # Extract point cloud (first 3 channels)
        pc = surface[:, :, :3]  # [B, N, 3]

        # Voxelize
        _, voxel_idx = voxelize_from_point(pc, model.token_num, resolution=model.voxel_res)

        wrapper = UltraShapeMeshWrapper(
            surface=surface,
            voxel_idx=voxel_idx,
            mesh_path=mesh_path,
            normalize_scale=normalize_scale
        )

        print(f"[UltraShape] Mesh loaded: surface={surface.shape}, voxel_idx={voxel_idx.shape}")
        return (wrapper,)


# ============================================================================
# Node 3: UltraShape Load Coarse Mesh (Path)
# ============================================================================

class UltraShapeLoadCoarseMeshPath:
    """Load coarse mesh from file path string"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("ULTRASHAPE_MODEL",),
                "mesh_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "normalize_scale": ("FLOAT", {"default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01}),
                "num_sharp_points": ("INT", {"default": 204800, "min": 10000, "max": 500000, "step": 10000}),
                "num_uniform_points": ("INT", {"default": 204800, "min": 10000, "max": 500000, "step": 10000}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MESH",)
    RETURN_NAMES = ("coarse_mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "UltraShape/Loaders"

    def load_mesh(self, model: UltraShapeModelWrapper, mesh_path: str,
                  normalize_scale=0.99, num_sharp_points=204800, num_uniform_points=204800):
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils import voxelize_from_point

        if not mesh_path or not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        print(f"[UltraShape] Loading coarse mesh: {mesh_path}")

        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=num_sharp_points,
            num_uniform_points=num_uniform_points,
        )

        surface = loader(mesh_path, normalize_scale=normalize_scale)
        surface = surface.to(model.device, dtype=model.dtype)

        pc = surface[:, :, :3]
        _, voxel_idx = voxelize_from_point(pc, model.token_num, resolution=model.voxel_res)

        wrapper = UltraShapeMeshWrapper(
            surface=surface,
            voxel_idx=voxel_idx,
            mesh_path=mesh_path,
            normalize_scale=normalize_scale
        )

        print(f"[UltraShape] Mesh loaded: surface={surface.shape}, voxel_idx={voxel_idx.shape}")
        return (wrapper,)


# ============================================================================
# Node 4: UltraShape Refine
# ============================================================================

class UltraShapeRefine:
    """Refine coarse mesh using image-guided diffusion"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("ULTRASHAPE_MODEL",),
                "coarse_mesh": ("ULTRASHAPE_MESH",),
                "image": ("IMAGE",),
            },
            "optional": {
                "steps": ("INT", {"default": 50, "min": 10, "max": 200, "step": 5}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "octree_resolution": ("INT", {"default": 384, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Mesh resolution. Higher=better quality but more VRAM. 384=~8GB, 512=~16GB, 1024=~48GB+"}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.01}),
                "box_v": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7fffffff}),
                "remove_bg": ("BOOLEAN", {"default": False}),
                "num_chunks": ("INT", {"default": 20000, "min": 1000, "max": 50000, "step": 1000,
                    "tooltip": "Query batch size for volume decoding. Lower=less VRAM but slower. Default 10000 is optimal balance."}),
                "z_slice_size": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32,
                    "tooltip": "Z-axis slice size for dilation ops. Lower=less VRAM but slower. 64 is balanced, 32 for tight VRAM, 128+ for fast high-VRAM GPUs."}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_OUTPUT",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "refine"
    CATEGORY = "UltraShape"

    def refine(self, model: UltraShapeModelWrapper, coarse_mesh: UltraShapeMeshWrapper,
               image, steps=50, guidance_scale=5.0, octree_resolution=1024,
               mc_level=0.0, box_v=1.0, seed=42, remove_bg=False, num_chunks=10000, z_slice_size=64):
        import comfy.utils

        # Convert ComfyUI image to PIL
        # ComfyUI image: (B, H, W, C) tensor, values 0-1
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Remove background if requested
        if remove_bg:
            try:
                from ultrashape.rembg import BackgroundRemover
                rembg = BackgroundRemover()
                pil_image = rembg(pil_image)
                print("[UltraShape] Background removed")
            except Exception as e:
                print(f"[UltraShape] Warning: Background removal failed: {e}")

        # Ensure RGBA mode for transparency
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        print(f"[UltraShape] Refining mesh: steps={steps}, guidance={guidance_scale}, octree_res={octree_resolution}, num_chunks={num_chunks}, z_slice_size={z_slice_size}")

        # Setup generator for reproducibility
        generator = torch.Generator(device=model.device).manual_seed(seed)

        # Progress bar
        pbar = comfy.utils.ProgressBar(steps)
        step_count = [0]

        def callback(step_idx, t, outputs):
            step_count[0] += 1
            pbar.update_absolute(step_count[0], steps)

        # Run diffusion refinement
        with torch.autocast(device_type="cuda", dtype=model.dtype):
            mesh, latents = model.pipeline(
                image=pil_image,
                voxel_cond=coarse_mesh.voxel_idx,
                generator=generator,
                box_v=box_v,
                mc_level=mc_level,
                octree_resolution=octree_resolution,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_chunks=num_chunks,
                z_slice_size=z_slice_size,
                callback=callback,
                callback_steps=1,
            )

        # mesh is a list, get first element
        output_mesh = mesh[0] if isinstance(mesh, list) else mesh

        wrapper = UltraShapeOutputWrapper(
            mesh=output_mesh,
            latents=latents
        )

        print(f"[UltraShape] Refinement complete: vertices={len(output_mesh.vertices)}, faces={len(output_mesh.faces)}")
        return (wrapper,)


# ============================================================================
# Node 5: UltraShape Save GLB
# ============================================================================

class UltraShapeSaveGLB:
    """Save refined mesh as GLB file"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "refined_mesh": ("ULTRASHAPE_OUTPUT",),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "ultrashape_output"}),
                "filename_prefix": ("STRING", {"default": "refined"}),
                "file_format": (["glb", "obj", "ply", "stl"], {"default": "glb"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "UltraShape"
    OUTPUT_NODE = True

    def save(self, refined_mesh: UltraShapeOutputWrapper, output_dir="ultrashape_output",
             filename_prefix="refined", file_format="glb"):
        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)

        ts = get_timestamp()
        uid = str(uuid.uuid4())[:8]
        filename = f"{filename_prefix}_{ts}_{uid}.{file_format}"
        save_path = os.path.join(out_dir, filename)

        # Export mesh
        refined_mesh.mesh.export(save_path)

        print(f"[UltraShape] Saved: {save_path}")

        # Return relative path
        rel_path = os.path.relpath(save_path, COMFY_OUTPUT_DIR)
        return (rel_path,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "UltraShapeLoadModel": UltraShapeLoadModel,
    "UltraShapeLoadCoarseMesh": UltraShapeLoadCoarseMesh,
    "UltraShapeLoadCoarseMeshPath": UltraShapeLoadCoarseMeshPath,
    "UltraShapeRefine": UltraShapeRefine,
    "UltraShapeSaveGLB": UltraShapeSaveGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraShapeLoadModel": "UltraShape Load Model",
    "UltraShapeLoadCoarseMesh": "UltraShape Load Coarse Mesh",
    "UltraShapeLoadCoarseMeshPath": "UltraShape Load Coarse Mesh (Path)",
    "UltraShapeRefine": "UltraShape Refine",
    "UltraShapeSaveGLB": "UltraShape Save GLB/OBJ",
}
