"""UltraShape model and mesh loading nodes."""

import os
from comfy_env import isolated

from common import (
    ULTRASHAPE_MODELS_DIR, CONFIG_DIR,
    COMFY_OUTPUT_DIR, ensure_ultrashape_checkpoint
)
from wrappers import UltraShapeModelWrapper, UltraShapeMeshWrapper


@isolated(env="ultrashape1", import_paths=["."])
class UltraShapeLoadModel:
    """Load UltraShape refinement model (VAE + DiT + Conditioner)"""

    @classmethod
    def INPUT_TYPES(s):
        # Scan for checkpoint files
        ckpt_files = []
        if os.path.exists(ULTRASHAPE_MODELS_DIR):
            for f in os.listdir(ULTRASHAPE_MODELS_DIR):
                if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensors"):
                    ckpt_files.append(f)

        # Auto-download if no checkpoints found
        if not ckpt_files:
            ensure_ultrashape_checkpoint()
            # Re-scan after download
            if os.path.exists(ULTRASHAPE_MODELS_DIR):
                for f in os.listdir(ULTRASHAPE_MODELS_DIR):
                    if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensors"):
                        ckpt_files.append(f)

        # Add placeholder if still empty
        if not ckpt_files:
            ckpt_files = ["(select file)"]

        # Scan for config files
        config_files = ["infer_dit_refine.yaml"]
        if os.path.exists(CONFIG_DIR):
            for f in os.listdir(CONFIG_DIR):
                if f.endswith(".yaml") and f not in config_files:
                    config_files.append(f)

        # Set default to first real checkpoint if available
        default_ckpt = ckpt_files[0] if ckpt_files and ckpt_files[0] != "(select file)" else "(select file)"

        return {
            "required": {
                "checkpoint": (ckpt_files, {"default": default_ckpt}),
            },
            "optional": {
                "config": (config_files, {"default": "infer_dit_refine.yaml"}),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "bfloat16"}),
                "attention_backend": (["sdpa", "sage_attn", "flash_attn"], {"default": "sdpa",
                    "tooltip": "Attention backend: sdpa (default, always works), sage_attn (faster, needs sageattention), flash_attn (needs flash-attn)"}),
                "low_vram": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable CPU offloading to reduce VRAM usage (slower but uses less memory)"}),
                "disk_offload": ("BOOLEAN", {"default": False,
                    "tooltip": "Extreme low VRAM mode (<6GB). Loads one model at a time from disk. Much slower but minimal VRAM."}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "UltraShape/Loaders"

    def load_model(self, checkpoint, config="infer_dit_refine.yaml", dtype="bfloat16",
                   attention_backend="sdpa", low_vram=False, disk_offload=False):
        # Set attention backend environment variable BEFORE importing ultrashape
        if attention_backend == "sage_attn":
            os.environ["USE_SAGEATTN"] = "1"
            print("[UltraShape] Using SageAttention backend")
        else:
            os.environ.pop("USE_SAGEATTN", None)

        # Lazy imports inside isolated subprocess
        import torch
        import comfy.model_management as model_management
        from omegaconf import OmegaConf
        from wrappers import UltraShapeModelWrapper

        if checkpoint == "(select file)":
            raise ValueError("Please select a checkpoint file. Place .pt files in ComfyUI/models/UltraShape/")

        # Determine paths
        ckpt_path = os.path.join(ULTRASHAPE_MODELS_DIR, checkpoint)
        config_path = os.path.join(CONFIG_DIR, config)

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

        # Extract config params
        token_num = cfg.model.params.vae_config.params.num_latents
        voxel_res = cfg.model.params.vae_config.params.voxel_query_res

        # Disk offload mode: don't load models, just store paths for lazy loading
        if disk_offload:
            print("[UltraShape] Disk offload mode enabled - models will be loaded on demand")
            wrapper = UltraShapeModelWrapper(
                pipeline=None,
                config=cfg,
                token_num=token_num,
                voxel_res=voxel_res,
                device=device,
                dtype=torch_dtype,
                disk_offload=True,
                ckpt_path=ckpt_path,
                config_path=config_path
            )
            print(f"[UltraShape] Config loaded: token_num={token_num}, voxel_res={voxel_res}")
            return (wrapper,)

        # Normal mode: load all models to GPU
        from ultrashape.pipelines import UltraShapePipeline
        from ultrashape.utils.misc import instantiate_from_config

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

        # Enable CPU offloading for low VRAM mode
        if low_vram:
            pipeline.enable_model_cpu_offload()
            print("[UltraShape] Low VRAM mode enabled (CPU offloading)")

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


@isolated(env="ultrashape1", import_paths=["."])
class UltraShapeLoadCoarseMesh:
    """Load and preprocess coarse mesh for refinement.

    Supports relative paths:
    - 'input/3d/mesh.glb' -> ComfyUI/input/3d/mesh.glb
    - 'output/3d/mesh.glb' -> ComfyUI/output/3d/mesh.glb
    - '3d/mesh.glb' -> defaults to ComfyUI/input/3d/mesh.glb
    - Absolute paths are also supported
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("ULTRASHAPE_MODEL",),
                "mesh_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to coarse mesh (.glb/.obj/.ply/.stl). Supports: 'input/mesh.glb', 'output/mesh.glb', or just 'mesh.glb' (defaults to input/). Absolute paths also work."
                }),
            },
            "optional": {
                "normalize_scale": ("FLOAT", {"default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01}),
                "num_sharp_points": ("INT", {"default": 204800, "min": 10000, "max": 500000, "step": 10000}),
                "num_uniform_points": ("INT", {"default": 204800, "min": 10000, "max": 500000, "step": 10000}),
                "num_latents": ("INT", {"default": 0, "min": 0, "max": 131072, "step": 1024,
                    "tooltip": "Number of latent tokens. 0=use config default (usually 32768). Higher=more detail but more VRAM"}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MESH",)
    RETURN_NAMES = ("coarse_mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "UltraShape/Loaders"

    def _resolve_mesh_path(self, mesh_path: str) -> str:
        """
        Resolve mesh path with support for relative paths.
        Rules:
        - If path starts with 'input/' or 'output/' -> resolve relative to ComfyUI root
        - If path has no such prefix (e.g., '3d/mesh.glb') -> assume 'input/' prefix
        - If path is absolute and exists -> use as is
        """
        if not mesh_path or not mesh_path.strip():
            return None

        path = mesh_path.strip().replace("\\", "/")

        # If absolute path and exists, use directly
        if os.path.isabs(path):
            if os.path.exists(path):
                return path
            return None

        # Get ComfyUI root directory (parent of output dir)
        comfy_root = os.path.dirname(COMFY_OUTPUT_DIR)

        # Check if path starts with input/ or output/
        if path.startswith("input/") or path.startswith("output/"):
            resolved = os.path.join(comfy_root, path)
        else:
            # Default to input/ prefix
            resolved = os.path.join(comfy_root, "input", path)

        # Normalize path
        resolved = os.path.normpath(resolved)

        if os.path.exists(resolved):
            return resolved
        else:
            print(f"[UltraShape] Mesh path not found: {resolved}")
            return None

    def load_mesh(self, model, mesh_path: str,
                  normalize_scale=0.99, num_sharp_points=204800, num_uniform_points=204800, num_latents=0):
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils import voxelize_from_point
        from wrappers import UltraShapeMeshWrapper

        # Resolve the mesh path
        resolved_path = self._resolve_mesh_path(mesh_path)
        if not resolved_path:
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        print(f"[UltraShape] Loading coarse mesh: {resolved_path}")

        # Initialize surface loader
        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=num_sharp_points,
            num_uniform_points=num_uniform_points,
        )

        # Load and process surface
        surface = loader(resolved_path, normalize_scale=normalize_scale)
        surface = surface.to(model.device, dtype=model.dtype)

        # Extract point cloud (first 3 channels)
        pc = surface[:, :, :3]  # [B, N, 3]

        # Use custom num_latents if specified, otherwise use config default
        token_num = num_latents if num_latents > 0 else model.token_num

        # Voxelize
        _, voxel_idx = voxelize_from_point(pc, token_num, resolution=model.voxel_res)

        wrapper = UltraShapeMeshWrapper(
            surface=surface,
            voxel_idx=voxel_idx,
            mesh_path=resolved_path,
            normalize_scale=normalize_scale
        )

        print(f"[UltraShape] Mesh loaded: surface={surface.shape}, voxel_idx={voxel_idx.shape}")
        return (wrapper,)
