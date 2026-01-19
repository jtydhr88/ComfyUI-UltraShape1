"""
Wrapper classes for UltraShape data types.

These are in a separate module so they can be imported by both
the main ComfyUI process and the isolated worker subprocess.
"""


class UltraShapeModelWrapper:
    """Wrapper for UltraShape pipeline components.

    In normal mode, holds the loaded pipeline.
    In disk_offload mode, holds paths for lazy loading.
    """
    def __init__(self, pipeline, config, token_num, voxel_res, device, dtype,
                 disk_offload=False, ckpt_path=None, config_path=None):
        self.pipeline = pipeline
        self.config = config
        self.token_num = token_num
        self.voxel_res = voxel_res
        self.device = device
        self.dtype = dtype
        # Disk offload mode - store paths for lazy loading
        self.disk_offload = disk_offload
        self.ckpt_path = ckpt_path
        self.config_path = config_path


class UltraShapeMeshWrapper:
    """Wrapper for mesh data"""
    def __init__(self, surface, voxel_idx, mesh_path, normalize_scale):
        self.surface = surface  # (B, N, 6+1) tensor
        self.voxel_idx = voxel_idx  # (B, K, 3) voxel indices
        self.mesh_path = mesh_path
        self.normalize_scale = normalize_scale
