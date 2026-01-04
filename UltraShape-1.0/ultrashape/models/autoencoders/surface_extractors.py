# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure

# cubvh is optional - fallback to skimage marching cubes if not available
try:
    import cubvh
    HAS_CUBVH = True
except ImportError:
    HAS_CUBVH = False
    print("[UltraShape] cubvh not found, using skimage marching cubes (slower but works)")



class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input 
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single 
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


def get_sparse_valid_voxels(grid_logit: torch.Tensor):

    if not isinstance(grid_logit, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if grid_logit.dim() != 3 or grid_logit.shape[0] != grid_logit.shape[1] or grid_logit.shape[0] != grid_logit.shape[2]:
        raise ValueError("Input tensor must have shape (N, N, N)")

    N = grid_logit.shape[0]
    device = grid_logit.device

    nan_mask = torch.isnan(grid_logit)

    invalid_voxel_mask = (
        nan_mask[:-1, :-1, :-1] |
        nan_mask[1:, :-1, :-1]  |
        nan_mask[:-1, 1:, :-1]  |
        nan_mask[:-1, :-1, 1:]  |
        nan_mask[1:, 1:, :-1]   |
        nan_mask[1:, :-1, 1:]   |
        nan_mask[:-1, 1:, 1:]   |
        nan_mask[1:, 1:, 1:]
    )
    
    valid_voxel_mask = ~invalid_voxel_mask

    sparse_coords = valid_voxel_mask.nonzero(as_tuple=False)

    if sparse_coords.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device), torch.empty((0, 8), dtype=grid_logit.dtype, device=device)

    x, y, z = sparse_coords[:, 0], sparse_coords[:, 1], sparse_coords[:, 2]

    sparse_vertex_logits = torch.stack([
        grid_logit[x,     y,     z],     # v0
        grid_logit[x + 1, y,     z],     # v1
        grid_logit[x + 1, y + 1, z],     # v2
        grid_logit[x,     y + 1, z],     # v3
        grid_logit[x,     y,     z + 1], # v4
        grid_logit[x + 1, y,     z + 1], # v5
        grid_logit[x + 1, y + 1, z + 1], # v6
        grid_logit[x,     y + 1, z + 1]  # v7
    ], dim=1)

    return sparse_coords, sparse_vertex_logits


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding 
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """

        grid_logit = grid_logit.detach().float()

        if HAS_CUBVH:
            # Use CUDA-accelerated sparse marching cubes
            sparse_coords, sparse_logits = get_sparse_valid_voxels(grid_logit)
            vertices, faces = cubvh.sparse_marching_cubes(sparse_coords, sparse_logits, mc_level)
            vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        else:
            # Fallback to skimage marching cubes (CPU, slower)
            grid_np = grid_logit.cpu().numpy()
            vertices, faces, normals, _ = measure.marching_cubes(
                grid_np, mc_level, method="lewiner", mask=(~np.isnan(grid_np)))

        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        """
        Extract surface mesh using Differentiable Marching Cubes (DMC) algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, centered and converted to numpy.
                - faces (np.ndarray): Extracted mesh faces (triangles), with reversed vertex order.
        
        Raises:
            ImportError: If the 'diso' package is not installed.
        """
        device = grid_logit.device
        if not hasattr(self, 'dmc'):
            try:
                from diso import DiffDMC
                self.dmc = DiffDMC(dtype=torch.float32).to(device)
            except:
                raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        verts = center_vertices(verts)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
}
