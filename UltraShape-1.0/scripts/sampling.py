import os
import trimesh
import numpy as np
from typing import List, Optional, Any, Tuple, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch3d.structures
import pytorch3d.ops
from scipy.stats import truncnorm
import json
import argparse
import cubvh

# import logging
# from tools.logger import init_log, set_all_log
# sys_logger = init_log("sampler", logging.DEBUG) 
# set_all_log(level=logging.DEBUG, path='./debug/logs')   
    
def load_mesh(mesh_path: str, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    if mesh_path.endswith(".npz"):
        mesh_np = np.load(mesh_path)
        vertices, faces = torch.tensor(mesh_np["vertices"], device=device), torch.tensor(mesh_np["faces"].astype('i8'), device=device)
    else:
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    if faces.shape[0] > 2 * 1e8:
        raise ValueError(f"too many faces {faces.shape}")
    return vertices, faces

def compute_mesh_features(vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = vertices.device
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_areas = torch.norm(face_normals, dim=1) * 0.5
    face_normals = face_normals / (face_areas.unsqueeze(1) * 2 + 1e-12)
    
    vertex_normals = torch.zeros_like(vertices)
    face_normals_weighted = face_normals * face_areas.unsqueeze(1)
    
    vertex_normals.scatter_add_(0, faces[:, 0:1].expand(-1, 3), face_normals_weighted)
    vertex_normals.scatter_add_(0, faces[:, 1:2].expand(-1, 3), face_normals_weighted)
    vertex_normals.scatter_add_(0, faces[:, 2:3].expand(-1, 3), face_normals_weighted)
    
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-12)
    
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)
    
    edges_unique, edges_inverse = torch.unique(torch.sort(edges, dim=1)[0], dim=0, return_inverse=True)
    edge_normals_diff = torch.norm(
        vertex_normals[edges[:, 0]] - vertex_normals[edges[:, 1]],
        dim=1
    )
    
    vertex_curvatures = torch.zeros(len(vertices), device=device)
    vertex_curvatures.scatter_add_(0, edges[:, 0], edge_normals_diff)
    vertex_curvatures.scatter_add_(0, edges[:, 1], edge_normals_diff)

    vertex_degrees = torch.zeros(len(vertices), device=device)
    vertex_degrees.scatter_add_(0, edges[:, 0], torch.ones_like(edge_normals_diff))
    vertex_degrees.scatter_add_(0, edges[:, 1], torch.ones_like(edge_normals_diff))
    
    vertex_curvatures = vertex_curvatures / (vertex_degrees + 1e-12)
    vertex_curvatures = (vertex_curvatures - vertex_curvatures.min()) / (
        vertex_curvatures.max() - vertex_curvatures.min() + 1e-12)
    
    return face_areas, vertex_curvatures

def sample_uniform_points(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    if random_seed is not None:
        torch.manual_seed(random_seed)
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces])
    
    points, normals = pytorch3d.ops.sample_points_from_meshes(
        mesh, num_samples=num_samples, return_normals=True)
    
    return points[0], normals[0]

def sample_surface_points(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    min_samples_per_face: int = 0,
    use_curvature: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Curvature-based surface sampling"""
    device = vertices.device
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Compute face areas and vertex curvatures
    face_areas, vertex_curvatures = compute_mesh_features(vertices, faces)
    
    # Compute average curvature of faces
    face_curvatures = torch.mean(vertex_curvatures[faces], dim=1)
    sampling_weights = face_curvatures  # Use only curvature as weights
    # Calculate number of sample points per face
    num_faces = len(faces)
    
    # Chunk forward
    if min_samples_per_face > 0:
        base_samples = torch.full((num_faces,), min_samples_per_face, device=device)
        remaining_samples = num_samples - torch.sum(base_samples).item()
        
        if remaining_samples > 0:
            # Block sampling to avoid large mesh issues
            if num_faces > 2**24:
                chunk_size = 1000000  # Process 1 million faces at a time
                additional_counts = torch.zeros(num_faces, device=device)
                
                for start in range(0, num_faces, chunk_size):
                    end = min(start + chunk_size, num_faces)
                    chunk_weights = sampling_weights[start:end]
                    chunk_probs = chunk_weights / chunk_weights.sum()
                    
                    # Proportinally allocate remaining samples
                    chunk_samples = int(remaining_samples * (end - start) / num_faces)
                    samples = torch.multinomial(chunk_probs, chunk_samples, replacement=True)
                    chunk_counts = torch.bincount(samples, minlength=chunk_size)
                    additional_counts[start:end] += chunk_counts[:end-start]
                
                sample_counts = additional_counts + base_samples
            else:
                probs = sampling_weights / sampling_weights.sum()
                additional_samples = torch.multinomial(probs, remaining_samples, replacement=True)
                sample_counts = torch.bincount(additional_samples, minlength=num_faces) + base_samples
        else:
            sample_counts = base_samples
    else:
        if num_faces > 2**24:
            # Chunk sampling strategy
            sample_counts = torch.zeros(num_faces, device=device)
            chunk_size = 1000000  # Process 1 million faces at a time
            chunk_samples = num_samples // ((num_faces + chunk_size - 1) // chunk_size)
            
            for start in range(0, num_faces, chunk_size):
                end = min(start + chunk_size, num_faces)
                chunk_weights = sampling_weights[start:end]
                chunk_probs = chunk_weights / chunk_weights.sum()
                
                samples = torch.multinomial(chunk_probs, chunk_samples, replacement=True)
                chunk_counts = torch.bincount(samples, minlength=chunk_size)
                sample_counts[start:end] += chunk_counts[:end-start]
        else:
            probs = sampling_weights / sampling_weights.sum()
            samples = torch.multinomial(probs, num_samples, replacement=True)
            sample_counts = torch.bincount(samples, minlength=num_faces)
    
    # Generate barycentric coordinates for sampled points
    total_samples = sample_counts.sum().item()
    r1 = torch.sqrt(torch.rand(total_samples, device=device))
    r2 = torch.rand(total_samples, device=device)
    
    barycentric_coords = torch.stack([
        1 - r1,
        r1 * (1 - r2),
        r1 * r2
    ], dim=1)
    
    # Generate face indices
    face_indices = torch.repeat_interleave(
        torch.arange(num_faces, device=device),
        sample_counts
    )
    
    # Get vertices of corresponding faces
    face_vertices = vertices[faces[face_indices]]
    
    # Compute 3D coordinates of sampled points
    points = (barycentric_coords.unsqueeze(1) @ face_vertices).squeeze(1)
    
    # Compute normal vectors of sampled points
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-12)
    
    return points, face_indices, normals

def normalize_points_and_mesh(vertices: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize mesh and point cloud to unit cube"""
    device = vertices.device
    vmin = vertices.min(dim=0)[0]
    vmax = vertices.max(dim=0)[0]
    center = (vmax + vmin) / 2
    scale = (vmax - vmin).max()
    margin = 0.01
    scale = scale * (1 + 2 * margin)
    
    vertices_normalized = (vertices - center) / scale + 0.5
    points_normalized = (points - center) / scale + 0.5
    
    return vertices_normalized, points_normalized, center, scale

def add_gaussian_noise(uniform_surface_points: torch.Tensor, curvature_surface_points: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to point cloud"""
    # noise = torch.randn_like(points) * sigma
    # print("u_num:",uniform_surface_points.shape)
    # print("c_num:",curvature_surface_points.shape)

    idx1 = torch.randperm(uniform_surface_points.shape[0])
    idx2 = torch.randperm(curvature_surface_points.shape[0])
    uniform_surface_points = uniform_surface_points[idx1]
    curvature_surface_points = curvature_surface_points[idx2]

    a, b = -0.25, 0.25
    mu = 0

    # get near points (add offset on surface points)
    offset1 = torch.tensor(truncnorm.rvs((a - mu) / 0.005, (b - mu) / 0.005, loc=mu, scale=0.005, size=(len(uniform_surface_points), 3)), 
                         dtype=uniform_surface_points.dtype, device=uniform_surface_points.device)
    offset2 = torch.tensor(truncnorm.rvs((a - mu) / 0.05, (b - mu) / 0.05, loc=mu, scale=0.05,  size=(len(uniform_surface_points), 3)), 
                         dtype=uniform_surface_points.dtype, device=uniform_surface_points.device)
    uniform_near_points = torch.cat([
        uniform_surface_points + offset1,
        uniform_surface_points + offset2
    ], dim=0)

    # Generate multi-scale noise for curvature sample points
    unit_num = curvature_surface_points.shape[0] // 6
    scales = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04]
    
    curvature_near_points = []
    for i in range(6):
        start = i * unit_num
        end = (i + 1) * unit_num if i < 5 else curvature_surface_points.shape[0]
        noise = torch.randn((end - start, 3), dtype=curvature_surface_points.dtype, 
                          device=curvature_surface_points.device) * scales[i]
        curvature_near_points.append(curvature_surface_points[start:end] + noise)
    
    curvature_near_points = torch.cat(curvature_near_points, dim=0)

    return uniform_near_points, curvature_near_points

def compute_points_value_bvh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    points: torch.Tensor,
    use_sdf: bool = True,
    batch_size: int = 100_00000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SDF or occupancy values for sampled points"""
    device = vertices.device
    
    # Normalize mesh and point cloud
    vertices_norm, points_norm, center, scale = normalize_points_and_mesh(vertices, points)
    
    BVH = cubvh.cuBVH(vertices_norm, faces)
    distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='watertight')
    values = distances
    
    return values, points_norm, center, scale

def save_point_cloud(
    points: torch.Tensor,
    output_path: str,
    normals: Optional[torch.Tensor] = None,
    colors: Optional[torch.Tensor] = None
) -> None:
    """Save point cloud to file"""
    points_np = points.cpu().numpy()
    normals_np = normals.cpu().numpy() if normals is not None else None
    colors_np = None
    
    if colors is not None:
        colors_np = colors.cpu().numpy()
        if colors_np.max() <= 1.0:
            colors_np = (colors_np * 255).astype(np.uint8)
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.txt':
        data_list = [points_np]
        if normals_np is not None:
            data_list.append(normals_np)
        if colors_np is not None:
            data_list.append(colors_np)
            
        combined_data = np.hstack(data_list)
        np.savetxt(output_path, combined_data, fmt='%.6f')
        
    elif ext == '.ply':
        cloud = trimesh.PointCloud(points_np, colors=colors_np)
        if normals_np is not None:
            cloud.metadata['normals'] = normals_np
        cloud.export(output_path)
        
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please use .txt or .ply")

def sample_points_in_bbox(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    num_samples: int,
    device: str = "cuda"
) -> torch.Tensor:
    """Uniformly sample points within bounding box"""
    points = torch.rand(num_samples, 3, device=device)
    points = points * (bbox_max - bbox_min) + bbox_min
    return points

def process_single_mesh(
    mesh_name:str,
    mesh_path: str,
    output_dir: str,
    data_type:str = 'mesh',
    surface_uniform_samples: int = 100000,      # surface上均匀采样点数
    surface_curvature_samples: int = 200000,    # surface上曲率采样点数
    space_samples: int = 300000,               # 空间中采样点数
    noise_sigma: float = 0.01,
    device: str = "cuda"
) -> None:
    """Process a single mesh file
    Args:
        mesh_path: Input mesh path
        output_dir: Output directory
        surface_uniform_samples: Number of uniform sample points on surface
        surface_curvature_samples: Number of curvature-based sample points on surface
        space_samples: Number of sample points in space
        noise_sigma: Gaussian noise standard deviation
        device: Computation device
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if data_type == "mesh":
        vertices, faces = load_mesh(mesh_path, device)
    elif data_type == "sparse_voxel":
        pass
    vertices_normalized, _, center, scale = normalize_points_and_mesh(vertices, vertices)
    
    space_points = torch.rand(space_samples, 3, device=device)
    
    uniform_surface_points, uniform_surface_normals = sample_uniform_points(
        vertices=vertices_normalized,
        faces=faces,
        num_samples=surface_uniform_samples
    )
    
    curvature_surface_points, _, curvature_surface_normals = sample_surface_points(
        vertices=vertices_normalized,
        faces=faces,
        num_samples=surface_curvature_samples,
        use_curvature=True
    )
    
    clean_surface_points = torch.cat([uniform_surface_points, curvature_surface_points], dim=0)
    clean_surface_normals = torch.cat([uniform_surface_normals, curvature_surface_normals], dim=0)

    surface_uni_save_path = os.path.join(output_dir, f"{mesh_name}_uni_surface")
    save_point_cloud(
        points=uniform_surface_points,
        output_path=f"{surface_uni_save_path}.ply",
        normals=uniform_surface_normals
    )   

    surface_cur_save_path = os.path.join(output_dir, f"{mesh_name}_cur_surface")
    save_point_cloud(
        points=curvature_surface_points,
        output_path=f"{surface_cur_save_path}.ply",
        normals=curvature_surface_normals
    )
    
    uniform_near_points, curvature_near_points = add_gaussian_noise(uniform_surface_points = uniform_surface_points.clone(),
                            curvature_surface_points = curvature_surface_points.clone(), sigma=noise_sigma)

    space_sdf, _, _, _ = compute_points_value_bvh(
        vertices=vertices_normalized,
        faces=faces,
        points=space_points,
        use_sdf=True,
        batch_size=1000_00000
    )
    
    # clean_surface_sdf = torch.zeros(len(clean_surface_points), device=device)
    uniform_near_sdf, _, _, _ = compute_points_value_bvh(
        vertices=vertices_normalized,
        faces=faces,
        points=uniform_near_points,
        use_sdf=True,
        batch_size=1000_00000
    )
    
    curvature_near_sdf, _, _, _ = compute_points_value_bvh(
        vertices=vertices_normalized,
        faces=faces,
        points=curvature_near_points,
        use_sdf=True,
        batch_size=1000_00000
    )
    
    print("sdf:",uniform_near_sdf.shape, curvature_near_sdf.shape)
    
    base_save_path = os.path.join(output_dir, mesh_name)
    
    np.savez(f"{base_save_path}.npz",
             space_points=space_points.cpu().numpy(),
             space_sdf=space_sdf.cpu().numpy(),
             clean_surface_points=clean_surface_points.cpu().numpy(),
             clean_surface_normals=clean_surface_normals.cpu().numpy(),
             uniform_near_points=uniform_near_points.cpu().numpy(),
             curvature_near_points=curvature_near_points.cpu().numpy(),
             uniform_near_sdf=uniform_near_sdf.cpu().numpy(),
             curvature_near_sdf=curvature_near_sdf.cpu().numpy(),
             center=center.cpu().numpy(),
             scale=scale.cpu().numpy())

class MeshDataset(Dataset):
    def __init__(self, mesh_json: str):
        with open(mesh_json, "r") as f:
            self.mesh_paths = json.load(f)
        # print(len(self.mesh_paths))
            
    def __len__(self) -> int:
        return len(self.mesh_paths)
    def __getitem__(self, idx: int) -> dict:
        mesh_path = self.mesh_paths[idx]
        mesh_name = os.path.basename(mesh_path)[:-4]
        mesh =  {
            "mesh_path": mesh_path,
            "mesh_name": mesh_name,
        }
        return mesh

class MeshProcessor(pl.LightningModule):
    def __init__(
        self,
        mesh_json: str,
        output_dir: str,
        data_type:str,
        surface_uniform_samples: int = 20000,
        surface_curvature_samples: int = 40000,
        space_samples: int = 300000,
        noise_sigma: float = 0.01,
        batch_size: int = 1,
        num_workers: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        os.makedirs(output_dir, exist_ok=True)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        mesh_path = batch["mesh_path"][0]
        mesh_name = batch["mesh_name"][0]
        
        # sys_logger.info(f"Processing {batch_idx}/{len(self.trainer.predict_dataloaders)}: {mesh_name} from {mesh_path}")
        
        output_subdir = self.hparams.output_dir
        
        try:
            filename = os.path.splitext(os.path.basename(mesh_path))[0]
            if os.path.exists(os.path.join(output_subdir, f"{filename}.npz")):
                # sys_logger.info(f"Skipping {mesh_name} as it already exists.")
                return {
                    "status": "success",
                    "mesh_name": mesh_name
                }
            process_single_mesh(
                mesh_name=mesh_name,
                mesh_path=mesh_path,
                output_dir=output_subdir,
                data_type = self.hparams.data_type,
                surface_uniform_samples=self.hparams.surface_uniform_samples,
                surface_curvature_samples=self.hparams.surface_curvature_samples,
                space_samples=self.hparams.space_samples,
                noise_sigma=self.hparams.noise_sigma,
                device=self.device
            )
            
            return {
                "status": "success",
                "mesh_name": mesh_name
            }
        
        except Exception as e:
                print(f"Error processing {mesh_name}: {str(e)}")
                return {
                    "status": "error",
                    "mesh_name": mesh_name,
                    "error": str(e)
                }

    def predict_dataloader(self) -> DataLoader:
        dataset = MeshDataset(
            self.hparams.mesh_json)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=False
        )

def process_mesh_directory(
    mesh_json: str,
    output_dir: str,
    data_type: str,
    surface_uniform_samples: int = 100000,
    surface_curvature_samples: int = 200000,
    space_samples: int = 300000,
    noise_sigma: float = 0.01,
    num_gpus: int = -1,
    batch_size: int = 1,
    num_workers: int = 4
) -> None:
    model = MeshProcessor(
        mesh_json=mesh_json,
        output_dir=output_dir,
        data_type=data_type,
        surface_uniform_samples=surface_uniform_samples,
        surface_curvature_samples=surface_curvature_samples,
        space_samples=space_samples,
        noise_sigma=noise_sigma,
        batch_size=batch_size,
        num_workers=num_workers
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp",
        precision=32,
        logger=False,
        enable_progress_bar=True
    )
    
    predictions = trainer.predict(model)
    
    success_count = sum(1 for p in predictions if p["status"] == "success")
    error_count = sum(1 for p in predictions if p["status"] == "error")
    
    print(f"\nProcessing completed:")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {error_count} files")
    
    if error_count > 0:
        print("\nFailed files:")
        for p in predictions:
            if p["status"] == "error":
                print(f"- {p['mesh_name']}: {p['error']}")
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Mesh Directory for Sampling")

    parser.add_argument("--mesh_json", type=str, default="test_mesh.json", help="Path to the mesh json file")
    parser.add_argument("--output_dir", type=str, default="ultrashape_test1", help="Directory to save outputs")

    parser.add_argument("--surface_uniform_samples", type=int, default=300000, help="Number of uniform samples on surface")
    parser.add_argument("--surface_curvature_samples", type=int, default=300000, help="Number of curvature-based samples on surface")
    parser.add_argument("--space_samples", type=int, default=400000, help="Number of samples in space")

    parser.add_argument("--noise_sigma", type=float, default=0.01, help="Sigma for Gaussian noise")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")

    args = parser.parse_args()
    # print(f"Arguments: {args}")

    process_mesh_directory(
        mesh_json=args.mesh_json,
        output_dir=args.output_dir,
        data_type='mesh',
        surface_uniform_samples=args.surface_uniform_samples,
        surface_curvature_samples=args.surface_curvature_samples,
        space_samples=args.space_samples,
        noise_sigma=args.noise_sigma,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
