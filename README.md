# ComfyUI-UltraShape1

A ComfyUI plugin based on [UltraShape 1.0](https://github.com/PKU-YuanGroup/UltraShape-1.0) for image-guided 3D mesh refinement.

## Features

- **Mesh Refinement**: Refine coarse 3D meshes using image guidance
- **High-Quality Output**: Generate detailed meshes with sharp edges
- **Multiple Export Formats**: Export to GLB, OBJ, PLY, STL formats
- **Flexible Input**: Load meshes from file picker or path string
- **Configurable Resolution**: Adjust output resolution based on available VRAM

## Installation

### 1. Install Dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-UltraShape1
pip install -r requirements.txt
```

### 2. Download Model Weights

Place model weights in ComfyUI's models directory:

```
ComfyUI/
└── models/
    └── UltraShape/
        └── ultrashape_refine.pt    # or other checkpoint name
```

Download from the official UltraShape repository or HuggingFace.

### 3. Prepare Input Files

Place coarse meshes in ComfyUI's input directory:

```
ComfyUI/
└── input/
    └── your_coarse_mesh.glb    # or .obj, .ply, .stl
```

## Node Documentation

### UltraShape Load Model
Load UltraShape refinement model (VAE + DiT + Conditioner).

| Parameter | Description |
|-----------|-------------|
| checkpoint | Select checkpoint file from `models/UltraShape/` |
| config | Config file (default: `infer_dit_refine.yaml`) |
| dtype | Model precision: `float16` / `bfloat16` / `float32` |

### UltraShape Load Coarse Mesh
Load and preprocess coarse mesh for refinement (file picker).

| Parameter | Description |
|-----------|-------------|
| model | Model from Load Model node |
| mesh_file | Select mesh file from dropdown |
| normalize_scale | Mesh normalization scale (default: 0.99) |
| num_sharp_points | Number of sharp edge sample points |
| num_uniform_points | Number of uniform sample points |

### UltraShape Load Coarse Mesh (Path)
Load coarse mesh from file path string.

| Parameter | Description |
|-----------|-------------|
| model | Model from Load Model node |
| mesh_path | Full path to mesh file |

### UltraShape Refine
Core refinement node - refine coarse mesh using image guidance.

| Parameter | Description |
|-----------|-------------|
| model | Model from Load Model node |
| coarse_mesh | Mesh from Load Coarse Mesh node |
| image | Reference image (IMAGE type) |
| steps | Diffusion steps (default: 50) |
| guidance_scale | Image guidance scale (default: 5.0) |
| octree_resolution | Output mesh resolution (default: 384) |
| mc_level | Marching cubes level (default: 0.0) |
| box_v | Bounding box scale (default: 1.0) |
| seed | Random seed |
| remove_bg | Remove image background before processing |

**VRAM Note for `octree_resolution`**:
| Resolution | Approx. VRAM |
|------------|--------------|
| 384 | ~8GB |
| 512 | ~16GB |
| 768 | ~32GB |
| 1024 | ~48GB+ |

### UltraShape Save GLB/OBJ
Save refined mesh to file.

| Parameter | Description |
|-----------|-------------|
| refined_mesh | Mesh from Refine node |
| output_dir | Output subdirectory (default: `ultrashape_output`) |
| filename_prefix | Output filename prefix |
| file_format | Export format: `glb` / `obj` / `ply` / `stl` |

## Example Workflow

```
[Load Image] ──────────────────────────────────────┐
                                                   │
[UltraShape Load Model] ──┬──> [UltraShape Load Coarse Mesh] ──┴──> [UltraShape Refine] ──> [UltraShape Save GLB/OBJ]
```

## Notes

1. **VRAM Requirements**:
   - Model loading: ~8-12GB VRAM
   - Refinement (octree_resolution=384): ~8GB additional
   - Refinement (octree_resolution=512): ~16GB additional
   - Total recommended: 16GB+ for default settings, 32GB+ for high resolution

2. **Input Requirements**:
   - UltraShape is a **refinement** model, not a generation model
   - You need to provide a coarse mesh (from other 3D generation tools like Hunyuan3D, InstantMesh, etc.)
   - Reference image should match the desired output appearance

3. **Optional Dependencies**:
   - `cubvh`: CUDA-accelerated marching cubes (falls back to skimage if not installed)
   - `flash_attn`: Flash Attention (falls back to PyTorch SDPA if not installed)
   - `pymeshlab`: Mesh post-processing (optional features disabled if not installed)

4. **Supported Mesh Formats**:
   - Input: GLB, GLTF, OBJ, PLY, STL
   - Output: GLB, OBJ, PLY, STL

## Troubleshooting

### Out of Memory (OOM)
- Reduce `octree_resolution` (try 384 or 256)
- Use `bfloat16` or `float16` dtype
- Close other GPU applications

### Tensor Size Mismatch
- This can happen with very simple meshes
- The plugin automatically pads voxels to match expected size

### Missing Dependencies
- `cubvh`: Optional, will use slower CPU marching cubes
- `flash_attn`: Optional, will use PyTorch SDPA
- `pymeshlab`: Optional, mesh post-processing will be skipped

## License

Please refer to the UltraShape 1.0 original project license.
