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

from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

from .attention_blocks import CrossAttentionDecoder
from .attention_processors import FlashVDMCrossAttentionProcessor, FlashVDMTopMCrossAttentionProcessor
from ...utils import logger


def dilate_3d_sliced(tensor: torch.Tensor, dilate_kernel: nn.Conv3d, slice_size: int = 64) -> torch.Tensor:
    """
    Apply 3D dilation to a large tensor by processing it in z-axis slices.
    This reduces memory usage for high-resolution volumes.
    """
    device = tensor.device
    # Get kernel dtype
    kernel_dtype = next(dilate_kernel.parameters()).dtype
    output_dtype = tensor.dtype if not torch.is_floating_point(tensor) else tensor.dtype

    if tensor.shape[0] <= slice_size:
        # Small enough to process directly
        result = dilate_kernel(tensor.unsqueeze(0).to(kernel_dtype)).squeeze(0)
        return result.to(output_dtype)

    slices = []
    for z_start in range(0, tensor.shape[0], slice_size):
        z_end = min(z_start + slice_size, tensor.shape[0])

        # Add overlap to handle boundary artifacts from the convolution
        z_start_with_pad = max(0, z_start - 1)
        z_end_with_pad = min(tensor.shape[0], z_end + 1)

        # Extract slice with overlap and convert to kernel dtype
        slice_data = tensor[z_start_with_pad:z_end_with_pad].unsqueeze(0).to(kernel_dtype)
        dilated_slice = dilate_kernel(slice_data).squeeze(0)

        # Convert back to original dtype if needed
        if output_dtype != kernel_dtype:
            dilated_slice = dilated_slice.to(output_dtype)

        # Remove the overlapping regions from the output
        overlap_start = z_start - z_start_with_pad
        overlap_end = overlap_start + (z_end - z_start)
        slices.append(dilated_slice[overlap_start:overlap_end])

        torch.cuda.empty_cache()

    return torch.cat(slices, dim=0)


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    D = input_tensor.shape[0]

    val = input_tensor + alpha
    valid_mask = val > -9000

    mask = torch.ones_like(val, dtype=torch.int32)
    sign = torch.sign(val.to(torch.float32))

    # Helper to compute neighbor for a single direction
    def check_neighbor_sign(shift, axis):
        if shift == 0:
            return

        pad_dims = [0, 0, 0, 0, 0, 0]
        if axis == 0:
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        padded = F.pad(val.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')

        slice_dims = [slice(None)] * 3
        if axis == 0:
            if shift > 0: slice_dims[0] = slice(shift, None)
            else: slice_dims[0] = slice(None, shift)
        elif axis == 1:
            if shift > 0: slice_dims[1] = slice(shift, None)
            else: slice_dims[1] = slice(None, shift)
        elif axis == 2:
            if shift > 0: slice_dims[2] = slice(shift, None)
            else: slice_dims[2] = slice(None, shift)

        padded = padded.squeeze(0).squeeze(0)
        neighbor = padded[tuple(slice_dims)]
        neighbor = torch.where(neighbor > -9000, neighbor, val)

        # Check sign consistency
        neighbor_sign = torch.sign(neighbor.to(torch.float32))
        return (neighbor_sign == sign)

    # Iteratively check neighbors and update mask
    # directions: (shift, axis)
    directions = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2)]

    for shift, axis in directions:
        is_same = check_neighbor_sign(shift, axis)
        mask = mask & is_same.to(torch.int32)

    # Invert mask: we want 1 where ANY neighbor has different sign
    mask = (~(mask.bool())).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_logits = []
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f"Volume Decoding",
                          disable=not enable_pbar):
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits


class HierarchicalVolumeDecoding:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        z_slice_size = kwargs.pop('z_slice_size', 64)

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_logits = []
        batch_size = latents.shape[0]
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks),
                          desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]"):
            queries = xyz_samples[start: start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=batch_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2]))

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate_3d_sliced(curr_points, dilate, z_slice_size)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1

            # Clear memory before dilation operations
            del curr_points
            torch.cuda.empty_cache()

            for i in range(2 - expand_num):
                next_index = dilate_3d_sliced(next_index, dilate, z_slice_size)
            nidx = torch.where(next_index > 0)

            # Store shape before deleting
            next_index_shape = next_index.shape
            del next_index
            torch.cuda.empty_cache()

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=next_points.dtype, device=device) +
                           torch.tensor(bbox_min, dtype=next_points.dtype, device=device))
            batch_logits = []
            for start in tqdm(range(0, next_points.shape[0], num_chunks),
                              desc=f"Hierarchical Volume Decoding [r{octree_depth_now + 1}]"):
                queries = next_points[start: start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                logits = geo_decoder(queries=batch_queries.to(latents.dtype), latents=latents)
                batch_logits.append(logits)

            # Delayed allocation of next_logits
            next_logits = torch.full(next_index_shape, -10000., dtype=dtype, device=device)
            grid_logits = torch.cat(batch_logits, dim=1)
            next_logits[nidx] = grid_logits[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)
        grid_logits[grid_logits == -10000.] = float('nan')

        return grid_logits


class FlashVDMVolumeDecoding:
    def __init__(self, topk_mode='mean'):
        if topk_mode not in ['mean', 'merge']:
            raise ValueError(f'Unsupported topk_mode {topk_mode}, available: {["mean", "merge"]}')

        if topk_mode == 'mean':
            self.processor = FlashVDMCrossAttentionProcessor()
        else:
            self.processor = FlashVDMTopMCrossAttentionProcessor()

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: CrossAttentionDecoder,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        mini_grid_num: int = 4,
        enable_pbar: bool = True,
        **kwargs,
    ):
        processor = self.processor
        geo_decoder.set_cross_attention_processor(processor)

        device = latents.device
        dtype = latents.dtype
        z_slice_size = kwargs.pop('z_slice_size', 64)

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)

        logger.info(f"FlashVDMVolumeDecoding Resolution: {resolutions}")

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)

        # 2. latents to 3d volume
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        batch_size = latents.shape[0]
        mini_grid_size = xyz_samples.shape[0] // mini_grid_num
        xyz_samples = xyz_samples.view(
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size, 3
        ).permute(
            0, 2, 4, 1, 3, 5, 6
        ).reshape(
            -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
        )
        batch_logits = []
        num_batchs = max(num_chunks // xyz_samples.shape[1], 1)
        for start in tqdm(range(0, xyz_samples.shape[0], num_batchs),
                          desc=f"FlashVDM Volume Decoding", disable=not enable_pbar):
            queries = xyz_samples[start: start + num_batchs, :]
            batch = queries.shape[0]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)
            processor.topk = True

            # Chunk queries along dim 1 if too large
            if queries.shape[1] > num_chunks:
                # print(f"Chunking queries: {queries.shape} with chunk size {num_chunks}")
                batch_logits_sub = []
                for sub_start in range(0, queries.shape[1], num_chunks):
                    sub_queries = queries[:, sub_start: sub_start + num_chunks, :]
                    logits = geo_decoder(queries=sub_queries, latents=batch_latents)
                    batch_logits_sub.append(logits)
                logits = torch.cat(batch_logits_sub, dim=1)
            else:
                logits = geo_decoder(queries=queries, latents=batch_latents)

            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=0).reshape(
            mini_grid_num, mini_grid_num, mini_grid_num,
            mini_grid_size, mini_grid_size,
            mini_grid_size
        ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])
        )

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate_3d_sliced(curr_points, dilate, z_slice_size)
                curr_points = dilate_3d_sliced(curr_points, dilate, z_slice_size)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)

            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1

            # Clear memory before dilation operations
            del curr_points
            torch.cuda.empty_cache()

            for i in range(2 - expand_num):
                next_index = dilate_3d_sliced(next_index, dilate, z_slice_size)
            nidx = torch.where(next_index > 0)

            # Store shape before deleting
            next_index_shape = next_index.shape
            del next_index
            torch.cuda.empty_cache()

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                           torch.tensor(bbox_min, dtype=torch.float32, device=device))

            query_grid_num = 6
            min_val = next_points.min(axis=0).values
            max_val = next_points.max(axis=0).values
            vol_queries_index = (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
            index = torch.floor(vol_queries_index).long()
            index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
            index = index.sort()
            next_points = next_points[index.indices].unsqueeze(0).contiguous()
            unique_values = torch.unique(index.values, return_counts=True)
            grid_logits = torch.zeros((next_points.shape[1]), dtype=latents.dtype, device=latents.device)
            input_grid = [[], []]
            logits_grid_list = []
            start_num = 0
            sum_num = 0
            for grid_index, count in zip(unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()):
                remaining_count = count
                while remaining_count > 0:
                    space_left = num_chunks - sum_num
                    # If buffer is full, flush it
                    if space_left <= 0:
                        processor.topk = input_grid
                        logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                        start_num = start_num + sum_num
                        logits_grid_list.append(logits_grid)
                        input_grid = [[], []]
                        sum_num = 0
                        space_left = num_chunks

                    take = min(remaining_count, space_left)
                    input_grid[0].append(grid_index)
                    input_grid[1].append(take)
                    sum_num += take
                    remaining_count -= take
            if sum_num > 0:
                processor.topk = input_grid
                logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                logits_grid_list.append(logits_grid)
            logits_grid = torch.cat(logits_grid_list, dim=1)
            grid_logits[index.indices] = logits_grid.squeeze(0).squeeze(-1)

            # Delayed allocation of next_logits
            next_logits = torch.full(next_index_shape, -10000., dtype=dtype, device=device)
            next_logits[nidx] = grid_logits
            grid_logits = next_logits.unsqueeze(0)

        grid_logits[grid_logits == -10000.] = float('nan')

        return grid_logits
