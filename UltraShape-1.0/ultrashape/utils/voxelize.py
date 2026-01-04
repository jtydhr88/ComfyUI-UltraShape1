import torch

def voxelize_from_point(pc, num_latents, resolution=128):

    B, N, D = pc.shape
    device = pc.device

    norm_pc = (pc + 1.0) / 2.0
    voxel_indices = torch.floor(norm_pc * resolution).long()
    voxel_indices = torch.clamp(voxel_indices, 0, resolution - 1) # (B, N, 3)

    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    flat_indices = torch.cat([batch_idx.unsqueeze(-1), voxel_indices], dim=-1).view(-1, 4)
    unique_voxels = torch.unique(flat_indices, dim=0)
    u_batch_ids = unique_voxels[:, 0]

    noise = torch.rand_like(u_batch_ids, dtype=torch.float)
    sort_keys = u_batch_ids.float() + noise
    perm = torch.argsort(sort_keys)
    shuffled_voxels = unique_voxels[perm]
    shuffled_batch_ids = shuffled_voxels[:, 0].contiguous()

    counts = torch.bincount(shuffled_batch_ids, minlength=B)
    min_count = counts.min().item()
    actual_k = min(num_latents, min_count)

    if actual_k < num_latents:
        print(f"[Info] Voxel count ({min_count}) < Target ({num_latents}). Padding with repeated samples.")

    batch_starts = torch.searchsorted(shuffled_batch_ids, torch.arange(B, device=device))
    offsets = torch.arange(actual_k, device=device).unsqueeze(0)
    gather_indices = batch_starts.unsqueeze(1) + offsets
    gather_indices = gather_indices.view(-1)
    selected_indices = shuffled_voxels[gather_indices]

    final_grid_coords = selected_indices[:, 1:]

    # Grid Index -> Voxel Center
    voxel_size = 2.0 / resolution
    final_centers = (final_grid_coords.float() + 0.5) * voxel_size - 1.0

    sampled_pc = final_centers.view(B, actual_k, 3)
    sampled_indices = final_grid_coords.view(B, actual_k, 3)

    # Pad to num_latents if we have fewer voxels than expected
    if actual_k < num_latents:
        pad_size = num_latents - actual_k
        # Repeat existing samples to fill the gap
        repeat_indices = torch.randint(0, actual_k, (B, pad_size), device=device)
        pad_pc = torch.gather(sampled_pc, 1, repeat_indices.unsqueeze(-1).expand(-1, -1, 3))
        pad_indices = torch.gather(sampled_indices, 1, repeat_indices.unsqueeze(-1).expand(-1, -1, 3))
        sampled_pc = torch.cat([sampled_pc, pad_pc], dim=1)
        sampled_indices = torch.cat([sampled_indices, pad_indices], dim=1)

    return sampled_pc, sampled_indices
