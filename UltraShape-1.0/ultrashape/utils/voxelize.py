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

    # Always aim for num_latents
    actual_k = num_latents

    if min_count < num_latents:
        print(f"[Info] Voxel count ({min_count}) < Target ({num_latents}). Sampling with replacement.")
        # If we don't have enough unique voxels, we need to sample with replacement/repetition
        # We can just repeat the indices to fill the gap

        batch_starts = torch.searchsorted(shuffled_batch_ids, torch.arange(B, device=device))

        # Create gathering indices that wrap around for each batch
        # For each batch element i, we want actual_k indices
        # They start at batch_starts[i] and go up to batch_starts[i] + counts[i]
        # We use modulo to wrap around: (j % counts[i]) + batch_starts[i]

        # Expand for broadcasting
        batch_starts_exp = batch_starts.unsqueeze(1) # [B, 1]
        counts_exp = counts.unsqueeze(1) # [B, 1]

        offsets = torch.arange(actual_k, device=device).unsqueeze(0) # [1, K]

        # Calculate offsets modulo the available count for each batch
        # This effectively repeats the available voxels to fill the desired size
        # We need to be careful about division by zero if a batch has 0 voxels (shouldn't happen with valid PC)
        counts_exp = torch.maximum(counts_exp, torch.tensor(1, device=device))

        wrapped_offsets = offsets % counts_exp
        gather_indices = batch_starts_exp + wrapped_offsets
        gather_indices = gather_indices.view(-1)

    else:
        # Standard case: enough points, just take the first k
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

    return sampled_pc, sampled_indices
