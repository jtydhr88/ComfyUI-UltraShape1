"""UltraShape mesh refinement node."""

from comfy_env import isolated


@isolated(env="ultrashape1", import_paths=["."])
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
                "octree_resolution": ("INT", {"default": 384, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Mesh resolution. Higher=better quality but more VRAM. 384=~8GB, 512=~16GB, 1024=~48GB+"}),
                "num_chunks": ("INT", {"default": 8000, "min": 1000, "max": 50000, "step": 1000,
                    "tooltip": "Chunk size for volume decoding. Lower=less VRAM but slower. Default 8000 works for most GPUs"}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.01}),
                "box_v": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7fffffff}),
                "remove_bg": ("BOOLEAN", {"default": False}),
                "sequential_cfg": ("BOOLEAN", {"default": False,
                    "tooltip": "Run CFG passes separately to halve VRAM. Slower but uses ~8GB instead of ~16GB during diffusion."}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "refine"
    CATEGORY = "UltraShape"

    def refine(self, model, coarse_mesh, image, steps=50, guidance_scale=5.0,
               octree_resolution=384, num_chunks=8000, mc_level=0.0, box_v=1.0,
               seed=42, remove_bg=False, sequential_cfg=False):
        # Check if disk offload mode
        if getattr(model, 'disk_offload', False):
            return self._refine_disk_offload(
                model, coarse_mesh, image, steps, guidance_scale,
                octree_resolution, num_chunks, mc_level, box_v, seed, remove_bg,
                sequential_cfg
            )
        return self._refine_normal(
            model, coarse_mesh, image, steps, guidance_scale,
            octree_resolution, num_chunks, mc_level, box_v, seed, remove_bg,
            sequential_cfg
        )

    def _refine_normal(self, model, coarse_mesh, image, steps=50, guidance_scale=5.0,
               octree_resolution=384, num_chunks=8000, mc_level=0.0, box_v=1.0,
               seed=42, remove_bg=False, sequential_cfg=False):
        # Lazy imports inside isolated subprocess
        import gc
        import numpy as np
        import torch
        from PIL import Image
        import comfy.utils

        def get_cuda_memory_str():
            """Get formatted CUDA memory usage string."""
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"VRAM: {allocated:.1f}GB alloc, {reserved:.1f}GB reserved"

        # Memory cleanup before inference
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[UltraShape] Initial {get_cuda_memory_str()}")

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

        print(f"[UltraShape] Refining mesh: steps={steps}, guidance={guidance_scale}, octree_res={octree_resolution}")
        print(f"[UltraShape] Before diffusion: {get_cuda_memory_str()}")

        # Setup generator for reproducibility
        generator = torch.Generator(device=model.device).manual_seed(seed)

        # Progress bar
        pbar = comfy.utils.ProgressBar(steps)
        step_count = [0]

        def callback(step_idx, t, outputs):
            step_count[0] += 1
            current = step_count[0]
            pbar.update_absolute(current, steps)

            # Print progress every 5 steps or at start/end
            if current == 1 or current % 5 == 0 or current == steps:
                mem = get_cuda_memory_str()
                print(f"[UltraShape] Step {current}/{steps} | {mem}")

        # Run diffusion refinement
        try:
            with torch.autocast(device_type="cuda", dtype=model.dtype):
                mesh, latents = model.pipeline(
                    image=pil_image,
                    voxel_cond=coarse_mesh.voxel_idx,
                    generator=generator,
                    box_v=box_v,
                    mc_level=mc_level,
                    octree_resolution=octree_resolution,
                    num_inference_steps=steps,
                    num_chunks=num_chunks,
                    guidance_scale=guidance_scale,
                    callback=callback,
                    callback_steps=1,
                )

            # mesh is a list, get first element
            output_mesh = mesh[0] if isinstance(mesh, list) else mesh

            print(f"[UltraShape] Mesh extracted: {get_cuda_memory_str()}")
            print(f"[UltraShape] Refinement complete: vertices={len(output_mesh.vertices)}, faces={len(output_mesh.faces)}")
            return (output_mesh,)
        finally:
            # Memory cleanup after inference
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[UltraShape] After cleanup: {get_cuda_memory_str()}")

    def _refine_disk_offload(self, model, coarse_mesh, image, steps=50, guidance_scale=5.0,
                             octree_resolution=384, num_chunks=8000, mc_level=0.0, box_v=1.0,
                             seed=42, remove_bg=False, sequential_cfg=False):
        """Disk offload mode: load one model at a time to minimize VRAM usage."""
        import gc
        import os
        import tempfile
        import shutil
        import numpy as np
        import torch
        from PIL import Image
        from tqdm import tqdm
        from omegaconf import OmegaConf
        from ultrashape.utils.misc import instantiate_from_config
        from ultrashape.pipelines import export_to_trimesh
        import comfy.utils

        def get_cuda_memory_str():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"VRAM: {allocated:.1f}GB alloc, {reserved:.1f}GB reserved"

        def cleanup():
            gc.collect()
            torch.cuda.empty_cache()

        # Create temp directory for intermediate tensors
        temp_dir = tempfile.mkdtemp(prefix="ultrashape_offload_")
        print(f"[UltraShape] Disk offload mode - temp dir: {temp_dir}")

        try:
            device = model.device
            dtype = model.dtype
            cfg = model.config

            # Prepare image
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            if remove_bg:
                try:
                    from ultrashape.rembg import BackgroundRemover
                    rembg = BackgroundRemover()
                    pil_image = rembg(pil_image)
                    print("[UltraShape] Background removed")
                    del rembg
                    cleanup()
                except Exception as e:
                    print(f"[UltraShape] Warning: Background removal failed: {e}")
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')

            # Helper to recursively move dict of tensors to device
            def move_to_device(obj, device, dtype=None):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device=device, dtype=dtype) if dtype else obj.to(device=device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, device, dtype) for k, v in obj.items()}
                return obj

            # Helper to recursively move dict of tensors to CPU
            def move_to_cpu(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu()
                elif isinstance(obj, dict):
                    return {k: move_to_cpu(v) for k, v in obj.items()}
                return obj

            # Helper to concatenate dicts of tensors for CFG
            def cat_cond(cond, uncond):
                if isinstance(cond, torch.Tensor):
                    return torch.cat([cond, uncond], dim=0)
                elif isinstance(cond, dict):
                    return {k: cat_cond(cond[k], uncond[k]) for k in cond.keys()}
                return cond

            # ================================================================
            # STAGE 1: Conditioner - encode image
            # ================================================================
            print(f"\n[UltraShape] === Stage 1/3: Image Encoding ===")
            print(f"[UltraShape] {get_cuda_memory_str()}")

            # Load conditioner
            print("[UltraShape] Loading conditioner...")
            conditioner = instantiate_from_config(cfg.model.params.conditioner_config)
            weights = torch.load(model.ckpt_path, map_location='cpu', weights_only=True)
            conditioner.load_state_dict(weights['conditioner'], strict=True)
            conditioner.eval().to(device, dtype=dtype)
            del weights['conditioner']
            print(f"[UltraShape] Conditioner loaded: {get_cuda_memory_str()}")

            # Load image processor
            image_processor = instantiate_from_config(cfg.model.params.image_processor_cfg)

            # Process image - image_processor returns dict with 'image' key
            # Conditioner returns dict with 'main' (and possibly 'additional') keys
            do_cfg = guidance_scale > 1.0
            with torch.inference_mode():
                processed = image_processor(pil_image)
                image_tensor = processed["image"].to(device=device, dtype=dtype)
                cond = conditioner(image=image_tensor)

                # For CFG, we need unconditional embeddings
                if do_cfg:
                    # Get token count from cond['main']
                    cond_token_num = cond["main"].shape[1]
                    uncond = conditioner.unconditional_embedding(1, num_tokens=cond_token_num)

                    if sequential_cfg:
                        # Save separately for sequential processing (lower VRAM)
                        torch.save(move_to_cpu(cond), os.path.join(temp_dir, "cond.pt"))
                        torch.save(move_to_cpu(uncond), os.path.join(temp_dir, "uncond.pt"))
                        print(f"[UltraShape] Sequential CFG: saved cond and uncond separately")
                    else:
                        # Concatenate cond and uncond for batched CFG
                        cond = cat_cond(cond, uncond)
                        torch.save(move_to_cpu(cond), os.path.join(temp_dir, "cond.pt"))
                        print(f"[UltraShape] Batched CFG: cond shape {cond['main'].shape}")
                else:
                    torch.save(move_to_cpu(cond), os.path.join(temp_dir, "cond.pt"))
                    print(f"[UltraShape] No CFG: saved cond embeddings")

            # Unload conditioner
            del conditioner, image_processor, image_tensor
            # cond/uncond may or may not exist depending on code path
            try:
                del cond
            except NameError:
                pass
            try:
                del uncond
            except NameError:
                pass
            del weights
            cleanup()
            print(f"[UltraShape] Stage 1 complete: {get_cuda_memory_str()}")

            # ================================================================
            # STAGE 2: DiT - run diffusion
            # ================================================================
            print(f"\n[UltraShape] === Stage 2/3: Diffusion ({steps} steps) ===")
            print(f"[UltraShape] {get_cuda_memory_str()}")

            # Load DiT
            print("[UltraShape] Loading DiT...")
            dit = instantiate_from_config(cfg.model.params.dit_cfg)
            weights = torch.load(model.ckpt_path, map_location='cpu', weights_only=True)
            dit.load_state_dict(weights['dit'], strict=True)
            dit.eval().to(device, dtype=dtype)
            del weights['dit']
            print(f"[UltraShape] DiT loaded: {get_cuda_memory_str()}")

            # Load scheduler
            scheduler = instantiate_from_config(cfg.model.params.scheduler_cfg)

            # Load cond from disk
            do_cfg = guidance_scale > 1.0
            if sequential_cfg and do_cfg:
                # Load separately for sequential processing
                cond = move_to_device(torch.load(os.path.join(temp_dir, "cond.pt")), device)
                uncond = move_to_device(torch.load(os.path.join(temp_dir, "uncond.pt")), device)
                print(f"[UltraShape] Sequential CFG: loaded cond and uncond separately")
            else:
                # Batched CFG or no CFG - cond already doubled if needed
                cond = move_to_device(torch.load(os.path.join(temp_dir, "cond.pt")), device)
                uncond = None

            # Prepare latents
            voxel_cond = coarse_mesh.voxel_idx
            num_tokens = voxel_cond.shape[1]
            latent_dim = cfg.model.params.vae_config.params.embed_dim
            latents = torch.randn(
                1, num_tokens, latent_dim,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed)
            )

            # Setup timesteps - use sigmas like the pipeline does
            sigmas = np.linspace(0, 1, steps)
            from ultrashape.pipelines import retrieve_timesteps
            timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas)

            # CFG setup for batched mode - double voxel_cond
            if do_cfg and not sequential_cfg:
                voxel_cond_batched = torch.cat([voxel_cond] * 2)
            else:
                voxel_cond_batched = voxel_cond

            # Progress bar
            pbar = comfy.utils.ProgressBar(steps)

            # Diffusion loop
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    for i, t in enumerate(tqdm(timesteps, desc="Diffusion")):
                        timestep_single = t.expand(1).to(dtype) / scheduler.config.num_train_timesteps

                        if sequential_cfg and do_cfg:
                            # Sequential CFG: run cond and uncond separately (half VRAM)
                            noise_pred_cond = dit(latents, timestep_single, cond, voxel_cond=voxel_cond)
                            noise_pred_uncond = dit(latents, timestep_single, uncond, voxel_cond=voxel_cond)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        elif do_cfg:
                            # Batched CFG: run both together
                            latent_input = torch.cat([latents] * 2)
                            timestep_batched = t.expand(2).to(dtype) / scheduler.config.num_train_timesteps
                            noise_pred = dit(latent_input, timestep_batched, cond, voxel_cond=voxel_cond_batched)
                            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            # No CFG
                            noise_pred = dit(latents, timestep_single, cond, voxel_cond=voxel_cond)

                        outputs = scheduler.step(noise_pred, t, latents)
                        latents = outputs.prev_sample

                        pbar.update_absolute(i + 1, steps)
                        if (i + 1) % 10 == 0:
                            print(f"[UltraShape] Step {i+1}/{steps} | {get_cuda_memory_str()}")

            # Save latents to disk
            torch.save(latents.cpu(), os.path.join(temp_dir, "latents.pt"))
            print(f"[UltraShape] Saved latents to disk")

            # Unload DiT
            del dit, scheduler, cond, latents, voxel_cond, voxel_cond_batched
            if uncond is not None:
                del uncond
            del weights
            cleanup()
            print(f"[UltraShape] Stage 2 complete: {get_cuda_memory_str()}")

            # ================================================================
            # STAGE 3: VAE - decode to mesh
            # ================================================================
            print(f"\n[UltraShape] === Stage 3/3: Mesh Decoding ===")
            print(f"[UltraShape] {get_cuda_memory_str()}")

            # Load VAE
            print("[UltraShape] Loading VAE...")
            vae = instantiate_from_config(cfg.model.params.vae_config)
            weights = torch.load(model.ckpt_path, map_location='cpu', weights_only=True)
            vae.load_state_dict(weights['vae'], strict=True)
            vae.eval().to(device, dtype=dtype)
            if hasattr(vae, 'enable_flashvdm_decoder'):
                vae.enable_flashvdm_decoder()
            del weights
            print(f"[UltraShape] VAE loaded: {get_cuda_memory_str()}")

            # Load latents from disk
            latents = torch.load(os.path.join(temp_dir, "latents.pt")).to(device, dtype=dtype)

            # Decode
            with torch.inference_mode():
                latents = 1. / vae.scale_factor * latents
                latents = vae(latents)
                outputs, _ = vae.latents2mesh(
                    latents,
                    bounds=box_v,
                    mc_level=mc_level,
                    num_chunks=num_chunks,
                    octree_resolution=octree_resolution,
                    mc_algo='mc',
                    enable_pbar=True,
                )

            output_mesh = export_to_trimesh(outputs)[0]
            print(f"[UltraShape] Mesh extracted: {get_cuda_memory_str()}")
            print(f"[UltraShape] Refinement complete: vertices={len(output_mesh.vertices)}, faces={len(output_mesh.faces)}")

            # Cleanup
            del vae, latents
            cleanup()
            print(f"[UltraShape] Final: {get_cuda_memory_str()}")

            return (output_mesh,)

        finally:
            # Always cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[UltraShape] Cleaned up temp dir")
