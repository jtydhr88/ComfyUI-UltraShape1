import os
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
import trimesh

from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model, instantiate_vae_model


def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output

class VAETrainer(pl.LightningModule):
    def __init__(
        self,
        *,
        vae_config,
        optimizer_cfg,
        loss_cfg,
        save_dir,
        mc_res,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
    ):
        super().__init__()

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg
        self.loss_cfg = loss_cfg
        self.ckpt_path = ckpt_path
        self.vae_model = instantiate_vae_model(vae_config, requires_grad=True)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.mc_res = mc_res
        self.save_root = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.vae_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    def init_from_ckpt(self, path, ignore_keys=()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' not in ckpt:
            # deepspeed ckpt
            state_dict = {}
            for k in ckpt.keys():
                new_k = k.replace('_forward_module.', '')
                state_dict[new_k] = ckpt[k]
        else:
            state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]
        
        # # ==================== Weight Surgery Start ====================
        # old_key_base = "vae_model.encoder.input_proj"
        # old_weight_key = f"{old_key_base}.weight"
        # old_bias_key = f"{old_key_base}.bias"

        # if old_weight_key in state_dict:
        #     print(f"[*] Detected legacy '{old_key_base}' in checkpoint. Performing weight surgery...")
            
        #     src_weight = state_dict[old_weight_key]
        #     src_bias = state_dict[old_bias_key]
            
        #     encoder = self.vae_model.encoder
        #     fourier_dim = encoder.fourier_embedder.out_dim

        #     # --- A. input_proj_kv ---
        #     # shape: [width, fourier_dim + point_feats]
        #     encoder.input_proj_kv.weight.data.copy_(src_weight)
        #     encoder.input_proj_kv.bias.data.copy_(src_bias)
        #     print(f"    -> Loaded input_proj_kv from {old_key_base}")

        #     # --- B. input_proj_q ---
        #     # shape: [width, fourier_dim]
        #     sliced_weight = src_weight[:, :fourier_dim]
        #     encoder.input_proj_q.weight.data.copy_(sliced_weight)
        #     encoder.input_proj_q.bias.data.copy_(src_bias)
        #     print(f"    -> Loaded input_proj_q (sliced) from {old_key_base}")

        #     del state_dict[old_weight_key]
        #     if old_bias_key in state_dict:
        #         del state_dict[old_bias_key]
        # # ==================== Weight Surgery End ====================

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")


    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.vae_model.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        return optimizers, schedulers

    def on_train_epoch_start(self) -> None:
        pl.seed_everything(self.trainer.global_rank)

    def forward(self, batch):
        sup_pc_s_list = [batch["sup_near_uniform"], batch["sup_near_sharp"], batch["sup_space"]]
        rand_points = [sup_pc_s[:,:,:3] for sup_pc_s in sup_pc_s_list]
        rand_points_val = [sup_pc_s[:,:,3:] for sup_pc_s in sup_pc_s_list]

        rand_points = torch.cat(rand_points, dim=1)
        target = torch.cat(rand_points_val, dim=1)[...,0]
        target = -target

        latents, posterior = self.vae_model.encode(
            batch['surface'], sample_posterior=True, need_kl=True)
        latents = self.vae_model.decode(latents)
        logits = self.vae_model.query(latents, rand_points)
        
        loss_kl = posterior.kl()
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

        criteria = torch.nn.MSELoss()
        criteria2 = torch.nn.L1Loss()
        loss_logits = criteria(logits, target).mean() + criteria2(logits, target).mean()
        loss = self.loss_cfg.lambda_logits * loss_logits + self.loss_cfg.lambda_kl * loss_kl

        loss_dict = {
            "loss": loss,
            "loss_logits": loss_logits,
            "loss_kl": loss_kl
        }
        return loss_dict, latents

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, latents = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}/total_loss": loss["loss"].detach(),
            f"{split}/loss_logits": loss["loss_logits"].detach(),
            f"{split}/loss_kl": loss["loss_kl"].detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss, latents = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}/total_loss": loss["loss"].detach(),
            f"{split}/loss_logits": loss["loss_logits"].detach(),
            f"{split}/loss_kl": loss["loss_kl"].detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        if self.trainer.global_rank < 2:
            with torch.no_grad():
                save_dir = f"{self.save_root}/gs{self.global_step:010d}_rank{self.trainer.global_rank}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                uids = batch.get('uid')
                for i, latent in enumerate(latents[:5]):
                    mesh, grid_logits = self.vae_model.latents2mesh(
                            latent[None],
                            output_type='trimesh',
                            bounds=1.01,
                            mc_level=0.0,
                            num_chunks=20000,
                            octree_resolution=self.mc_res,
                            mc_algo='mc',
                            enable_pbar=True
                        )
        
                    mesh = export_to_trimesh(mesh[0])
                    
                    save_path = f"{save_dir}/recon_{os.path.splitext(os.path.basename(uids[i]))[0]}_mc{self.mc_res}.obj"               
                    mesh.export(save_path)

        return loss
