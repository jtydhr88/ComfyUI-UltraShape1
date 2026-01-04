
import os
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
from ultrashape.pipelines import export_to_trimesh

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model, instantiate_vae_model, instantiate_vae_model_local


class Diffuser(pl.LightningModule):
    def __init__(
        self,
        *,
        vae_config,
        cond_config,
        dit_cfg,
        scheduler_cfg,
        optimizer_cfg,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
    ):
        super().__init__()

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg

        # ========= init diffusion scheduler ========= #
        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        # ========= init the model ========= #
        self.dit_cfg = dit_cfg
        self.model = instantiate_from_config(dit_cfg, device=None, dtype=None)
        
        self.cond_stage_model = instantiate_from_config(cond_config)

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # ========= config lora model ========= #
        if lora_config is not None:
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                target_modules=lora_config.get('target_modules')
            )
            self.model = get_peft_model(self.model, loraconfig)

        # ========= config ema model ========= #
        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config.ema_model == 'DSEma':
                # from michelangelo.models.modules.ema_deepspeed import DSEma
                from ..utils.ema_deepspeed import DSEma
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            #do not initilize EMA weight from ckpt path, since I need to change moe layers
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #
        self.first_stage_model = instantiate_vae_model_local(vae_config)
        self.first_stage_model.enable_flashvdm_decoder()

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # ========= init pipeline for inference ========= #
        self.image_processor_cfg = image_processor_cfg
        self.image_processor = None
        if self.image_processor_cfg is not None:
            self.image_processor = instantiate_from_config(self.image_processor_cfg)
        self.pipeline_cfg = pipeline_cfg
        from ...schedulers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.pipeline = instantiate_from_config(
            pipeline_cfg,
            vae=self.first_stage_model,
            model=self.model,
            scheduler=scheduler,
            conditioner=self.cond_stage_model,
            image_processor=self.image_processor,
        )

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)
            torch.nn.Module.compile(self.cond_stage_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)
                if context is not None:
                    print(f"{context}: Restored training weights")

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

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_load_checkpoint(self, checkpoint):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.model.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']


        if self.optimizer_cfg.get('train_image_encoder', False):
            image_encoder_parameters = list(self.cond_stage_model.named_parameters())
            image_encoder_parameters_decay = [param for name, param in image_encoder_parameters if
                                              not any((no_decay_name in name) for no_decay_name in no_decay)]
            image_encoder_parameters_nodecay = [param for name, param in image_encoder_parameters if
                                                any((no_decay_name in name) for no_decay_name in no_decay)]
            # filter trainable params
            image_encoder_parameters_decay = [param for param in image_encoder_parameters_decay if
                                              param.requires_grad]
            image_encoder_parameters_nodecay = [param for param in image_encoder_parameters_nodecay if
                                                param.requires_grad]

            print(f"Image Encoder Params: {len(image_encoder_parameters_decay)} decay, ")
            print(f"Image Encoder Params: {len(image_encoder_parameters_nodecay)} nodecay, ")

            image_encoder_lr = self.optimizer_cfg['image_encoder_lr']
            image_encoder_lr_multiply = self.optimizer_cfg.get('image_encoder_lr_multiply', 1.0)
            image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else lr * image_encoder_lr_multiply
            params_list.append(
                {'params': image_encoder_parameters_decay, 'lr': image_encoder_lr,
                 'weight_decay': 0.05})
            params_list.append(
                {'params': image_encoder_parameters_nodecay, 'lr': image_encoder_lr,
                 'weight_decay': 0.})

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


    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)

    def on_train_epoch_start(self) -> None:
        pl.seed_everything(self.trainer.global_rank)

    def forward(self, batch, disable_drop):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): #float32 for text
            contexts = self.cond_stage_model(image=batch.get('image'), text=batch.get('text'), mask=batch.get('mask'), disable_drop=disable_drop)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                latents, voxel_idx = self.first_stage_model.encode(batch["surface"], sample_posterior=True, need_voxel=True)
                latents = self.z_scale_factor * latents
                # print(latents.shape)
                
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.transport.training_losses(self.model, latents, 
                    dict(contexts=contexts, voxel_cond=voxel_idx))["loss"].mean()

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch, disable_drop=False)
        split = 'train'
        loss_dict = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch, disable_drop=True)
        split = 'val'
        loss_dict = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    @torch.no_grad()
    def sample(self, batch, output_type='trimesh', **kwargs):
        self.cond_stage_model.disable_drop = True

        generator = torch.Generator().manual_seed(0)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    print("### USING PIPELINE ###")
                    print(f'device: {self.device} dtype : {self.dtype}')
                    additional_params = {'output_type':output_type}

                    image = batch.get("image", None)
                    mask = batch.get('mask', None)
                    
                    outputs = self.pipeline(image=image, 
                                            mask=mask,
                                            generator=generator,
                                            box_v=1.0,
                                            mc_level=0.0,
                                            octree_resolution=1024,
                                            **additional_params)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Unexpected {e=}, {type(e)=}")
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = [None]

        self.cond_stage_model.disable_drop = False
        return [outputs]
