# -*- coding: utf-8 -*-

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

import torch
import torch.distributed as dist
from typing import Union
from .utils import logger
import os


def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if config.get("from_pretrained", None):
        return cls.from_pretrained(
                    config["from_pretrained"], 
                    use_safetensors=config.get('use_safetensors', False),
                    variant=config.get('variant', 'fp16'))

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance


def instantiate_vae_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if config.get("from_pretrained", None):
        return cls.from_pretrained(
                    config["from_pretrained"], 
                    params=config.get("params", dict()),
                    use_safetensors=config.get('use_safetensors', False),
                    variant=config.get('variant', 'fp16'))

    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance

def instantiate_vae_from_config_local(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if not config.get("from_pretrained", None):
        raise FileNotFoundError(f"Need from_pretrained!")
    
    ckpt_path = config["from_pretrained"]
            
    logger.info(f"Loading model from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file {ckpt_path} not found")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    if 'state_dict' not in ckpt:
        # deepspeed ckpt
        state_dict = {}
        for k in ckpt.keys():
            new_k = k.replace('vae_model.', '')
            state_dict[new_k] = ckpt[k]
    else:
        state_dict = ckpt["state_dict"]

    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)


    missing, unexpected = instance.load_state_dict(state_dict)
    print(f"VAE Missing Keys: {missing}")
    print(f"VAE Unexpected Keys: {unexpected}")

    return instance

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def instantiate_non_trainable_model(config):
    model = instantiate_from_config(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    return model


def instantiate_vae_model(config, requires_grad=False):
    model = instantiate_vae_from_config(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = requires_grad

    return model

def instantiate_vae_model_local(config, requires_grad=False):
    model = instantiate_vae_from_config_local(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = requires_grad

    return model

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
