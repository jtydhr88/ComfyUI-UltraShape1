# -*- coding: utf-8 -*-

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

import math
import os
import json
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from ultrashape.utils.typing import *
import pandas as pd
import cv2
import torchvision.transforms as transforms
from pytorch_lightning.utilities import rank_zero_info

def padding(image, mask, center=True, padding_ratio_range=[1.15, 1.15]):
    """
    Pad the input image and mask to a square shape with padding ratio.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        mask (np.ndarray): Corresponding mask array of shape (H, W).
        center (bool): Whether to center the original image in the padded output.
        padding_ratio_range (list): Range [min, max] to randomly select padding ratio.

    Returns:
        newimg (np.ndarray): Padded image of shape (resize_side, resize_side, 3).
        newmask (np.ndarray): Padded mask of shape (resize_side, resize_side).
    """
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Select padding ratio either fixed or randomly within the given range
    if padding_ratio_range[0] == padding_ratio_range[1]:
        padding_ratio = padding_ratio_range[0]
    else:
        padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    resize_side = int(max_side * padding_ratio)

    pad_h = resize_side - h
    pad_w = resize_side - w
    if center:
        start_h = pad_h // 2
    else:
        start_h = pad_h - resize_side // 20
        
    start_w = pad_w // 2

    # Create new white image and black mask with padded size
    newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    newmask = np.zeros((resize_side, resize_side), dtype=np.uint8)
    
    # Place original image and mask into the padded canvas
    newimg[start_h:start_h + h, start_w:start_w + w] = image
    newmask[start_h:start_h + h, start_w:start_w + w] = mask
    
    return newimg, newmask


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        data_json,
        sample_root,
        image_path,
        image_transform = None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        padding = True,
        padding_ratio_range=[1.15, 1.15],
    ):
        super().__init__()

        self.uids = json.load(open(data_json))
        self.sample_root = sample_root
        self.image_paths = json.load(open(image_path))
        self.image_transform = image_transform
        
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
        print(f"Loaded {len(self.uids)} uids from {data_json}.")

        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Dataset Infos:')
        rank_zero_info(f'# of 3D file: {len(self.uids)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'*' * 50)

    def __len__(self):
        return len(self.uids)

    def _load_shape(self, index: int) -> Dict[str, Any]:

        data = np.load(f'{self.sample_root}/{self.uids[index]}.npz')

        surface_og = (np.asarray(data['clean_surface_points'])-0.5) * 2 
        normal = np.asarray(data['clean_surface_normals']) 
        surface_og_n = np.concatenate([surface_og, normal], axis=1) 
        rng = np.random.default_rng()

        # hard code: first 300k are uniform, last 300k are sharp
        assert surface_og_n.shape[0] == 600000, f"assume that suface points = 30w uniform + 30w curvature, but {len(surface_og_n)=}"
        coarse_surface = surface_og_n[:300000]
        sharp_surface = surface_og_n[300000:]

        surface_normal = []
        rng = np.random.default_rng()
        if self.pc_size > 0:
            ind = rng.choice(coarse_surface.shape[0], self.pc_size // 2, replace=False)
            coarse_surface = coarse_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size // 2, 1))
                coarse_surface = np.concatenate((coarse_surface, sharpedge_label), axis=1)
            surface_normal.append(coarse_surface)

            ind_sharpedge = rng.choice(sharp_surface.shape[0], self.pc_size // 2, replace=False)
            sharp_surface = sharp_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_size // 2, 1))
                sharp_surface = np.concatenate((sharp_surface, sharpedge_label), axis=1)
            surface_normal.append(sharp_surface)
        
        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size

        geo_points = 0.0
        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)

        ret = {
                "uid": self.uids[index],
                "surface": surface,
                "geo_points": geo_points
            }
        return ret

        
    def _load_image(self, index: int) -> Dict[str, Any]:
        ret = {}
        sel_idx = random.randint(0, 15)
        ret["sel_image_idx"] = sel_idx
        obj_name = self.uids[index]
        img_path = f'{self.image_paths[obj_name]}/{os.path.basename(self.image_paths[obj_name])}/rgba/' + f"{sel_idx:03d}.png"
 
        images, masks = [], []
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert image.shape[2] == 4
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        forground = image[:, :, :3]
        background = np.ones_like(forground) * 255
        img_new = forground * alpha + background * (1 - alpha)
        image = img_new.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (alpha[:, :, 0] * 255).astype(np.uint8)

        if self.padding:
            h, w = image.shape[:2]
            binary = mask > 0.3
            non_zero_coords = np.argwhere(binary)
            x_min, y_min = non_zero_coords.min(axis=0)
            x_max, y_max = non_zero_coords.max(axis=0)
            image, mask = padding(
                image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                center=True, padding_ratio_range=self.padding_ratio_range)
        
        if self.image_transform:
            image = self.image_transform(image)
            mask = np.stack((mask, mask, mask), axis=-1)
            mask = self.image_transform(mask)
        
        images.append(image)
        masks.append(mask)
        ret["image"] = torch.cat(images, dim=0)
        ret["mask"] = torch.cat(masks, dim=0)[:1, ...]
        
        return ret

    def get_data(self, index):
        ret = self._load_shape(index)
        ret.update(self._load_image(index))
        return ret
        
    def __getitem__(self, index):
        try:
            return self.get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch


class ObjaverseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        val_num_workers: int = 2,
        training_data_list: str = None,
        sample_pcd_dir: str = None,
        image_data_json: str = None,
        image_size: int = 224,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False, 
        padding = True,
        padding_ratio_range=[1.15, 1.15]
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        self.training_data_list = training_data_list
        self.sample_pcd_dir = sample_pcd_dir
        self.image_data_json = image_data_json
        
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.val_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range

    def train_dataloader(self):
        asl_params = {
            "data_json": f'{self.training_data_list}/train.json',
            "sample_root": self.sample_pcd_dir,
            "image_path": self.image_data_json,
            "image_transform": self.train_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range,
        }
        dataset = ObjaverseDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        asl_params = {
            "data_json": f'{self.training_data_list}/val.json',
            "sample_root": self.sample_pcd_dir,
            "image_path": self.image_data_json,
            "image_transform": self.val_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal, 
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range,
        }
        dataset = ObjaverseDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
        )
