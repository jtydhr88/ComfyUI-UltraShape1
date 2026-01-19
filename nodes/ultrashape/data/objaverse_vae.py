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


import os
import cv2
import json
import math
import random
import imageio
import pickle
import numpy as np
from PIL import Image
import pandas as pd
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pytorch_lightning.utilities import rank_zero_info
from ultrashape.utils.typing import *

class ObjaverseDataset(Dataset):
    def __init__(
        self,
        data_json,
        sample_root,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sup_near_uni_size: int = 4096,
        sup_near_sharp_size: int = 4096,
        sup_space_size: int = 4096,
        tsdf_threshold: float = 0.05,
        sharpedge_label: bool = False,
        return_normal: bool = False,
    ):
        super().__init__()

        self.uids = json.load(open(data_json))
        self.sample_root = sample_root
        
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.sup_near_uni_size = sup_near_uni_size
        self.sup_near_sharp_size = sup_near_sharp_size
        self.sup_space_size = sup_space_size
        self.tsdf_threshold = tsdf_threshold
        
        print(f"Loaded {len(self.uids)} uids from {data_json}.")

        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Dataset Infos:')
        rank_zero_info(f'# of 3D file: {len(self.uids)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'# of Uniform Near-Surface Sup-Points: {self.sup_near_uni_size}')
        rank_zero_info(f'# of Sharpedge Near-Surface Sup-Points: {self.sup_near_sharp_size}')
        rank_zero_info(f'# of Random Space Sup-Points: {self.sup_space_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'*' * 50)

    def __len__(self):
        return len(self.uids)

    def _load_shape(self, index: int) -> Dict[str, Any]:
        rng = np.random.default_rng()

        data = np.load(f'{self.sample_root}/{self.uids[index]}.npz')
        
        ##################### sup pcd&sdf ######################
        uniform_near_points =  (np.asarray(data['uniform_near_points'])-0.5) * 2
        curvature_near_points = (np.asarray(data['curvature_near_points'])-0.5) * 2
        space_points = (np.asarray(data['space_points'])-0.5) * 2 
        uniform_near_sdf = np.asarray(data['uniform_near_sdf']) * 2 
        curvature_near_sdf = np.asarray(data['curvature_near_sdf']) * 2
        space_sdf = np.asarray(data['space_sdf']) * 2 

        uni_noisy_idx = rng.choice(uniform_near_points.shape[0], self.sup_near_uni_size, replace=False)
        cur_noisy_idx = rng.choice(curvature_near_points.shape[0], self.sup_near_sharp_size, replace=False)
        space_idx = rng.choice(space_points.shape[0], self.sup_space_size, replace=False)

        uniform_near_points = uniform_near_points[uni_noisy_idx]
        curvature_near_points = curvature_near_points[cur_noisy_idx]
        space_points = space_points[space_idx]
        uniform_near_sdf = uniform_near_sdf[uni_noisy_idx]
        curvature_near_sdf = curvature_near_sdf[cur_noisy_idx]
        space_sdf = space_sdf[space_idx]

        uniform_near_sdf, curvature_near_sdf, space_sdf = map(self._clip_to_tsdf, (uniform_near_sdf, curvature_near_sdf, space_sdf))

        surface_og = (np.asarray(data['clean_surface_points'])-0.5) * 2 
        normal = np.asarray(data['clean_surface_normals'])
        surface_og_n = np.concatenate([surface_og, normal], axis=1) 
        rng = np.random.default_rng()

        # hard code: first 300k are uniform, last 300k are sharp
        assert surface_og_n.shape[0] == 600000, f"assume that suface points = 30w uniform + 30w curvature, but {len(surface_og_n)=}"
        coarse_surface = surface_og_n[:300000]
        sharp_surface = surface_og_n[300000:]

        surface_normal = []
        
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
                "sup_near_uniform": np.concatenate([uniform_near_points, uniform_near_sdf[...,None]], axis=1), 
                "sup_near_sharp": np.concatenate([curvature_near_points, curvature_near_sdf[...,None]], axis=1), 
                "sup_space": np.concatenate([space_points, space_sdf[...,None]], axis=1),
                "geo_points": geo_points
            }
        return ret
    
    def _clip_to_tsdf(self, sdf: np.array):
        nan_mask = np.isnan(sdf)
        if np.any(nan_mask):
            sdf=np.nan_to_num(sdf, nan=1.0, posinf=1.0, neginf=-1.0)
        return sdf.flatten().astype(np.float32).clip(-self.tsdf_threshold, self.tsdf_threshold) / self.tsdf_threshold

    def get_data(self, index):
        ret = self._load_shape(index)
        return ret
        
    def __getitem__(self, index):
        return self.get_data(index)

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
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sup_near_uni_size: int = 4096,
        sup_near_sharp_size: int = 4096,
        sup_space_size: int = 4096,
        tsdf_threshold: float = 0.05,
        sharpedge_label: bool = False,
        return_normal: bool = False, 
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        self.training_data_list = training_data_list
        self.sample_pcd_dir = sample_pcd_dir

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.sup_near_uni_size = sup_near_uni_size
        self.sup_near_sharp_size = sup_near_sharp_size
        self.sup_space_size = sup_space_size
        self.tsdf_threshold = tsdf_threshold

    def train_dataloader(self):
        asl_params = {
            "data_json": f'{self.training_data_list}/train.json',
            "sample_root": self.sample_pcd_dir,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sup_near_uni_size": self.sup_near_uni_size,
            "sup_near_sharp_size": self.sup_near_sharp_size,
            "sup_space_size": self.sup_space_size,
            "tsdf_threshold": self.tsdf_threshold,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
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
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sup_near_uni_size": self.sup_near_uni_size,
            "sup_near_sharp_size": self.sup_near_sharp_size,
            "sup_space_size": self.sup_space_size,
            "tsdf_threshold": self.tsdf_threshold,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal, 
        }
        dataset = ObjaverseDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
        )
