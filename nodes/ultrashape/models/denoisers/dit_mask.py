# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================

# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

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
import yaml
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .moe_layers import MoEBlock
from ...utils import logger, synchronize_timer, smart_load_model

# flash_attn is optional - fallback to PyTorch SDPA if not available
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None
    print("[UltraShape] flash_attn not found, using PyTorch SDPA (may be slower for variable-length attention)")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Timesteps(nn.Module):
    def __init__(self,
                 num_channels: int,
                 downscale_freq_shift: float = 0.0,
                 scale: int = 1,
                 max_period: int = 10000
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):

        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        # t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)


    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2) - may contain padding (marked with -1)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]

        # Detect padding tokens: check if all values in the feature dimension are -1
        # y_mask: [b, s2], True for valid tokens, False for padding
        y_mask = (y != -1).any(dim=-1)  # [b, s2]
        has_padding = not y_mask.all()

        _, s2, c = y.shape  # [b, s2, 1024]
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        k = k.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]
        v = v.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if has_padding and HAS_FLASH_ATTN:
            seqlens_k = y_mask.sum(dim=1).int()
            q_flat = q.reshape(-1, self.num_heads, self.head_dim)

            # For k, v: only keep valid tokens (remove padding)
            # Create indices for valid tokens
            valid_indices = []
            cu_seqlens_k = [0]
            for i in range(b):
                valid_len = seqlens_k[i].item()
                batch_indices = torch.arange(valid_len, device=y.device) + i * s2
                valid_indices.append(batch_indices)
                cu_seqlens_k.append(cu_seqlens_k[-1] + valid_len)

            valid_indices = torch.cat(valid_indices)
            k_flat = k.reshape(b * s2, self.num_heads, self.head_dim)[valid_indices]  # [total_k, h, d]
            v_flat = v.reshape(b * s2, self.num_heads, self.head_dim)[valid_indices]  # [total_k, h, d]

            # Create cumulative sequence lengths
            cu_seqlens_q = torch.arange(0, (b + 1) * s1, s1, dtype=torch.int32, device=x.device)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=x.device)

            # Call flash attention varlen
            q_flat = q_flat.to(torch.bfloat16)
            k_flat = k_flat.to(torch.bfloat16)
            v_flat = v_flat.to(torch.bfloat16)

            context = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q, cu_seqlens_k,
                s1, seqlens_k.max().item(),
                dropout_p=0.0,
                softmax_scale=None,
                causal=False
            )
            context = context.reshape(b, s1, -1)
        elif has_padding:
            # Fallback for variable-length attention without flash_attn
            # Use masking with PyTorch SDPA
            q_t = rearrange(q, 'b n h d -> b h n d')
            k_t = rearrange(k, 'b n h d -> b h n d')
            v_t = rearrange(v, 'b n h d -> b h n d')

            # Create attention mask from y_mask: [b, s2] -> [b, 1, 1, s2]
            attn_mask = y_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, s1, -1)
            # Convert to float mask for SDPA (0 for valid, -inf for masked)
            attn_mask = torch.where(attn_mask, torch.zeros_like(attn_mask, dtype=q.dtype),
                                    torch.full_like(attn_mask, float('-inf'), dtype=q.dtype))

            context = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask
            ).transpose(1, 2).reshape(b, s1, -1)
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads), (q, k, v))
                
                attn_mask = None
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask
                ).transpose(1, 2).reshape(b, s1, -1)

        out = self.out_proj(context)

        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    # def forward(self, x):
    def forward(self, x, rotary_cos=None, rotary_sin=None):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, s, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        # ========================= Apply RoPE =========================
        if rotary_cos is not None:
            q = apply_rotary_emb(q, rotary_cos, rotary_sin)
            k = apply_rotary_emb(k, rotary_cos, rotary_sin)
        # ==============================================================

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        init_scale=1.0,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               norm_layer=qk_norm_layer)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer, init_scale=init_scale)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.use_moe = use_moe
        if self.use_moe:
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None, rotary_cos=None, rotary_sin=None):

        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-Attention
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        attn_out = self.attn1(self.norm1(x), rotary_cos=rotary_cos, rotary_sin=rotary_sin)

        x = x + attn_out

        # Cross-Attention
        x = x + self.attn2(self.norm2(x), text_states)

        # FFN Layer
        mlp_inputs = self.norm3(x)

        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)

        return x


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)

        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


class RefineDiT(nn.Module):

    @classmethod
    @synchronize_timer('Refine Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading model from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        if 'model' in ckpt:
            ckpt = ckpt['model']
        if 'model' in config:
            config = config['model']

        model_kwargs = config['params']
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt)
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type='layer',
        qk_norm_type='rms',
        qk_norm=False,
        text_len=257,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        voxel_query_res: int = 128,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.voxel_query_res = voxel_query_res
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim

        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.text_len = text_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=context_dim,
                            qk_norm=qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > depth // 2,
                            qkv_bias=qkv_bias,
                            use_moe=True if depth - layer <= num_moe_layers else False,
                            num_experts=num_experts,
                            moe_top_k=moe_top_k
                            )
            for layer in range(depth)
        ])
        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, contexts, **kwargs):
        cond = contexts['main']

        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)
        c = t

        ##########################################
        head_dim = self.blocks[0].attn1.head_dim
        num_cond_tokens = c.shape[1] if c.dim() == 3 else 1

        device = x.device
        cond_cos = torch.ones(x.shape[0], num_cond_tokens, head_dim, device=device)
        cond_sin = torch.zeros(x.shape[0], num_cond_tokens, head_dim, device=device)

        voxel_cond = kwargs.get('voxel_cond')
        rotary_cos_vox, rotary_sin_vox = precompute_freqs_cis_3d_interpolated(
            head_dim, voxel_cond, current_res=self.voxel_query_res)

        rotary_cos = torch.cat([cond_cos, rotary_cos_vox], dim=1)
        rotary_sin = torch.cat([cond_sin, rotary_sin_vox], dim=1)
        ##########################################

        x = torch.cat([c, x], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, rotary_cos=rotary_cos, rotary_sin=rotary_sin, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x


def apply_rotary_emb(x, cos, sin):
    """
    x: [B, H, N, D]
    cos, sin: [B, N, D]
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
        
    return (x * cos) + (rotate_half(x) * sin)


def precompute_freqs_cis_3d(dim: int, grid_indices: torch.Tensor, theta: float = 10000.0):
    """
    grid_indices: [B, N, 3] voxel idx
    """
    dim_x = dim // 3
    dim_y = dim // 3
    dim_z = dim - dim_x - dim_y 
    
    device = grid_indices.device
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim_x, 2, device=device).float() / dim_x))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim_y, 2, device=device).float() / dim_y))
    freqs_z = 1.0 / (theta ** (torch.arange(0, dim_z, 2, device=device).float() / dim_z))
    
    x_idx = grid_indices[..., 0].float()
    y_idx = grid_indices[..., 1].float()
    z_idx = grid_indices[..., 2].float()

    args_x = x_idx.unsqueeze(-1) * freqs_x.unsqueeze(0).unsqueeze(0)
    args_y = y_idx.unsqueeze(-1) * freqs_y.unsqueeze(0).unsqueeze(0)
    args_z = z_idx.unsqueeze(-1) * freqs_z.unsqueeze(0).unsqueeze(0)

    args = torch.cat([args_x, args_y, args_z], dim=-1)
    args = torch.cat([args, args], dim=-1)

    return torch.cos(args), torch.sin(args)


def precompute_freqs_cis_3d_interpolated(
    dim: int,
    grid_indices: torch.Tensor,
    theta: float = 10000.0,
    trained_res: float = 128.0,  # training resolution
    current_res: float = 256.0,  # inference resolution
):
    scale_factor = current_res / trained_res

    dim_x = dim // 3
    dim_y = dim // 3
    dim_z = dim - dim_x - dim_y

    device = grid_indices.device

    freqs_x = 1.0 / (theta ** (torch.arange(0, dim_x, 2, device=device).float() / dim_x))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim_y, 2, device=device).float() / dim_y))
    freqs_z = 1.0 / (theta ** (torch.arange(0, dim_z, 2, device=device).float() / dim_z))

    num_freqs_x = dim_x // 2 + (dim_x % 2)
    num_freqs_y = dim_y // 2 + (dim_y % 2)
    target_len = dim // 2
    freqs_x = freqs_x[:num_freqs_x]
    freqs_y = freqs_y[:num_freqs_y]
    freqs_z = freqs_z[:(target_len - len(freqs_x) - len(freqs_y))]

    input_x = grid_indices[..., 0].float()
    input_y = grid_indices[..., 1].float()
    input_z = grid_indices[..., 2].float()

    # Apply Scaling
    pos_x = input_x / scale_factor
    pos_y = input_y / scale_factor
    pos_z = input_z / scale_factor

    # pos * freq
    args_x = pos_x.unsqueeze(-1) * freqs_x.unsqueeze(0).unsqueeze(0)
    args_y = pos_y.unsqueeze(-1) * freqs_y.unsqueeze(0).unsqueeze(0)
    args_z = pos_z.unsqueeze(-1) * freqs_z.unsqueeze(0).unsqueeze(0)

    args = torch.cat([args_x, args_y, args_z], dim=-1)
    args = torch.cat([args, args], dim=-1)

    return torch.cos(args), torch.sin(args)
