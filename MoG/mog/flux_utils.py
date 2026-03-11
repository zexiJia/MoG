"""Flux 模型的辅助工具函数

包含 latent packing/unpacking、image ID 准备和 mu 计算等。
"""

import torch
import math


def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    """将 (B, C, H, W) 的 latent 打包为 Flux 所需的 (B, L, D) 格式"""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def flux_unpack_latents(latents, height, width, vae_scale_factor):
    """将 Flux 的 (B, L, D) 格式解包为 (B, C, H, W)"""
    batch_size, num_patches, channels = latents.shape
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents


def flux_prepare_img_ids(batch_size, height, width, device, dtype):
    """生成 Flux 所需的 image positional IDs"""
    h_seq = height // 2
    w_seq = width // 2
    ids = torch.zeros(h_seq, w_seq, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h_seq, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w_seq, device=device, dtype=dtype)[None, :]
    ids = ids.reshape(1, -1, 3)
    ids = ids.repeat(batch_size, 1, 1)
    return ids


def calculate_flux_mu(scheduler, height, width):
    """计算 Flux 的时间步偏移参数 mu"""
    base_seq_len = getattr(scheduler.config, "base_image_seq_len", 256)
    max_seq_len = getattr(scheduler.config, "max_image_seq_len", 4096)
    base_shift = getattr(scheduler.config, "base_shift", 0.5)
    max_shift = getattr(scheduler.config, "max_shift", 1.15)
    seq_len = (height // 16) * (width // 16)
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = base_shift + slope * (seq_len - base_seq_len)
    return mu
