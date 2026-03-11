"""
MoG (Manifold-Optimal Guidance) — 核心处理器

实现了 Riemannian Natural Gradient 引导算法：
  1. 将无条件与条件预测的差分 (delta) 分解为平行与垂直分量
  2. 通过各向异性度量张量 M 进行重加权（抑制垂直分量，放大平行分量）
  3. Auto-MOG 模式下自动计算能量平衡缩放因子，无需手动调参
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MOGConfig:
    """MoG 配置参数

    Attributes:
        mode: 引导模式
            - "std_cfg": 标准 Classifier-Free Guidance
            - "auto_mog": 自适应 MoG (推荐, 无需调参)
        guidance_scale: CFG 引导强度 (仅在 std_cfg 模式下生效)
        lambda_parallel: 平行方向度量参数 (越小 → 平行方向移动越快)
        lambda_perp: 垂直方向度量参数 (越大 → 垂直方向移动越慢)
        auto_gamma: Auto-MOG 能量平衡系数
        auto_clamp_min: 动态缩放因子下限
        auto_clamp_max: 动态缩放因子上限
    """
    mode: str = "auto_mog"
    guidance_scale: float = 4.5
    lambda_parallel: float = 1.0
    lambda_perp: float = 5.0
    auto_gamma: float = 1.0
    auto_clamp_min: float = 1.0
    auto_clamp_max: float = 20.0


class MOGProcessor:
    """MoG 引导处理器

    用于在扩散模型的去噪循环中替换标准 CFG 的引导计算。

    支持的架构:
        - UNet-based (SDXL): delta/s0 维度为 (B, C, H, W)
        - Transformer-based (SD3/3.5): delta/s0 维度为 (B, C, H, W)
        - Flux: delta/s0 维度为 (B, L, D)

    Usage:
        >>> config = MOGConfig(mode="auto_mog")
        >>> processor = MOGProcessor(config)
        >>> guided = processor.step(noise_uncond, noise_cond)
    """

    def __init__(self, config: MOGConfig):
        self.cfg = config

    def _compute_rmog_score(self, delta: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        """计算 Riemannian Natural Gradient 方向

        v_nat = M^{-1} * delta
             = (1/λ_perp) * delta + (1/λ_para - 1/λ_perp) * <s0, delta>/<s0, s0> * s0

        Args:
            delta: 条件与无条件预测的差分 (cond - uncond)
            s0: 无条件预测 (作为当前流形方向的估计)

        Returns:
            v_nat: 黎曼自然梯度方向
        """
        delta_f, s0_f = delta.float(), s0.float()

        # 自动检测维度格式
        # UNet/Transformer: (B, C, H, W) → sum over (C, H, W)
        # Flux packed: (B, L, D) → sum over (L, D)
        reduce_dims = tuple(range(1, delta_f.ndim))

        s0_norm_sq = torch.sum(s0_f ** 2, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        dot_prod = torch.sum(s0_f * delta_f, dim=reduce_dims, keepdim=True)

        c_perp = 1.0 / self.cfg.lambda_perp
        c_para = 1.0 / self.cfg.lambda_parallel

        v_nat = (c_perp * delta_f) + (c_para - c_perp) * (dot_prod / s0_norm_sq) * s0_f
        return v_nat.to(delta.dtype)

    def _compute_energy_norm(self, vec: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        """计算各向异性能量范数 ||vec||_M

        ||vec||_M = sqrt(λ_para * v_para^2 + λ_perp * v_perp^2)

        Args:
            vec: 待计算范数的向量
            s0: 参考方向 (用于定义平行/垂直分解)

        Returns:
            能量范数，shape 为 (B, 1, ..., 1)
        """
        vec_f, s0_f = vec.float(), s0.float()
        reduce_dims = tuple(range(1, vec_f.ndim))

        s0_norm = torch.norm(s0_f, p=2, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        s0_unit = s0_f / s0_norm

        v_para = torch.sum(vec_f * s0_unit, dim=reduce_dims, keepdim=True)
        v_perp_sq = torch.sum(vec_f ** 2, dim=reduce_dims, keepdim=True) - v_para ** 2

        energy = self.cfg.lambda_parallel * (v_para ** 2) + self.cfg.lambda_perp * v_perp_sq.abs()
        return torch.sqrt(energy + 1e-8)

    def _compute_l2_norm(self, vec: torch.Tensor) -> torch.Tensor:
        """计算 L2 范数"""
        reduce_dims = tuple(range(1, vec.ndim))
        return torch.norm(vec.float(), p=2, dim=reduce_dims, keepdim=True)

    def step(self, uncond: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """执行一步 MoG 引导

        Args:
            uncond: 无条件模型预测
            cond: 条件模型预测

        Returns:
            引导后的预测结果
        """
        delta = cond - uncond

        if self.cfg.mode == "std_cfg":
            # 标准 Classifier-Free Guidance
            return uncond + self.cfg.guidance_scale * delta

        elif self.cfg.mode == "auto_mog":
            # Auto-MOG: 自适应能量平衡
            v_nat = self._compute_rmog_score(delta, uncond)

            # 计算能量平衡缩放因子: scale = γ * ||uncond||_M / ||v_nat||_M
            norm_s0 = self._compute_l2_norm(uncond)
            norm_v = self._compute_l2_norm(v_nat)

            scale = self.cfg.auto_gamma * (norm_s0 / (norm_v + 1e-8))
            scale = torch.clamp(
                scale,
                self.cfg.auto_clamp_min,
                self.cfg.auto_clamp_max
            ).to(uncond.dtype)

            return uncond + scale * v_nat

        else:
            raise ValueError(
                f"Unknown mode: '{self.cfg.mode}'. "
                f"Supported modes: 'std_cfg', 'auto_mog'"
            )
