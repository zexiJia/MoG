"""
MoG (Manifold-Optimal Guidance) — 核心算法模块

基于黎曼几何的扩散模型引导方法，替换传统 CFG，
在不增加推理开销的前提下，显著提升生成质量。
"""

from .processor import MOGConfig, MOGProcessor

__all__ = ["MOGConfig", "MOGProcessor"]
