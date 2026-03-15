太好了，这些信息足够把 README 里的论文标题、作者、摘要式介绍和引用都补完整。下面给你一份可直接复制粘贴的 **完整单文件 `README.md`**。

````md
<div align="center">

# 🧭 MoG

### Manifold-Optimal Guidance for Diffusion Models

<a href="https://arxiv.org/abs/2603.11509" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.11509-b31b1b?logo=arxiv" height="25" />
</a>
<a href="https://doi.org/10.48550/arXiv.2603.11509" target="_blank">
    <img alt="DOI" src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2603.11509-blue" height="25" />
</a>
<a href="LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" height="25" />
</a>

<br />

**A training-free, plug-and-play guidance framework that replaces standard classifier-free guidance (CFG) with geometry-aware manifold-optimal guidance.**

<br />

[**Paper**](https://arxiv.org/abs/2603.11509) ·
[**DOI**](https://doi.org/10.48550/arXiv.2603.11509) ·
[**Citation**](#citation) ·
[**Quick Start**](#-quick-start)

</div>

---

## 🌟 Overview

MoG (**Manifold-Optimal Guidance**) is a training-free guidance framework for diffusion models.

Instead of applying classifier-free guidance as a Euclidean extrapolation in ambient space, MoG reformulates guidance as a **local Riemannian optimal control problem**, yielding a geometry-aware update that better follows the data manifold and reduces common high-scale CFG artifacts.

This repository provides two practical variants:

| Method | Full Name | Level | Description | Direction |
|--------|-----------|-------|-------------|-----------|
| **MOG-Score** | Score-based Manifold-Optimal Guidance | Step-level guidance | Uses an anisotropic score-induced metric to suppress off-manifold drift during sampling | Better quality / alignment |
| **Auto-MOG** | Automatic Manifold-Optimal Guidance | Step-level guidance | Dynamically balances guidance strength via an energy-based scaling rule, reducing manual tuning | Better quality / alignment |

### Key Features

- **🧠 Geometry-Aware Guidance**: Replaces Euclidean CFG with a Riemannian natural-gradient style update
- **🪶 Training-Free**: No retraining, fine-tuning, or extra learnable parameters
- **🔌 Plug-and-Play**: Integrates into existing diffusion sampling pipelines with minimal code changes
- **🌍 Architecture-Agnostic**: Applicable to latent diffusion, DiT-style models, and flow-based transformers
- **⚡ Lightweight**: Adds only negligible computation beyond standard CFG
- **🎛️ Auto-Adaptive**: Auto-MOG automatically calibrates guidance strength during sampling

---

## 📌 Motivation

Classifier-Free Guidance (CFG) serves as the de facto control mechanism for conditional diffusion, yet high guidance scales notoriously induce:

- oversaturation
- texture artifacts
- structural collapse

We attribute this failure to a geometric mismatch: standard CFG performs **Euclidean extrapolation in ambient space**, which can drive sampling trajectories away from the high-density data manifold.

MoG addresses this issue by applying guidance under a **Riemannian metric**, which penalizes off-manifold directions while preserving efficient progress along the manifold.

In practice, MoG replaces the standard CFG update:

```python
guided = uncond + cfg_scale * (cond - uncond)
````

with a geometry-aware manifold-optimal update derived from a local optimal control formulation.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/KlingTeam/MoG.git
cd MoG
pip install -r requirements.txt
```

### Basic Usage

```python
from mog import MOGConfig, MOGProcessor

# Create processor (auto-adaptive mode, no tuning needed)
config = MOGConfig(mode="auto_mog")
processor = MOGProcessor(config)

# In your denoising loop, replace CFG with:
# OLD: guided = uncond + cfg_scale * (cond - uncond)
# NEW:
guided = processor.step(uncond, cond)
```

### Integration with a Diffusion Pipeline

```python
import torch
from diffusers import StableDiffusion3Pipeline
from mog import MOGConfig, MOGProcessor

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
).to("cuda")

config = MOGConfig(mode="auto_mog")
processor = MOGProcessor(config)

# Encode prompt
prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
    prompt="A majestic mountain at sunrise",
    negative_prompt="",
    device="cuda"
)

# Prepare latents and scheduler
latents = torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.bfloat16)
pipe.scheduler.set_timesteps(30, device="cuda")

# Manual denoising loop with MoG
for t in pipe.scheduler.timesteps:
    input_cat = torch.cat([latents] * 2)
    noise_pred = pipe.transformer(
        hidden_states=input_cat,
        timestep=t.expand(2),
        encoder_hidden_states=torch.cat([neg_embeds, prompt_embeds]),
        pooled_projections=torch.cat([neg_pooled, pooled]),
        return_dict=False
    )[0]

    uncond, cond = noise_pred.chunk(2)
    guided = processor.step(uncond, cond)  # <-- MoG replaces CFG here
    latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]

image = pipe.vae.decode(
    latents / pipe.vae.config.scaling_factor,
    return_dict=False
)[0]
```

---

## 🧱 Project Structure

```text
MoG/
├── mog/                        # Core MoG algorithm
│   ├── __init__.py             # Package exports
│   ├── processor.py            # MOGConfig & MOGProcessor
│   └── flux_utils.py           # FLUX-specific utilities
│
├── examples/                   # Ready-to-run inference scripts
│   ├── sample_sdxl.py          # SDXL + MoG
│   ├── sample_sd3.py           # SD3/SD3.5 + MoG
│   └── sample_flux.py          # FLUX.1-dev + MoG
│
├── experiments/                # Paper reproduction
│   └── toy_experiment.py       # Spiral manifold visualization (Figure 2)
│
├── assets/                     # Documentation images
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🏗️ Supported Architectures

| Architecture                                     | Example Script            | Default Steps |
| ------------------------------------------------ | ------------------------- | ------------- |
| **SDXL 1.0** (UNet, 4-channel latent)            | `examples/sample_sdxl.py` | 25            |
| **SD3 Medium** (MMDiT, 16-channel latent)        | `examples/sample_sd3.py`  | 30            |
| **SD3.5 Medium** (MMDiT, 16-channel latent)      | `examples/sample_sd3.py`  | 30            |
| **FLUX.1-dev** (Flux Transformer, packed latent) | `examples/sample_flux.py` | 28            |

---

## ⚙️ Configuration Reference

```python
from mog import MOGConfig

# Auto-MOG (recommended — no tuning needed)
config = MOGConfig(
    mode="auto_mog",        # Automatic energy-balanced guidance
    lambda_parallel=1.0,    # Parallel direction metric (smaller = faster)
    lambda_perp=5.0,        # Perpendicular direction metric (larger = more stable)
    auto_gamma=1.0,         # Energy balance coefficient
    auto_clamp_min=1.0,     # Min dynamic scale
    auto_clamp_max=20.0,    # Max dynamic scale
)

# Standard CFG (baseline)
config = MOGConfig(
    mode="std_cfg",
    guidance_scale=7.5,     # Traditional CFG scale
)
```

### Recommended Hyperparameters

| Architecture | `lambda_perp` | `auto_clamp_max` | Notes                |
| ------------ | ------------- | ---------------- | -------------------- |
| SDXL         | 5.0           | 20.0             | Default works well   |
| SD3 / SD3.5  | 5.0           | 15.0             | Slightly lower clamp |
| FLUX.1       | 5.0           | 20.0             | Default works well   |

> In most cases, the default `MOGConfig(mode="auto_mog")` works out of the box.

---

## 🧪 Toy Experiment

Reproduce the spiral manifold visualization from the paper:

```bash
cd experiments
python toy_experiment.py
# Outputs: toy_experiment.pdf, toy_experiment.png
```

This experiment illustrates how:

* **CFG** (red) diverges from the manifold due to Euclidean guidance
* **APG** (orange) stays on-manifold but converges slowly because tangent projection loses energy
* **MOG** (blue) stays on-manifold **and** converges quickly through Riemannian preconditioning

---

## 🧠 How It Works

### Mathematical Formulation

Given the unconditional prediction $s_0 = s_\theta(x_t, \emptyset)$ and conditional prediction $s_c = s_\theta(x_t, c)$:

1. Compute the conditional difference:

   $$
   \delta = s_c - s_0
   $$

2. Apply the Riemannian metric inverse $M^{-1}$:

   $$
   v_{\text{nat}} =
   \frac{1}{\lambda_\perp}\delta
   +
   \left(
   \frac{1}{\lambda_\parallel} - \frac{1}{\lambda_\perp}
   \right)
   \frac{\langle s_0, \delta \rangle}{|s_0|^2} s_0
   $$

3. Compute automatic energy balancing:

   $$
   \text{scale} = \gamma \cdot \frac{|s_0|}{|v_{\text{nat}}|}
   $$

4. Produce the final guided score:

   $$
   s_{\text{guided}} = s_0 + \text{clamp}(\text{scale}) \cdot v_{\text{nat}}
   $$

### Intuition

* $\lambda_\parallel < \lambda_\perp$ means moving along the score direction is cheaper than moving perpendicular to it
* The metric inverse $M^{-1}$ acts as a preconditioner that rotates and rescales the guidance direction
* Auto-scaling adapts the guidance magnitude to local geometry, reducing the need for manual CFG tuning

---

## 📚 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{jia2026manifold,
  title={Manifold-Optimal Guidance: A Unified Riemannian Control View of Diffusion Guidance},
  author={Jia, Zexi and Luo, Pengcheng and Fang, Zhengyao and Zhang, Jinchao and Zhou, Jie},
  journal={arXiv preprint arXiv:2603.11509},
  year={2026},
  doi={10.48550/arXiv.2603.11509},
  url={https://arxiv.org/abs/2603.11509}
}
```


## 🙏 Acknowledgments

* **[Diffusers](https://github.com/huggingface/diffusers)** — Diffusion inference framework
* **[Stable Diffusion](https://github.com/Stability-AI/generative-models)** — Foundation generation architectures
* **[FLUX](https://github.com/black-forest-labs/flux)** — Rectified flow transformer

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

`Updates`、`TODO`、`Results`、`BibTeX` 高亮区、`Teaser Figure` 占位、`Contact`。
```
