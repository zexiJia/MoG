<div align="center">

# 🧭 MoG

### Manifold-Optimal Guidance for Diffusion Models

<a href="" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-TBD-red?logo=arxiv" height="25" />
</a>
<a href="LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" height="25" />
</a>

**A training-free, plug-and-play guidance framework that replaces standard classifier-free guidance (CFG) with geometry-aware manifold-optimal guidance.**

</div>

---

## 🌟 Overview

MoG (**Manifold-Optimal Guidance**) is a training-free guidance framework for diffusion models.  
Instead of applying classifier-free guidance as a Euclidean extrapolation in ambient space, MoG reformulates guidance as a **local Riemannian optimal control problem**, yielding a geometry-aware update that better follows the data manifold and reduces common high-scale CFG artifacts.

This repository provides two practical variants:

| Method | Full Name | Level | Description | Direction |
|--------|-----------|-------|-------------|-----------|
| **MOG-Score** | Score-based Manifold-Optimal Guidance | Step-level guidance | Uses an anisotropic score-induced metric to suppress off-manifold drift during sampling | Better quality/alignment |
| **Auto-MOG** | Automatic Manifold-Optimal Guidance | Step-level guidance | Dynamically balances guidance strength via an energy-based scaling rule, reducing manual tuning | Better quality/alignment |

### Key Features

- **🧠 Geometry-Aware Guidance**: Replaces Euclidean CFG with a Riemannian natural-gradient style update
- **🪶 Training-Free**: No retraining, fine-tuning, or extra learnable parameters
- **🔌 Plug-and-Play**: Can be integrated into existing diffusion sampling pipelines with minimal code changes
- **🌍 Architecture-Agnostic**: Applicable to multiple diffusion backbones, including latent diffusion, DiT-style models, and flow-based transformers
- **⚡ Lightweight**: Introduces only negligible extra computation beyond standard CFG
- **🎛️ Auto-Adaptive**: Auto-MOG automatically calibrates guidance strength during sampling

---

## 📌 Motivation

Standard classifier-free guidance (CFG) works by linearly extrapolating between unconditional and conditional predictions.  
While effective, large guidance scales often push trajectories away from the high-density data manifold, leading to:

- oversaturation
- unnatural textures
- structural distortion
- poor fidelity/alignment trade-offs

MoG addresses this by applying guidance under a **Riemannian metric**, which penalizes off-manifold directions and preserves efficient progress along the manifold.

In practice, MoG replaces:

```python
guided = uncond + cfg_scale * (cond - uncond)

## Quick Start

### Installation

```bash
git clone https://github.com/KlingTeam/MoG.git
cd MoG
pip install -r requirements.txt
```

### Basic Usage (3 Lines of Code)

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

### Integration with Any Diffusion Pipeline

```python
import torch
from diffusers import StableDiffusion3Pipeline
from mog import MOGConfig, MOGProcessor

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
).to("cuda")

config = MOGConfig(mode="auto_mog")
processor = MOGProcessor(config)

# Manual denoising loop with MoG
prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
    prompt="A majestic mountain at sunrise", negative_prompt="", device="cuda"
)
latents = torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.bfloat16)
pipe.scheduler.set_timesteps(30, device="cuda")

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

image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
```

## Project Structure

```
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

## Supported Architectures

| Architecture | Example Script | Default Steps |
|-------------|----------------|---------------|
| **SDXL 1.0** (UNet, 4ch latent) | `examples/sample_sdxl.py` | 25 |
| **SD3 Medium** (MMDiT, 16ch latent) | `examples/sample_sd3.py` | 30 |
| **SD3.5 Medium** (MMDiT, 16ch latent) | `examples/sample_sd3.py` | 30 |
| **FLUX.1-dev** (Flux Transformer, packed latent) | `examples/sample_flux.py` | 28 |

## Configuration Reference

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

| Architecture | lambda_perp | auto_clamp_max | Notes |
|-------|------------|----------------|-------|
| SDXL | 5.0 | 20.0 | Default works well |
| SD3/3.5 | 5.0 | 15.0 | Slightly lower clamp |
| FLUX.1 | 5.0 | 20.0 | Default works well |

> In most cases, the default `MOGConfig(mode="auto_mog")` works out of the box.

## Toy Experiment

Reproduce the spiral manifold visualization from the paper:

```bash
cd experiments
python toy_experiment.py
# Outputs: toy_experiment.pdf, toy_experiment.png
```

This demonstrates how:
- **CFG** (red) diverges from the manifold due to Euclidean guidance
- **APG** (orange) stays on-manifold but converges slowly (tangent projection loses energy)
- **MOG** (blue) stays on-manifold AND converges fast (Riemannian preconditioning)

## How It Works

### Mathematical Formulation

Given the unconditional prediction $s_0 = s_\theta(x_t, \emptyset)$ and conditional prediction $s_c = s_\theta(x_t, c)$:

1. **Compute delta**: $\delta = s_c - s_0$

2. **Apply Riemannian metric inverse** $M^{-1}$:

$$v_{\text{nat}} = \frac{1}{\lambda_\perp} \delta + \left(\frac{1}{\lambda_\parallel} - \frac{1}{\lambda_\perp}\right) \frac{\langle s_0, \delta \rangle}{\|s_0\|^2} s_0$$

3. **Auto-scale** (energy balance):

$$\text{scale} = \gamma \cdot \frac{\|s_0\|}{\|v_{\text{nat}}\|}$$

4. **Final guidance**:

$$s_{\text{guided}} = s_0 + \text{clamp}(\text{scale}) \cdot v_{\text{nat}}$$

### Intuition

- $\lambda_\parallel < \lambda_\perp$: Moving along $s_0$ (parallel to manifold) is "cheaper" than moving perpendicular
- The metric $M^{-1}$ acts as a preconditioner: it rotates and scales the gradient to follow the manifold geometry
- Auto-scaling ensures the guidance magnitude adapts to the local curvature, eliminating the need for manual CFG tuning

## Acknowledgments

- **[Diffusers](https://github.com/huggingface/diffusers)** — Diffusion inference framework
- **[Stable Diffusion](https://github.com/Stability-AI/generative-models)** — Foundation generation architectures
- **[FLUX](https://github.com/black-forest-labs/flux)** — Rectified flow transformer

## License

This project is licensed under the [Apache License 2.0](LICENSE).
