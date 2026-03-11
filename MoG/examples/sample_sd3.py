"""SD3 / SD3.5 + MoG 推理示例

使用 Stable Diffusion 3 或 3.5 结合 MoG 引导生成高质量图像。
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusion3Pipeline

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mog import MOGConfig, MOGProcessor


@torch.no_grad()
def sample_sd3(pipeline, prompt, seed, mog_config, num_steps=30):
    """SD3/SD3.5 + MoG 采样

    Args:
        pipeline: StableDiffusion3Pipeline 实例
        prompt: 文本提示
        seed: 随机种子
        mog_config: MOGConfig 配置
        num_steps: 推理步数

    Returns:
        PIL Image
    """
    device = pipeline.device
    dtype = pipeline.transformer.dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    neg_prompt = "cartoon, anime, 3d render, low quality, distorted, bad anatomy, watermark, text"

    # SD3 使用三路 prompt 编码
    (p_emb, n_emb, p_pool, n_pool) = pipeline.encode_prompt(
        prompt=prompt, prompt_2=prompt, prompt_3=prompt,
        negative_prompt=neg_prompt, negative_prompt_2=neg_prompt, negative_prompt_3=neg_prompt,
        device=device
    )

    # SD3 使用 16 通道 latent
    num_ch = pipeline.transformer.config.in_channels
    latents = torch.randn((1, num_ch, 128, 128), generator=generator, device=device, dtype=dtype)

    pipeline.scheduler.set_timesteps(num_steps, device=device)
    processor = MOGProcessor(mog_config)

    # 去噪循环
    for t in pipeline.scheduler.timesteps:
        input_cat = torch.cat([latents] * 2)

        noise_pred = pipeline.transformer(
            hidden_states=input_cat,
            timestep=t.expand(input_cat.shape[0]),
            encoder_hidden_states=torch.cat([n_emb, p_emb], dim=0),
            pooled_projections=torch.cat([n_pool, p_pool], dim=0),
            return_dict=False
        )[0]

        noise_uncond, noise_cond = noise_pred.chunk(2)
        guided = processor.step(noise_uncond, noise_cond)
        latents = pipeline.scheduler.step(guided, t, latents, return_dict=False)[0]

    # 解码
    latents = latents / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def main():
    # =====================
    # 模型加载 (SD3.5 Medium)
    # =====================
    model_id = "stabilityai/stable-diffusion-3.5-medium"

    print(f"Loading SD3.5 from {model_id}...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("cuda")

    # =====================
    # 采样配置
    # =====================
    prompts = [
        "Close up portrait of an elderly woman with silver hair, softly lit by window light, detailed skin texture, realistic photography.",
        "Majestic snow-capped mountain reflected in a calm alpine lake at sunrise, photorealistic landscape.",
    ]
    seed = 42

    methods = [
        ("CFG (7.0)",  MOGConfig(mode="std_cfg", guidance_scale=7.0)),
        ("Auto-MOG",   MOGConfig(mode="auto_mog", auto_gamma=1.0)),
    ]

    # =====================
    # 生成对比
    # =====================
    fig, axes = plt.subplots(len(prompts), len(methods), figsize=(6 * len(methods), 6 * len(prompts)))

    for i, prompt in enumerate(prompts):
        for j, (name, config) in enumerate(methods):
            print(f"  [{i+1}/{len(prompts)}] Sampling '{prompt[:40]}...' with {name}")
            img = sample_sd3(pipeline, prompt, seed, config)
            axes[i, j].imshow(img)
            axes[i, j].set_title(name, fontsize=14, fontweight="bold")
            axes[i, j].axis("off")

    plt.tight_layout()
    output_path = "sd35_mog_comparison.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
