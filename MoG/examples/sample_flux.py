"""FLUX.1-dev + MoG 推理示例

使用 FLUX.1-dev 结合 MoG 引导生成高质量图像。
FLUX 使用 packed latent 格式，需要特殊的手动循环。
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from diffusers import FluxPipeline

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mog import MOGConfig, MOGProcessor
from mog.flux_utils import (
    flux_pack_latents,
    flux_unpack_latents,
    flux_prepare_img_ids,
    calculate_flux_mu,
)


@torch.no_grad()
def sample_flux(pipeline, prompt, seed, config: MOGConfig, height=512, width=512, num_steps=28):
    """FLUX + MoG 采样

    Args:
        pipeline: FluxPipeline 实例
        prompt: 文本提示
        seed: 随机种子
        config: MOGConfig 配置
        height: 图像高度
        width: 图像宽度
        num_steps: 推理步数

    Returns:
        PIL Image
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 标准 CFG 模式: 直接调用官方 Pipeline
    if config.mode == "std_cfg":
        return pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=config.guidance_scale,
            num_inference_steps=num_steps,
            max_sequence_length=512,
            generator=generator,
            output_type="pil"
        ).images[0]

    # Auto-MOG 模式: 手动循环
    device = torch.device("cuda")
    dtype = pipeline.transformer.dtype

    # 编码文本
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, device=device
    )
    neg_prompt_embeds, neg_pooled_prompt_embeds, neg_text_ids = pipeline.encode_prompt(
        prompt="", prompt_2=None, device=device
    )

    # 初始化 latent
    num_channels_latents = 16
    vae_scale_factor = 8
    shape = (1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)

    # Pack latents & 准备 IDs
    latents = flux_pack_latents(
        latents, 1, num_channels_latents,
        height // vae_scale_factor, width // vae_scale_factor
    )
    img_ids = flux_prepare_img_ids(
        1, height // vae_scale_factor, width // vae_scale_factor, device, dtype
    )

    # 设置调度器
    mu = calculate_flux_mu(pipeline.scheduler, height, width)
    pipeline.scheduler.set_timesteps(num_steps, device=device, mu=mu)

    processor = MOGProcessor(config)
    guidance_vec = torch.tensor([3.5], device=device, dtype=dtype).expand(2)

    # 去噪循环
    for t in pipeline.scheduler.timesteps:
        latents_input = torch.cat([latents] * 2)
        prompt_embeds_input = torch.cat([neg_prompt_embeds, prompt_embeds])
        pooled_input = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
        text_ids_input = torch.cat([neg_text_ids, text_ids])
        img_ids_input = torch.cat([img_ids, img_ids])

        t_expand = t.to(device=device, dtype=dtype).expand(latents_input.shape[0]) / 1000.0

        noise_pred = pipeline.transformer(
            hidden_states=latents_input,
            timestep=t_expand,
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_input,
            txt_ids=text_ids_input,
            img_ids=img_ids_input,
            guidance=guidance_vec,
            return_dict=False
        )[0]

        noise_uncond, noise_cond = noise_pred.chunk(2)
        guided = processor.step(noise_uncond, noise_cond)

        latents = pipeline.scheduler.step(guided, t, latents, return_dict=False)[0]
        torch.cuda.empty_cache()

    # 解码
    latents = flux_unpack_latents(
        latents, height // vae_scale_factor, width // vae_scale_factor, vae_scale_factor
    )
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(device)

    image = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def main():
    # =====================
    # 模型加载
    # =====================
    model_id = "black-forest-labs/FLUX.1-dev"

    print(f"Loading FLUX.1-dev from {model_id}...")
    pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()

    # =====================
    # 采样配置
    # =====================
    prompts = [
        "Realistic photography of Tokyo streets at night in rain, neon signs reflecting on wet asphalt, cinematic lighting, 8k.",
        "A vibrant Betta fish swimming in crystal clear water, long flowing fins, macro underwater photography, high detail.",
    ]
    seed = 42

    methods = [
        ("CFG (3.5)", MOGConfig(mode="std_cfg", guidance_scale=3.5)),
        ("Auto-MOG",  MOGConfig(mode="auto_mog", auto_gamma=1.0)),
    ]

    # =====================
    # 生成对比
    # =====================
    fig, axes = plt.subplots(len(prompts), len(methods), figsize=(6 * len(methods), 6 * len(prompts)))

    for i, prompt in enumerate(prompts):
        for j, (name, config) in enumerate(methods):
            print(f"  [{i+1}/{len(prompts)}] Sampling '{prompt[:40]}...' with {name}")
            img = sample_flux(pipeline, prompt, seed, config)
            axes[i, j].imshow(img)
            axes[i, j].set_title(name, fontsize=14, fontweight="bold")
            axes[i, j].axis("off")

    plt.tight_layout()
    output_path = "flux_mog_comparison.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
