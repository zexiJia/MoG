"""SDXL + MoG 推理示例

使用 Stable Diffusion XL 结合 MoG 引导生成高质量图像。
对比标准 CFG 与 Auto-MOG 在不同去噪步骤下的效果。
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mog import MOGConfig, MOGProcessor


def get_sdxl_time_ids(original_size, crops_coords_top_left, target_size, dtype, device, batch_size):
    """生成 SDXL 所需的时间嵌入 IDs"""
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids


@torch.no_grad()
def sample_sdxl(pipeline, prompt, negative_prompt, seed, mog_config, num_steps=25):
    """SDXL + MoG 采样

    Args:
        pipeline: StableDiffusionXLPipeline 实例
        prompt: 文本提示
        negative_prompt: 负向提示
        seed: 随机种子
        mog_config: MOGConfig 配置
        num_steps: 推理步数

    Returns:
        PIL Image
    """
    device = pipeline.device
    dtype = pipeline.unet.dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    # 编码文本
    (prompt_embeds, negative_prompt_embeds, 
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, device=device, negative_prompt=negative_prompt
    )

    # 初始化 latent
    latents = torch.randn((1, 4, 128, 128), generator=generator, device=device, dtype=dtype)

    # 设置调度器
    pipeline.scheduler.set_timesteps(num_steps, device=device)
    latents = latents * pipeline.scheduler.init_noise_sigma

    # 准备额外条件
    add_time_ids = get_sdxl_time_ids((1024, 1024), (0, 0), (1024, 1024), dtype, device, 1)
    prompt_embeds_cat = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds_cat = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    add_time_ids_cat = torch.cat([add_time_ids, add_time_ids], dim=0)

    processor = MOGProcessor(mog_config)

    # 去噪循环
    for t in pipeline.scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipeline.unet(
            latent_model_input, t,
            encoder_hidden_states=prompt_embeds_cat,
            added_cond_kwargs={"text_embeds": add_text_embeds_cat, "time_ids": add_time_ids_cat},
            return_dict=False
        )[0]

        noise_uncond, noise_cond = noise_pred.chunk(2)
        guided = processor.step(noise_uncond, noise_cond)
        latents = pipeline.scheduler.step(guided, t, latents, return_dict=False)[0]

    # 解码
    pipeline.vae.to(dtype=torch.float32)
    latents_fp32 = latents.to(dtype=torch.float32) / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents_fp32, return_dict=False)[0]
    pipeline.vae.to(dtype=torch.float16)

    return pipeline.image_processor.postprocess(image, output_type="pil")[0]


def main():
    # =====================
    # 模型加载
    # =====================
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    print(f"Loading SDXL from {model_id}...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    # =====================
    # 采样配置
    # =====================
    prompt = "A cute cat with blue eyes, sitting on a windowsill, photorealistic, sunny day, high detail"
    neg_prompt = "cartoon, drawing, anime, low quality, blurry, distorted, ugly"
    seed = 1337

    methods = [
        ("Standard CFG", MOGConfig(mode="std_cfg", guidance_scale=8.0)),
        ("Auto-MOG",     MOGConfig(mode="auto_mog", auto_gamma=1.0)),
    ]

    # =====================
    # 生成对比
    # =====================
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]

    for idx, (name, config) in enumerate(methods):
        print(f"  Sampling with {name}...")
        img = sample_sdxl(pipeline, prompt, neg_prompt, seed, config, num_steps=25)
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=14, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle(f"Prompt: {prompt}", fontsize=10, y=0.02)
    plt.tight_layout()

    output_path = "sdxl_mog_comparison.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
