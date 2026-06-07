#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sd_lora_hard_ici_sampling_refined.py

使用困难样本和 Stable Diffusion LoRA 模型生成类别条件图像的脚本。

核心工作流：
1. 读取困难样本 CSV。
2. 按 (true_label_name, nearest_wrong_class_name) 分组样本。
3. 使用 DDIM 反演将两个源图像映射到潜在空间。
4. 对两个反演得到的潜变量做圆形插值（circle interpolation）。
5. 使用真实类别的提示词对插值后的潜变量进行 DDIM 去噪生成图像。
6. 保存生成的图像及其元数据。

"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from peft import LoraConfig
except ImportError:  # Give a clearer error when build_sd_lora_unet is called.
    LoraConfig = None


ISIC_PROMPTS = {
    "MEL": "a dermoscopic image of melanoma",
    "NV": "a dermoscopic image of melanocytic nevus",
    "BCC": "a dermoscopic image of basal cell carcinoma",
    "AKIEC": "a dermoscopic image of actinic keratosis or intraepithelial carcinoma",
    "BKL": "a dermoscopic image of benign keratosis-like lesion",
    "DF": "a dermoscopic image of dermatofibroma",
    "VASC": "a dermoscopic image of vascular lesion",
}

# Kept for CLI compatibility with the old script. The current sampler uses only plain class prompts.
DEFAULT_ISIC_SUFFIXES = [
    "with realistic dermoscopic texture",
    "with natural skin color variation",
    "with fine lesion boundary details",
    "with clinical dermoscopy illumination",
    "with subtle local color variation",
]


# -----------------------------------------------------------------------------
# Arguments and small utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 SD LoRA 的困难样本约束 ICI / Diff-II 风格采样。"
    )

    # Stable Diffusion / LoRA
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--sd_lora_ckpt_path", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
    )

    # 困难样本
    parser.add_argument("--hard_csv", type=str, required=True)
    parser.add_argument(
        "--image_root",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
        help="Image root. Images are resolved as image_root / (sample_id + image_ext).",
    )
    parser.add_argument("--image_ext", type=str, default=".jpg")
    parser.add_argument(
        "--classes_to_generate",
        type=str,
        nargs="+",
        default=None,
        help="Generate only these true_label_name classes. Default: all classes in CSV.",
    )
    parser.add_argument(
        "--nearest_wrong_classes",
        type=str,
        nargs="+",
        default=None,
        help="Use only these nearest_wrong_class_name groups. Default: all groups in CSV.",
    )
    parser.add_argument(
        "--top_k_per_group",
        type=int,
        default=50,
        help="Use the K smallest distance_margin samples inside each group.",
    )
    parser.add_argument(
        "--num_per_group",
        type=int,
        default=5,
        help="Number of generated images for each valid (true_label, nearest_wrong) group.",
    )

    # DDIM / ICI（反演与采样相关参数）
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.3,
        help="Compatibility argument. The current sampler does not use two-stage prompts.",
    )
    parser.add_argument(
        "--disable_two_stage",
        action="store_true",
        help="Compatibility argument. The current sampler already uses plain prompts only.",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="*",
        default=None,
        help="Compatibility argument. The current sampler already uses plain prompts only.",
    )

    # 批量与多样性相关参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for denoising interpolated latents inside the same group.",
    )
    parser.add_argument(
        "--latent_noise_std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std added to interpolated latents. 0 disables it.",
    )

    # 运行时相关参数
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="experiments/sd_lora_ici_outputs"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache_inversions",
        action="store_true",
        help="Cache per-image DDIM inversion latents under output_dir/inversions.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch_size 必须 >= 1")
    if args.latent_noise_std < 0:
        raise ValueError("--latent_noise_std 必须 >= 0")
    if args.top_k_per_group < 2:
        raise ValueError("--top_k_per_group 必须 >= 2，因为插值需要至少两个样本")
    if args.num_per_group < 1:
        raise ValueError("--num_per_group 必须 >= 1")


def get_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(text: Any) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def chunks(seq: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def build_sd_lora_unet(
    args: argparse.Namespace, device: torch.device, weight_dtype: torch.dtype
):
    """加载基础 UNet，附加 LoRA 适配器，然后载入训练好的 LoRA 检查点（checkpoint）。"""
    if LoraConfig is None:
        raise ImportError("需要 peft 库。请通过 pip install peft 安装")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora_config)

    ckpt = torch.load(args.sd_lora_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    print(
        f"[INFO] 已加载 LoRA UNet: missing={len(missing)}, unexpected={len(unexpected)}"
    )

    unet.to(device=device, dtype=weight_dtype)
    unet.eval()
    return unet


def build_pipe(args: argparse.Namespace, device: torch.device):
    weight_dtype = get_weight_dtype(args.mixed_precision)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    ).to(device=device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(
        device=device,
        dtype=weight_dtype,
    )
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    unet = build_sd_lora_unet(args, device, weight_dtype)

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # 可选的内存优化。若失败不应中断采样流程。
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[INFO] 已启用 xformers 内存高效注意力")
    except Exception as exc:
        print(f"[INFO] 未启用 xformers: {exc}")

    return pipe, inverse_scheduler, weight_dtype


# -----------------------------------------------------------------------------
# CSV, image path, and image encoding helpers
# -----------------------------------------------------------------------------


def is_missing_value(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip().lower() in {"", "nan", "none", "null"}


def normalize_image_ext(image_ext: str | None) -> str:
    if is_missing_value(image_ext):
        return ".jpg"
    image_ext = str(image_ext).strip()
    return image_ext if image_ext.startswith(".") else f".{image_ext}"


def resolve_image_path(
    row: dict[str, Any], image_root: str | None, image_ext: str = ".jpg"
) -> str:
    """严格根据 sample_id 解析图像路径，而不使用 CSV 中的 image_path 列。"""
    if is_missing_value(image_root):
        raise FileNotFoundError(
            "--image_root 是必需的，例如 dataset/ISIC2018_Task3_Training_Input"
        )

    sample_id = row.get("sample_id")
    if is_missing_value(sample_id):
        raise FileNotFoundError(f"行中存在无效的 sample_id: {row}")

    sid = str(sample_id).strip()
    if sid.endswith(".0") and sid[:-2].isdigit():
        sid = sid[:-2]

    known_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    filename = (
        sid
        if sid.lower().endswith(known_exts)
        else sid + normalize_image_ext(image_ext)
    )
    image_path = os.path.normpath(os.path.join(str(image_root), filename))

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"未找到图像: {image_path}\n"
            f"请检查 --image_root 是否正确，以及 sample_id={sample_id} 是否存在。"
        )
    return image_path


def load_image_tensor(
    image_path: str, resolution: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.BICUBIC)

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = arr[None].transpose(0, 3, 1, 2)  # [1, H, W, C] -> [1, C, H, W]
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
    return tensor * 2.0 - 1.0  # Stable Diffusion VAE 期望输入范围为 [-1, 1]


def load_hard_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.hard_csv)
    required_cols = [
        "sample_id",
        "true_label_name",
        "nearest_wrong_class_name",
        "distance_margin",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"hard_csv 缺少必需的列: {missing_cols}")

    if "image_path" in df.columns:
        print(f"[INFO] image_path NaN 行数: {df['image_path'].isna().sum()}/{len(df)}")
    print(f"[INFO] sample_id NaN 行数: {df['sample_id'].isna().sum()}/{len(df)}")

    if args.classes_to_generate is not None:
        df = df[df["true_label_name"].isin(args.classes_to_generate)].copy()
    if args.nearest_wrong_classes is not None:
        df = df[df["nearest_wrong_class_name"].isin(args.nearest_wrong_classes)].copy()

    return df.sort_values("distance_margin", ascending=True).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Prompt encoding, inversion, interpolation, and denoising
# -----------------------------------------------------------------------------


def encode_prompts(
    pipe: StableDiffusionPipeline, prompts: str | list[str], guidance: bool = True
) -> torch.Tensor:
    """编码提示词（prompts）。如果使用 classifier-free guidance，则返回 [uncond_batch, cond_batch]。"""
    if isinstance(prompts, str):
        prompts = [prompts]

    tokenizer = pipe.tokenizer
    device = pipe.device

    cond_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_embeds = pipe.text_encoder(cond_input.input_ids.to(device))[0]

    if not guidance:
        return cond_embeds

    uncond_input = tokenizer(
        [""] * len(prompts),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeds = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    return torch.cat([uncond_embeds, cond_embeds], dim=0)


def predict_noise(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """UNet 的噪声预测，支持可选的 classifier-free guidance。"""
    batch_size = latents.shape[0]
    use_guidance = prompt_embeds.shape[0] == batch_size * 2

    latent_input = torch.cat([latents, latents], dim=0) if use_guidance else latents
    latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)

    noise_pred = pipe.unet(
        latent_input, timestep, encoder_hidden_states=prompt_embeds
    ).sample
    if not use_guidance:
        return noise_pred

    noise_uncond, noise_text = noise_pred.chunk(2)
    return noise_uncond + guidance_scale * (noise_text - noise_uncond)


@torch.no_grad()
def ddim_invert_image(
    pipe: StableDiffusionPipeline,
    inverse_scheduler: DDIMInverseScheduler,
    image_tensor: torch.Tensor,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    """使用 DDIMInverseScheduler 将单张图像反演为高噪声潜变量（latent）。"""
    scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    # ddim_invert_image 里改掉 sample()
    latents = pipe.vae.encode(image_tensor).latent_dist.mean * scaling
    # latents = pipe.vae.encode(image_tensor).latent_dist.sample * scaling
    prompt_embeds = encode_prompts(pipe, prompt, guidance=True)

    old_scheduler = pipe.scheduler
    pipe.scheduler = inverse_scheduler
    inverse_scheduler.set_timesteps(num_steps, device=pipe.device)

    for timestep in inverse_scheduler.timesteps:
        noise_pred = predict_noise(
            pipe, latents, timestep, prompt_embeds, guidance_scale
        )
        latents = inverse_scheduler.step(noise_pred, timestep, latents).prev_sample

    pipe.scheduler = old_scheduler
    return latents.detach()


def circle_interpolate(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-7):
    """原脚本使用的圆形插值（slerp-like）。返回插值后的潜变量、lambda 与角度 theta。"""
    z1_flat = z1.reshape(-1).float()
    z2_flat = z2.reshape(-1).float()

    cos = torch.sum(z1_flat * z2_flat) / (
        torch.norm(z1_flat) * torch.norm(z2_flat) + eps
    )
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos)

    if torch.isnan(theta) or theta.item() < eps:
        return 0.5 * z1 + 0.5 * z2, 0.5, float(theta.item())

    lam = random.uniform(0.0, 2.0 * math.pi / float(theta.item()))
    z = (
        torch.sin((1.0 + lam) * theta) / torch.sin(theta) * z1
        - torch.sin(lam * theta) / torch.sin(theta) * z2
    )
    return z.to(dtype=z1.dtype), lam, float(theta.item())

def safe_slerp(z1: torch.Tensor, z2: torch.Tensor, t: float | None = None, eps: float = 1e-7):
    if t is None:
        t = random.uniform(0.25, 0.75)

    z1_flat = z1.reshape(-1).float()
    z2_flat = z2.reshape(-1).float()

    cos = torch.sum(z1_flat * z2_flat) / (torch.norm(z1_flat) * torch.norm(z2_flat) + eps)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos)

    if torch.isnan(theta) or theta.item() < eps:
        z = (1 - t) * z1 + t * z2
        return z.to(dtype=z1.dtype), t, float(theta.item())

    z = (
        torch.sin((1 - t) * theta) / torch.sin(theta) * z1
        + torch.sin(t * theta) / torch.sin(theta) * z2
    )
    return z.to(dtype=z1.dtype), t, float(theta.item())

def tensor_pair_metrics(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """计算两个反演 latent 的距离指标，用于 metadata 回溯坏图来源。"""
    a = z1.detach().reshape(-1).float()
    b = z2.detach().reshape(-1).float()
    diff = a - b

    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    l2 = torch.norm(diff)
    l1 = torch.mean(torch.abs(diff))
    mse = torch.mean(diff * diff)
    cosine = torch.sum(a * b) / (a_norm * b_norm + eps)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angle = torch.acos(cosine)

    relative_l2 = l2 / ((a_norm + b_norm) * 0.5 + eps)

    return {
        "latent_l2_distance": float(l2.item()),
        "latent_relative_l2_distance": float(relative_l2.item()),
        "latent_l1_mean_distance": float(l1.item()),
        "latent_mse_distance": float(mse.item()),
        "latent_cosine_similarity": float(cosine.item()),
        "latent_angle_rad": float(angle.item()),
        "source_a_latent_norm": float(a_norm.item()),
        "source_b_latent_norm": float(b_norm.item()),
    }

@torch.no_grad()
def denoise_from_latents_batch(
    pipe: StableDiffusionPipeline,
    init_latents: torch.Tensor,
    prompts: list[str],
    num_steps: int,
    guidance_scale: float,
) -> list[Image.Image]:
    """对一批插值后的潜变量进行去噪并解码为 PIL 图像列表。"""
    pipe.scheduler.set_timesteps(num_steps, device=pipe.device)
    prompt_embeds = encode_prompts(pipe, prompts, guidance=True)

    latents = init_latents * pipe.scheduler.init_noise_sigma
    for timestep in pipe.scheduler.timesteps:
        noise_pred = predict_noise(
            pipe, latents, timestep, prompt_embeds, guidance_scale
        )
        latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

    scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    images = pipe.vae.decode(latents / scaling).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


# 为兼容旧代码/测试保留的包装函数。
@torch.no_grad()
def denoise_from_latent(
    pipe: StableDiffusionPipeline,
    init_latents: torch.Tensor,
    plain_prompt: str,
    suffixed_prompt: str,
    num_steps: int,
    guidance_scale: float,
    split_ratio: float,
    disable_two_stage: bool,
) -> Image.Image:
    return denoise_from_latents_batch(
        pipe, init_latents, [plain_prompt], num_steps, guidance_scale
    )[0]


# -----------------------------------------------------------------------------
# Task building and cache handling
# -----------------------------------------------------------------------------


def build_grouped_tasks(
    df: pd.DataFrame, args: argparse.Namespace
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    grouped = df.groupby(["true_label_name", "nearest_wrong_class_name"], sort=True)
    for (true_cls, near_wrong), sub in grouped:
        if true_cls not in ISIC_PROMPTS:
            raise ValueError(
                f"ISIC_PROMPTS 中没有类 {true_cls} 的提示词。请在采样前添加。"
            )

        sub = (
            sub.sort_values("distance_margin", ascending=True)
            .head(args.top_k_per_group)
            .reset_index(drop=True)
        )
        if len(sub) < 2:
            print(
                f"[SKIP] 组=({true_cls}, near {near_wrong}) 样本数为 {len(sub)}，需要 >= 2。"
            )
            continue

        for local_idx in range(args.num_per_group):
            idx_a, idx_b = random.sample(range(len(sub)), 2)
            tasks.append(
                {
                    "true_label_name": true_cls,
                    "nearest_wrong_class_name": near_wrong,
                    "row_a": sub.iloc[idx_a].to_dict(),
                    "row_b": sub.iloc[idx_b].to_dict(),
                    "local_idx": local_idx,
                    "plain_prompt": ISIC_PROMPTS[true_cls],
                }
            )

    return tasks


def group_tasks_by_key(tasks: list[dict[str, Any]]):
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []

    for task in tasks:
        key = (task["true_label_name"], task["nearest_wrong_class_name"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(task)

    return [(key, grouped[key]) for key in order]


def inversion_cache_path(
    cache_dir: Path, row: dict[str, Any], prompt: str, resolution: int, num_steps: int
) -> Path:
    sample_id = row.get("sample_id", row.get("index", row.get("image_path")))
    key = stable_hash(
        f"{sample_id}|{row.get('image_path')}|{prompt}|{resolution}|{num_steps}"
    )
    return cache_dir / f"{key}.pt"


def get_or_compute_inversion(
    pipe: StableDiffusionPipeline,
    inverse_scheduler: DDIMInverseScheduler,
    row: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    weight_dtype: torch.dtype,
    cache_dir: Path,
) -> torch.Tensor:
    cache_path = inversion_cache_path(
        cache_dir, row, prompt, args.resolution, args.num_inference_steps
    )
    if args.cache_inversions and cache_path.exists():
        return torch.load(cache_path, map_location=device).to(
            device=device, dtype=weight_dtype
        )

    image_path = resolve_image_path(row, args.image_root, args.image_ext)
    image_tensor = load_image_tensor(image_path, args.resolution, device, weight_dtype)
    inverted = ddim_invert_image(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        image_tensor=image_tensor,
        prompt=prompt,
        num_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    if args.cache_inversions:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(inverted.detach().cpu(), cache_path)

    return inverted


def build_output_path(
    output_dir: Path,
    true_cls: str,
    near_wrong: str,
    local_idx: int,
    global_idx: int,
    overwrite: bool,
) -> Path:
    class_dir = output_dir / true_cls
    class_dir.mkdir(parents=True, exist_ok=True)

    out_path = class_dir / f"{true_cls}_near_{near_wrong}_{local_idx:04d}.png"
    if out_path.exists() and not overwrite:
        out_path = (
            class_dir
            / f"{true_cls}_near_{near_wrong}_{local_idx:04d}_g{global_idx:06d}.png"
        )
    return out_path


def make_metadata_row(
    out_path: Path,
    task: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_a = task["row_a"]
    row_b = task["row_b"]

    return {
        "output_path": str(out_path),
        "true_label_name": task["true_label_name"],
        "nearest_wrong_class_name": task["nearest_wrong_class_name"],
        # "plain_prompt": record["plain_prompt"],
        # "suffixed_prompt": record["suffixed_prompt"],
        "latent_relative_l2_distance": record.get("latent_relative_l2_distance", ""),
        "latent_cosine_similarity": record.get("latent_cosine_similarity", ""),
        "source_margin_abs_diff": record.get("source_margin_abs_diff", ""),
        "source_a_image_path": resolve_image_path(
            row_a, args.image_root, args.image_ext
        ),
        "source_b_image_path": resolve_image_path(
            row_b, args.image_root, args.image_ext
        ),
        "source_a_sample_id": row_a.get("sample_id", ""),
        "source_b_sample_id": row_b.get("sample_id", ""),
        "source_a_margin": row_a.get("distance_margin", ""),
        "source_b_margin": row_b.get("distance_margin", ""),
        "circle_lambda": record["circle_lambda"],
        "theta": record["theta"],
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "split_ratio": args.split_ratio,
        "batch_size": args.batch_size,
        "latent_noise_std": args.latent_noise_std,
    }


# -----------------------------------------------------------------------------
# Main sampling loop
# -----------------------------------------------------------------------------


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def sample_one_batch(
    pipe: StableDiffusionPipeline,
    inverse_scheduler: DDIMInverseScheduler,
    batch_tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    weight_dtype: torch.dtype,
    cache_dir: Path,
):
    """为一个任务批次创建插值潜变量，然后一起去噪生成图像。"""
    latents: list[torch.Tensor] = []
    prompts: list[str] = []
    records: list[dict[str, Any]] = []

    # 反演仍然是按图像执行。批量加速来自对插值潜变量的批量去噪。
    for task in batch_tasks:
        prompt = task["plain_prompt"]

        inv_a = get_or_compute_inversion(
            pipe,
            inverse_scheduler,
            task["row_a"],
            prompt,
            args,
            device,
            weight_dtype,
            cache_dir,
        )
        inv_b = get_or_compute_inversion(
            pipe,
            inverse_scheduler,
            task["row_b"],
            prompt,
            args,
            device,
            weight_dtype,
            cache_dir,
        )

        pair_metrics = tensor_pair_metrics(inv_a, inv_b)

        # 暂时换成保守 slerp
        z, circle_lambda, theta = safe_slerp(inv_a, inv_b)
        # z, circle_lambda, theta = circle_interpolate(inv_a, inv_b)
        if args.latent_noise_std > 0:
            z = z + args.latent_noise_std * torch.randn_like(z)

        latents.append(z)
        prompts.append(prompt)
        records.append(
            {
                "plain_prompt": prompt,
                "suffixed_prompt": prompt,
                "circle_lambda": circle_lambda,
                "theta": theta,
                **pair_metrics,
            }
        )

    init_latents = torch.cat(latents, dim=0).to(device=device, dtype=weight_dtype)
    images = denoise_from_latents_batch(
        pipe=pipe,
        init_latents=init_latents,
        prompts=prompts,
        num_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    return images, records


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "inversions"
    prepare_output_dir(output_dir, args.overwrite)

    print(f"[INFO] device={device}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] batch_size={args.batch_size}")
    print(f"[INFO] latent_noise_std={args.latent_noise_std}")

    pipe, inverse_scheduler, weight_dtype = build_pipe(args, device)

    df = load_hard_dataframe(args)
    tasks = build_grouped_tasks(df, args)
    grouped_tasks = group_tasks_by_key(tasks)
    num_batches = sum(
        math.ceil(len(group_tasks) / args.batch_size)
        for _, group_tasks in grouped_tasks
    )

    print(f"[INFO] 有效生成任务数: {len(tasks)}")
    print(f"[INFO] 有效组数: {len(grouped_tasks)}")
    print(f"[INFO] 组内生成批次数: {num_batches}")

    metadata: list[dict[str, Any]] = []
    global_idx = 0

    pbar = tqdm(total=num_batches, desc="ICI 采样组批次")
    for (_true_cls, _near_wrong), group_tasks in grouped_tasks:
        for batch_tasks in chunks(group_tasks, args.batch_size):
            images, records = sample_one_batch(
                pipe=pipe,
                inverse_scheduler=inverse_scheduler,
                batch_tasks=batch_tasks,
                args=args,
                device=device,
                weight_dtype=weight_dtype,
                cache_dir=cache_dir,
            )

            for image, task, record in zip(images, batch_tasks, records):
                out_path = build_output_path(
                    output_dir=output_dir,
                    true_cls=task["true_label_name"],
                    near_wrong=task["nearest_wrong_class_name"],
                    local_idx=task["local_idx"],
                    global_idx=global_idx,
                    overwrite=args.overwrite,
                )
                image.save(out_path)
                metadata.append(make_metadata_row(out_path, task, record, args))
                global_idx += 1

            pbar.update(1)
    pbar.close()

    metadata_csv = output_dir / "metadata.csv"
    pd.DataFrame(metadata).to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] generated={len(metadata)}, metadata={metadata_csv}")


if __name__ == "__main__":
    main()
