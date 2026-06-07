import argparse
import os
import random
import torch
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from peft import LoraConfig
except ImportError:
    LoraConfig = None

# -----------------------------------------------------------------------------
# Prompt dictionary：类别名必须和 CSV / Dataset 的列名一致
# 每类生成图像的 prompt，txt2img 生成直接使用
# -----------------------------------------------------------------------------
ISIC_PROMPTS = {
    "MEL": "a dermoscopic image of melanoma",
    "NV": "a dermoscopic image of melanocytic nevus",
    "BCC": "a dermoscopic image of basal cell carcinoma",
    "AKIEC": "a dermoscopic image of actinic keratosis or intraepithelial carcinoma",
    "BKL": "a dermoscopic image of benign keratosis-like lesion",
    "DF": "a dermoscopic image of dermatofibroma",
    "VASC": "a dermoscopic image of vascular lesion",
}


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure txt2img generation for ISIC classes with LoRA"
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

    # 生成控制参数
    parser.add_argument(
        "--classes_to_generate",
        type=str,
        nargs="+",
        required=True,
        help="Specify which classes to generate, e.g., MEL NV BCC",
    )
    parser.add_argument(
        "--num_per_class", type=int, default=300, help="Number of images per class"
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)

    # Runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_weight_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def stable_int_hash(text, modulo):
    """Stable hash for reproducible per-image generation seeds."""
    import hashlib

    value = int(hashlib.md5(str(text).encode("utf-8")).hexdigest(), 16)
    return value % modulo


def make_output_path(run_dir, class_name, idx):
    """Make output directory and filename."""
    class_dir = os.path.join(run_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    filename = f"{class_name}_{idx:04d}.png"
    return os.path.join(class_dir, filename)


# -----------------------------------------------------------------------------
# LoRA UNet & txt2img pipeline
# -----------------------------------------------------------------------------
def build_sd_lora_unet(args, device, weight_dtype):
    """Build UNet and attach LoRA weights."""
    if LoraConfig is None:
        raise ImportError("peft is required. pip install peft")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
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
    state_dict = ckpt["model_state_dict"]
    unet.load_state_dict(state_dict, strict=False)

    unet.to(device=device, dtype=weight_dtype)
    unet.eval()
    return unet


def build_txt2img_pipe(args, device):
    """Assemble StableDiffusionPipeline for txt2img."""
    weight_dtype = get_weight_dtype(args.mixed_precision)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device, dtype=weight_dtype)
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
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
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


# -----------------------------------------------------------------------------
# Build tasks: pure txt2img
# -----------------------------------------------------------------------------
def build_tasks(classes_to_generate, num_per_class, seed):
    """Build a list of generation tasks per class."""
    tasks = []
    for cls in classes_to_generate:
        for idx in range(num_per_class):
            gen_seed = seed + stable_int_hash(f"{cls}_{idx}", 1000000)
            tasks.append(
                {
                    "class_name": cls,
                    "idx": idx,
                    "prompt": ISIC_PROMPTS[cls],
                    "gen_seed": gen_seed,
                }
            )
    return tasks


# -----------------------------------------------------------------------------
# Run txt2img sampling
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_txt2img(args, pipe, tasks, run_dir, device):
    """Generate images in batches and save with metadata."""
    metadata_rows = []

    for start in tqdm(range(0, len(tasks), args.batch_size), desc="txt2img generation"):
        batch = tasks[start : start + args.batch_size]
        prompts = [t["prompt"] for t in batch]
        generators = [
            torch.Generator(device=device).manual_seed(t["gen_seed"]) for t in batch
        ]

        result = pipe(
            prompt=prompts,
            height=args.resolution,
            width=args.resolution,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generators,
        )

        for t, img in zip(batch, result.images):
            out_path = make_output_path(run_dir, t["class_name"], t["idx"])
            img.save(out_path)
            metadata_rows.append(
                {
                    "class_name": t["class_name"],
                    "idx": t["idx"],
                    "seed": t["gen_seed"],
                    "output_path": out_path,
                }
            )

    return pd.DataFrame(metadata_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_dir = (
        args.output_dir or Path(args.sd_lora_ckpt_path).parent / "txt2img_sampling"
    )
    if args.overwrite and os.path.exists(run_dir):
        import shutil

        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Output directory: {run_dir}")

    # 构建 pipeline
    pipe = build_txt2img_pipe(args, device)

    # 构建任务
    tasks = build_tasks(args.classes_to_generate, args.num_per_class, args.seed)
    print(f"[INFO] Total tasks: {len(tasks)}")

    # 生成图像
    metadata_df = run_txt2img(args, pipe, tasks, run_dir, device)
    metadata_csv = os.path.join(run_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)
    print(
        f"[DONE] Generated {len(metadata_df)} images, metadata saved to {metadata_csv}"
    )


if __name__ == "__main__":
    main()
