import argparse
import gc
import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# 这里导入你原脚本里的函数
# 如果你的原脚本文件名不是 sd_lora_img2img_sampling.py，请改这里
from .sd_lora_img2img_sampling import (
    ISIC_PROMPTS,
    get_weight_dtype,
    read_isic_gt,
    select_random_seeds_excluding_existing,
    build_sampling_tasks,
    load_init_image,
    build_img2img_pipe,  # 这是你原脚本里的 LoRA pipe 构建函数
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original pretrained SD img2img vs pretrained SD + LoRA img2img."
    )

    # ===== Stable Diffusion / LoRA =====
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--sd_lora_ckpt_path", type=str, required=True)

    # 必须和 LoRA 训练时一致
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
    )

    # ===== ISIC 数据 =====
    parser.add_argument(
        "--gt_csv_path",
        type=str,
        default=r"dataset\ISIC2018_Task3_Training_GroundTruth.csv",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=r"dataset\ISIC2018_Task3_Training_Input",
    )

    # ===== seed 选择 =====
    parser.add_argument(
        "--seed_csv_path",
        type=str,
        default=None,
        help=(
            "推荐使用你原脚本生成的 selected_seeds_random.csv 或 selected_seeds_hard.csv。"
            "如果不提供，就每类随机选图。"
        ),
    )
    parser.add_argument("--num_seed_per_class", type=int, default=5)
    parser.add_argument("--num_aug_per_seed", type=int, default=3)

    # 这些参数是为了兼容你原来的 build_sampling_tasks / build_img2img_pipe
    parser.add_argument(
        "--seed_strategy",
        type=str,
        default="compare",
        help="这里只是为了兼容原脚本里的任务字段，不参与 hard/random 选择。",
    )
    parser.add_argument("--exclude_seed_csv", type=str, default=None)

    # ===== 采样参数 =====
    parser.add_argument("--batch_size_sampling", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=100)

    # ===== 运行设置 =====
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="compare_base_vs_lora_img2img",
    )

    return parser.parse_args()


def build_base_img2img_pipe(args, device):
    """
    构建原始预训练 Stable Diffusion img2img pipeline。

    这个 pipe 和 LoRA pipe 的区别只有一个：
        - base pipe: 不 add_adapter，不加载 LoRA checkpoint
        - lora pipe: 使用你原脚本 build_img2img_pipe() 加载 LoRA

    这样才是公平对照。
    """
    weight_dtype = get_weight_dtype(args.mixed_precision)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    ).to(device=device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    ).to(device=device, dtype=weight_dtype)

    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    unet.requires_grad_(False)
    unet = unet.to(device=device, dtype=weight_dtype)
    unet.eval()

    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe


def load_or_select_seed_df(args):
    """
    优先读取你原脚本生成的 seed CSV。
    如果没有提供 seed_csv_path，就从 ground truth CSV 里每类随机选图。
    """
    if args.seed_csv_path is not None:
        seed_df = pd.read_csv(args.seed_csv_path)

        required_cols = {"image", "label", "label_idx"}
        missing = required_cols - set(seed_df.columns)
        if missing:
            raise ValueError(
                f"seed_csv_path 缺少必要列: {missing}. "
                f"至少需要 image, label, label_idx。"
            )

        print(f"[INFO] loaded seed csv: {args.seed_csv_path}")
        print(f"[INFO] seed images: {len(seed_df)}")
        return seed_df

    gt_df, class_names = read_isic_gt(args.gt_csv_path)

    seed_df = select_random_seeds_excluding_existing(
        gt_df=gt_df,
        class_names=class_names,
        num_seed_per_class=args.num_seed_per_class,
        seed=args.seed,
        exclude_seed_csv=args.exclude_seed_csv,
    )

    print(
        "[INFO] randomly selected seed images because --seed_csv_path was not provided."
    )
    print(seed_df.groupby("label").size())

    return seed_df


@torch.no_grad()
def run_sampling_stage(args, pipe, tasks, stage_name, output_dir, device):
    """
    stage_name:
        base
        lora
    """
    stage_dir = os.path.join(output_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    metadata_rows = []

    for start in range(0, len(tasks), args.batch_size_sampling):
        batch_tasks = tasks[start : start + args.batch_size_sampling]

        prompts = []
        init_images = []
        generators = []

        for task in batch_tasks:
            init_image = load_init_image(
                img_dir=args.img_dir,
                image_id=task["image_id"],
                resolution=args.resolution,
            )

            prompts.append(task["prompt"])
            init_images.append(init_image)

            # 关键：base 和 lora 必须用同一个 generator seed
            generators.append(
                torch.Generator(device=device).manual_seed(task["gen_seed"])
            )

        result = pipe(
            prompt=prompts,
            image=init_images,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generators,
        )

        for task, out_image in zip(batch_tasks, result.images):
            image_id = task["image_id"]
            class_name = task["class_name"]
            aug_idx = task["aug_idx"]

            class_out_dir = os.path.join(stage_dir, class_name)
            os.makedirs(class_out_dir, exist_ok=True)

            filename = (
                f"{image_id}"
                f"_label-{class_name}"
                f"_aug-{aug_idx:03d}"
                f"_seed-{task['gen_seed']}"
                f"_{stage_name}.png"
            )

            out_path = os.path.join(class_out_dir, filename)
            out_image.save(out_path)

            metadata_rows.append(
                {
                    "stage": stage_name,
                    "source_image": image_id,
                    "label": class_name,
                    "label_idx": task["label_idx"],
                    "prompt": task["prompt"],
                    "aug_idx": aug_idx,
                    "generator_seed": task["gen_seed"],
                    "strength": args.strength,
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "output_path": out_path,
                }
            )

            print(f"[{stage_name}] saved: {out_path}")

    return pd.DataFrame(metadata_rows)


def make_side_by_side_grids(base_meta, lora_meta, output_dir):
    """
    把同一个 source_image + aug_idx 的 base/lora 拼成对比图。

    左：base
    右：lora
    """
    grid_dir = os.path.join(output_dir, "side_by_side")
    os.makedirs(grid_dir, exist_ok=True)

    key_cols = ["source_image", "label", "aug_idx", "generator_seed"]

    base_df = base_meta.copy()
    lora_df = lora_meta.copy()

    base_df = base_df.rename(columns={"output_path": "base_path"})
    lora_df = lora_df.rename(columns={"output_path": "lora_path"})

    merged = pd.merge(
        base_df[key_cols + ["base_path"]],
        lora_df[key_cols + ["lora_path"]],
        on=key_cols,
        how="inner",
    )

    rows = []

    for _, row in merged.iterrows():
        base_img = Image.open(row["base_path"]).convert("RGB")
        lora_img = Image.open(row["lora_path"]).convert("RGB")

        w, h = base_img.size
        title_h = 32

        canvas = Image.new("RGB", (w * 2, h + title_h), color=(255, 255, 255))
        canvas.paste(base_img, (0, title_h))
        canvas.paste(lora_img, (w, title_h))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), "BASE pretrained SD", fill=(0, 0, 0))
        draw.text((w + 10, 8), "BASE pretrained SD + LoRA", fill=(0, 0, 0))

        out_name = (
            f"{row['source_image']}"
            f"_label-{row['label']}"
            f"_aug-{int(row['aug_idx']):03d}"
            f"_seed-{int(row['generator_seed'])}"
            f"_compare.png"
        )

        out_path = os.path.join(grid_dir, out_name)
        canvas.save(out_path)

        rows.append(
            {
                "source_image": row["source_image"],
                "label": row["label"],
                "aug_idx": row["aug_idx"],
                "generator_seed": row["generator_seed"],
                "base_path": row["base_path"],
                "lora_path": row["lora_path"],
                "compare_path": out_path,
            }
        )

    compare_df = pd.DataFrame(rows)
    compare_csv = os.path.join(output_dir, "compare_pairs.csv")
    compare_df.to_csv(compare_csv, index=False)

    print(f"[DONE] side-by-side images saved to: {grid_dir}")
    print(f"[DONE] compare csv saved to: {compare_csv}")


def cleanup_pipe(pipe):
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] device: {device}")
    print(f"[INFO] output_dir: {args.output_dir}")
    print(f"[INFO] strength: {args.strength}")
    print(f"[INFO] guidance_scale: {args.guidance_scale}")
    print(f"[INFO] num_inference_steps: {args.num_inference_steps}")

    seed_df = load_or_select_seed_df(args)

    seed_csv_out = os.path.join(args.output_dir, "used_seed_images.csv")
    seed_df.to_csv(seed_csv_out, index=False)
    print(f"[INFO] saved used seeds to: {seed_csv_out}")

    tasks = build_sampling_tasks(seed_df, args)
    print(f"[INFO] total tasks: {len(tasks)}")

    # ============================================================
    # 1. 原始预训练模型采样
    # ============================================================
    print("\n[STAGE 1] Sampling with original pretrained SD base model...")
    base_pipe = build_base_img2img_pipe(args, device=device)

    base_meta = run_sampling_stage(
        args=args,
        pipe=base_pipe,
        tasks=tasks,
        stage_name="base",
        output_dir=args.output_dir,
        device=device,
    )

    base_meta_path = os.path.join(args.output_dir, "metadata_base.csv")
    base_meta.to_csv(base_meta_path, index=False)
    print(f"[DONE] base metadata saved to: {base_meta_path}")

    cleanup_pipe(base_pipe)

    # ============================================================
    # 2. 原始预训练模型 + LoRA 采样
    # ============================================================
    print("\n[STAGE 2] Sampling with pretrained SD base model + LoRA...")
    lora_pipe = build_img2img_pipe(args, device=device)

    lora_meta = run_sampling_stage(
        args=args,
        pipe=lora_pipe,
        tasks=tasks,
        stage_name="lora",
        output_dir=args.output_dir,
        device=device,
    )

    lora_meta_path = os.path.join(args.output_dir, "metadata_lora.csv")
    lora_meta.to_csv(lora_meta_path, index=False)
    print(f"[DONE] lora metadata saved to: {lora_meta_path}")

    cleanup_pipe(lora_pipe)

    # ============================================================
    # 3. 拼接对比图
    # ============================================================
    make_side_by_side_grids(
        base_meta=base_meta,
        lora_meta=lora_meta,
        output_dir=args.output_dir,
    )

    print("\n[DONE] comparison finished.")


if __name__ == "__main__":
    main()
