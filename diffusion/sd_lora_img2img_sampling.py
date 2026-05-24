import argparse
import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from peft import LoraConfig
except ImportError:
    LoraConfig = None


ISIC_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

ISIC_PROMPTS = {
    "MEL": "a dermoscopic image of melanoma",
    "NV": "a dermoscopic image of melanocytic nevus",
    "BCC": "a dermoscopic image of basal cell carcinoma",
    "AKIEC": "a dermoscopic image of actinic keratosis or intraepithelial carcinoma",
    "BKL": "a dermoscopic image of benign keratosis-like lesion",
    "DF": "a dermoscopic image of dermatofibroma",
    "VASC": "a dermoscopic image of vascular lesion",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion LoRA img2img / SDEdit sampling for ISIC2018"
    )

    # ========== Stable Diffusion / LoRA ==========
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Stable Diffusion 预训练模型路径，必须和训练 sd_lora 时一致。",
    )
    parser.add_argument(
        "--sd_lora_ckpt_path",
        type=str,
        required=True,
        help="你训练得到的 sd_lora checkpoint，例如 experiments/xxx/checkpoints/last.pth.tar。",
    )

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
    )

    # ========== ISIC 数据 ==========
    parser.add_argument(
        "--gt_csv_path",
        type=str,
        required=True,
        help="ISIC ground truth CSV，例如 ISIC2018_Task3_Training_GroundTruth.csv。",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="ISIC 图片目录，例如 ISIC2018_Task3_Training_Input。",
    )

    # ========== seed 选择 ==========
    parser.add_argument(
        "--seed_strategy",
        type=str,
        choices=["random", "hard"],
        required=True,
        help="random: 每类随机选图；hard: 每类从低置信度困难样本中选图。",
    )
    parser.add_argument(
        "--hard_csv_path",
        type=str,
        default=None,
        help="困难样本 CSV。seed_strategy=hard 时必须提供。列为 image,label,confidence。",
    )
    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.2,
        help="每类选择低置信度前多少比例作为困难样本池。默认 0.2，即每类最低置信度 20%。",
    )

    # ========== 采样参数 ==========
    parser.add_argument(
        "--num_seed_per_class",
        type=int,
        default=20,
        help="每个类别选择多少张 seed 原图。",
    )
    parser.add_argument(
        "--num_aug_per_seed",
        type=int,
        default=1,
        help="每张 seed 图生成几张增强图。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="img2img 输入和输出分辨率。建议和训练 sd_lora 时一致。",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.45,
        help=(
            "img2img 加噪强度。越大越不像原图。" "医学图像建议先试 0.35 / 0.45 / 0.55。"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="CFG guidance scale。医学图像不要盲目设太大，先试 3.0 / 5.0 / 7.5。",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="反向去噪步数。",
    )

    # ========== 运行设置 ==========
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="增强图保存目录。",
    )

    return parser.parse_args()


def get_weight_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def read_isic_gt(gt_csv_path):
    """
    读取 ISIC one-hot 标签 CSV，输出三列：
        image, label_idx, label
    """
    df = pd.read_csv(gt_csv_path)

    class_columns = [c for c in df.columns if c != "image"]

    # 你的训练代码也是 one-hot -> argmax 得到整数标签。
    # 这里保持一致。
    df["label_idx"] = df[class_columns].values.argmax(axis=1)
    df["label"] = df["label_idx"].apply(lambda x: class_columns[int(x)])

    return df[["image", "label_idx", "label"]], class_columns


def select_random_seeds(gt_df, class_names, num_seed_per_class, seed):
    """
    每类随机选择 seed 图像。
    """
    rng = random.Random(seed)
    selected_rows = []

    for class_name in class_names:
        class_df = gt_df[gt_df["label"] == class_name].copy()

        if len(class_df) == 0:
            print(f"[WARN] class {class_name} has no images, skipped.")
            continue

        records = class_df.to_dict("records")
        rng.shuffle(records)

        selected_rows.extend(records[: min(num_seed_per_class, len(records))])

    return pd.DataFrame(selected_rows)


def select_hard_seeds(
    gt_df, hard_csv_path, class_names, hard_ratio, num_seed_per_class, seed
):
    """
    每类选择 baseline 分类器置信度最低的前 hard_ratio，
    再从这些困难样本中随机抽 num_seed_per_class 张。

    hard_csv 必须包含：
        image,label,confidence
    """
    if hard_csv_path is None:
        raise ValueError("--seed_strategy hard requires --hard_csv_path.")

    hard_df = pd.read_csv(hard_csv_path)

    required_cols = {"image", "label", "confidence"}
    missing = required_cols - set(hard_df.columns)
    if missing:
        raise ValueError(
            f"hard_csv_path missing columns: {missing}. "
            f"Required columns are: {required_cols}"
        )

    # 统一 image 为字符串，避免 CSV 读入后类型不一致。
    hard_df["image"] = hard_df["image"].astype(str)
    gt_df = gt_df.copy()
    gt_df["image"] = gt_df["image"].astype(str)

    # 只保留确实存在于 GT 里的样本，避免路径找不到。
    hard_df = hard_df.merge(
        gt_df[["image", "label_idx"]],
        on="image",
        how="inner",
        suffixes=("", "_gt"),
    )

    rng = random.Random(seed)
    selected_rows = []

    for class_name in class_names:
        class_df = hard_df[hard_df["label"] == class_name].copy()

        if len(class_df) == 0:
            print(f"[WARN] hard csv has no images for class {class_name}, skipped.")
            continue

        # 低 confidence = 困难样本
        class_df = class_df.sort_values("confidence", ascending=True)

        k = max(1, int(len(class_df) * hard_ratio))
        hard_pool = class_df.iloc[:k].copy()

        records = hard_pool.to_dict("records")
        rng.shuffle(records)

        selected_rows.extend(records[: min(num_seed_per_class, len(records))])

    selected_df = pd.DataFrame(selected_rows)

    # 保证列名和 random 策略一致
    selected_df = selected_df[["image", "label_idx", "label", "confidence"]]

    return selected_df


def build_sd_lora_unet(args, device, weight_dtype):
    """
    重新构造训练时的 sd_lora UNet，并加载 .pth.tar 里的 model_state_dict。

    注意：
    这里不是 pipe.load_lora_weights。
    因为你现在保存的是工程 checkpoint，不是 Diffusers LoRA 权重目录。
    """
    if LoraConfig is None:
        raise ImportError(
            "sd_lora img2img requires peft. Please install: pip install peft"
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # 和训练时保持一致：先冻结 base UNet，再注入 LoRA。
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

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)

    # strict=False 是为了兼容不同 diffusers/peft 版本可能出现的少量非关键键。
    # 但是如果 LoRA key 没加载上，这里必须警告。
    lora_keys = [k for k in state_dict.keys() if "lora" in k.lower()]
    if len(lora_keys) == 0:
        raise ValueError(
            "No LoRA keys found in checkpoint['model_state_dict']. "
            "你这个 checkpoint 可能不是 sd_lora 训练出来的。"
        )

    print(f"[INFO] loaded checkpoint: {args.sd_lora_ckpt_path}")
    print(f"[INFO] lora keys in checkpoint: {len(lora_keys)}")
    print(f"[INFO] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    unet.to(device=device, dtype=weight_dtype)
    unet.eval()

    return unet


def build_img2img_pipe(args, device):
    """
    组装 StableDiffusionImg2ImgPipeline。
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

    unet = build_sd_lora_unet(
        args=args,
        device=device,
        weight_dtype=weight_dtype,
    )

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
    pipe.set_progress_bar_config(disable=False)

    # 省显存；不是必须。
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe


def load_init_image(img_dir, image_id, resolution):
    """
    读取 seed 图片，并 resize 到训练分辨率。
    """
    img_path = os.path.join(img_dir, f"{image_id}.jpg")

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.BILINEAR)

    return image


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_df, class_names = read_isic_gt(args.gt_csv_path)

    # 保险：优先使用 CSV 里的类别顺序。
    # 如果 ISIC CSV 标准，这里应该和 ISIC_CLASS_NAMES 一致。
    print(f"[INFO] class_names from CSV: {class_names}")

    if args.seed_strategy == "random":
        seed_df = select_random_seeds(
            gt_df=gt_df,
            class_names=class_names,
            num_seed_per_class=args.num_seed_per_class,
            seed=args.seed,
        )
    else:
        seed_df = select_hard_seeds(
            gt_df=gt_df,
            hard_csv_path=args.hard_csv_path,
            class_names=class_names,
            hard_ratio=args.hard_ratio,
            num_seed_per_class=args.num_seed_per_class,
            seed=args.seed,
        )

    if len(seed_df) == 0:
        raise ValueError("No seed images selected. Check your CSV paths and labels.")

    # 保存本次实际使用了哪些 seed，后面做实验对比必须留痕。
    seed_csv_out = os.path.join(
        args.output_dir, f"selected_seeds_{args.seed_strategy}.csv"
    )
    seed_df.to_csv(seed_csv_out, index=False)
    print(f"[INFO] saved selected seeds to: {seed_csv_out}")

    pipe = build_img2img_pipe(args, device=device)

    metadata_rows = []

    for row in tqdm(
        seed_df.to_dict("records"), desc=f"img2img sampling [{args.seed_strategy}]"
    ):
        image_id = str(row["image"])
        class_name = str(row["label"])
        label_idx = int(row["label_idx"])

        if class_name not in ISIC_PROMPTS:
            raise ValueError(
                f"Unknown class_name={class_name}. "
                f"Expected one of {list(ISIC_PROMPTS.keys())}"
            )

        prompt = ISIC_PROMPTS[class_name]

        init_image = load_init_image(
            img_dir=args.img_dir,
            image_id=image_id,
            resolution=args.resolution,
        )

        class_out_dir = os.path.join(args.output_dir, args.seed_strategy, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        for aug_idx in range(args.num_aug_per_seed):
            # 每张增强图用不同 seed，保证可复现。
            gen_seed = (
                args.seed
                + label_idx * 100000
                + aug_idx * 1000
                + abs(hash(image_id)) % 1000
            )
            generator = torch.Generator(device=device).manual_seed(gen_seed)

            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )

            out_image = result.images[0]

            out_name = (
                f"{image_id}"
                f"_label-{class_name}"
                f"_strategy-{args.seed_strategy}"
                f"_strength-{args.strength}"
                f"_gs-{args.guidance_scale}"
                f"_aug-{aug_idx:03d}.png"
            )

            out_path = os.path.join(class_out_dir, out_name)
            out_image.save(out_path)

            metadata_rows.append(
                {
                    "source_image": image_id,
                    "label": class_name,
                    "label_idx": label_idx,
                    "seed_strategy": args.seed_strategy,
                    "strength": args.strength,
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "aug_idx": aug_idx,
                    "generator_seed": gen_seed,
                    "output_path": out_path,
                    "source_confidence": row.get("confidence", None),
                }
            )

    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(args.output_dir, f"metadata_{args.seed_strategy}.csv")
    meta_df.to_csv(meta_path, index=False)

    print(f"[DONE] generated images: {len(meta_df)}")
    print(f"[DONE] metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
