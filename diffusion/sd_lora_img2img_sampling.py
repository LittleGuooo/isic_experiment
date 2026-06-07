import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from classifier.dataset import ISICResNetDataset
from classifier.trainer import build_transforms

try:
    from peft import LoraConfig
except ImportError:
    LoraConfig = None


# 类别名必须和 CSV / Dataset 里的列名一致；后面 build_sampling_tasks 会用 label 查 prompt。
# 这里的 prompt 只负责给 img2img 提供类别文本条件，不会自动保证生成图标签正确。
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
        description=(
            "Select random/hard ISIC seed images, then run Stable Diffusion "
            "LoRA img2img / SDEdit-style sampling."
        )
    )

    # Stable Diffusion / LoRA：必须和训练 LoRA 时使用的底模、LoRA 结构保持一致。
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

    # ISIC data
    parser.add_argument(
        "--gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
    )

    # Seed selection：random 直接从 GT 里抽；hard 需要先跑分类器算 true-class confidence。
    parser.add_argument(
        "--seed_strategy",
        type=str,
        choices=["random", "hard"],
        required=True,
        help="random: per-class random seeds; hard: per-class low true-class confidence seeds.",
    )
    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.3,
        help="In hard mode, only the lowest hard_ratio samples per class are used as the default hard pool.",
    )
    parser.add_argument(
        "--num_seed_per_class",
        type=int,
        default=30,
        help="Maximum number of seed images selected for each class.",
    )
    parser.add_argument(
        "--expand_hard_pool_if_needed",
        action="store_true",
        help=(
            "Hard mode only. If the hard_ratio pool has fewer than num_seed_per_class "
            "samples, expand the pool up to num_seed_per_class."
        ),
    )
    parser.add_argument(
        "--exclude_seed_csv",
        type=str,
        default=None,
        help=(
            "Optional seed CSV with an 'image' column. Both random and hard mode "
            "will exclude these images before selecting new seeds."
        ),
    )
    parser.add_argument(
        "--exclude_classes",
        type=str,
        nargs="+",
        default=[],
        help="Classes to exclude from img2img generation, e.g. MEL NV.",
    )

    # Classifier used by hard mode：hard 模式用它判断“哪些真实类别置信度低”。
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=None,
        help="Required when --seed_strategy hard. Baseline classifier checkpoint.",
    )
    parser.add_argument(
        "--classifier_arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    parser.add_argument(
        "--classifier_resolution",
        type=int,
        default=256,
        help="Classifier input resolution. Must match baseline training/evaluation.",
    )
    parser.add_argument("--classifier_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    # Sampling parameters：这些参数直接传给 Diffusers img2img pipeline。
    parser.add_argument("--num_aug_per_seed", type=int, default=5)
    parser.add_argument("--batch_size_sampling", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=100)

    # Runtime：output_dir 建议在 Windows 上设短路径，避免保存图片时路径过长。
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
        default=None,
        help=(
            "Output root. If omitted, use the experiment directory inferred from "
            "sd_lora_ckpt_path/sampling_img2img."
        ),
    )
    parser.add_argument(
        "--overwrite_run_dir",
        action="store_true",
        help="Delete the current run directory before sampling.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def get_weight_dtype(mixed_precision):
    # 只控制 SD 组件的 dtype；分类器仍按 PyTorch 默认 float32 推理。
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def format_tag_value(value):
    """Make parameter values safe for directory names."""
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value).replace(".", "p").replace("/", "-").replace("\\", "-")


def stable_int_hash(text, modulo):
    """Stable hash for reproducible per-image generation seeds."""
    value = int(hashlib.md5(str(text).encode("utf-8")).hexdigest(), 16)
    return value % modulo


def resolve_sampling_root(args):
    """Infer default output root from the LoRA checkpoint location."""
    # 用户手动指定输出根目录时，直接使用它，避免深层 experiment 路径过长。
    if args.output_dir is not None:
        return str(args.output_dir)

    ckpt_path = Path(args.sd_lora_ckpt_path)
    # 兼容两种常见 checkpoint 放法：exp/checkpoints/xxx.pt 或 exp/xxx.pt。
    exp_dir = (
        ckpt_path.parent.parent
        if ckpt_path.parent.name == "checkpoints"
        else ckpt_path.parent
    )
    return str(exp_dir / "sampling_img2img")


def build_run_dir(args, sampling_root):
    """Build the run-specific image output directory from sampling parameters."""
    # 目录名记录关键采样参数，便于区分实验；图片文件名则保持短，避免 Windows 路径过长。
    common_tag = (
        f"res{args.resolution}"
        f"_seed{args.num_seed_per_class}"
        f"_aug{args.num_aug_per_seed}"
        f"_s{format_tag_value(args.strength)}"
        f"_gs{format_tag_value(args.guidance_scale)}"
        f"_steps{args.num_inference_steps}"
        f"_runseed{args.seed}"
    )

    if args.seed_strategy == "hard":
        # hard_ratio 和是否 expand 会改变 seed 池，所以必须写进 run_dir。
        hard_tag = f"hr{format_tag_value(args.hard_ratio)}"
        hard_tag += "_expand" if args.expand_hard_pool_if_needed else "_noexpand"
        common_tag = f"{common_tag}_{hard_tag}"

    return os.path.join(sampling_root, args.seed_strategy, common_tag)


def read_exclude_images(exclude_seed_csv):
    """Read image IDs from a seed CSV. Return None when no CSV is provided."""
    if exclude_seed_csv is None:
        return None

    exclude_df = pd.read_csv(exclude_seed_csv)
    # 这里只认 image 列；不依赖 label，避免旧 CSV 的列格式影响排除逻辑。
    if "image" not in exclude_df.columns:
        raise ValueError(f"'image' column not found in {exclude_seed_csv}")

    return set(exclude_df["image"].astype(str).tolist())


def exclude_images_by_id(df, exclude_seed_csv, context):
    """Exclude rows whose image ID appears in exclude_seed_csv."""
    exclude_images = read_exclude_images(exclude_seed_csv)
    if exclude_images is None:
        return df

    before = len(df)
    # random 模式传入 GT 表；hard 模式传入 conf_df。两者都有 image 列，所以可以复用。
    kept_df = df[~df["image"].astype(str).isin(exclude_images)].copy()
    after = len(kept_df)

    print(f"[INFO] exclude_seed_csv for {context}: {exclude_seed_csv}")
    print(f"[INFO] excluded {before - after} images from {context} candidate pool.")
    return kept_df.reset_index(drop=True)


def make_output_dirs(args):
    sampling_root = resolve_sampling_root(args)
    run_dir = build_run_dir(args, sampling_root)

    os.makedirs(sampling_root, exist_ok=True)
    # overwrite 只清当前参数对应的 run_dir，不会删除整个 sampling_root。
    if args.overwrite_run_dir and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    return sampling_root, run_dir


# -----------------------------------------------------------------------------
# Seed selection: random mode
# -----------------------------------------------------------------------------


def read_isic_gt(gt_csv_path):
    """Read ISIC one-hot labels and return columns: image, label_idx, label."""
    df = pd.read_csv(gt_csv_path)
    # ISIC GT 是 one-hot 表：除 image 外的列就是类别名，列顺序也决定 label_idx。
    class_columns = [c for c in df.columns if c != "image"]

    # argmax 把 one-hot 转成整数标签；必须和 Dataset / 分类器训练时的类别顺序一致。
    df["label_idx"] = df[class_columns].values.argmax(axis=1)
    df["label"] = df["label_idx"].apply(lambda x: class_columns[int(x)])
    df["image"] = df["image"].astype(str)

    return df[["image", "label_idx", "label"]], class_columns


def select_random_seeds_excluding_existing(
    gt_df,
    class_names,
    num_seed_per_class,
    seed,
    exclude_seed_csv=None,
):
    """Randomly select up to num_seed_per_class images per class after optional exclusion."""
    rng = random.Random(seed)
    # 先排除旧 seed，再按类别随机抽；否则可能重复使用上一批 seed。
    gt_df = exclude_images_by_id(gt_df, exclude_seed_csv, context="random mode")

    selected_rows = []
    for class_name in class_names:
        class_df = gt_df[gt_df["label"] == class_name].copy()

        if len(class_df) == 0:
            print(f"[WARN] class {class_name} has no available images, skipped.")
            continue

        if len(class_df) < num_seed_per_class:
            print(
                f"[WARN] class {class_name} only has {len(class_df)} available images, "
                f"less than num_seed_per_class={num_seed_per_class}."
            )

        records = class_df.to_dict("records")
        # random 模式才 shuffle；hard 模式故意不 shuffle，而是按 confidence 排序。
        rng.shuffle(records)
        selected_rows.extend(records[: min(num_seed_per_class, len(records))])

    return pd.DataFrame(selected_rows)


# -----------------------------------------------------------------------------
# Seed selection: hard mode
# -----------------------------------------------------------------------------


def build_classifier(arch, num_classes, device):
    """Build a ResNet classifier with a replaced final fc layer."""
    # weights=None：这里加载的是你自己的 checkpoint，不使用 torchvision 预训练权重。
    model = models.__dict__[arch](weights=None)

    if not (hasattr(model, "fc") and isinstance(model.fc, nn.Linear)):
        raise ValueError(
            f"Only ResNet models with model.fc are supported, got arch={arch}"
        )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def load_classifier_checkpoint(model, checkpoint_path, device):
    """Load classifier weights from common checkpoint formats."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容不同训练脚本保存 checkpoint 的常见 key 名。
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def export_sample_confidences(model, loader, class_names, device):
    """Compute true-class confidence for each image; lower means harder."""
    rows = []
    model.eval()

    for images, labels, image_ids in tqdm(loader, desc="Exporting confidences"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        # 用 softmax 概率衡量分类器置信度；hard 样本看真实类别概率，而不是最大预测概率。
        probs = torch.softmax(logits, dim=1)

        pred_conf, preds = probs.max(dim=1)
        # true_conf 越低，说明分类器越不相信真实标签，这张图越 hard。
        true_conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)

        for i in range(images.size(0)):
            label_idx = int(labels[i].item())
            pred_idx = int(preds[i].item())
            rows.append(
                {
                    "image": str(image_ids[i]),
                    "label": class_names[label_idx],
                    "label_idx": label_idx,
                    "confidence": float(true_conf[i].item()),
                    "pred": class_names[pred_idx],
                    "pred_idx": pred_idx,
                    "pred_confidence": float(pred_conf[i].item()),
                    "correct": int(pred_idx == label_idx),
                }
            )

    conf_df = pd.DataFrame(rows)

    # A normal ISIC image should appear once. Deduplicate defensively.
    before = len(conf_df)
    conf_df = conf_df.sort_values("confidence", ascending=True).drop_duplicates(
        subset=["image"], keep="first"
    )
    after = len(conf_df)
    if after != before:
        print(
            f"[WARN] duplicated image ids in confidence table: {before - after} rows dropped."
        )

    return conf_df.reset_index(drop=True)


def select_hard_per_class(conf_df, class_names, hard_ratio):
    """Save-facing hard pool: lowest hard_ratio true-class confidence samples per class."""
    hard_rows = []

    for class_name in class_names:
        class_df = conf_df[conf_df["label"] == class_name].copy()
        if len(class_df) == 0:
            print(f"[WARN] class {class_name} has no samples, skipped.")
            continue

        # 每类单独排序，防止大类把小类的 hard 样本挤掉。
        class_df = class_df.sort_values("confidence", ascending=True)
        k = max(1, int(len(class_df) * hard_ratio))
        hard_rows.append(class_df.iloc[:k])

        print(
            f"[hard csv] {class_name}: total={len(class_df)}, "
            f"hard={k}, max_hard_conf={class_df.iloc[:k]['confidence'].max():.4f}"
        )

    if len(hard_rows) == 0:
        raise ValueError("No hard samples selected.")

    return pd.concat(hard_rows, axis=0).reset_index(drop=True)


def select_hard_seeds_from_confidences(
    conf_df,
    class_names,
    hard_ratio,
    num_seed_per_class,
    seed,
    expand_hard_pool_if_needed,
):
    """
    Sampling-facing hard seeds.

    Deterministic rule:
    - sort each class by true-class confidence ascending;
    - build the allowed pool from hard_ratio;
    - optionally expand that pool to num_seed_per_class;
    - take the lowest-confidence seeds from the allowed pool.

    The seed argument is kept for API compatibility; hard selection itself is not shuffled.
    """
    _ = seed
    selected_rows = []

    print("\n[INFO] hard seed selection stats:")
    print(
        f"{'class':<8} {'total':>8} {'ratio_pool':>11} "
        f"{'used_pool':>10} {'selected':>10} {'expanded':>9}"
    )

    for class_name in class_names:
        class_df = conf_df[conf_df["label"] == class_name].copy()
        if len(class_df) == 0:
            print(f"{class_name:<8} {0:>8} {0:>11} {0:>10} {0:>10} {'no':>9}")
            continue

        # 排序后的前面就是最 hard 的样本；这里不随机，保证 hard seed 可解释、可复现。
        class_df = class_df.sort_values("confidence", ascending=True)
        ratio_k = max(1, int(len(class_df) * hard_ratio))

        if expand_hard_pool_if_needed:
            # 允许突破 hard_ratio，尽量补够 num_seed_per_class，但不能超过该类总数。
            used_k = min(max(ratio_k, num_seed_per_class), len(class_df))
        else:
            # 不 expand 时严格受 hard_ratio 限制；seed 不够也不会报错，只会少选。
            used_k = ratio_k

        hard_pool = class_df.iloc[:used_k].copy()
        # 最终 seed 仍然从允许池里取最低 confidence 的前 num_seed_per_class 张。
        selected = hard_pool.iloc[: min(num_seed_per_class, len(hard_pool))]
        selected_rows.extend(selected.to_dict("records"))

        expanded = "yes" if used_k > ratio_k else "no"
        print(
            f"{class_name:<8} {len(class_df):>8} {ratio_k:>11} "
            f"{used_k:>10} {len(selected):>10} {expanded:>9}"
        )

    if len(selected_rows) == 0:
        raise ValueError("No seed images selected.")

    selected_df = pd.DataFrame(selected_rows)
    return selected_df[
        [
            "image",
            "label_idx",
            "label",
            "confidence",
            "pred",
            "pred_idx",
            "pred_confidence",
            "correct",
        ]
    ]


def build_classifier_loader(args, device):
    """Build the ISIC dataset and DataLoader used for hard-sample mining."""
    # 必须和 baseline 分类器训练/验证时的预处理一致，否则 hard 分数没有意义。
    _, eval_transform = build_transforms(args)

    # Dataset 负责返回 image tensor、label、image_id；class_columns 用作类别顺序。
    dataset = ISICResNetDataset(
        gt_csv_path=args.gt_csv_path,
        img_dir=args.img_dir,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,  # hard 分数导出不需要打乱，方便排查 image_id 对应关系。
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    return dataset, loader


def export_confidences_and_select_hard_seeds(args, device, sampling_root):
    """Hard-mode entry: export confidence CSVs and return the selected seed DataFrame."""
    if args.classifier_checkpoint is None:
        raise ValueError("--seed_strategy hard requires --classifier_checkpoint.")
    if not (0 < args.hard_ratio <= 1):
        raise ValueError(f"--hard_ratio must be in (0, 1], got {args.hard_ratio}")

    dataset, loader = build_classifier_loader(args, device)
    class_names = dataset.class_columns
    num_classes = len(class_names)

    print(f"[INFO] classifier class_names: {class_names}")
    print(f"[INFO] classifier dataset size: {len(dataset)}")

    model = build_classifier(args.classifier_arch, num_classes, device)
    model = load_classifier_checkpoint(model, args.classifier_checkpoint, device)

    conf_df = export_sample_confidences(model, loader, class_names, device)
    # hard 模式也支持排除旧 seed：排除发生在 hard pool 计算之前。
    conf_df = exclude_images_by_id(conf_df, args.exclude_seed_csv, context="hard mode")
    if len(conf_df) == 0:
        raise ValueError("No samples left after applying exclude_seed_csv.")

    # hard_df 是“记录用”的 hard pool；seed_df 是“本次采样用”的最终 seed。
    hard_df = select_hard_per_class(conf_df, class_names, args.hard_ratio)

    ratio_tag = str(args.hard_ratio).replace(".", "p")
    all_csv = os.path.join(sampling_root, f"hard_all_confidences_ratio_{ratio_tag}.csv")
    hard_csv = os.path.join(sampling_root, f"hard_samples_ratio_{ratio_tag}.csv")

    conf_df.to_csv(all_csv, index=False)
    hard_df.to_csv(hard_csv, index=False)
    print(f"[INFO] saved all confidences to: {all_csv}")
    print(f"[INFO] saved hard samples to: {hard_csv}")

    seed_df = select_hard_seeds_from_confidences(
        conf_df=conf_df,
        class_names=class_names,
        hard_ratio=args.hard_ratio,
        num_seed_per_class=args.num_seed_per_class,
        seed=args.seed,
        expand_hard_pool_if_needed=args.expand_hard_pool_if_needed,
    )
    return seed_df, class_names


# -----------------------------------------------------------------------------
# Stable Diffusion LoRA img2img pipeline
# -----------------------------------------------------------------------------


def build_sd_lora_unet(args, device, weight_dtype):
    """Build the base UNet, attach a LoRA adapter, and load the trained LoRA weights."""
    if LoraConfig is None:
        raise ImportError(
            "sd_lora img2img requires peft. Please install: pip install peft"
        )

    # 先加载底模 UNet，再挂 LoRA adapter；底模必须和训练 LoRA 时一致。
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    unet.requires_grad_(False)

    # adapter 结构必须和训练时一致，否则 checkpoint key 会对不上。
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora_config)

    # 先在 CPU 读权重，检查通过后再移动到目标 device/dtype。
    ckpt = torch.load(args.sd_lora_ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError(
            "checkpoint does not contain 'model_state_dict'; it may not be your sd_lora checkpoint."
        )

    state_dict = ckpt["model_state_dict"]
    lora_keys = [k for k in state_dict.keys() if "lora" in k.lower()]
    if len(lora_keys) == 0:
        raise ValueError("No LoRA keys found in checkpoint['model_state_dict'].")

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    # 底模参数 missing/unexpected 不重要；这里专门检查 LoRA 相关 key。
    missing_lora = [k for k in missing if "lora" in k.lower()]
    unexpected_lora = [k for k in unexpected if "lora" in k.lower()]

    if len(missing_lora) > 0 or len(unexpected_lora) > 0:
        raise ValueError(
            "LoRA checkpoint does not match the current UNet adapter.\n"
            f"missing_lora_keys={missing_lora}\n"
            f"unexpected_lora_keys={unexpected_lora}\n"
            "Check pretrained_model_name_or_path, lora_rank, lora_alpha, and lora_target_modules."
        )

    lora_tensor_count = 0
    lora_param_count = 0
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            lora_tensor_count += 1
            lora_param_count += param.numel()

    print(f"[INFO] loaded sd_lora checkpoint: {args.sd_lora_ckpt_path}")
    print(f"[INFO] lora keys in checkpoint: {len(lora_keys)}")
    print(f"[INFO] loaded LoRA tensors in UNet: {lora_tensor_count}")
    print(f"[INFO] loaded LoRA params in UNet: {lora_param_count}")

    unet.to(device=device, dtype=weight_dtype)
    unet.eval()
    return unet


def build_img2img_pipe(args, device):
    """Assemble StableDiffusionImg2ImgPipeline with the trained LoRA UNet."""
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
    # DDIM 是常见 img2img 推理 scheduler；比 DDPM 更适合这里的采样使用。
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    unet = build_sd_lora_unet(args, device, weight_dtype)

    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,  # 医学研究采样通常不需要 NSFW safety checker。
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    try:
        # 降低 VAE 显存压力；失败也不影响主流程。
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


def load_init_image(img_dir, image_id, resolution):
    """Load a seed image and resize it to the generation resolution."""
    # ISIC 原图文件名按 image_id.jpg 查找；如果你的数据是 png，需要改这里。
    img_path = os.path.join(img_dir, f"{image_id}.jpg")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    return image.resize((resolution, resolution), resample=Image.BILINEAR)


def build_sampling_tasks(seed_df, args):
    """Expand selected seed images into per-augmentation sampling tasks."""
    tasks = []

    for row in seed_df.to_dict("records"):
        image_id = str(row["image"])
        class_name = str(row["label"])
        label_idx = int(row["label_idx"])

        # label 必须能找到 prompt；否则生成时没有类别文本条件。
        if class_name not in ISIC_PROMPTS:
            raise ValueError(
                f"Unknown class_name={class_name}. Expected one of {list(ISIC_PROMPTS.keys())}"
            )

        for aug_idx in range(args.num_aug_per_seed):
            # 每张 seed 的每个 aug_idx 使用确定性 seed，保证重复运行可复现。
            gen_seed = (
                args.seed
                + label_idx * 100000
                + aug_idx * 1000
                + stable_int_hash(image_id, 1000)
            )

            tasks.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "label_idx": label_idx,
                    "prompt": ISIC_PROMPTS[class_name],
                    "aug_idx": aug_idx,
                    "gen_seed": gen_seed,
                    "source_confidence": row.get("confidence", None),
                    "pred": row.get("pred", None),
                    "pred_confidence": row.get("pred_confidence", None),
                    "correct": row.get("correct", None),
                }
            )

    return tasks


def build_batch_inputs(batch_tasks, args, device):
    """Prepare prompts, PIL init images, and per-sample torch.Generator objects."""
    prompts = []
    init_images = []
    generators = []

    for task in batch_tasks:
        # Diffusers img2img 接收 PIL 图像列表，不是已经归一化的 tensor。
        init_images.append(
            load_init_image(
                img_dir=args.img_dir,
                image_id=task["image_id"],
                resolution=args.resolution,
            )
        )
        prompts.append(task["prompt"])
        # 每个样本一个 Generator，避免同一 batch 内所有图共用随机噪声。
        generators.append(torch.Generator(device=device).manual_seed(task["gen_seed"]))

    return prompts, init_images, generators


def make_output_path(run_dir, class_name, image_id, aug_idx):
    """Use a short filename; full metadata is saved in metadata_*.csv."""
    class_out_dir = os.path.join(run_dir, class_name)
    # 按类别分文件夹，后续可直接统计每类生成结果。
    os.makedirs(class_out_dir, exist_ok=True)
    # 文件名保持短；strength/guidance/seed 等完整信息保存在 metadata CSV。
    out_name = f"{image_id}_aug{aug_idx:03d}.png"
    return os.path.join(class_out_dir, out_name)


def build_metadata_row(task, args, out_path):
    return {
        "source_image": task["image_id"],
        "label": task["class_name"],
        "label_idx": task["label_idx"],
        "seed_strategy": args.seed_strategy,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "aug_idx": task["aug_idx"],
        "generator_seed": task["gen_seed"],
        "output_path": out_path,
        "source_confidence": task["source_confidence"],
        "pred": task["pred"],
        "pred_confidence": task["pred_confidence"],
        "correct": task["correct"],
    }


def run_img2img_sampling(args, pipe, tasks, run_dir, device):
    """Run batched Stable Diffusion img2img sampling and save images plus metadata."""
    metadata_rows = []

    for start in tqdm(
        range(0, len(tasks), args.batch_size_sampling),
        desc=f"img2img batch sampling [{args.seed_strategy}]",
    ):
        batch_tasks = tasks[start : start + args.batch_size_sampling]
        prompts, init_images, generators = build_batch_inputs(batch_tasks, args, device)

        # 这一步就是 SDEdit-style img2img：原图 latent 加噪，再按 prompt 反向去噪。
        result = pipe(
            prompt=prompts,
            image=init_images,
            strength=args.strength,  # 越大越偏离原图；医学增强不宜盲目设太高。
            guidance_scale=args.guidance_scale,  # 文本条件引导强度。
            num_inference_steps=args.num_inference_steps,
            generator=generators,
        )

        for task, out_image in zip(batch_tasks, result.images):
            out_path = make_output_path(
                run_dir=run_dir,
                class_name=task["class_name"],
                image_id=task["image_id"],
                aug_idx=task["aug_idx"],
            )
            out_image.save(out_path)
            metadata_rows.append(build_metadata_row(task, args, out_path))

    return pd.DataFrame(metadata_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def select_seed_dataframe(args, device, sampling_root):
    """Dispatch to random or hard seed selection."""
    if args.seed_strategy == "random":
        gt_df, class_names = read_isic_gt(args.gt_csv_path)
        print(f"[INFO] class_names from CSV: {class_names}")
        # random 模式：从 one-hot GT 表中每类随机抽 seed。
        seed_df = select_random_seeds_excluding_existing(
            gt_df=gt_df,
            class_names=class_names,
            num_seed_per_class=args.num_seed_per_class,
            seed=args.seed,
            exclude_seed_csv=args.exclude_seed_csv,
        )
        return seed_df, class_names

    # hard 模式：先跑分类器导出 confidence，再按低 confidence 选 seed。
    return export_confidences_and_select_hard_seeds(args, device, sampling_root)


def print_seed_summary(seed_df):
    print("\n[INFO] selected seeds per class:")
    print(seed_df.groupby("label").size())
    print("\n[INFO] unique selected seed images per class:")
    print(seed_df.groupby("label")["image"].nunique())


def main():
    args = parse_args()

    # 控制 Python 和 PyTorch 的随机性；生成阶段还会为每张图单独构造 Generator。
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    sampling_root, run_dir = make_output_dirs(args)

    print(f"[INFO] sampling_root: {sampling_root}")
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] device: {device}")

    # seed_df 是后续生成任务的唯一来源；为空就没有任何图可生成。
    seed_df, _ = select_seed_dataframe(args, device, sampling_root)

    # 排除指定类别，例如 --exclude_classes MEL NV
    if len(args.exclude_classes) > 0:
        before = len(seed_df)
        seed_df = seed_df[~seed_df["label"].isin(args.exclude_classes)].copy()
        after = len(seed_df)

        print(f"[INFO] exclude_classes: {args.exclude_classes}")
        print(f"[INFO] excluded {before - after} seed images by class.")

    if len(seed_df) == 0:
        raise ValueError("No seed images selected after applying exclude_classes.")

    seed_csv_out = os.path.join(run_dir, f"selected_seeds_{args.seed_strategy}.csv")
    seed_df.to_csv(seed_csv_out, index=False)
    print(f"[INFO] saved selected seeds to: {seed_csv_out}")
    print_seed_summary(seed_df)

    # 一张 seed 会扩展成 num_aug_per_seed 个任务。
    tasks = build_sampling_tasks(seed_df, args)
    print(f"[INFO] total generation tasks: {len(tasks)}")

    # pipeline 只构建一次；真正耗时的是下面的批量 img2img 采样。
    pipe = build_img2img_pipe(args, device=device)
    meta_df = run_img2img_sampling(args, pipe, tasks, run_dir, device)

    meta_path = os.path.join(run_dir, f"metadata_{args.seed_strategy}.csv")
    meta_df.to_csv(meta_path, index=False)

    print("\n[DONE] generated images per class:")
    print(meta_df.groupby("label").size())
    print(f"[DONE] generated images: {len(meta_df)}")
    print(f"[DONE] metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
