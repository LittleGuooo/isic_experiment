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
    StableDiffusionImg2ImgPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from classifier.dataset import ISICResNetDataset

try:
    from peft import LoraConfig
except ImportError:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export hard samples with a trained ISIC classifier, then run "
            "Stable Diffusion LoRA img2img / SDEdit sampling."
        )
    )

    # ========== Stable Diffusion / LoRA ==========
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

    # ========== ISIC 数据 ==========
    parser.add_argument(
        "--gt_csv_path",
        type=str,
        default="dataset\ISIC2018_Task3_Training_GroundTruth.csv",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="dataset\ISIC2018_Task3_Training_Input",
    )

    # ========== seed 选择 ==========
    parser.add_argument(
        "--seed_strategy",
        type=str,
        choices=["random", "hard"],
        required=True,
        help="random: 每类随机选图；hard: 先用分类器计算真实类别置信度，再选低置信度样本。",
    )
    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.2,
        help="每类最低置信度比例。若候选 seed 不够，可自动扩大候选池到 num_seed_per_class。",
    )
    parser.add_argument(
        "--num_seed_per_class",
        type=int,
        default=20,
        help="每个类别最终选择多少张 seed 原图。",
    )
    parser.add_argument(
        "--expand_hard_pool_if_needed",
        action="store_true",
        help=(
            "hard 模式下，如果最低 hard_ratio 的样本数不足 num_seed_per_class，"
            "就突破 hard_ratio，把候选池扩大到 num_seed_per_class。"
        ),
    )

    # ========== baseline classifier，用于 hard 模式 ==========
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=None,
        help="baseline 分类器 checkpoint。seed_strategy=hard 时必须提供。",
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
        help="分类器输入分辨率。必须和 baseline 训练/验证时一致。",
    )
    parser.add_argument("--classifier_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    # ========== 采样参数 ==========
    parser.add_argument("--num_aug_per_seed", type=int, default=5)
    parser.add_argument("--batch_size_sampling", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=100)

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
        default=None,
        help=(
            "输出根目录。若不指定，默认放到 sd_lora_ckpt_path 所在实验目录的 sampling_img2img 下。"
        ),
    )
    parser.add_argument(
        "--overwrite_run_dir",
        action="store_true",
        help="清空当前参数对应的生成图片目录，避免旧图片累计影响统计。",
    )

    return parser.parse_args()


def get_weight_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_sampling_root(args):
    """
    默认输出根目录：
        experiments/xxx/sampling_img2img

    hard_samples.csv 和 all_confidences.csv 会保存在这个目录下；
    生成图片会保存在这个目录的子目录中。
    """
    if args.output_dir is not None:
        return str(args.output_dir)

    ckpt_path = Path(args.sd_lora_ckpt_path)
    if ckpt_path.parent.name == "checkpoints":
        exp_dir = ckpt_path.parent.parent
    else:
        exp_dir = ckpt_path.parent

    return str(exp_dir / "sampling_img2img")


def build_run_dir(args, sampling_root):
    """
    当前采样参数对应的图片输出目录。
    """
    return os.path.join(
        sampling_root,
        args.seed_strategy,
    )


def read_isic_gt(gt_csv_path):
    """
    读取 ISIC one-hot 标签 CSV，输出三列：
        image, label_idx, label
    """
    df = pd.read_csv(gt_csv_path)
    class_columns = [c for c in df.columns if c != "image"]

    df["label_idx"] = df[class_columns].values.argmax(axis=1)
    df["label"] = df["label_idx"].apply(lambda x: class_columns[int(x)])

    df["image"] = df["image"].astype(str)
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


def build_classifier(arch, num_classes, device):
    """
    构建 ResNet，并替换最后 fc 层。
    必须和 baseline 训练脚本保持一致。
    """
    model = models.__dict__[arch](weights=None)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"当前脚本只支持带 model.fc 的 ResNet，当前模型={arch}")

    return model.to(device)


def load_classifier_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    """
    输出每个样本的真实类别置信度。
    hard sample 用 confidence 排序，不用 pred_confidence 排序。
    """
    rows = []
    model.eval()

    for images, labels, image_ids in tqdm(loader, desc="Exporting confidences"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        pred_conf, preds = probs.max(dim=1)
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

    # 正常 ISIC 数据一张图应该只出现一次。这里主动去重，避免后面 seed 假性变多。
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
    """
    保存用 hard_df：
    每个类别内部，按真实类别 confidence 从低到高排序，
    只取最低 hard_ratio。
    """
    hard_rows = []

    for class_name in class_names:
        class_df = conf_df[conf_df["label"] == class_name].copy()

        if len(class_df) == 0:
            print(f"[WARN] class {class_name} has no samples, skipped.")
            continue

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
    采样用 seed_df：
    先按 confidence 从低到高排序。
    默认从最低 hard_ratio 的候选池里抽 seed。
    如果候选池不足 num_seed_per_class，并且 expand_hard_pool_if_needed=True，
    就突破 hard_ratio，把候选池扩大到 num_seed_per_class。

    注意：
    保存的 hard_samples.csv 仍然是原始 hard_ratio；
    这里的“扩大”只影响本次采样 seed，不篡改 hard_samples.csv 的含义。
    """
    rng = random.Random(seed)
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

        class_df = class_df.sort_values("confidence", ascending=True)
        ratio_k = max(1, int(len(class_df) * hard_ratio))

        if expand_hard_pool_if_needed:
            used_k = max(ratio_k, num_seed_per_class)
            used_k = min(used_k, len(class_df))
        else:
            used_k = ratio_k

        hard_pool = class_df.iloc[:used_k].copy()
        records = hard_pool.to_dict("records")
        rng.shuffle(records)

        selected = records[: min(num_seed_per_class, len(records))]
        selected_rows.extend(selected)

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


def export_confidences_and_select_hard_seeds(args, device, sampling_root):
    """
    hard 模式入口：
    1. 计算全部训练样本的真实类别置信度 conf_df；
    2. 保存 all_confidences.csv；
    3. 保存 hard_samples.csv；
    4. 直接用内存里的 conf_df 选择 seed_df，不重新读取磁盘。
    """
    if args.classifier_checkpoint is None:
        raise ValueError("--seed_strategy hard requires --classifier_checkpoint.")

    if not (0 < args.hard_ratio <= 1):
        raise ValueError(f"--hard_ratio must be in (0, 1], got {args.hard_ratio}")

    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.classifier_resolution, args.classifier_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ISICResNetDataset(
        gt_csv_path=args.gt_csv_path,
        img_dir=args.img_dir,
        transform=eval_transform,
    )

    class_names = dataset.class_columns
    num_classes = len(class_names)

    print(f"[INFO] classifier class_names: {class_names}")
    print(f"[INFO] classifier dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.classifier_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(
        arch=args.classifier_arch,
        num_classes=num_classes,
        device=device,
    )
    model = load_classifier_checkpoint(
        model=model,
        checkpoint_path=args.classifier_checkpoint,
        device=device,
    )

    conf_df = export_sample_confidences(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
    )

    hard_df = select_hard_per_class(
        conf_df=conf_df,
        class_names=class_names,
        hard_ratio=args.hard_ratio,
    )

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


def build_sd_lora_unet(args, device, weight_dtype):
    """
    重新构造训练时的 sd_lora UNet，并加载 .pth.tar 里的 model_state_dict。
    """
    if LoraConfig is None:
        raise ImportError(
            "sd_lora img2img requires peft. Please install: pip install peft"
        )

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
    state_dict = ckpt["model_state_dict"]

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)

    lora_keys = [k for k in state_dict.keys() if "lora" in k.lower()]
    if len(lora_keys) == 0:
        raise ValueError(
            "No LoRA keys found in checkpoint['model_state_dict']. "
            "你这个 checkpoint 可能不是 sd_lora 训练出来的。"
        )

    print(f"[INFO] loaded sd_lora checkpoint: {args.sd_lora_ckpt_path}")
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

    unet = build_sd_lora_unet(args=args, device=device, weight_dtype=weight_dtype)

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


def load_init_image(img_dir, image_id, resolution):
    """
    读取 seed 图片，并 resize 到生成分辨率。
    """
    img_path = os.path.join(img_dir, f"{image_id}.jpg")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.BILINEAR)
    return image


def stable_int_hash(text, modulo):
    """
    Python 内置 hash() 每个进程可能不同。
    这里用 md5 让同一个 image_id 的生成 seed 跨运行稳定。
    """
    value = int(hashlib.md5(str(text).encode("utf-8")).hexdigest(), 16)
    return value % modulo


def build_sampling_tasks(seed_df, args):
    tasks = []

    for row in seed_df.to_dict("records"):
        image_id = str(row["image"])
        class_name = str(row["label"])
        label_idx = int(row["label_idx"])

        if class_name not in ISIC_PROMPTS:
            raise ValueError(
                f"Unknown class_name={class_name}. "
                f"Expected one of {list(ISIC_PROMPTS.keys())}"
            )

        for aug_idx in range(args.num_aug_per_seed):
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


def run_img2img_sampling(args, pipe, tasks, run_dir, device):
    metadata_rows = []

    for start in tqdm(
        range(0, len(tasks), args.batch_size_sampling),
        desc=f"img2img batch sampling [{args.seed_strategy}]",
    ):
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
            label_idx = task["label_idx"]
            aug_idx = task["aug_idx"]

            class_out_dir = os.path.join(run_dir, class_name)
            os.makedirs(class_out_dir, exist_ok=True)

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
                    "generator_seed": task["gen_seed"],
                    "output_path": out_path,
                    "source_confidence": task["source_confidence"],
                    "pred": task["pred"],
                    "pred_confidence": task["pred_confidence"],
                    "correct": task["correct"],
                }
            )

    return pd.DataFrame(metadata_rows)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    sampling_root = resolve_sampling_root(args)
    run_dir = build_run_dir(args, sampling_root)

    os.makedirs(sampling_root, exist_ok=True)

    if args.overwrite_run_dir and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] sampling_root: {sampling_root}")
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] device: {device}")

    if args.seed_strategy == "random":
        gt_df, class_names = read_isic_gt(args.gt_csv_path)
        print(f"[INFO] class_names from CSV: {class_names}")

        seed_df = select_random_seeds(
            gt_df=gt_df,
            class_names=class_names,
            num_seed_per_class=args.num_seed_per_class,
            seed=args.seed,
        )
    else:
        seed_df, class_names = export_confidences_and_select_hard_seeds(
            args=args,
            device=device,
            sampling_root=sampling_root,
        )

    if len(seed_df) == 0:
        raise ValueError("No seed images selected.")

    seed_csv_out = os.path.join(run_dir, f"selected_seeds_{args.seed_strategy}.csv")
    seed_df.to_csv(seed_csv_out, index=False)
    print(f"[INFO] saved selected seeds to: {seed_csv_out}")

    print("\n[INFO] selected seeds per class:")
    print(seed_df.groupby("label").size())
    print("\n[INFO] unique selected seed images per class:")
    print(seed_df.groupby("label")["image"].nunique())

    tasks = build_sampling_tasks(seed_df, args)
    print(f"[INFO] total generation tasks: {len(tasks)}")

    pipe = build_img2img_pipe(args, device=device)

    meta_df = run_img2img_sampling(
        args=args,
        pipe=pipe,
        tasks=tasks,
        run_dir=run_dir,
        device=device,
    )

    meta_path = os.path.join(run_dir, f"metadata_{args.seed_strategy}.csv")
    meta_df.to_csv(meta_path, index=False)

    print("\n[DONE] generated images per class:")
    print(meta_df.groupby("label").size())

    print(f"[DONE] generated images: {len(meta_df)}")
    print(f"[DONE] metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
