import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler

from torchmetrics.image.fid import FrechetInceptionDistance


# =========================================================
# 1. 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DDPM baseline for MILK10k dermoscopy images")

    # -------------------------
    # 数据路径
    # -------------------------
    parser.add_argument("--meta_csv_path", type=str, default="dataset/MILK10k_Training_Metadata.csv",
                        help="Path to MILK10k_Training_Metadata.csv")
    parser.add_argument("--gt_csv_path", type=str, default="dataset/MILK10k_Training_GroundTruth.csv",
                        help="Path to MILK10k_Training_GroundTruth.csv")
    parser.add_argument("--img_dir", type=str, default="dataset/MILK10k_Training_Input/MILK10k_Training_Input",
                        help="Path to MILK10k_Training_Input root")

    # -------------------------
    # 数据模式
    # mode=all:
    #   不区分label，做 train/val 划分
    # mode=single_label:
    #   只取一个label训练，不划分 train/val
    # -------------------------
    parser.add_argument("--data_mode", type=str, default="all", choices=["all", "single_label"],
                        help="all: use all dermoscopic images and split train/val; "
                             "single_label: only use one class and do not split train/val")
    parser.add_argument("--target_label", type=str, default=None,
                        help="Used when data_mode=single_label. Can be class name or class index string.")

    # -------------------------
    # 输出目录
    # -------------------------
    parser.add_argument("--output_root", type=str, default="experiments",
                        help="Root directory for experiments")

    # -------------------------
    # 图像和训练超参数
    # -------------------------
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # -------------------------
    # DDPM 参数
    # -------------------------
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")

    # -------------------------
    # 保存与评估
    # -------------------------
    parser.add_argument("--save_images_epochs", type=int, default=5,
                        help="Save sample images every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_epochs", type=int, default=5,
                        help="Run FID evaluation every N epochs")
    parser.add_argument("--num_fid_samples_train", type=int, default=512,
                        help="How many generated samples to compare against training set FID")
    parser.add_argument("--num_fid_samples_val", type=int, default=512,
                        help="How many generated samples to compare against validation set FID")

    # -------------------------
    # 复现
    # -------------------------
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =========================================================
# 2. 实验目录和日志工具
# =========================================================
def make_experiment_name(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = args.data_mode
    label_tag = f"label_{args.target_label}" if args.data_mode == "single_label" else "all_labels"
    exp_name = f"{timestamp}_ddpm_{mode_tag}_{label_tag}_res{args.resolution}_bs{args.train_batch_size}_seed{args.seed}"
    return exp_name


def setup_experiment_folders(base_dir, exp_name):
    exp_dir = os.path.join(base_dir, exp_name)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    metadata_dir = os.path.join(exp_dir, "metadata")
    samples_dir = os.path.join(exp_dir, "samples")
    fid_dir = os.path.join(exp_dir, "fid")
    fid_generated_dir = os.path.join(exp_dir, "fid_generated_images")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(fid_dir, exist_ok=True)
    os.makedirs(fid_generated_dir, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "metadata_dir": metadata_dir,
        "samples_dir": samples_dir,
        "fid_dir": fid_dir,
        "fid_generated_dir": fid_generated_dir
    }


def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def update_epoch_metrics_csv(metrics_csv_path, row_dict):
    row_df = pd.DataFrame([row_dict])
    if os.path.exists(metrics_csv_path):
        old_df = pd.read_csv(metrics_csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df
    new_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")


def update_epoch_metrics_json(metrics_json_path, row_dict):
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(row_dict)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def count_labels_from_indices(labels, indices, class_names):
    counter = Counter([labels[i] for i in indices])
    count_dict = {}
    for class_idx, class_name in enumerate(class_names):
        count_dict[class_name] = int(counter.get(class_idx, 0))
    return count_dict


def print_class_distribution(title, count_dict):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    total_count = 0
    for class_name, count in count_dict.items():
        print(f"{class_name}: {count}")
        total_count += count
    print(f"Total: {total_count}")
    print(f"{'=' * 60}\n")


def save_checkpoint(state, is_best, save_dir, filename="last.pth.tar"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    best_filepath = os.path.join(save_dir, "model_best.pth.tar")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_filepath)


# =========================================================
# 3. 数据集
# =========================================================
class MILK10kDDPMDataset(Dataset):
    """
    基于你代码1的真实数据结构：
    - merge metadata 和 ground truth
    - 只保留 dermoscopic 图像
    - drop_duplicates(lesion_id)
    - 支持两种模式：
        1) all: 不按 label 过滤
        2) single_label: 只取某一个类别
    """
    def __init__(
        self,
        meta_csv_path,
        gt_csv_path,
        img_dir,
        transform=None,
        data_mode="all",
        target_label=None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.data_mode = data_mode
        self.target_label = target_label

        meta_df = pd.read_csv(meta_csv_path)
        gt_df = pd.read_csv(gt_csv_path)

        df = pd.merge(meta_df, gt_df, on="lesion_id", how="inner")

        # 只保留 dermoscopic
        if "image_type" in df.columns:
            derm_mask = df["image_type"].astype(str).str.contains("dermoscopic", case=False, na=False)
            df = df[derm_mask].copy()

        # 保持和代码1一致：每个 lesion_id 只保留一条
        df = df.drop_duplicates(subset=["lesion_id"], keep="first").reset_index(drop=True)

        self.class_columns = [c for c in gt_df.columns if c != "lesion_id"]

        # 先统一生成整数标签
        if "label" in df.columns:
            df["label_int"] = df["label"].astype(int)
        else:
            df["label_int"] = df[self.class_columns].values.argmax(axis=1)

        # single_label 模式：只取指定类别
        if data_mode == "single_label":
            if target_label is None:
                raise ValueError("When data_mode='single_label', --target_label must be provided.")

            # target_label 可以是类别名，也可以是数字字符串
            if str(target_label).isdigit():
                target_label_idx = int(target_label)
                if target_label_idx < 0 or target_label_idx >= len(self.class_columns):
                    raise ValueError(f"target_label index out of range: {target_label_idx}")
            else:
                if target_label not in self.class_columns:
                    raise ValueError(f"target_label '{target_label}' not found in class columns: {self.class_columns}")
                target_label_idx = self.class_columns.index(target_label)

            df = df[df["label_int"] == target_label_idx].copy().reset_index(drop=True)
            self.selected_label_idx = target_label_idx
            self.selected_label_name = self.class_columns[target_label_idx]
        else:
            self.selected_label_idx = None
            self.selected_label_name = None

        self.df = df.reset_index(drop=True)
        self.labels = self.df["label_int"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id_col = "isic_id" if "isic_id" in row.index else ("image_id" if "image_id" in row.index else None)
        if img_id_col is None:
            raise KeyError("Cannot find 'isic_id' or 'image_id' in csv.")

        lesion_id = str(row["lesion_id"]) if "lesion_id" in row.index else None
        img_name = f"{row[img_id_col]}.jpg"

        if lesion_id:
            img_path = os.path.join(self.img_dir, lesion_id, img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_int"])
        sample_id = str(row["lesion_id"]) if "lesion_id" in row.index else str(idx)

        # 返回风格尽量贴近 diffusion 常见写法
        return {
            "input": image,
            "label": label,
            "sample_id": sample_id
        }


# =========================================================
# 4. FID 工具
# =========================================================
def tensor_to_uint8_for_fid(x):
    """
    输入 x: [-1, 1] 的 float tensor, shape [B, C, H, W]
    输出: [0, 255] 的 uint8 tensor，供 TorchMetrics FID 使用
    """
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x


def save_sample_images(pipeline, device, save_dir, epoch, eval_batch_size, num_inference_steps):
    """
    每个评估 epoch 保存一批可视化图片
    """
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(0)

    images = pipeline(
        batch_size=eval_batch_size,
        generator=generator,
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images

    for i, image in enumerate(images):
        image.save(os.path.join(epoch_dir, f"sample_{i:03d}.png"))

    return epoch_dir


@torch.no_grad()
def compute_fid_for_loader(
    accelerator,
    pipeline,
    real_loader,
    num_gen_samples,
    fid_save_dir,
    epoch,
    split_name,
    num_inference_steps,
):
    """
    计算某个数据划分上的 FID：
    - real images 来自 real_loader
    - fake images 由 pipeline 生成
    - 使用标准 ImageNet-InceptionV3 特征（TorchMetrics 默认实现）
    """
    device = accelerator.device

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # -------------------------
    # 1) 先喂真实图像
    # -------------------------
    real_count = 0
    for batch in real_loader:
        real_images = batch["input"].to(device)

        # 转成 uint8 [0,255]
        real_images_uint8 = tensor_to_uint8_for_fid(real_images)
        fid.update(real_images_uint8, real=True)

        real_count += real_images.size(0)
        if real_count >= num_gen_samples:
            break

    real_count = min(real_count, num_gen_samples)

    # -------------------------
    # 2) 再生成相同数量的 fake 图像
    # -------------------------
    generated_dir = os.path.join(fid_save_dir, f"epoch_{epoch:03d}_{split_name}_generated")
    os.makedirs(generated_dir, exist_ok=True)

    fake_count = 0
    batch_idx = 0
    while fake_count < real_count:
        cur_bs = min(real_loader.batch_size if real_loader.batch_size is not None else 16, real_count - fake_count)

        generator = torch.Generator(device=device).manual_seed(1000 + epoch * 100 + batch_idx)

        fake_pil_images = pipeline(
            batch_size=cur_bs,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="pil",
        ).images

        fake_tensors = []
        for i, img in enumerate(fake_pil_images):
            # 保存一份用于复查
            img.save(os.path.join(generated_dir, f"{split_name}_sample_{fake_count + i:05d}.png"))

            arr = np.array(img).astype(np.uint8)  # HWC, uint8, [0,255]
            ten = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
            fake_tensors.append(ten)

        fake_tensors = torch.stack(fake_tensors, dim=0).to(device)
        fid.update(fake_tensors, real=False)

        fake_count += cur_bs
        batch_idx += 1

    fid_value = fid.compute().item()

    fid_json_path = os.path.join(fid_save_dir, f"epoch_{epoch:03d}_{split_name}_fid.json")
    save_json({
        "epoch": epoch,
        "split": split_name,
        "num_real_images": int(real_count),
        "num_fake_images": int(fake_count),
        "fid": float(fid_value),
        "generated_dir": generated_dir
    }, fid_json_path)

    return float(fid_value), generated_dir, fid_json_path


# =========================================================
# 5. 主函数
# =========================================================
def main(args):
    # -------------------------
    # 固定随机种子
    # -------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # -------------------------
    # 实验目录
    # -------------------------
    exp_name = make_experiment_name(args)
    exp_folders = setup_experiment_folders(args.output_root, exp_name)

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(exp_folders["metadata_dir"], "experiment_metadata.json")
    best_model_path = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

    # -------------------------
    # 图像预处理
    # diffusion 常用：[-1,1]
    # -------------------------
    image_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # -------------------------
    # 数据集
    # -------------------------
    full_dataset = MILK10kDDPMDataset(
        meta_csv_path=args.meta_csv_path,
        gt_csv_path=args.gt_csv_path,
        img_dir=args.img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label
    )

    gt_df = pd.read_csv(args.gt_csv_path)
    class_names = [c for c in gt_df.columns if c != "lesion_id"]
    num_classes = len(class_names)

    print(f"Full filtered dataset size: {len(full_dataset)}")

    # -------------------------
    # 两种模式的数据划分
    # -------------------------
    if args.data_mode == "all":
        total_size = len(full_dataset)
        all_indices = np.arange(total_size)
        all_labels = np.array(full_dataset.labels)

        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=0.1,
            random_state=args.seed,
            stratify=all_labels
        )

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        full_class_distribution = count_labels_from_indices(full_dataset.labels, all_indices, class_names)
        train_class_distribution = count_labels_from_indices(full_dataset.labels, train_indices, class_names)
        val_class_distribution = count_labels_from_indices(full_dataset.labels, val_indices, class_names)

        print_class_distribution("Full Dataset Class Distribution", full_class_distribution)
        print_class_distribution("Train Dataset Class Distribution", train_class_distribution)
        print_class_distribution("Validation Dataset Class Distribution", val_class_distribution)

    else:
        # single_label 模式：不划分 train/val
        total_size = len(full_dataset)
        all_indices = np.arange(total_size)

        train_dataset = full_dataset
        val_dataset = None

        single_count_dict = {
            full_dataset.selected_label_name: total_size
        }

        print(f"\nUsing single_label mode. Selected label: {full_dataset.selected_label_name}")
        print(f"Training samples: {total_size}\n")

        full_class_distribution = single_count_dict
        train_class_distribution = single_count_dict
        val_class_distribution = {}

    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # FID 对 train split 的 real dataloader
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # FID 对 val split 的 real dataloader
    if val_dataset is not None:
        val_eval_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        val_eval_loader = None

    # -------------------------
    # 模型
    # -------------------------
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 注意：FID 的 eval loader 不参与训练，不需要 prepare
    # 这样更简单，也避免跨进程重复统计 real images

    # -------------------------
    # metadata 初始化
    # -------------------------
    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "data": {
            "meta_csv_path": args.meta_csv_path,
            "gt_csv_path": args.gt_csv_path,
            "img_dir": args.img_dir,
            "data_mode": args.data_mode,
            "target_label": args.target_label,
            "num_classes": num_classes,
            "class_names": class_names,
            "full_dataset_size": len(full_dataset),
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
            "class_distribution": {
                "full_dataset": full_class_distribution,
                "train_dataset": train_class_distribution,
                "val_dataset": val_class_distribution
            }
        },
        "model": {
            "resolution": args.resolution,
            "ddpm_num_steps": args.ddpm_num_steps,
            "ddpm_num_inference_steps": args.ddpm_num_inference_steps,
            "ddpm_beta_schedule": args.ddpm_beta_schedule,
        },
        "training": {
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "checkpoints_dir": exp_folders["checkpoints_dir"],
            "samples_dir": exp_folders["samples_dir"],
            "fid_dir": exp_folders["fid_dir"],
            "fid_generated_dir": exp_folders["fid_generated_dir"]
        },
        "best_result": {
            "best_epoch_by_val_fid": -1,
            "best_val_fid": None,
            "best_epoch_by_train_fid": -1,
            "best_train_fid": None,
            "best_model_path": ""
        }
    }

    if accelerator.is_main_process:
        save_json(experiment_metadata, metadata_json_path)

    # -------------------------
    # 训练
    # -------------------------
    global_step = 0
    best_val_fid = float("inf")
    best_train_fid = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}"
        )

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]

            noise = torch.randn_like(clean_images)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 记录训练损失
            loss_item = loss.detach().item()
            epoch_loss_sum += loss_item * clean_images.size(0)
            epoch_loss_count += clean_images.size(0)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({
                    "loss": f"{loss_item:.6f}",
                    "step": global_step
                })

        progress_bar.close()
        accelerator.wait_for_everyone()

        train_loss_epoch = epoch_loss_sum / max(epoch_loss_count, 1)

        # -------------------------
        # 每隔若干 epoch 做一次评估
        # -------------------------
        need_save_images = ((epoch + 1) % args.save_images_epochs == 0) or (epoch == args.num_epochs - 1)
        need_save_model = ((epoch + 1) % args.save_model_epochs == 0) or (epoch == args.num_epochs - 1)
        need_eval = ((epoch + 1) % args.eval_epochs == 0) or (epoch == args.num_epochs - 1)

        fid_train_value = None
        fid_val_value = None
        train_fid_json_path = ""
        val_fid_json_path = ""
        train_generated_dir = ""
        val_generated_dir = ""
        sample_dir = ""

        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
            pipeline = pipeline.to(accelerator.device)

            # 1) 保存可视化样本
            if need_save_images:
                sample_dir = save_sample_images(
                    pipeline=pipeline,
                    device=accelerator.device,
                    save_dir=exp_folders["samples_dir"],
                    epoch=epoch + 1,
                    eval_batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps
                )
                print(f"Sample images saved to: {sample_dir}")

            # 2) 计算 FID：train 和 val 都算
            if need_eval:
                print(f"Running FID evaluation at epoch {epoch + 1} ...")

                fid_train_value, train_generated_dir, train_fid_json_path = compute_fid_for_loader(
                    accelerator=accelerator,
                    pipeline=pipeline,
                    real_loader=train_eval_loader,
                    num_gen_samples=args.num_fid_samples_train,
                    fid_save_dir=exp_folders["fid_dir"],
                    epoch=epoch + 1,
                    split_name="train",
                    num_inference_steps=args.ddpm_num_inference_steps,
                )

                print(f"Train FID: {fid_train_value:.6f}")

                if val_eval_loader is not None:
                    fid_val_value, val_generated_dir, val_fid_json_path = compute_fid_for_loader(
                        accelerator=accelerator,
                        pipeline=pipeline,
                        real_loader=val_eval_loader,
                        num_gen_samples=args.num_fid_samples_val,
                        fid_save_dir=exp_folders["fid_dir"],
                        epoch=epoch + 1,
                        split_name="val",
                        num_inference_steps=args.ddpm_num_inference_steps,
                    )
                    print(f"Val FID: {fid_val_value:.6f}")
                else:
                    print("Val FID skipped because current mode has no validation split.")

            # 3) 保存模型
            # best 模型优先按 val_fid 选；如果没有 val，则按 train_fid
            is_best = False
            if fid_val_value is not None:
                if fid_val_value < best_val_fid:
                    best_val_fid = fid_val_value
                    is_best = True
                    experiment_metadata["best_result"]["best_epoch_by_val_fid"] = epoch + 1
                    experiment_metadata["best_result"]["best_val_fid"] = float(fid_val_value)
            elif fid_train_value is not None:
                if fid_train_value < best_train_fid:
                    best_train_fid = fid_train_value
                    is_best = True
                    experiment_metadata["best_result"]["best_epoch_by_train_fid"] = epoch + 1
                    experiment_metadata["best_result"]["best_train_fid"] = float(fid_train_value)

            if need_save_model or is_best:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_val_fid": best_val_fid,
                    "best_train_fid": best_train_fid,
                    "args": vars(args),
                }, is_best=is_best, save_dir=exp_folders["checkpoints_dir"], filename="last.pth.tar")

                if is_best:
                    experiment_metadata["best_result"]["best_model_path"] = best_model_path

                # 也保存 diffusers 格式，方便后面直接 pipeline.from_pretrained
                pipeline.save_pretrained(exp_folders["exp_dir"])

        accelerator.wait_for_everyone()

        # -------------------------
        # 每个评估 epoch 记录信息（不记录 lr）
        # -------------------------
        if accelerator.is_main_process and need_eval:
            epoch_row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss_epoch),
                "train_fid": float(fid_train_value) if fid_train_value is not None else None,
                "val_fid": float(fid_val_value) if fid_val_value is not None else None,
                "sample_dir": sample_dir,
                "train_fid_json_path": train_fid_json_path,
                "val_fid_json_path": val_fid_json_path,
                "train_generated_dir": train_generated_dir,
                "val_generated_dir": val_generated_dir,
                "checkpoint_path": os.path.join(exp_folders["checkpoints_dir"], "last.pth.tar")
            }
            update_epoch_metrics_csv(metrics_csv_path, epoch_row)
            update_epoch_metrics_json(metrics_json_path, epoch_row)

            experiment_metadata["last_epoch_finished"] = epoch + 1
            experiment_metadata["updated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_json(experiment_metadata, metadata_json_path)

    accelerator.end_training()


# =========================================================
# 6. 运行入口
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)