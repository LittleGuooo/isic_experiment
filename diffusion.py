import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from accelerate import Accelerator
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler

from torchmetrics.image.fid import FrechetInceptionDistance

# =========================================================
# 1. 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DDPM baseline for MILK10k dermoscopy images")

    # -------------------------
    # 从已有权重继续训练
    # -------------------------
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a saved checkpoint .pth.tar for resuming training")
    
    # -------------------------
    # DDIM 采样参数
    # 训练仍然使用 DDPM
    # 这里只控制“采样/生成图片”时是否使用 DDIM
    # -------------------------
    parser.add_argument("--use_ddim_sampling", action="store_true",
                        help="Use DDIM instead of DDPM during sampling/evaluation.")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta. Usually 0.0 for deterministic sampling.")

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
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
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


def disable_pipeline_progress_bar(pipeline):
    """
    关闭 diffusers pipeline 自带的内部进度条
    """
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)


def save_diffusers_model_index_copy(exp_dir, metadata_dir):
    """
    复制一份更明确命名的 model_index.json 到 metadata 文件夹中。

    说明：
    - 标准的 model_index.json 必须保留在 exp_dir 中，供 Diffusers from_pretrained() 使用
    - 这里额外复制一份到 metadata 下，便于你查看和管理
    """
    src = os.path.join(exp_dir, "model_index.json")
    dst = os.path.join(metadata_dir, "diffusers_pipeline_model_index.json")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    return dst


# =========================================================
# 3. 数据集
# =========================================================
class MILK10kDDPMDataset(Dataset):
    """
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

        if "image_type" in df.columns:
            derm_mask = df["image_type"].astype(str).str.contains("dermoscopic", case=False, na=False)
            df = df[derm_mask].copy()

        df = df.drop_duplicates(subset=["lesion_id"], keep="first").reset_index(drop=True)

        self.class_columns = [c for c in gt_df.columns if c != "lesion_id"]

        if "label" in df.columns:
            df["label_int"] = df["label"].astype(int)
        else:
            df["label_int"] = df[self.class_columns].values.argmax(axis=1)

        if data_mode == "single_label":
            if target_label is None:
                raise ValueError("When data_mode='single_label', --target_label must be provided.")

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


@torch.no_grad()
def collect_real_uint8_images(real_loader, device, num_samples):
    """
    从 real_loader 中取出最多 num_samples 张真实图像，并转换成 FID 需要的 uint8 格式
    """
    real_batches = []
    real_count = 0

    for batch in real_loader:
        real_images = batch["input"].to(device)
        real_images_uint8 = tensor_to_uint8_for_fid(real_images)
        real_batches.append(real_images_uint8)

        real_count += real_images.size(0)
        if real_count >= num_samples:
            break

    if len(real_batches) == 0:
        return torch.empty(0, 3, 0, 0, dtype=torch.uint8, device=device), 0

    real_images_all = torch.cat(real_batches, dim=0)[:num_samples]
    return real_images_all, real_images_all.size(0)


@torch.no_grad()
def generate_fake_images_for_fid(
    accelerator,
    pipeline,
    num_gen_samples,
    fake_save_root,
    epoch,
    num_inference_steps,
    eval_batch_size,
    use_ddim_sampling=False,
    ddim_eta=0.0,
):
    """
    统一生成一批 fake images，供 train FID 和 val FID 共用
    """
    device = accelerator.device
    disable_pipeline_progress_bar(pipeline)

    generated_dir = os.path.join(fake_save_root, f"epoch_{epoch:03d}_shared_generated")
    os.makedirs(generated_dir, exist_ok=True)

    fake_uint8_batches = []
    fake_count = 0
    batch_idx = 0

    progress_bar = tqdm(
        total=num_gen_samples,
        desc="FID generated images",
        leave=True
    )

    while fake_count < num_gen_samples:
        cur_bs = min(eval_batch_size, num_gen_samples - fake_count)
        generator = torch.Generator(device=device).manual_seed(1000 + epoch * 100 + batch_idx)

        if use_ddim_sampling:
            fake_pil_images = pipeline(
                batch_size=cur_bs,
                generator=generator,
                num_inference_steps=num_inference_steps,
                eta=ddim_eta,
                output_type="pil",
            ).images
        else:
            fake_pil_images = pipeline(
                batch_size=cur_bs,
                generator=generator,
                num_inference_steps=num_inference_steps,
                output_type="pil",
            ).images

        fake_tensors = []
        for i, img in enumerate(fake_pil_images):
            global_idx = fake_count + i
            img.save(os.path.join(generated_dir, f"fid_sample_{global_idx:05d}.png"))

            arr = np.array(img).astype(np.uint8)
            ten = torch.from_numpy(arr).permute(2, 0, 1)
            fake_tensors.append(ten)

        fake_tensors = torch.stack(fake_tensors, dim=0).to(device)
        fake_uint8_batches.append(fake_tensors)

        fake_count += cur_bs
        batch_idx += 1
        progress_bar.update(cur_bs)

    progress_bar.close()

    fake_images_all = torch.cat(fake_uint8_batches, dim=0)
    return fake_images_all, generated_dir


@torch.no_grad()
def compute_fid_from_real_and_fake(real_images_uint8, fake_images_uint8, device):
    """
    使用已经准备好的 real / fake uint8 图像直接计算 FID
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(real_images_uint8, real=True)
    fid.update(fake_images_uint8[:real_images_uint8.size(0)], real=False)
    return float(fid.compute().item())


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
        total_size = len(full_dataset)
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

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

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

    # -------------------------
    # 断点继续训练相关状态
    # -------------------------
    start_epoch = 0
    global_step = 0
    best_val_fid = float("inf")
    best_train_fid = float("inf")

    # -------------------------
    # 如果指定了 checkpoint，则从已有权重继续训练
    # -------------------------
    if args.resume_from_checkpoint is not None:
        print(f"Loading checkpoint from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")

        # 恢复模型权重
        model.load_state_dict(checkpoint["model_state_dict"])

        # 恢复优化器和学习率调度器状态
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # 恢复训练进度
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", 0)
        best_val_fid = checkpoint.get("best_val_fid", float("inf"))
        best_train_fid = checkpoint.get("best_train_fid", float("inf"))

        print(f"Resume training from epoch {start_epoch + 1}")
        print(f"Recovered global_step = {global_step}")
        print(f"Recovered best_val_fid = {best_val_fid}")
        print(f"Recovered best_train_fid = {best_train_fid}")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # -------------------------
    # metadata 初始化
    # -------------------------
    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "resume_from_checkpoint": args.resume_from_checkpoint,
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
            "use_ddim_sampling": args.use_ddim_sampling,
            "ddim_eta": args.ddim_eta,
        },
        "training": {
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "start_epoch": start_epoch,
            "initial_global_step": global_step,
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "checkpoints_dir": exp_folders["checkpoints_dir"],
            "samples_dir": exp_folders["samples_dir"],
            "fid_dir": exp_folders["fid_dir"],
            "fid_generated_dir": exp_folders["fid_generated_dir"],
            "diffusers_model_index_copy": os.path.join(exp_folders["metadata_dir"], "diffusers_pipeline_model_index.json")
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
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Train Epoch [{epoch + 1}/{args.num_epochs}]"
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

            loss_item = loss.detach().item()
            epoch_loss_sum += loss_item * clean_images.size(0)
            epoch_loss_count += clean_images.size(0)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({
                    "loss": f"{loss_item:.6f}",
                    "update": global_step
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
        shared_generated_dir = ""
        sample_dir = ""
        diffusers_model_index_copy_path = ""

        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)

            if args.use_ddim_sampling:
                ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
                pipeline = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
            else:
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

            pipeline = pipeline.to(accelerator.device)

            # 1) 保存可视化样本
            if need_save_images:
                disable_pipeline_progress_bar(pipeline)

                epoch_dir = os.path.join(exp_folders["samples_dir"], f"epoch_{epoch + 1:03d}")
                os.makedirs(epoch_dir, exist_ok=True)

                generator = torch.Generator(device=accelerator.device).manual_seed(0)
                sample_progress_bar = tqdm(
                    total=args.eval_batch_size,
                    desc="Generating samples",
                    leave=True
                )

                if args.use_ddim_sampling:
                    images = pipeline(
                        batch_size=args.eval_batch_size,
                        generator=generator,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        eta=args.ddim_eta,
                        output_type="pil",
                    ).images
                else:
                    images = pipeline(
                        batch_size=args.eval_batch_size,
                        generator=generator,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        output_type="pil",
                    ).images

                for i, image in enumerate(images):
                    image.save(os.path.join(epoch_dir, f"sample_{i:03d}.png"))
                    sample_progress_bar.update(1)

                sample_progress_bar.close()
                sample_dir = epoch_dir
                print(f"Samples saved to: {sample_dir}")

            # 2) 计算 FID：先统一生成一批 fake，再分别算 train / val
            if need_eval:
                print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
                print("-" * 60)

                target_fake_count = args.num_fid_samples_train
                if val_eval_loader is not None:
                    target_fake_count = max(args.num_fid_samples_train, args.num_fid_samples_val)

                fake_images_uint8, shared_generated_dir = generate_fake_images_for_fid(
                    accelerator=accelerator,
                    pipeline=pipeline,
                    num_gen_samples=target_fake_count,
                    fake_save_root=exp_folders["fid_generated_dir"],
                    epoch=epoch + 1,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    eval_batch_size=args.eval_batch_size,
                    use_ddim_sampling=args.use_ddim_sampling,
                    ddim_eta=args.ddim_eta,
                )

                train_real_uint8, train_real_count = collect_real_uint8_images(
                    real_loader=train_eval_loader,
                    device=accelerator.device,
                    num_samples=args.num_fid_samples_train,
                )

                fid_train_value = compute_fid_from_real_and_fake(
                    real_images_uint8=train_real_uint8,
                    fake_images_uint8=fake_images_uint8,
                    device=accelerator.device,
                )

                train_fid_json_path = os.path.join(exp_folders["fid_dir"], f"epoch_{epoch + 1:03d}_train_fid.json")
                save_json({
                    "epoch": epoch + 1,
                    "split": "train",
                    "num_real_images": int(train_real_count),
                    "num_fake_images": int(train_real_count),
                    "fid": float(fid_train_value),
                    "generated_dir": shared_generated_dir,
                    "sampler": "ddim" if args.use_ddim_sampling else "ddpm",
                    "num_inference_steps": int(args.ddpm_num_inference_steps),
                    "ddim_eta": float(args.ddim_eta) if args.use_ddim_sampling else None,
                    "shared_fake_images": True,
                }, train_fid_json_path)

                print(f"Train FID: {fid_train_value:.6f}")

                if val_eval_loader is not None:
                    val_real_uint8, val_real_count = collect_real_uint8_images(
                        real_loader=val_eval_loader,
                        device=accelerator.device,
                        num_samples=args.num_fid_samples_val,
                    )

                    fid_val_value = compute_fid_from_real_and_fake(
                        real_images_uint8=val_real_uint8,
                        fake_images_uint8=fake_images_uint8,
                        device=accelerator.device,
                    )

                    val_fid_json_path = os.path.join(exp_folders["fid_dir"], f"epoch_{epoch + 1:03d}_val_fid.json")
                    save_json({
                        "epoch": epoch + 1,
                        "split": "val",
                        "num_real_images": int(val_real_count),
                        "num_fake_images": int(val_real_count),
                        "fid": float(fid_val_value),
                        "generated_dir": shared_generated_dir,
                        "sampler": "ddim" if args.use_ddim_sampling else "ddpm",
                        "num_inference_steps": int(args.ddpm_num_inference_steps),
                        "ddim_eta": float(args.ddim_eta) if args.use_ddim_sampling else None,
                        "shared_fake_images": True,
                    }, val_fid_json_path)

                    print(f"Val FID: {fid_val_value:.6f}")
                else:
                    print("Val FID skipped (no validation split).")

            # 3) 保存模型
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

                # 保存 diffusers 格式，供后续 from_pretrained 使用
                pipeline.save_pretrained(exp_folders["exp_dir"])

                # 额外复制一份更明确命名的 model_index.json 到 metadata 文件夹
                diffusers_model_index_copy_path = save_diffusers_model_index_copy(
                    exp_dir=exp_folders["exp_dir"],
                    metadata_dir=exp_folders["metadata_dir"]
                )

                print(f"Checkpoint saved to: {os.path.join(exp_folders['checkpoints_dir'], 'last.pth.tar')}")
                print(f"Diffusers model index copy saved to: {diffusers_model_index_copy_path}")
                if is_best:
                    print("New best model saved.")

        accelerator.wait_for_everyone()

        # -------------------------
        # 每个评估 epoch 记录信息
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
                "shared_fid_generated_dir": shared_generated_dir,
                "checkpoint_path": os.path.join(exp_folders["checkpoints_dir"], "last.pth.tar"),
                "diffusers_model_index_copy_path": diffusers_model_index_copy_path
            }
            update_epoch_metrics_csv(metrics_csv_path, epoch_row)
            update_epoch_metrics_json(metrics_json_path, epoch_row)

            experiment_metadata["last_epoch_finished"] = epoch + 1
            experiment_metadata["updated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment_metadata["paths"]["diffusers_model_index_copy"] = os.path.join(
                exp_folders["metadata_dir"], "diffusers_pipeline_model_index.json"
            )
            save_json(experiment_metadata, metadata_json_path)

    accelerator.end_training()


# =========================================================
# 6. 运行入口
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)