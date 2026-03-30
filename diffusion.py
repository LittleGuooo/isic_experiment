import argparse
import json
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils as vutils
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from sklearn.model_selection import train_test_split


# =========================================================
# 1. 命令行参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a classic DDPM baseline on MILK10k dermoscopy images.")

    # -----------------------------
    # 数据路径：改成适配你代码1中的真实数据组织形式
    # -----------------------------
    parser.add_argument("--meta_csv_path", type=str,default='dataset/MILK10k_Training_Metadata.csv',
                        help="Path to MILK10k_Training_Metadata.csv")
    parser.add_argument("--gt_csv_path", type=str, default='dataset/MILK10k_Training_GroundTruth.csv',
                        help="Path to MILK10k_Training_GroundTruth.csv")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Root image directory, e.g. dataset/MILK10k_Training_Input/MILK10k_Training_Input")

    # -----------------------------
    # 输出目录
    # -----------------------------
    parser.add_argument("--output_root", type=str, default="experiments",
                        help="Root directory to save experiment outputs")

    # -----------------------------
    # 基础训练参数
    # -----------------------------
    parser.add_argument("--resolution", type=int, default=64,
                        help="Input image resolution for DDPM")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size used for generation / FID feature extraction")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Total training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="LR warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        help="Scheduler type for training")
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # -----------------------------
    # DDPM 参数
    # -----------------------------
    parser.add_argument("--ddpm_num_steps", type=int, default=1000,
                        help="Number of diffusion train timesteps")
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000,
                        help="Number of inference steps for image generation")
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear",
                        help="Beta schedule for DDPM scheduler")

    # -----------------------------
    # 数据增强参数
    # -----------------------------
    parser.add_argument("--random_flip", action="store_true",
                        help="Use random horizontal flip")
    parser.add_argument("--center_crop", action="store_true",
                        help="Use center crop instead of random crop")

    # -----------------------------
    # 评估与保存参数
    # -----------------------------
    parser.add_argument("--save_images_epochs", type=int, default=10,
                        help="Save sample images every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_epochs", type=int, default=10,
                        help="Run FID evaluation every N epochs")
    parser.add_argument("--num_sample_visualize", type=int, default=16,
                        help="How many images to generate for visualization")
    parser.add_argument("--num_fid_gen_train", type=int, default=512,
                        help="How many generated images to use for train FID")
    parser.add_argument("--num_fid_gen_val", type=int, default=512,
                        help="How many generated images to use for val FID")

    # -----------------------------
    # 其他
    # -----------------------------
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm")

    return parser.parse_args()


# =========================================================
# 2. 数据集定义
# =========================================================
class MILK10kDDPMDataset(Dataset):
    def __init__(self, meta_csv_path, gt_csv_path, img_dir, transform=None):
        """
        适配 MILK10k 单模态（皮肤镜图像）的 DDPM 数据集。
        保留你代码1中的核心逻辑：
        1) metadata 与 ground truth 按 lesion_id merge
        2) 只保留 dermoscopic 图像
        3) lesion_id 去重
        4) 用 lesion_id / isic_id(.jpg) 或 image_id(.jpg) 找图
        这里不返回 label，只返回图像和 sample_id。
        """
        self.img_dir = img_dir
        self.transform = transform

        meta_df = pd.read_csv(meta_csv_path)
        gt_df = pd.read_csv(gt_csv_path)

        # 按 lesion_id 合并，完全沿用你代码1的思路
        df = pd.merge(meta_df, gt_df, on="lesion_id", how="inner")

        # 只保留 dermoscopic 图像
        if "image_type" in df.columns:
            derm_mask = df["image_type"].astype(str).str.contains("dermoscopic", case=False, na=False)
            df = df[derm_mask].copy()

        # 按 lesion_id 去重，与你代码1保持一致
        df = df.drop_duplicates(subset=["lesion_id"], keep="first").reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)

    def _build_img_path(self, row):
        """
        根据你代码1的逻辑构建图像路径：
        img_dir / lesion_id / isic_id.jpg
        或
        img_dir / lesion_id / image_id.jpg
        """
        img_id_col = "isic_id" if "isic_id" in row.index else ("image_id" if "image_id" in row.index else None)
        if img_id_col is None:
            raise KeyError("无法在 CSV 中找到 'isic_id' 或 'image_id' 列。")

        lesion_id = str(row["lesion_id"]) if "lesion_id" in row.index else None
        img_name = f"{row[img_id_col]}.jpg"

        if lesion_id is not None:
            img_path = os.path.join(self.img_dir, lesion_id, img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)

        return img_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._build_img_path(row)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample_id = str(row["lesion_id"]) if "lesion_id" in row.index else str(idx)

        # 这里只返回生成模型需要的内容，不返回 label
        return {
            "input": image,
            "sample_id": sample_id
        }


# =========================================================
# 3. 实验目录与保存工具
# =========================================================
def make_experiment_name(args):
    """
    生成实验名。
    尽量保持和你代码1相似的可读性。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{timestamp}_ddpm_baseline"
        f"_res{args.resolution}"
        f"_lr{args.learning_rate}"
        f"_bs{args.train_batch_size}"
        f"_seed{args.seed}"
    )
    return exp_name


def setup_experiment_folders(base_dir, exp_name):
    """
    创建实验目录。
    参考你代码1的实验目录组织方式，但替换成生成模型需要的目录。
    """
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
        "fid_generated_dir": fid_generated_dir,
    }


def save_json(data, json_path):
    """
    保存 JSON 文件。
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def update_epoch_metrics_csv(metrics_csv_path, row_dict):
    """
    追加保存每个 epoch 的训练/评估指标到 CSV。
    """
    row_df = pd.DataFrame([row_dict])
    if os.path.exists(metrics_csv_path):
        old_df = pd.read_csv(metrics_csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df
    new_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")


def update_epoch_metrics_json(metrics_json_path, row_dict):
    """
    追加保存每个 epoch 的训练/评估指标到 JSON。
    """
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(row_dict)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_checkpoint(state, is_best, save_dir, filename="last.pth.tar"):
    """
    保存 checkpoint。
    参考你代码1的做法：保存 last，并在更优时覆盖 model_best。
    """
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    best_filepath = os.path.join(save_dir, "model_best.pth.tar")

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, best_filepath)


# =========================================================
# 4. 一些简单统计工具
# =========================================================
def print_dataset_statistics(title, dataset_size):
    """
    打印数据集大小。
    因为你要求放弃 label 相关内容，所以这里只打印样本总量。
    """
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"Total samples: {dataset_size}")
    print(f"{'=' * 60}\n")


def set_seed(seed):
    """
    固定随机种子，增强复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 5. FID 计算相关
#    用标准 ImageNet-InceptionV3 提取特征
# =========================================================
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        """
        使用 torchvision 官方标准 ImageNet 预训练 InceptionV3。
        我们去掉最后的全连接层，取倒数特征（2048维）。
        """
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False, aux_logits=False)

        # 去掉最后的 fc，保留到 avgpool 之前/之后的特征提取路径
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.eval()

    def forward(self, x):
        """
        输入 x: [B, 3, H, W]，输出 [B, 2048]
        这里要求输入已经被 resize 到 299x299，并按 ImageNet 标准归一化。
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def matrix_sqrt_torch(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    用特征值分解计算矩阵平方根。
    FID 需要 sqrtm(sigma1 * sigma2)。
    这里实现一个 torch 版，避免依赖 scipy。
    """
    # 保证对称
    mat = 0.5 * (mat + mat.T)

    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_mat = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    return sqrt_mat


def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    根据两组高斯统计量计算 FID。
    FID = ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = mu1.double()
    mu2 = mu2.double()
    sigma1 = sigma1.double()
    sigma2 = sigma2.double()

    diff = mu1 - mu2

    cov_prod = sigma1 @ sigma2
    covmean = matrix_sqrt_torch(cov_prod, eps=eps)

    # 数值稳定
    if torch.isnan(covmean).any() or torch.isinf(covmean).any():
        eye = torch.eye(sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype)
        covmean = matrix_sqrt_torch((sigma1 + eps * eye) @ (sigma2 + eps * eye), eps=eps)

    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid.item())


def get_inception_transform():
    """
    给 InceptionV3 提取特征时的输入预处理。
    注意：
    - DDPM 训练图像是 [-1,1]
    - Inception 需要 [0,1] 再按 ImageNet mean/std 标准化
    """
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_real_features(dataloader, feature_extractor, device):
    """
    从真实图像 dataloader 提取 Inception 特征。
    dataloader batch 要包含 batch["input"]，且 input 已是 [-1,1] tensor。
    """
    inception_transform = get_inception_transform()
    all_features = []

    feature_extractor.eval()

    for batch in tqdm(dataloader, desc="Extract real features", leave=False):
        images = batch["input"].to(device)

        # 从 [-1,1] 转回 [0,1]
        images = (images + 1.0) / 2.0
        images = inception_transform(images)

        feats = feature_extractor(images)
        all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)
    return all_features


@torch.no_grad()
def generate_images_for_fid(
    pipeline,
    num_images,
    batch_size,
    device,
    num_inference_steps,
    save_dir=None,
    save_prefix="gen"
):
    """
    用当前 DDPM 模型生成一批图片，返回 tensor 列表。
    同时可选地把生成图保存到指定目录，方便你检查用于 FID 的样本质量。
    """
    all_images = []
    generator = torch.Generator(device=device).manual_seed(0)

    os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

    total_saved = 0
    num_rounds = math.ceil(num_images / batch_size)

    for r in tqdm(range(num_rounds), desc="Generate images for FID", leave=False):
        cur_bs = min(batch_size, num_images - len(all_images))
        if cur_bs <= 0:
            break

        outputs = pipeline(
            batch_size=cur_bs,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="pt",
        )
        # 输出范围通常是 [0,1]
        images = outputs.images  # [B, C, H, W]
        all_images.append(images.cpu())

        if save_dir is not None:
            for i in range(images.size(0)):
                img = transforms.ToPILImage()(images[i].cpu().clamp(0, 1))
                img.save(os.path.join(save_dir, f"{save_prefix}_{total_saved:06d}.png"))
                total_saved += 1

    all_images = torch.cat(all_images, dim=0)[:num_images]
    return all_images


@torch.no_grad()
def extract_generated_features(generated_images, feature_extractor, device):
    """
    从生成图片 tensor 中提取 Inception 特征。
    generated_images: [N, 3, H, W], range [0,1]
    """
    inception_transform = get_inception_transform()
    all_features = []

    feature_extractor.eval()

    batch_size = 64
    for start in tqdm(range(0, generated_images.size(0), batch_size), desc="Extract generated features", leave=False):
        batch = generated_images[start:start + batch_size].to(device)
        batch = inception_transform(batch)
        feats = feature_extractor(batch)
        all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)
    return all_features


def compute_mean_cov(features: torch.Tensor):
    """
    根据特征矩阵计算均值和协方差。
    features: [N, D]
    """
    features = features.double()
    mu = torch.mean(features, dim=0)

    diff = features - mu.unsqueeze(0)
    # 无偏协方差
    sigma = diff.T @ diff / (features.size(0) - 1)
    return mu, sigma


def save_fid_result_json(save_path, result_dict):
    """
    保存某次评估的 FID 结果。
    """
    save_json(result_dict, save_path)


# =========================================================
# 6. 可视化生成样本
# =========================================================
@torch.no_grad()
def save_sample_images(
    pipeline,
    epoch,
    save_dir,
    device,
    num_images,
    num_inference_steps,
):
    """
    每隔若干 epoch 生成一组可视化图片并保存。
    """
    os.makedirs(save_dir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(0)

    outputs = pipeline(
        batch_size=num_images,
        generator=generator,
        num_inference_steps=num_inference_steps,
        output_type="pt",
    )
    images = outputs.images  # [B, C, H, W], [0,1]

    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 保存单张图
    for i in range(images.size(0)):
        img = transforms.ToPILImage()(images[i].cpu().clamp(0, 1))
        img.save(os.path.join(epoch_dir, f"sample_{i:03d}.png"))

    # 再保存一张 grid 总览图，方便快速看训练效果
    grid = vutils.make_grid(images.cpu(), nrow=int(math.sqrt(num_images)) if num_images >= 4 else num_images)
    grid_img = transforms.ToPILImage()(grid)
    grid_path = os.path.join(epoch_dir, f"grid_epoch_{epoch:03d}.png")
    grid_img.save(grid_path)

    return epoch_dir, grid_path


# =========================================================
# 7. 训练一个 epoch
# =========================================================
def train_one_epoch(
    accelerator,
    model,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    train_dataloader,
    epoch,
    args,
):
    """
    训练一个 epoch。
    按你的新要求：训练时不输出 lr。
    这里只记录/显示 loss、step。
    """
    model.train()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )

    running_loss = 0.0
    loss_count = 0
    global_step_in_epoch = 0

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
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # 记录 loss
        loss_scalar = loss.detach().item()
        running_loss += loss_scalar
        loss_count += 1

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step_in_epoch += 1
            progress_bar.set_postfix({
                "loss": f"{loss_scalar:.6f}",
                "step": global_step_in_epoch,
            })

    progress_bar.close()

    mean_train_loss = running_loss / max(loss_count, 1)
    return {
        "train_loss": float(mean_train_loss)
    }


# =========================================================
# 8. FID 评估
# =========================================================
@torch.no_grad()
def evaluate_fid(
    accelerator,
    model,
    noise_scheduler,
    feature_extractor,
    train_eval_loader,
    val_eval_loader,
    epoch,
    exp_folders,
    args,
):
    """
    分别对 train / val 计算 FID。
    你的要求是两个都要算。
    """
    if not accelerator.is_main_process:
        return None

    unet = accelerator.unwrap_model(model)
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    pipeline = pipeline.to(accelerator.device)

    # -------------------------------------------------
    # 1) 先提取 train 真实图像特征
    # -------------------------------------------------
    real_train_features = extract_real_features(
        dataloader=train_eval_loader,
        feature_extractor=feature_extractor,
        device=accelerator.device
    )
    mu_train_real, sigma_train_real = compute_mean_cov(real_train_features)

    # -------------------------------------------------
    # 2) 生成用于 train FID 的图片并提取特征
    # -------------------------------------------------
    train_gen_save_dir = os.path.join(
        exp_folders["fid_generated_dir"], f"epoch_{epoch:03d}_train"
    )
    generated_train_images = generate_images_for_fid(
        pipeline=pipeline,
        num_images=args.num_fid_gen_train,
        batch_size=args.eval_batch_size,
        device=accelerator.device,
        num_inference_steps=args.ddpm_num_inference_steps,
        save_dir=train_gen_save_dir,
        save_prefix="train_gen"
    )
    gen_train_features = extract_generated_features(
        generated_images=generated_train_images,
        feature_extractor=feature_extractor,
        device=accelerator.device
    )
    mu_train_gen, sigma_train_gen = compute_mean_cov(gen_train_features)

    fid_train = compute_fid_from_stats(
        mu1=mu_train_real,
        sigma1=sigma_train_real,
        mu2=mu_train_gen,
        sigma2=sigma_train_gen
    )

    # -------------------------------------------------
    # 3) 再提取 val 真实图像特征
    # -------------------------------------------------
    real_val_features = extract_real_features(
        dataloader=val_eval_loader,
        feature_extractor=feature_extractor,
        device=accelerator.device
    )
    mu_val_real, sigma_val_real = compute_mean_cov(real_val_features)

    # -------------------------------------------------
    # 4) 生成用于 val FID 的图片并提取特征
    # -------------------------------------------------
    val_gen_save_dir = os.path.join(
        exp_folders["fid_generated_dir"], f"epoch_{epoch:03d}_val"
    )
    generated_val_images = generate_images_for_fid(
        pipeline=pipeline,
        num_images=args.num_fid_gen_val,
        batch_size=args.eval_batch_size,
        device=accelerator.device,
        num_inference_steps=args.ddpm_num_inference_steps,
        save_dir=val_gen_save_dir,
        save_prefix="val_gen"
    )
    gen_val_features = extract_generated_features(
        generated_images=generated_val_images,
        feature_extractor=feature_extractor,
        device=accelerator.device
    )
    mu_val_gen, sigma_val_gen = compute_mean_cov(gen_val_features)

    fid_val = compute_fid_from_stats(
        mu1=mu_val_real,
        sigma1=sigma_val_real,
        mu2=mu_val_gen,
        sigma2=sigma_val_gen
    )

    # -------------------------------------------------
    # 5) 保存本次 FID 结果
    # -------------------------------------------------
    fid_result = {
        "epoch": int(epoch),
        "fid_train": float(fid_train),
        "fid_val": float(fid_val),
        "num_real_train": int(len(train_eval_loader.dataset)),
        "num_real_val": int(len(val_eval_loader.dataset)),
        "num_generated_train": int(args.num_fid_gen_train),
        "num_generated_val": int(args.num_fid_gen_val),
        "train_generated_dir": train_gen_save_dir,
        "val_generated_dir": val_gen_save_dir,
    }

    fid_json_path = os.path.join(exp_folders["fid_dir"], f"epoch_{epoch:03d}_fid.json")
    save_fid_result_json(fid_json_path, fid_result)

    print(
        f" * FID Eval => Epoch {epoch} | "
        f"FID(train): {fid_train:.4f} | FID(val): {fid_val:.4f}"
    )
    print(f" * FID result saved to: {fid_json_path}")
    print(f" * Train FID generated images saved to: {train_gen_save_dir}")
    print(f" * Val FID generated images saved to: {val_gen_save_dir}")

    return {
        "fid_train": float(fid_train),
        "fid_val": float(fid_val),
        "fid_json_path": fid_json_path,
        "train_generated_dir": train_gen_save_dir,
        "val_generated_dir": val_gen_save_dir,
    }


# =========================================================
# 9. 主函数
# =========================================================
def main(args):
    # -----------------------------
    # 随机种子
    # -----------------------------
    if args.seed is not None:
        set_seed(args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This may improve reproducibility but can slightly reduce speed."
        )

    # -----------------------------
    # Accelerator
    # -----------------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # -----------------------------
    # 实验目录
    # -----------------------------
    exp_name = make_experiment_name(args)
    exp_folders = setup_experiment_folders(base_dir=args.output_root, exp_name=exp_name)

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(exp_folders["metadata_dir"], "experiment_metadata.json")
    best_model_path = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

    if accelerator.is_main_process:
        print(f"Experiment directory: {exp_folders['exp_dir']}")

    # -----------------------------
    # 图像预处理
    # 训练 diffusion 时归一化到 [-1,1]
    # -----------------------------
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # -----------------------------
    # 数据集：按你代码1中的真实 CSV + 图片目录结构读取
    # -----------------------------
    full_dataset = MILK10kDDPMDataset(
        meta_csv_path=args.meta_csv_path,
        gt_csv_path=args.gt_csv_path,
        img_dir=args.img_dir,
        transform=image_transforms
    )

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size

    # 这里按你的新要求放弃 label 相关内容
    # 所以简单用随机划分，不再做 stratify
    all_indices = np.arange(total_size)
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=args.seed,
        shuffle=True,
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    if accelerator.is_main_process:
        print_dataset_statistics("Full Dataset", len(full_dataset))
        print_dataset_statistics("Train Dataset", len(train_dataset))
        print_dataset_statistics("Validation Dataset", len(val_dataset))

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # 下面这两个 dataloader 是给 FID 提取真实图像特征用的
    # 不需要打乱顺序
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # -----------------------------
    # 模型：保留你代码2的经典 DDPM baseline 主体
    # -----------------------------
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

    # -----------------------------
    # 交给 accelerator 托管
    # 这里只托管训练相关对象
    # FID 的 feature extractor 只放主进程单独使用，不需要 prepare
    # -----------------------------
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # -----------------------------
    # InceptionV3 特征提取器
    # 只在主进程创建和使用，避免多卡重复评估
    # -----------------------------
    feature_extractor = None
    if accelerator.is_main_process:
        feature_extractor = InceptionV3FeatureExtractor().to(accelerator.device)
        feature_extractor.eval()

    # -----------------------------
    # 初始化实验元信息
    # -----------------------------
    best_fid_val = float("inf")
    best_epoch = -1

    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": "DDPM baseline for dermoscopy image generation",
        "model": {
            "name": "UNet2DModel",
            "sample_size": args.resolution,
            "in_channels": 3,
            "out_channels": 3,
            "layers_per_block": 2,
            "block_out_channels": [128, 128, 256, 256, 512, 512],
        },
        "scheduler": {
            "name": "DDPMScheduler",
            "num_train_timesteps": args.ddpm_num_steps,
            "beta_schedule": args.ddpm_beta_schedule,
            "prediction_type": "epsilon",
        },
        "training": {
            "num_epochs": args.num_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "lr_warmup_steps": args.lr_warmup_steps,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_weight_decay": args.adam_weight_decay,
            "adam_epsilon": args.adam_epsilon,
            "mixed_precision": args.mixed_precision,
        },
        "data": {
            "meta_csv_path": args.meta_csv_path,
            "gt_csv_path": args.gt_csv_path,
            "img_dir": args.img_dir,
            "full_dataset_size": total_size,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "split_ratio": f"{1.0 - args.val_ratio:.2f} / {args.val_ratio:.2f}",
            "filtering": [
                "metadata and ground truth merged by lesion_id",
                "keep only dermoscopic images if image_type exists",
                "drop_duplicates by lesion_id",
                "no label used in DDPM training pipeline",
            ],
        },
        "evaluation": {
            "main_metric": "FID",
            "fid_feature_extractor": "torchvision InceptionV3 with standard ImageNet pretrained weights",
            "fid_against": ["train", "val"],
            "save_images_epochs": args.save_images_epochs,
            "save_model_epochs": args.save_model_epochs,
            "eval_epochs": args.eval_epochs,
            "num_sample_visualize": args.num_sample_visualize,
            "num_fid_gen_train": args.num_fid_gen_train,
            "num_fid_gen_val": args.num_fid_gen_val,
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "checkpoints_dir": exp_folders["checkpoints_dir"],
            "samples_dir": exp_folders["samples_dir"],
            "fid_dir": exp_folders["fid_dir"],
            "fid_generated_dir": exp_folders["fid_generated_dir"],
            "best_checkpoint": best_model_path,
        },
        "best_result": {
            "best_epoch": best_epoch,
            "best_fid_val": None,
            "best_model_path": "",
        }
    }

    if accelerator.is_main_process:
        save_json(experiment_metadata, metadata_json_path)

    # =====================================================
    # 开始训练
    # =====================================================
    for epoch in range(1, args.num_epochs + 1):
        # -----------------------------
        # 1) 训练一个 epoch
        # -----------------------------
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            epoch=epoch,
            args=args,
        )

        accelerator.wait_for_everyone()

        # -----------------------------
        # 2) 每隔若干 epoch 保存可视化生成图
        # -----------------------------
        sample_epoch_dir = ""
        sample_grid_path = ""

        if accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs):
            unet = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler).to(accelerator.device)

            sample_epoch_dir, sample_grid_path = save_sample_images(
                pipeline=pipeline,
                epoch=epoch,
                save_dir=exp_folders["samples_dir"],
                device=accelerator.device,
                num_images=args.num_sample_visualize,
                num_inference_steps=args.ddpm_num_inference_steps,
            )

            print(f" * Sample images saved to: {sample_epoch_dir}")
            print(f" * Sample grid saved to: {sample_grid_path}")

        accelerator.wait_for_everyone()

        # -----------------------------
        # 3) 每隔若干 epoch 做 FID 评估
        # -----------------------------
        fid_metrics = {
            "fid_train": None,
            "fid_val": None,
            "fid_json_path": "",
            "train_generated_dir": "",
            "val_generated_dir": "",
        }

        if epoch % args.eval_epochs == 0 or epoch == args.num_epochs:
            fid_result = evaluate_fid(
                accelerator=accelerator,
                model=model,
                noise_scheduler=noise_scheduler,
                feature_extractor=feature_extractor,
                train_eval_loader=train_eval_loader,
                val_eval_loader=val_eval_loader,
                epoch=epoch,
                exp_folders=exp_folders,
                args=args,
            )

            if accelerator.is_main_process and fid_result is not None:
                fid_metrics = fid_result

                # 用 val FID 作为 best model 选择标准
                current_fid_val = fid_result["fid_val"]
                is_best = current_fid_val < best_fid_val
                if is_best:
                    best_fid_val = current_fid_val
                    best_epoch = epoch
                else:
                    is_best = False
            else:
                is_best = False
        else:
            is_best = False

        accelerator.wait_for_everyone()

        # -----------------------------
        # 4) 每隔若干 epoch 保存 checkpoint
        #    若本轮评估更优，也保存 best
        # -----------------------------
        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs or is_best):
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "best_fid_val": best_fid_val,
                    "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "args": vars(args),
                },
                is_best=is_best,
                save_dir=exp_folders["checkpoints_dir"],
                filename="last.pth.tar"
            )

            # 同时保存 diffusers pipeline，方便你后续直接加载生成
            unet = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
            pipeline.save_pretrained(exp_folders["checkpoints_dir"])

        # -----------------------------
        # 5) 保存每个 epoch 的 metrics
        #    按你的新要求：不记录 lr
        # -----------------------------
        if accelerator.is_main_process:
            epoch_row = {
                "epoch": int(epoch),
                "train_loss": float(train_metrics["train_loss"]),
                "fid_train": None if fid_metrics["fid_train"] is None else float(fid_metrics["fid_train"]),
                "fid_val": None if fid_metrics["fid_val"] is None else float(fid_metrics["fid_val"]),
                "sample_epoch_dir": sample_epoch_dir,
                "sample_grid_path": sample_grid_path,
                "fid_json_path": fid_metrics["fid_json_path"],
                "train_generated_dir_for_fid": fid_metrics["train_generated_dir"],
                "val_generated_dir_for_fid": fid_metrics["val_generated_dir"],
                "checkpoint_dir": exp_folders["checkpoints_dir"],
            }

            update_epoch_metrics_csv(metrics_csv_path, epoch_row)
            update_epoch_metrics_json(metrics_json_path, epoch_row)

            # 更新 metadata
            experiment_metadata["best_result"]["best_epoch"] = best_epoch
            experiment_metadata["best_result"]["best_fid_val"] = None if best_epoch == -1 else float(best_fid_val)
            experiment_metadata["best_result"]["best_model_path"] = best_model_path if os.path.exists(best_model_path) else ""
            experiment_metadata["last_epoch_finished"] = epoch
            experiment_metadata["updated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_json(experiment_metadata, metadata_json_path)

        accelerator.wait_for_everyone()

    accelerator.end_training()


# =========================================================
# 10. 入口
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)