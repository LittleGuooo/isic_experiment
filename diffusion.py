import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as tv_models

from accelerate import Accelerator
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler

from torchmetrics.image.fid import FrechetInceptionDistance

# =========================================================
# 1. 参数
# =========================================================
def parse_args():
   parser = argparse.ArgumentParser(description="DDPM baseline for ISIC2018 dermoscopy images")

   # -------------------------
   # 断点续训
   # -------------------------
   parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="指定 .pth.tar checkpoint 路径以从上次训练中断处继续，"
                            "会自动复用原实验文件夹（metrics/samples/fid 等目录接续写入）")

   # -------------------------
   # 采样器选择
   # 训练始终使用 DDPM；这里只控制推理/评估时是否切换为 DDIM
   # DDIM 推理步数可远少于训练步数（如 50 步），速度更快
   # -------------------------
   parser.add_argument("--use_ddim_sampling", action="store_true",
                       help="推理/评估时使用 DDIM 采样器替代 DDPM，可大幅加快生成速度")
   parser.add_argument("--ddim_eta", type=float, default=0.0,
                       help="DDIM 随机性系数：0.0 为完全确定性采样，1.0 退化为 DDPM")

   # -------------------------
   # 数据路径
   # -------------------------
   parser.add_argument("--train_gt_csv_path", type=str,
                       default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
                       help="训练集 GroundTruth CSV 路径，列格式：image,MEL,NV,BCC,AKIEC,BKL,DF,VASC")
   parser.add_argument("--val_gt_csv_path", type=str,
                       default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
                       help="验证集 GroundTruth CSV 路径，格式同训练集")
   parser.add_argument("--train_img_dir", type=str,
                       default="dataset/ISIC2018_Task3_Training_Input",
                       help="训练集图片目录，图片命名格式：<image_id>.jpg")
   parser.add_argument("--val_img_dir", type=str,
                       default="dataset/ISIC2018_Task3_Validation_Input",
                       help="验证集图片目录，图片命名格式：<image_id>.jpg")

   # -------------------------
   # 数据过滤模式
   # all:          使用全部 7 个类别的图片
   # single_label: 只使用 --target_label 指定的单个类别（用于类别专属生成模型）
   # -------------------------
   parser.add_argument("--data_mode", type=str, default="all",
                       choices=["all", "single_label"],
                       help="all: 使用全部类别; single_label: 只使用一个类别（需配合 --target_label）")
   parser.add_argument("--target_label", type=str, default=None,
                       choices=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC",
                                "0", "1", "2", "3", "4", "5", "6"],
                       help="当 data_mode=single_label 时指定目标类别；"
                            "可用类别名（MEL/NV/BCC/AKIEC/BKL/DF/VASC）"
                            "或对应索引（0~6）")

   # -------------------------
   # 输出目录
   # -------------------------
   parser.add_argument("--output_root", type=str, default="experiments",
                       help="所有实验结果的根目录，每次新运行会在此下创建带时间戳的子文件夹")

   # -------------------------
   # 图像参数
   # -------------------------
   parser.add_argument("--resolution", type=int, default=128,
                       help="训练和生成图像的分辨率（正方形），同时决定 UNet 输入尺寸")

   # -------------------------
   # 训练超参数
   # -------------------------
   parser.add_argument("--train_batch_size", type=int, default=32,
                       help="训练时每个 GPU 的 batch size")
   parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="评估/生成时的 batch size（同时用于 FID 采样和可视化样本生成）")
   parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="DataLoader 的并行读取进程数")
   parser.add_argument("--num_epochs", type=int, default=40,
                       help="总训练轮数（含 resume 续训时已完成的 epoch）")
   parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数，等效 batch size = train_batch_size × gradient_accumulation_steps")

   # -------------------------
   # 优化器参数（AdamW）
   # -------------------------
   parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="AdamW 基础学习率")
   parser.add_argument("--adam_beta1", type=float, default=0.95,
                       help="AdamW 一阶矩估计的指数衰减率")
   parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="AdamW 二阶矩估计的指数衰减率")
   parser.add_argument("--adam_weight_decay", type=float, default=1e-6,
                       help="AdamW 权重衰减系数")
   parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                       help="AdamW 数值稳定性 epsilon，防止除零")

   # -------------------------
   # 学习率调度器
   # -------------------------
   parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       help="学习率调度策略，传入 diffusers get_scheduler 的 name 参数"
                            "（如 cosine / linear / constant 等）")
   parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="学习率从 0 线性 warmup 到 learning_rate 所需的优化步数")

   # -------------------------
   # 混合精度
   # -------------------------
   parser.add_argument("--mixed_precision", type=str, default="no",
                       choices=["no", "fp16", "bf16"],
                       help="混合精度训练：no=全 fp32；fp16=半精度（适合消费级 GPU）；"
                            "bf16=BFloat16（适合 A100/H100）")

   # -------------------------
   # DDPM 噪声调度参数
   # -------------------------
   parser.add_argument("--ddpm_num_steps", type=int, default=1000,
                       help="DDPM 训练时的总扩散步数 T")
   parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000,
                       help="推理/采样时的去噪步数；使用 DDIM 时可设为 50~200 以加速")
   parser.add_argument("--ddpm_beta_schedule", type=str, default="linear",
                       help="噪声调度方案：linear（原始 DDPM）或 squaredcos_cap_v2（改进版）")

   # -------------------------
   # 保存与评估频率
   # -------------------------
   parser.add_argument("--save_images_epochs", type=int, default=10,
                       help="每隔多少 epoch 保存一批可视化生成样本")
   parser.add_argument("--save_model_epochs", type=int, default=10,
                       help="每隔多少 epoch 保存一次模型 checkpoint")
   parser.add_argument("--eval_epochs", type=int, default=10,
                       help="每隔多少 epoch 计算一次 FID 和 Precision/Recall")

   # -------------------------
   # FID 采样数量
   # 设为 0 可跳过对应 split 的 FID 计算
   # -------------------------
   parser.add_argument("--num_fid_samples_train", type=int, default=1024,
                       help="用于计算训练集 FID 的生成图片数量；0 表示跳过训练集 FID")
   parser.add_argument("--num_fid_samples_val", "--num_fid_samples_valid",
                       dest="num_fid_samples_val", type=int, default=194,
                       help="用于计算验证集 FID 的生成图片数量；0 表示跳过验证集 FID。"
                            "（--num_fid_samples_valid 为兼容旧命令的别名）")

   # -------------------------
   # Manifold Improved Precision & Recall 参数
   # -------------------------
   parser.add_argument("--ipr_k", type=int, default=3,
                       help="流形估计的 k 近邻数；k=3 为论文默认值，"
                            "值越大流形越宽松，precision 倾向偏高")

   # -------------------------
   # 随机种子（复现）
   # -------------------------
   parser.add_argument("--seed", type=int, default=42,
                       help="全局随机种子，固定后可复现训练结果")

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
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)


def save_diffusers_model_index_copy(exp_dir, metadata_dir):
    src = os.path.join(exp_dir, "model_index.json")
    dst = os.path.join(metadata_dir, "diffusers_pipeline_model_index.json")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    return dst


# =========================================================
# 从 checkpoint 路径中提取实验文件夹路径
#
# 约定：checkpoint 保存在 <exp_dir>/checkpoints/last.pth.tar
# 因此 exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
# 同时 checkpoint 内的 "args" 字段也保存了 exp_dir，两路互为备份
# =========================================================
def recover_exp_dir_from_checkpoint(checkpoint_path, checkpoint_data):
    """
    优先从 checkpoint 内保存的 args 中读取 exp_dir（更可靠）
    如果没有，则用文件路径推算
    """
    if "exp_dir" in checkpoint_data:
        return checkpoint_data["exp_dir"]
    # 备用：从路径推断 <exp_dir>/checkpoints/last.pth.tar
    checkpoints_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    exp_dir = os.path.dirname(checkpoints_dir)
    return exp_dir


# =========================================================
# 3. 数据集
# =========================================================
class ISIC2018DDPMDataset(Dataset):
    def __init__(self, gt_csv_path, img_dir, transform=None, data_mode="all", target_label=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data_mode = data_mode
        self.target_label = target_label

        df = pd.read_csv(gt_csv_path)
        self.class_columns = [c for c in df.columns if c != "image"]
        df["label_int"] = df[self.class_columns].values.argmax(axis=1)

        if data_mode == "single_label":
            if target_label is None:
                raise ValueError("When data_mode='single_label', --target_label must be provided.")
            if str(target_label).isdigit():
                target_label_idx = int(target_label)
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
        image_id = str(row["image"])
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_int"])
        sample_id = image_id
        return {"input": image, "label": label, "sample_id": sample_id}


# =========================================================
# 4. FID 工具
# =========================================================
def tensor_to_uint8_for_fid(x):
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x


@torch.no_grad()
def collect_real_uint8_images(real_loader, device, num_samples):
    if num_samples <= 0:
        return torch.empty(0, 3, 0, 0, dtype=torch.uint8, device=device), 0
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
def generate_fake_images_for_fid(accelerator, pipeline, num_gen_samples, fake_save_root,
                                  epoch, num_inference_steps, eval_batch_size,
                                  use_ddim_sampling=False, ddim_eta=0.0):
    if num_gen_samples <= 0:
        return None, ""
    device = accelerator.device
    disable_pipeline_progress_bar(pipeline)
    generated_dir = os.path.join(fake_save_root, f"epoch_{epoch:03d}_shared_generated")
    os.makedirs(generated_dir, exist_ok=True)
    fake_uint8_batches = []
    fake_count = 0
    batch_idx = 0
    progress_bar = tqdm(total=num_gen_samples, desc="FID generated images", leave=True)
    while fake_count < num_gen_samples:
        cur_bs = min(eval_batch_size, num_gen_samples - fake_count)
        generator = torch.Generator(device=device).manual_seed(1000 + epoch * 100 + batch_idx)
        if use_ddim_sampling:
            fake_pil_images = pipeline(batch_size=cur_bs, generator=generator,
                                       num_inference_steps=num_inference_steps,
                                       eta=ddim_eta, output_type="pil").images
        else:
            fake_pil_images = pipeline(batch_size=cur_bs, generator=generator,
                                       num_inference_steps=num_inference_steps,
                                       output_type="pil").images
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
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(real_images_uint8, real=True)
    fid.update(fake_images_uint8[:real_images_uint8.size(0)], real=False)
    return float(fid.compute().item())


# =========================================================
# Manifold Improved Precision & Recall
#
# 算法来自 Kynkäänniemi et al. NeurIPS 2019 (arXiv:1904.06991)
# 实现参考 yj-uh/improved-precision-and-recall-metric-pytorch
#
# 核心思路：
#   1. 用预训练模型的 fc2 层（classifier[:4]）提取 4096 维特征
#   2. 对 real 和 fake 各自构建 k-NN 流形（每个样本的半径 = 到第 k 个近邻的距离）
#   3. Precision = fake 中有多少落在 real 流形内（衡量生成质量）
#   4. Recall    = real 中有多少落在 fake 流形内（衡量覆盖度）
# =========================================================

# 用 namedtuple 存流形（特征向量 + 各样本的 k-NN 半径）
Manifold = namedtuple("Manifold", ["features", "radii"])


def _build_vgg16_feature_extractor(device):
    """
    加载预训练 VGG16，只保留到 classifier[3]（即 fc2 的输出，4096 维）
    使用原生 torchvision，不需要 timm
    """
    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    vgg = vgg.to(device).eval()
    return vgg


@torch.no_grad()
def _extract_vgg16_features(images_uint8, vgg16, device, batch_size=64):
    """
    从 uint8 图像张量中提取 VGG16 fc2 特征

    参数:
        images_uint8: shape [N, 3, H, W]，dtype=uint8，值域 [0,255]
        vgg16: 预训练 VGG16 模型
        device: cuda / cpu
        batch_size: 每批处理多少张图

    返回:
        features: np.ndarray, shape [N, 4096]
    """
    # VGG16 ImageNet 标准化参数
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_feats = []
    n = images_uint8.shape[0]
    for start in range(0, n, batch_size):
        batch = images_uint8[start:start + batch_size].to(device)
        # uint8 [0,255] → float [0,1]
        batch = batch.float() / 255.0
        # resize 到 VGG16 输入尺寸 224x224
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        # ImageNet 归一化
        batch = (batch - mean) / std
        # 提取 conv 特征
        conv_feat = vgg16.features(batch)                    # [B, 512, 7, 7]
        conv_feat = vgg16.avgpool(conv_feat)                 # [B, 512, 7, 7]
        conv_feat = conv_feat.view(conv_feat.size(0), -1)   # [B, 25088]
        # 提取 fc1 + relu + fc2（classifier[0..3]）= 4096 维
        fc_feat = vgg16.classifier[:4](conv_feat)           # [B, 4096]
        all_feats.append(fc_feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)  # [N, 4096]


def _compute_pairwise_distances(X, Y=None):
    """
    计算两组特征向量之间的欧氏距离矩阵

    参数:
        X: np.ndarray [N, D]
        Y: np.ndarray [M, D]，若为 None 则计算 X 自身的距离矩阵

    返回:
        distances: np.ndarray [N, M]
    """
    X = X.astype(np.float64)
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # [N, 1]

    if Y is None:
        Y = X
        Y_sq = X_sq
    else:
        Y = Y.astype(np.float64)
        Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)  # [M, 1]

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
    diff_sq = X_sq + Y_sq.T - 2.0 * X.dot(Y.T)
    diff_sq = np.clip(diff_sq, 0, None)  # 防止数值误差导致负数
    return np.sqrt(diff_sq)


def _distances_to_radii(distances, k):
    """
    对每个样本，取它到第 k 个近邻的距离作为流形半径
    注意：distances[i, i] = 0（自身），所以第 k 近邻实际上是第 k+1 小的值

    参数:
        distances: np.ndarray [N, N] 自距离矩阵
        k: int，近邻数

    返回:
        radii: np.ndarray [N]
    """
    n = distances.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        # kth NN = 排除自身(0)后的第 k 小距离
        kth_idx = np.argpartition(distances[i], k + 1)[:k + 1]
        radii[i] = distances[i][kth_idx].max()
    return radii


def _build_manifold(features, k):
    """
    给定特征向量，构建流形（计算 k-NN 半径）

    参数:
        features: np.ndarray [N, D]
        k: int

    返回:
        Manifold(features, radii)
    """
    distances = _compute_pairwise_distances(features)
    radii = _distances_to_radii(distances, k)
    return Manifold(features, radii)


def _compute_precision_or_recall(manifold_ref, feats_query):
    """
    计算 query 中有多少样本落在 ref 流形内

    Precision: ref=real,  query=fake  → 生成质量
    Recall:    ref=fake,  query=real  → 覆盖度

    参数:
        manifold_ref:  Manifold(features [N, D], radii [N])
        feats_query:   np.ndarray [M, D]

    返回:
        score: float in [0, 1]
    """
    # dist[i, j] = 第 j 个 query 到第 i 个 ref 的距离
    dist = _compute_pairwise_distances(manifold_ref.features, feats_query)  # [N, M]
    # query[j] 在 ref 流形内 ⟺ 至少有一个 ref[i] 的超球覆盖了它
    # 即 dist[i, j] < radii[i] 对某个 i 成立
    in_manifold = (dist < manifold_ref.radii[:, None]).any(axis=0)  # [M]
    return float(in_manifold.mean())


@torch.no_grad()
def compute_manifold_precision_recall(real_images_uint8, fake_images_uint8, device, k=3, vgg_batch_size=64):
    """
    计算 Improved Precision 和 Recall（Kynkäänniemi et al. NeurIPS 2019）

    参数:
        real_images_uint8: torch.Tensor [N, 3, H, W], uint8
        fake_images_uint8: torch.Tensor [M, 3, H, W], uint8
        device: torch.device
        k: int，流形估计的近邻数，默认 3
        vgg_batch_size: int，VGG16 特征提取的批大小

    返回:
        precision: float，生成图像质量（fake 落在 real 流形内的比例）
        recall:    float，覆盖度（real 落在 fake 流形内的比例）
    """
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None, None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None, None

    print("  [IPR] Loading VGG16 for Improved Precision/Recall...")
    vgg16 = _build_vgg16_feature_extractor(device)

    print("  [IPR] Extracting features for real images...")
    real_feats = _extract_vgg16_features(real_images_uint8, vgg16, device, batch_size=vgg_batch_size)

    print("  [IPR] Extracting features for fake images...")
    # fake 只取和 real 等量的样本，保持 precision/recall 的对称性
    n_real = real_images_uint8.size(0)
    fake_feats = _extract_vgg16_features(
        fake_images_uint8[:n_real], vgg16, device, batch_size=vgg_batch_size
    )

    print("  [IPR] Building manifolds (k={})...".format(k))
    manifold_real = _build_manifold(real_feats, k)
    manifold_fake = _build_manifold(fake_feats, k)

    print("  [IPR] Computing precision and recall...")
    precision = _compute_precision_or_recall(manifold_real, fake_feats)
    recall    = _compute_precision_or_recall(manifold_fake, real_feats)

    # 释放 VGG 显存
    del vgg16
    torch.cuda.empty_cache()

    return precision, recall


# =========================================================
# 5. 主函数
# =========================================================
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # =========================================================
    #  实验目录逻辑
    #   - 如果 --resume_from_checkpoint 为 None，新建文件夹
    #   - 如果指定了 checkpoint，则从 checkpoint 中恢复 exp_dir，
    #     复用原来的文件夹（metrics/samples/fid 等目录都接着用）
    # =========================================================
    if args.resume_from_checkpoint is not None:
        # 先把 checkpoint 加载进来，从里面取 exp_dir
        print(f"Loading checkpoint from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        # 恢复原实验文件夹路径
        recovered_exp_dir = recover_exp_dir_from_checkpoint(args.resume_from_checkpoint, checkpoint)
        # exp_name 从路径中提取（只是用于显示，不影响目录）
        exp_name = os.path.basename(recovered_exp_dir)
        # 直接重建文件夹引用（文件夹本身已存在，exist_ok=True 不会覆盖内容）
        exp_folders = setup_experiment_folders(
            base_dir=os.path.dirname(recovered_exp_dir),
            exp_name=exp_name
        )
        print(f"Resuming experiment at: {recovered_exp_dir}")
    else:
        checkpoint = None  # 没有 checkpoint
        exp_name = make_experiment_name(args)
        exp_folders = setup_experiment_folders(args.output_root, exp_name)

    metrics_csv_path  = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(exp_folders["metadata_dir"], "experiment_metadata.json")
    best_model_path   = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

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
    train_dataset = ISIC2018DDPMDataset(
        gt_csv_path=args.train_gt_csv_path,
        img_dir=args.train_img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label
    )
    val_dataset = ISIC2018DDPMDataset(
        gt_csv_path=args.val_gt_csv_path,
        img_dir=args.val_img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label
    )

    gt_df = pd.read_csv(args.train_gt_csv_path)
    class_names = [c for c in gt_df.columns if c != "image"]
    num_classes = len(class_names)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_indices = np.arange(len(train_dataset))
    val_indices   = np.arange(len(val_dataset))
    train_class_distribution = count_labels_from_indices(train_dataset.labels, train_indices, class_names)
    val_class_distribution   = count_labels_from_indices(val_dataset.labels, val_indices, class_names)
    print_class_distribution("Train Dataset Class Distribution", train_class_distribution)
    print_class_distribution("Validation Dataset Class Distribution", val_class_distribution)

    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers, pin_memory=pin_memory, drop_last=True,
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_eval_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.dataloader_num_workers, pin_memory=pin_memory, drop_last=False,
    )

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
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    # -------------------------
    # 断点继续训练相关状态
    # -------------------------
    start_epoch   = 0
    global_step   = 0
    best_val_fid  = float("inf")
    best_train_fid = float("inf")

    # =========================================================
    # checkpoint 恢复权重
    # checkpoint 已在上面加载，这里直接使用
    # =========================================================
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch    = checkpoint["epoch"]
        global_step    = checkpoint.get("global_step", 0)
        best_val_fid   = checkpoint.get("best_val_fid", float("inf"))
        best_train_fid = checkpoint.get("best_train_fid", float("inf"))
        print(f"Resume training from epoch {start_epoch + 1}")
        print(f"Recovered global_step = {global_step}")
        print(f"Recovered best_val_fid = {best_val_fid}")
        print(f"Recovered best_train_fid = {best_train_fid}")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # -------------------------
    # metadata 初始化（复用时从磁盘读取，新建时从头写）
    # -------------------------
    if args.resume_from_checkpoint is not None and os.path.exists(metadata_json_path):
        # 复用：加载原有 metadata，后续只追加更新
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            experiment_metadata = json.load(f)
        print(f"Loaded existing metadata from: {metadata_json_path}")
    else:
        experiment_metadata = {
            "experiment_name": exp_name,
            "experiment_dir": exp_folders["exp_dir"],
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "mixed_precision": args.mixed_precision,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "data": {
                "train_gt_csv_path": args.train_gt_csv_path,
                "val_gt_csv_path": args.val_gt_csv_path,
                "train_img_dir": args.train_img_dir,
                "val_img_dir": args.val_img_dir,
                "data_mode": args.data_mode,
                "target_label": args.target_label,
                "num_classes": num_classes,
                "class_names": class_names,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "class_distribution": {
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

        epoch_loss_sum   = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],), device=clean_images.device,
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
            epoch_loss_sum   += loss_item * clean_images.size(0)
            epoch_loss_count += clean_images.size(0)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": f"{loss_item:.6f}", "update": global_step})

        progress_bar.close()
        accelerator.wait_for_everyone()

        train_loss_epoch = epoch_loss_sum / max(epoch_loss_count, 1)

        need_save_images = ((epoch + 1) % args.save_images_epochs == 0) or (epoch == args.num_epochs - 1)
        need_save_model  = ((epoch + 1) % args.save_model_epochs == 0) or (epoch == args.num_epochs - 1)

        enable_train_fid = args.num_fid_samples_train > 0
        enable_val_fid   = args.num_fid_samples_val > 0

        need_eval = (
            (((epoch + 1) % args.eval_epochs == 0) or (epoch == args.num_epochs - 1))
            and (enable_train_fid or enable_val_fid)
        )

        fid_train_value  = None
        fid_val_value    = None
        # Precision/Recall 结果变量
        train_precision  = None
        train_recall     = None
        val_precision    = None
        val_recall       = None
        train_fid_json_path  = ""
        val_fid_json_path    = ""
        shared_generated_dir = ""
        sample_dir           = ""
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
                sample_progress_bar = tqdm(total=args.eval_batch_size, desc="Generating samples", leave=True)
                if args.use_ddim_sampling:
                    images = pipeline(batch_size=args.eval_batch_size, generator=generator,
                                      num_inference_steps=args.ddpm_num_inference_steps,
                                      eta=args.ddim_eta, output_type="pil").images
                else:
                    images = pipeline(batch_size=args.eval_batch_size, generator=generator,
                                      num_inference_steps=args.ddpm_num_inference_steps,
                                      output_type="pil").images
                for i, image in enumerate(images):
                    image.save(os.path.join(epoch_dir, f"sample_{i:03d}.png"))
                    sample_progress_bar.update(1)
                sample_progress_bar.close()
                sample_dir = epoch_dir
                print(f"Samples saved to: {sample_dir}")

            # 2) 计算 FID 和 Precision/Recall
            if need_eval:
                print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
                print("-" * 60)

                fake_targets = []
                if enable_train_fid:
                    fake_targets.append(args.num_fid_samples_train)
                if enable_val_fid:
                    fake_targets.append(args.num_fid_samples_val)

                target_fake_count = max(fake_targets) if fake_targets else 0

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

                # --- Train split ---
                if enable_train_fid:
                    train_real_uint8, train_real_count = collect_real_uint8_images(
                        real_loader=train_eval_loader,
                        device=accelerator.device,
                        num_samples=args.num_fid_samples_train,
                    )
                    fid_train_value = compute_fid_from_real_and_fake(
                        train_real_uint8, fake_images_uint8, accelerator.device
                    )

                    # 计算 train split 的 Manifold Precision/Recall
                    train_precision, train_recall = compute_manifold_precision_recall(
                        real_images_uint8=train_real_uint8,
                        fake_images_uint8=fake_images_uint8,
                        device=accelerator.device,
                        k=args.ipr_k,
                    )

                    if fid_train_value is not None:
                        train_fid_json_path = os.path.join(
                            exp_folders["fid_dir"], f"epoch_{epoch + 1:03d}_train_fid.json"
                        )
                        save_json({
                            "epoch": epoch + 1,
                            "split": "train",
                            "num_real_images": int(train_real_count),
                            "num_fake_images": int(train_real_count),
                            "fid": float(fid_train_value),
                            # 写入 JSON
                            "precision": float(train_precision) if train_precision is not None else None,
                            "recall":    float(train_recall)    if train_recall    is not None else None,
                            "ipr_k": args.ipr_k,
                            "generated_dir": shared_generated_dir,
                            "sampler": "ddim" if args.use_ddim_sampling else "ddpm",
                            "num_inference_steps": int(args.ddpm_num_inference_steps),
                            "ddim_eta": float(args.ddim_eta) if args.use_ddim_sampling else None,
                            "shared_fake_images": True,
                        }, train_fid_json_path)
                        print(f"Train FID: {fid_train_value:.6f}  "
                              f"Precision: {train_precision:.4f}  Recall: {train_recall:.4f}")
                    else:
                        print("Train FID skipped.")
                else:
                    print("Train FID skipped (--num_fid_samples_train 0).")

                # --- Val split ---
                if enable_val_fid:
                    val_real_uint8, val_real_count = collect_real_uint8_images(
                        real_loader=val_eval_loader,
                        device=accelerator.device,
                        num_samples=args.num_fid_samples_val,
                    )
                    fid_val_value = compute_fid_from_real_and_fake(
                        val_real_uint8, fake_images_uint8, accelerator.device
                    )

                    #  计算 val split 的 Manifold Precision/Recall
                    val_precision, val_recall = compute_manifold_precision_recall(
                        real_images_uint8=val_real_uint8,
                        fake_images_uint8=fake_images_uint8,
                        device=accelerator.device,
                        k=args.ipr_k,
                    )

                    if fid_val_value is not None:
                        val_fid_json_path = os.path.join(
                            exp_folders["fid_dir"], f"epoch_{epoch + 1:03d}_val_fid.json"
                        )
                        save_json({
                            "epoch": epoch + 1,
                            "split": "val",
                            "num_real_images": int(val_real_count),
                            "num_fake_images": int(val_real_count),
                            "fid": float(fid_val_value),
                            # 写入 JSON
                            "precision": float(val_precision) if val_precision is not None else None,
                            "recall":    float(val_recall)    if val_recall    is not None else None,
                            "ipr_k": args.ipr_k,
                            "generated_dir": shared_generated_dir,
                            "sampler": "ddim" if args.use_ddim_sampling else "ddpm",
                            "num_inference_steps": int(args.ddpm_num_inference_steps),
                            "ddim_eta": float(args.ddim_eta) if args.use_ddim_sampling else None,
                            "shared_fake_images": True,
                        }, val_fid_json_path)
                        print(f"Val FID: {fid_val_value:.6f}  "
                              f"Precision: {val_precision:.4f}  Recall: {val_recall:.4f}")
                    else:
                        print("Val FID skipped.")
                else:
                    print("Val FID skipped (--num_fid_samples_val 0).")

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
                #  checkpoint 中额外保存 exp_dir，供 resume 时读取
                save_checkpoint({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_val_fid": best_val_fid,
                    "best_train_fid": best_train_fid,
                    "args": vars(args),
                    "exp_dir": exp_folders["exp_dir"],  # 供 resume 时定位实验文件夹
                }, is_best=is_best, save_dir=exp_folders["checkpoints_dir"], filename="last.pth.tar")

                if is_best:
                    experiment_metadata["best_result"]["best_model_path"] = best_model_path

                pipeline.save_pretrained(exp_folders["exp_dir"])
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
                # Precision/Recall 写入 CSV 和 JSON
                "train_precision": float(train_precision) if train_precision is not None else None,
                "train_recall":    float(train_recall)    if train_recall    is not None else None,
                "val_precision":   float(val_precision)   if val_precision   is not None else None,
                "val_recall":      float(val_recall)      if val_recall      is not None else None,
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