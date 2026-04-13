import argparse
from html import parser
import importlib
import math
import os
import random
import shutil
import time
import warnings
from enum import Enum
from collections import Counter

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

import torchvision.models as models
import torchvision.transforms as transforms

import json
from datetime import datetime

import numpy as np
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 新增：使用 tqdm 改善训练/验证进度条

from diffusers import (
    DDPMPipeline,
    DDIMPipeline,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from accelerate import Accelerator

from collections import Counter

# =========================================================
# 1. 支持的模型名称
# =========================================================
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


# =========================================================
# 2. 命令行参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch ResNet50 Training for ISIC2018 Task3"
    )

    # 1. 基础训练参数
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=model_names,
        help="模型架构，支持torchvision内置模型 (default: resnet50)",
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="训练总轮数 (default: 20)"
    )
    parser.add_argument(
        "--train_batch_size",
        default=128,
        type=int,
        help="每个mini-batch的样本数 (default: 128)",
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="数据加载进程数 (default: 4)"
    )

    parser.add_argument(
        "--eval-freq",
        default=1,
        type=int,
        help="每隔多少个 epoch 做一次验证，默认每个 epoch 都验证一次",
    )

    parser.add_argument(
        "--save-every-eval",
        action="store_true",
        help="若设置，则每次进行验证时都额外保存一个对应 epoch 的 checkpoint",
    )

    # 2. 优化器与学习率相关参数
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        dest="lr",
        help="初始学习率 (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="动量参数 (default: 0.9)"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        dest="weight_decay",
        help="权重衰减 (default: 1e-4)",
    )

    # 3. 运行控制参数
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        help="打印频率（保留参数，主流程用tqdm进度条）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="恢复训练的checkpoint路径 (default: None)",
    )
    parser.add_argument("--evaluate", action="store_true", help="仅在验证集上评估模型")

    # 4. 预训练权重与复现性参数
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="预训练权重名称，如 DEFAULT / IMAGENET1K_V1 / IMAGENET1K_V2",
    )
    parser.add_argument("--seed", default=42, type=int, help="随机种子，保证实验可复现")

    # 5. 设备相关参数
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="使用的GPU编号，如0。未设置时自动选择cuda/cpu",
    )

    # 6. 数据路径参数（可选，便于灵活指定数据集位置）
    parser.add_argument(
        "--train-gt-csv",
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        type=str,
        help="训练集标签CSV路径 (default: dataset/ISIC2018_Task3_Training_GroundTruth.csv)",
    )
    parser.add_argument(
        "--val-gt-csv",
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        type=str,
        help="验证集标签CSV路径 (default: dataset/ISIC2018_Task3_Validation_GroundTruth.csv)",
    )
    parser.add_argument(
        "--train-img-dir",
        default="dataset/ISIC2018_Task3_Training_Input",
        type=str,
        help="训练集图片文件夹路径 (default: dataset/ISIC2018_Task3_Training_Input)",
    )
    parser.add_argument(
        "--val-img-dir",
        default="dataset/ISIC2018_Task3_Validation_Input",
        type=str,
        help="验证集图片文件夹路径 (default: dataset/ISIC2018_Task3_Validation_Input)",
    )

    # 7. 扩散模型生成参数
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        default=None,
        help="扩散模型checkpoint路径",
    )
    parser.add_argument(
        "--ddpm_num_steps", type=int, default=1000, help="DDPM 训练时的总扩散步数 T"
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        help="噪声调度方案：linear（原始 DDPM）或 squaredcos_cap_v2（改进版）",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=7,
        help="类别数量",
    )
    parser.add_argument(
        "--time_scale_shift",
        type=str,
        default="default",
        help="时间尺度移位策略",
    )
    parser.add_argument(
        "--scheduler_config",
        type=str,
        default=None,
        help="采样调度器配置路径",
    )

    # 扩充比例
    # ISIC2018 Task3 类别对照：
    #   0=MEL(Melanoma), 1=NV(Melanocytic nevus), 2=BCC(Basal cell carcinoma)
    #   3=AKIEC(Actinic keratosis/Bowen's disease), 4=BKL(Benign keratosis)
    #   5=DF(Dermatofibroma), 6=VASC(Vascular lesion)
    parser.add_argument(
        "--ratios",
        type=str,
        nargs="+",
        default=None,
        help="例: '0:0.5 1:1.0 2:2.0 3:0.0 4:0.5 5:1.0 6:1.0'",
    )

    # 用 arg 指定要导入哪个 diffusion 模块
    parser.add_argument(
        "--diffusion_module",
        type=str,
        default="diffusion",
        help="要导入的 diffusion 模块名",
    )

    # 批次大小
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=32,
        help="生成批次大小和classifier 训练批次大小",
    )

    # 7.加速器相关参数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "混合精度模式，自动选择最佳选项（默认: None）。"
            "在支持 bf16 的 GPU 上使用 'bf16'，否则使用 'fp16'。"
            "设置为 'no' 可禁用混合精度。"
        ),
    )

    return parser.parse_args()


best_acc1 = 0.0


# =========================================================
# 3. 数据集定义
# =========================================================
class ISICResNetDataset(Dataset):
    def __init__(self, gt_csv_path, img_dir, transform=None):
        """
        初始化 ISIC2018 Task3 单模态分类数据集
        图片平铺在一个目录中，CSV 第一列为 image，后面是 7 个 one-hot / confidence 类别列
        """
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(gt_csv_path)

        self.class_columns = [c for c in df.columns if c != "image"]
        self.df = df.reset_index(drop=True)

        # 提前计算整数标签，后面统计类别分布会用到
        self.labels = self.df[self.class_columns].values.argmax(axis=1).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["image"])
        img_name = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_array = row[self.class_columns].values.astype(float)
        label = torch.tensor(label_array.argmax(), dtype=torch.long)

        sample_id = image_id

        return image, label, sample_id


class SyntheticISICDataset(Dataset):
    """
    用扩散模型生成合成图片的数据集。
    返回格式与 ISICResNetDataset 完全一致: (image, label, sample_id)，
    可直接用 ConcatDataset 与原始数据集混合。
    """

    def __init__(
        self,
        diffusion_model,
        sampling_scheduler,
        device,
        resolution=64,
        num_inference_steps=50,
        ddim_eta=0.0,
        gen_ratios=None,  # dict {class_idx: ratio}，ratio>0 表示额外生成原数量的多少倍
        original_class_counts=None,  # dict {class_idx: count}，原始各类别样本数量
        class_names=None,
        transform=None,
        seed=42,
        sample_images_with_model=None,
        batch_size=20,
    ):
        self.device = device
        self.resolution = resolution
        self.transform = transform

        self.gen_ratios = gen_ratios or {}
        self.original_class_counts = original_class_counts or {}

        self.num_classes = len(class_names) if class_names else 7
        self.class_names = class_names or [
            f"class_{i}" for i in range(self.num_classes)
        ]

        # 只对明确指定的类别生成，未指定类别默认 0.0
        self.gen_targets = {}
        for c in range(self.num_classes):
            ratio = self.gen_ratios.get(c, 0.0)
            original_count = self.original_class_counts.get(c, 0)
            self.gen_targets[c] = int(original_count * ratio)

        if all(v == 0 for v in self.gen_targets.values()):
            self.samples = []
            print("[SyntheticISICDataset] 没有需要生成的样本。")
            return

        generator = torch.Generator(device=device).manual_seed(seed)
        all_generated_images = []
        all_generated_labels = []
        all_generated_ids = []

        # 预先整理需要生成的类别信息
        active_classes = []
        total_targets = 0
        for class_idx, target_count in self.gen_targets.items():
            if target_count > 0:
                num_batches = (target_count + batch_size - 1) // batch_size
                active_classes.append(
                    {
                        "class_idx": class_idx,
                        "class_name": self.class_names[class_idx],
                        "target_count": target_count,
                        "num_batches": num_batches,
                    }
                )
                total_targets += target_count

        if len(active_classes) == 0:
            self.samples = []
            print("[SyntheticISICDataset] 没有需要生成的样本。")
            return

        # -------- 日志输出：开始前只打印简短摘要 --------
        print("\n[SyntheticISICDataset] 开始生成")
        print(f"  分辨率: {self.resolution}")
        print(f"  推理步数: {num_inference_steps}")
        print(f"  批次大小: {batch_size}")
        print(f"  总生成张数: {total_targets}")
        print(f"  生成类别数: {len(active_classes)}")

        overall_start = time.time()
        total_generated = 0

        with torch.inference_mode():
            for item in active_classes:
                class_idx = item["class_idx"]
                class_name = item["class_name"]
                target_count = item["target_count"]
                num_batches = item["num_batches"]

                generated_this_class = 0
                class_start = time.time()

                # 每个类别单独一个进度条，不在同一个条里显示所有信息
                pbar = tqdm(
                    total=target_count,
                    desc=f"[生成 {class_name}]",
                    unit="img",
                    dynamic_ncols=True,
                    leave=True,
                )

                for _ in range(num_batches):
                    current_batch_size = min(
                        batch_size, target_count - generated_this_class
                    )
                    if current_batch_size <= 0:
                        break

                    generated_batch = sample_images_with_model(
                        model=diffusion_model,
                        sampling_scheduler=sampling_scheduler,
                        device=device,
                        resolution=resolution,
                        batch_size=current_batch_size,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        use_class_conditioning=True,
                        class_labels=torch.tensor(
                            [class_idx] * current_batch_size, device=device
                        ),
                        ddim_eta=ddim_eta,
                    )

                    if isinstance(generated_batch, torch.Tensor):
                        gen_np = (
                            generated_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                    else:
                        gen_np = np.asarray(generated_batch)

                    if gen_np.dtype != np.uint8:
                        raise ValueError(
                            "sample_images_with_model 的输出不是 uint8。"
                            "请先在 sample_images_with_model 内部把图像转换到 [0,255] 并转成 uint8。"
                        )

                    for i in range(current_batch_size):
                        img_pil = Image.fromarray(gen_np[i])
                        all_generated_images.append(img_pil)
                        all_generated_labels.append(class_idx)
                        sample_id = f"synth_{class_name}_{generated_this_class + i:06d}"
                        all_generated_ids.append(sample_id)

                    generated_this_class += current_batch_size
                    total_generated += current_batch_size
                    pbar.update(current_batch_size)

                pbar.close()

                class_time = time.time() - class_start
                print(
                    f"[SyntheticISICDataset] {class_name}: "
                    f"{generated_this_class} 张, {class_time:.1f}s"
                )

        overall_elapsed = time.time() - overall_start
        avg_time = overall_elapsed / total_generated if total_generated > 0 else 0.0

        print(
            f"[SyntheticISICDataset] 生成完成: "
            f"{total_generated} 张, 总耗时 {overall_elapsed:.1f}s, "
            f"平均 {avg_time:.2f}s/张"
        )

        self.samples = list(
            zip(all_generated_images, all_generated_labels, all_generated_ids)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label, sample_id = self.samples[idx]

        image = image.copy()

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label, sample_id


def concat_datasets(original_dataset, output_dir, args, device):
    num_classes = args.num_classes
    class_names = list(original_dataset.class_columns)

    # ---------- 解析扩充比例 ----------
    gen_ratios = {c: 0.0 for c in range(num_classes)}
    if args.ratios is not None:
        for item in args.ratios:
            class_idx, ratio = item.split(":")
            gen_ratios[int(class_idx)] = float(ratio)

    active_ratios = {k: v for k, v in gen_ratios.items() if v > 0}
    print("\n[concat_datasets] 构建扩增数据集")
    print(f"  输出目录: {output_dir}")
    print(f"  生效类别比例: {active_ratios if active_ratios else '无'}")

    # =====================================================
    # 先尝试复用已经生成好的图片
    # 条件：
    #   1) 传入了 --resume
    #   2) output_dir 存在
    #   3) 目录下确实已经有图片
    # =====================================================
    can_reuse = False
    if args.resume is not None and os.path.isdir(output_dir):
        for class_name in class_names:
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            has_img = any(
                fname.lower().endswith((".jpg", ".jpeg", ".png"))
                for fname in os.listdir(class_dir)
            )
            if has_img:
                can_reuse = True
                break

    if can_reuse:
        print("[concat_datasets] 检测到 --resume，且已有合成图片，直接复用。")

        synth_dataset = SavedSyntheticISICDataset(
            root_dir=output_dir,
            class_names=class_names,
            transform=original_dataset.transform,
        )

        print(f"[concat_datasets] 复用合成数据集大小: {len(synth_dataset)}")
        print(
            f"[concat_datasets] 拼接后总大小: {len(original_dataset) + len(synth_dataset)}"
        )

        return ConcatDataset([original_dataset, synth_dataset])

    # =====================================================
    # 如果不能复用，走原来的生成流程
    # =====================================================
    if args.diffusion_checkpoint is not None:
        print(f"[concat_datasets] 加载 checkpoint: {args.diffusion_checkpoint}")
        checkpoint = torch.load(args.diffusion_checkpoint, map_location="cpu")
    else:
        raise ValueError("必须提供扩散模型 checkpoint 路径")

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
        num_class_embeds=num_classes,
        resnet_time_scale_shift=args.time_scale_shift,
    )

    sampling_scheduler = DDIMScheduler.from_pretrained(
        args.scheduler_config, subfolder="scheduler"
    )
    sampling_scheduler.set_timesteps(args.num_inference_steps)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    diffusion_module = importlib.import_module(args.diffusion_module)
    original_class_counts = get_class_counts_from_dataset(original_dataset)

    synth_dataset = SyntheticISICDataset(
        diffusion_model=model,
        sampling_scheduler=sampling_scheduler,
        device=device,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        ddim_eta=args.ddim_eta,
        gen_ratios=gen_ratios,
        original_class_counts=original_class_counts,
        class_names=class_names,
        transform=original_dataset.transform,
        seed=args.seed,
        sample_images_with_model=diffusion_module.sample_images_with_model,
        batch_size=args.gen_batch_size,
    )

    # ---------- 保存合成图片到磁盘 ----------
    os.makedirs(output_dir, exist_ok=True)

    for img_pil, label, sample_id in synth_dataset.samples:
        class_name = synth_dataset.class_names[label]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_pil.save(os.path.join(class_dir, f"{sample_id}.jpg"))

    print(f"[concat_datasets] 合成数据集大小: {len(synth_dataset)}")
    print(f"[concat_datasets] 保存完成: {output_dir}")
    print(
        f"[concat_datasets] 拼接后总大小: {len(original_dataset) + len(synth_dataset)}"
    )

    return ConcatDataset([original_dataset, synth_dataset])


def get_class_counts_from_dataset(dataset):
    """统计原始数据集各类别样本数量。"""
    if not hasattr(dataset, "labels"):
        raise AttributeError("dataset 必须有 labels 属性，当前数据集不满足。")
    return dict(Counter(dataset.labels))


# =========================================================
# 作用：
#   - 当 --resume 被使用时，如果 train_augmented_data 目录下已经有合成图
#   - 直接从磁盘读取这些图，不再重新调用扩散模型生成
# =========================================================
class SavedSyntheticISICDataset(Dataset):
    """
    从磁盘读取已经生成好的合成图片。
    目录结构要求：
        output_dir/
            MEL/
                synth_MEL_xxxxxx.jpg
            NV/
                synth_NV_xxxxxx.jpg
            ...
    返回格式与 ISICResNetDataset / SyntheticISICDataset 一致：
        (image, label, sample_id)
    """

    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.samples = []

        # 按 class_names 的顺序扫描每个类别目录
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in sorted(os.listdir(class_dir)):
                # 只读取常见图片格式
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, fname)
                    sample_id = os.path.splitext(fname)[0]
                    self.samples.append((img_path, label, sample_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, sample_id = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label, sample_id


# =========================================================
# 4. 实验目录与日志工具函数
# =========================================================
def make_experiment_name(args):
    """
    生成规范实验名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_tag = args.weights if args.weights is not None else "scratch"
    seed_tag = f"seed{args.seed}" if args.seed is not None else "seedNone"
    exp_name = f"{timestamp}_{args.arch}_{weights_tag}_lr{args.lr}_bs{args.train_batch_size}_{seed_tag}"
    return exp_name


def setup_experiment_folders(base_dir, exp_name):
    """
    创建规范实验目录
    """
    exp_dir = os.path.join(base_dir, exp_name)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    metadata_dir = os.path.join(exp_dir, "metadata")
    roc_dir = os.path.join(exp_dir, "roc_curves")
    cm_dir = os.path.join(exp_dir, "confusion_matrices")
    predictions_dir = os.path.join(exp_dir, "predictions")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "metadata_dir": metadata_dir,
        "roc_dir": roc_dir,
        "cm_dir": cm_dir,
        "predictions_dir": predictions_dir,
    }


def reuse_experiment_folders(exp_dir):
    """
    复用已有实验目录
    这个函数专门给 resume 场景使用：
    checkpoint 中会保存 exp_dir，这里直接根据旧目录恢复所有子目录路径
    """
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    metadata_dir = os.path.join(exp_dir, "metadata")
    roc_dir = os.path.join(exp_dir, "roc_curves")
    cm_dir = os.path.join(exp_dir, "confusion_matrices")
    predictions_dir = os.path.join(exp_dir, "predictions")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "metadata_dir": metadata_dir,
        "roc_dir": roc_dir,
        "cm_dir": cm_dir,
        "predictions_dir": predictions_dir,
    }


def save_json(data, json_path):
    """
    保存 JSON 文件
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def update_epoch_metrics_csv(metrics_csv_path, row_dict):
    """
    追加保存每轮汇总指标到 CSV
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
    追加保存每轮汇总指标到 JSON
    """
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(row_dict)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_detailed_metrics_json(detailed_metrics, save_dir, epoch):
    """
    保存每个 epoch 的详细指标（包括 per-class 指标）
    """
    json_path = os.path.join(save_dir, f"epoch_{epoch:03d}_detailed_metrics.json")
    save_json(detailed_metrics, json_path)
    return json_path


def save_confusion_matrix_artifacts(y_true, y_pred, class_names, save_dir, epoch):
    """
    保存混淆矩阵图和混淆矩阵原始 CSV
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(save_dir, f"epoch_{epoch:03d}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()

    cm_png_path = os.path.join(save_dir, f"epoch_{epoch:03d}_confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cm_csv_path, cm_png_path


def save_multiclass_roc_artifacts(y_true, y_prob, class_names, save_dir, epoch):
    """
    保存多分类 ROC 曲线图及其原始点数据
    """
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    roc_points = {
        "epoch": epoch,
        "num_classes": num_classes,
        "class_names": class_names,
        "per_class": {},
        "macro_average_auc": None,
    }

    plt.figure(figsize=(10, 8))

    valid_class_count = 0
    all_fpr = []

    for i in range(num_classes):
        class_name = class_names[i]
        y_true_i = y_true_bin[:, i]
        y_prob_i = y_prob[:, i]

        unique_vals = np.unique(y_true_i)
        if len(unique_vals) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_i, y_prob_i)
        roc_auc = auc(fpr, tpr)

        roc_points["per_class"][class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc),
        }

        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={roc_auc:.3f})")
        valid_class_count += 1
        all_fpr.append(fpr)

    if valid_class_count > 0:
        mean_fpr = np.unique(np.concatenate(all_fpr))
        mean_tpr = np.zeros_like(mean_fpr)

        valid_for_macro = 0
        for i in range(num_classes):
            class_name = class_names[i]
            if class_name not in roc_points["per_class"]:
                continue

            fpr = np.array(roc_points["per_class"][class_name]["fpr"])
            tpr = np.array(roc_points["per_class"][class_name]["tpr"])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            valid_for_macro += 1

        if valid_for_macro > 0:
            mean_tpr /= valid_for_macro
            macro_auc = auc(mean_fpr, mean_tpr)
            roc_points["macro_average_auc"] = float(macro_auc)

            plt.plot(
                mean_fpr,
                mean_tpr,
                linestyle="--",
                linewidth=3,
                color="navy",
                label=f"Macro-average ROC (AUC={macro_auc:.3f})",
            )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Epoch {epoch}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_png_path = os.path.join(save_dir, f"epoch_{epoch:03d}_roc.png")
    plt.savefig(roc_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    roc_json_path = os.path.join(save_dir, f"epoch_{epoch:03d}_roc_points.json")
    save_json(roc_points, roc_json_path)

    return roc_png_path, roc_json_path


def save_val_predictions_csv(
    sample_ids, y_true, y_pred, y_prob, class_names, save_dir, epoch
):
    """
    保存样本级验证预测结果
    每个样本一行：
    case_id, y_true, y_pred, prob_MEL, prob_NV, ...
    """
    data = {
        "case_id": list(sample_ids),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

    num_classes = y_prob.shape[1]
    for i in range(num_classes):
        data[f"prob_{class_names[i]}"] = y_prob[:, i].tolist()

    pred_df = pd.DataFrame(data)
    pred_csv_path = os.path.join(save_dir, f"val_predictions_epoch_{epoch:02d}.csv")
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    return pred_csv_path


def count_labels_from_dataset(labels, class_names):
    """
    统计一个数据集里各类别样本数
    """
    counter = Counter(labels)
    count_dict = {}
    for class_idx, class_name in enumerate(class_names):
        count_dict[class_name] = int(counter.get(class_idx, 0))
    return count_dict


def print_class_distribution(title, class_distribution):
    """
    class_distribution: dict，格式例如
    {
        "MEL": 1113,
        "NV": 6705,
        ...
    }
    """
    total = sum(class_distribution.values())

    print(f"\n{title}")
    print("-" * 60)
    for class_name, count in class_distribution.items():
        ratio = (count / total * 100.0) if total > 0 else 0.0
        print(f"{class_name:<20}: {count:>6} ({ratio:6.2f}%)")
    print("-" * 60)
    print(f"{'Total':<20}: {total:>6} (100.00%)")


def format_progress_bar(current, total, prefix="", width=30):
    """
    生成简单单行进度条
    """
    ratio = current / total
    filled = int(width * ratio)
    bar = "=" * filled + "." * (width - filled)
    return f"{prefix} [{bar}] {current}/{total}"


def print_single_line_progress(current, total, prefix, extra_info):
    """
    单行更新进度，避免输出一堆乱日志
    说明：保留这个函数，但主流程已改为 tqdm，不再使用
    """
    line = f"\r{format_progress_bar(current, total, prefix=prefix)} | {extra_info}"
    print(line, end="", flush=True)
    if current == total:
        print("")


def safe_div(a, b):
    """
    安全除法，避免 0 除错误
    """
    return float(a) / float(b) if b != 0 else float("nan")


def compute_auc80(y_true_binary, y_score, sensitivity_low=0.8):
    """
    计算 Melanoma 专用 AUC80：
    只积分 sensitivity(TPR) 在 [0.8, 1.0] 区间上的 ROC 面积

    这里返回“原始部分 ROC 面积”，不做额外归一化。
    """
    unique_vals = np.unique(y_true_binary)
    if len(unique_vals) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)

    # 选出 TPR >= 0.8 的部分
    keep_idx = np.where(tpr >= sensitivity_low)[0]
    if len(keep_idx) == 0:
        return 0.0

    start_idx = keep_idx[0]

    # 如果第一个满足条件的点不是刚好 0.8，则插值补一个边界点
    if tpr[start_idx] > sensitivity_low and start_idx > 0:
        x0, y0 = fpr[start_idx - 1], tpr[start_idx - 1]
        x1, y1 = fpr[start_idx], tpr[start_idx]
        if y1 != y0:
            interp_fpr = x0 + (sensitivity_low - y0) * (x1 - x0) / (y1 - y0)
        else:
            interp_fpr = x1

        fpr_part = np.concatenate([[interp_fpr], fpr[start_idx:]])
        tpr_part = np.concatenate([[sensitivity_low], tpr[start_idx:]])
    else:
        fpr_part = fpr[start_idx:]
        tpr_part = tpr[start_idx:]

    if len(fpr_part) < 2:
        return 0.0

    return float(auc(fpr_part, tpr_part))


def compute_detailed_classification_metrics(y_true, y_pred, y_prob, class_names):
    """
    计算你要求的所有额外指标：
    1. normalized multi-class accuracy metric = macro recall = balanced accuracy
    2. sensitivity / specificity / accuracy / AUC / mAP / F1 / PPV / NPV
    3. Melanoma AUC80
    4. average AUC across all diagnoses
    5. malignant vs benign AUC
    """
    num_classes = len(class_names)
    labels = np.arange(num_classes)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    y_true_bin = label_binarize(y_true, classes=labels)

    per_class_metrics = {}
    per_class_auc_list = []
    per_class_sensitivity_list = []
    per_class_specificity_list = []
    per_class_accuracy_list = []
    per_class_ap_list = []
    per_class_f1_list = []
    per_class_ppv_list = []
    per_class_npv_list = []

    total_samples = cm.sum()

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total_samples - tp - fn - fp

        sensitivity = safe_div(tp, tp + fn)  # recall
        specificity = safe_div(tn, tn + fp)
        accuracy_i = safe_div(tp + tn, total_samples)
        ppv = safe_div(tp, tp + fp)  # precision
        npv = safe_div(tn, tn + fn)

        if np.isnan(ppv) or np.isnan(sensitivity) or (ppv + sensitivity) == 0:
            f1_i = float("nan")
        else:
            f1_i = 2 * ppv * sensitivity / (ppv + sensitivity)

        y_true_i = y_true_bin[:, i]
        y_prob_i = y_prob[:, i]

        if len(np.unique(y_true_i)) < 2:
            auc_i = float("nan")
            ap_i = float("nan")
        else:
            auc_i = float(roc_auc_score(y_true_i, y_prob_i))
            ap_i = float(average_precision_score(y_true_i, y_prob_i))

        melanoma_auc80 = float("nan")
        if class_name == "MEL":
            melanoma_auc80 = compute_auc80(y_true_i, y_prob_i, sensitivity_low=0.8)

        per_class_metrics[class_name] = {
            "sensitivity": (
                float(sensitivity) if not np.isnan(sensitivity) else float("nan")
            ),
            "specificity": (
                float(specificity) if not np.isnan(specificity) else float("nan")
            ),
            "accuracy": float(accuracy_i) if not np.isnan(accuracy_i) else float("nan"),
            "auc": float(auc_i) if not np.isnan(auc_i) else float("nan"),
            "mean_average_precision": (
                float(ap_i) if not np.isnan(ap_i) else float("nan")
            ),
            "f1_score": float(f1_i) if not np.isnan(f1_i) else float("nan"),
            "ppv": float(ppv) if not np.isnan(ppv) else float("nan"),
            "npv": float(npv) if not np.isnan(npv) else float("nan"),
            "auc80": (
                float(melanoma_auc80) if not np.isnan(melanoma_auc80) else float("nan")
            ),
        }

        per_class_auc_list.append(auc_i)
        per_class_sensitivity_list.append(sensitivity)
        per_class_specificity_list.append(specificity)
        per_class_accuracy_list.append(accuracy_i)
        per_class_ap_list.append(ap_i)
        per_class_f1_list.append(f1_i)
        per_class_ppv_list.append(ppv)
        per_class_npv_list.append(npv)

    # -----------------------------
    # 聚合指标
    # -----------------------------
    overall_accuracy = accuracy_score(y_true, y_pred) * 100.0

    # 这个就是你要的 normalized multi-class accuracy metric
    balanced_macro_recall = balanced_accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = balanced_macro_recall

    # 宏平均 AUC / AP / Sens / Spec / Acc / PPV / NPV
    mean_auc_all_diagnoses = float(np.nanmean(per_class_auc_list))
    mean_ap_all_diagnoses = float(np.nanmean(per_class_ap_list))
    mean_sensitivity = float(np.nanmean(per_class_sensitivity_list))
    mean_specificity = float(np.nanmean(per_class_specificity_list))
    mean_accuracy = float(np.nanmean(per_class_accuracy_list))
    mean_f1 = float(np.nanmean(per_class_f1_list))
    mean_ppv = float(np.nanmean(per_class_ppv_list))
    mean_npv = float(np.nanmean(per_class_npv_list))

    # 你之前已经在算的多分类 OVR macro AUC
    try:
        multiclass_macro_auc_ovr = float(
            roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        multiclass_macro_auc_ovr = float("nan")

    # 恶性 vs 良性 AUC
    # 这里按常见临床分组推断：
    # malignant = MEL / BCC / AKIEC
    # benign    = NV / BKL / DF / VASC
    malignant_names = [name for name in ["MEL", "BCC", "AKIEC"] if name in class_names]
    malignant_indices = [class_names.index(name) for name in malignant_names]

    if len(malignant_indices) > 0:
        y_true_malignant = np.isin(y_true, malignant_indices).astype(int)
        y_prob_malignant = y_prob[:, malignant_indices].sum(axis=1)

        if len(np.unique(y_true_malignant)) < 2:
            malignant_vs_benign_auc = float("nan")
        else:
            malignant_vs_benign_auc = float(
                roc_auc_score(y_true_malignant, y_prob_malignant)
            )
    else:
        malignant_vs_benign_auc = float("nan")

    melanoma_auc80 = (
        per_class_metrics["MEL"]["auc80"]
        if "MEL" in per_class_metrics
        else float("nan")
    )

    metrics = {
        "overall": {
            "accuracy": float(overall_accuracy),  # 保持和你原代码一致：百分比
            "balanced_multiclass_accuracy": float(
                balanced_macro_recall
            ),  # macro recall
            "macro_recall": float(macro_recall),
            "macro_precision": float(macro_precision),
            "macro_f1": float(macro_f1),
            "multiclass_macro_auc_ovr": float(multiclass_macro_auc_ovr),
            "mean_auc_all_diagnoses": float(mean_auc_all_diagnoses),
            "mean_average_precision_all_diagnoses": float(mean_ap_all_diagnoses),
            "mean_sensitivity": float(mean_sensitivity),
            "mean_specificity": float(mean_specificity),
            "mean_accuracy": float(mean_accuracy),
            "mean_f1": float(mean_f1),
            "mean_ppv": float(mean_ppv),
            "mean_npv": float(mean_npv),
            "melanoma_auc80": (
                float(melanoma_auc80) if not np.isnan(melanoma_auc80) else float("nan")
            ),
            "malignant_vs_benign_auc": (
                float(malignant_vs_benign_auc)
                if not np.isnan(malignant_vs_benign_auc)
                else float("nan")
            ),
        },
        "per_class": per_class_metrics,
        "malignant_definition_used": malignant_names,
    }

    return metrics


# =========================================================
# 5. 主函数
# =========================================================
def main():
    global best_acc1
    args = parse_args()
    # -----------------------------
    # 固定随机种子
    # -----------------------------
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        warnings.warn(
            "You have chosen to seed training. "
            "This may slow down training a bit, but improves reproducibility."
        )

    # -----------------------------
    # 选择设备
    # -----------------------------
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -----------------------------
    # ISIC2018 Task3 路径
    # -----------------------------
    train_gt_csv_path = args.train_gt_csv
    val_gt_csv_path = args.val_gt_csv
    train_img_dir = args.train_img_dir
    val_img_dir = args.val_img_dir

    # -----------------------------
    # 先准备 start_epoch / checkpoint 变量
    # -----------------------------
    start_epoch = 0
    best_epoch = -1
    checkpoint = None

    # -----------------------------
    # 创建或复用实验目录
    # -----------------------------
    if args.resume is not None and os.path.isfile(args.resume):
        print(
            f"=> loading checkpoint metadata from '{args.resume}' to recover experiment folder"
        )
        checkpoint = torch.load(args.resume, map_location=device)

        if "exp_dir" not in checkpoint:
            raise ValueError(
                "resume 的 checkpoint 中没有 'exp_dir'，无法复用旧实验目录。"
            )

        exp_folders = reuse_experiment_folders(checkpoint["exp_dir"])
        exp_name = os.path.basename(checkpoint["exp_dir"])
        print(f"=> reusing experiment folder: {exp_folders['exp_dir']}")
    else:
        exp_name = make_experiment_name(args)
        exp_folders = setup_experiment_folders(
            base_dir="experiments", exp_name=exp_name
        )

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(
        exp_folders["metadata_dir"], "experiment_metadata.json"
    )

    # -----------------------------
    # 创建模型
    # -----------------------------
    if args.weights is not None:
        print(f"=> using weights '{args.weights}' for model '{args.arch}'")
        weights_enum = models.get_model_weights(args.arch)
        weights = weights_enum[args.weights]
        model = models.__dict__[args.arch](weights=weights)
    else:
        print(f"=> creating model '{args.arch}' with random initialization")
        model = models.__dict__[args.arch](weights=None)

    # -----------------------------
    # 读取类别信息并修改最后分类层
    # -----------------------------
    gt_df = pd.read_csv(train_gt_csv_path)
    class_columns = [c for c in gt_df.columns if c != "image"]
    num_classes = len(class_columns)
    class_names = class_columns

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"当前代码只处理带有 model.fc 的模型，当前模型 {args.arch} 不满足该条件。"
        )

    model = model.to(device)

    # -----------------------------
    # 损失函数、优化器、学习率调度器
    # -----------------------------
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # -----------------------------
    # 断点恢复
    # 删除了 start-epoch 参数后，这里由 checkpoint 自动恢复 start_epoch
    # -----------------------------
    best_model_path = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

    if checkpoint is not None:
        print(f"=> loading checkpoint state from '{args.resume}'")

        # checkpoint 里的 epoch 表示“已经完成到哪一轮”
        # 所以下一轮训练应该从这个 epoch 开始继续
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        if "best_epoch" in checkpoint:
            best_epoch = checkpoint["best_epoch"]

        print(
            f"=> loaded checkpoint '{args.resume}' (finished epoch {checkpoint['epoch']})"
        )
        print(f"=> training will continue from epoch {start_epoch + 1}")

    elif args.resume is not None:
        print(f"=> no checkpoint found at '{args.resume}'")

    # =====================================================
    # 数据加载部分
    # =====================================================
    resnet_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ISICResNetDataset(
        gt_csv_path=train_gt_csv_path,
        img_dir=train_img_dir,
        transform=resnet_transforms,
    )

    val_dataset = ISICResNetDataset(
        gt_csv_path=val_gt_csv_path, img_dir=val_img_dir, transform=resnet_transforms
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")

    train_class_distribution = count_labels_from_dataset(
        labels=train_dataset.labels, class_names=class_names
    )

    val_class_distribution = count_labels_from_dataset(
        labels=val_dataset.labels, class_names=class_names
    )

    print_class_distribution(
        "Train Dataset Class Distribution", train_class_distribution
    )
    print_class_distribution(
        "Validation Dataset Class Distribution", val_class_distribution
    )

    pin_memory = device.type == "cuda"

    # =====================================================
    # 用扩散模型数据增强部分
    # =====================================================
    output_dir = os.path.join(exp_folders["exp_dir"], "train_augmented_data")
    os.makedirs(output_dir, exist_ok=True)

    train_concat_dataset = concat_datasets(
        train_dataset, output_dir, args=args, device=device
    )

    # 从 ConcatDataset 中取出合成数据集
    # 注意：train_concat_dataset.datasets[0] 是原始 train_dataset
    #      train_concat_dataset.datasets[1] 是 synth_dataset
    synth_dataset = train_concat_dataset.datasets[1]

    # 统计合成数据集类别分布
    synth_labels = [label for _, label, _ in synth_dataset.samples]
    synth_class_distribution = count_labels_from_dataset(
        labels=synth_labels, class_names=class_names
    )

    # 统计增强后的训练集类别分布
    augmented_train_labels = train_dataset.labels + synth_labels
    augmented_train_class_distribution = count_labels_from_dataset(
        labels=augmented_train_labels, class_names=class_names
    )

    print(f"\nSynthetic dataset size: {len(synth_dataset)}")
    print(f"Augmented train dataset size: {len(train_concat_dataset)}")

    print_class_distribution(
        "Synthetic Dataset Class Distribution", synth_class_distribution
    )
    print_class_distribution(
        "Augmented Train Dataset Class Distribution",
        augmented_train_class_distribution,
    )

    pin_memory = device.type == "cuda"
    persistent_workers = (args.workers > 0,)

    train_loader = DataLoader(
        train_concat_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # -----------------------------
    # 初始化实验元信息
    # resume 时会写回原实验目录中的 metadata
    # -----------------------------
    active_ratios = {}
    if args.ratios is not None:
        for item in args.ratios:
            class_idx, ratio = item.split(":")
            ratio = float(ratio)
            if ratio > 0:
                active_ratios[class_names[int(class_idx)]] = ratio

    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.arch,
        "weights": args.weights,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "data": {
            "train_gt_csv_path": train_gt_csv_path,
            "val_gt_csv_path": val_gt_csv_path,
            "train_img_dir": train_img_dir,
            "val_img_dir": val_img_dir,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "synthetic_dataset_size": len(synth_dataset),
            "augmented_train_dataset_size": len(train_concat_dataset),
            "split_ratio": "official train / official val",
            "num_classes": num_classes,
            "class_names": class_names,
            "class_distribution": {
                "train_dataset": train_class_distribution,
                "synthetic_dataset": synth_class_distribution,
                "augmented_train_dataset": augmented_train_class_distribution,
                "val_dataset": val_class_distribution,
            },
        },
        "eval": {
            "eval_freq": args.eval_freq,
            "save_every_eval": args.save_every_eval,
        },
        "diffusion_generation": {
            "enabled": len(synth_dataset) > 0,
            "diffusion_checkpoint": args.diffusion_checkpoint,
            "scheduler_config": args.scheduler_config,
            "diffusion_module": args.diffusion_module,
            "resolution": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "ddim_eta": args.ddim_eta,
            "num_classes": args.num_classes,
            "time_scale_shift": args.time_scale_shift,
            "generation_batch_size": args.gen_batch_size,
            "seed": args.seed,
            "ratios_raw": args.ratios,
            "active_ratios_by_class_name": active_ratios,
            "output_dir": output_dir,
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "last_checkpoint": os.path.join(
                exp_folders["checkpoints_dir"], "last.pth.tar"
            ),
            "best_checkpoint": best_model_path,
            "roc_dir": exp_folders["roc_dir"],
            "confusion_matrix_dir": exp_folders["cm_dir"],
            "predictions_dir": exp_folders["predictions_dir"],
        },
        "resume": {"resume_path": args.resume, "start_epoch": start_epoch},
        "best_result": {
            "best_epoch": best_epoch,
            "best_val_balanced_acc": float(best_acc1),
            "best_model_path": (
                best_model_path if os.path.exists(best_model_path) else ""
            ),
        },
    }
    save_json(experiment_metadata, metadata_json_path)

    # -----------------------------
    # 仅评估模式
    # -----------------------------
    if args.evaluate:
        eval_epoch = start_epoch if checkpoint is not None else 0

        val_metrics = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            args=args,
            num_classes=num_classes,
            class_names=class_names,
            epoch=eval_epoch,
            roc_dir=exp_folders["roc_dir"],
            cm_dir=exp_folders["cm_dir"],
            predictions_dir=exp_folders["predictions_dir"],
            metrics_dir=exp_folders["metrics_dir"],
        )

        eval_row = {
            "epoch": eval_epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": None,
            "train_acc": None,
            "val_loss": float(val_metrics["val_loss"]),
            "val_acc": float(val_metrics["overall"]["accuracy"]),
            "val_balanced_acc": float(
                val_metrics["overall"]["balanced_multiclass_accuracy"]
            ),
            "val_macro_recall": float(val_metrics["overall"]["macro_recall"]),
            "val_macro_f1": float(val_metrics["overall"]["macro_f1"]),
            "val_macro_precision": float(val_metrics["overall"]["macro_precision"]),
            "val_auc_macro_ovr": float(
                val_metrics["overall"]["multiclass_macro_auc_ovr"]
            ),
            "val_mean_auc_all_diagnoses": float(
                val_metrics["overall"]["mean_auc_all_diagnoses"]
            ),
            "val_mean_ap_all_diagnoses": float(
                val_metrics["overall"]["mean_average_precision_all_diagnoses"]
            ),
            "val_mean_sensitivity": float(val_metrics["overall"]["mean_sensitivity"]),
            "val_mean_specificity": float(val_metrics["overall"]["mean_specificity"]),
            "val_mean_ppv": float(val_metrics["overall"]["mean_ppv"]),
            "val_mean_npv": float(val_metrics["overall"]["mean_npv"]),
            "val_melanoma_auc80": float(val_metrics["overall"]["melanoma_auc80"]),
            "val_malignant_vs_benign_auc": float(
                val_metrics["overall"]["malignant_vs_benign_auc"]
            ),
            "roc_curve_path": val_metrics["roc_curve_path"],
            "confusion_matrix_path": val_metrics["confusion_matrix_png_path"],
            "val_predictions_path": val_metrics["val_predictions_csv_path"],
            "detailed_metrics_path": val_metrics["detailed_metrics_json_path"],
        }
        update_epoch_metrics_csv(metrics_csv_path, eval_row)
        update_epoch_metrics_json(metrics_json_path, eval_row)
        return

    # -----------------------------
    # 训练循环
    # 训练起点改为 start_epoch（由 checkpoint 自动恢复）
    # -----------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{args.epochs} {'=' * 25}")

        train_metrics = train(
            train_loader, model, criterion, optimizer, epoch, device, args
        )

        do_eval = ((epoch + 1) % args.eval_freq == 0) or ((epoch + 1) == args.epochs)

        val_metrics = None
        is_best = False

        if do_eval:
            val_metrics = validate(
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                device=device,
                args=args,
                num_classes=num_classes,
                class_names=class_names,
                epoch=epoch + 1,
                roc_dir=exp_folders["roc_dir"],
                cm_dir=exp_folders["cm_dir"],
                predictions_dir=exp_folders["predictions_dir"],
                metrics_dir=exp_folders["metrics_dir"],
            )

            # 用 balanced multi-class accuracy 作为 best 指标
            current_score = val_metrics["overall"]["balanced_multiclass_accuracy"]
            is_best = current_score > best_acc1
            if is_best:
                best_acc1 = current_score
                best_epoch = epoch + 1

        scheduler.step()

        # 每个 epoch 都保存 last，方便 resume
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "exp_dir": exp_folders["exp_dir"],
                "is_eval_epoch": do_eval,
            },
            is_best,
            save_dir=exp_folders["checkpoints_dir"],
            filename="last.pth.tar",
        )

        # 只有 eval epoch 才额外保存一个历史 checkpoint
        if do_eval and args.save_every_eval:
            eval_ckpt_filename = f"checkpoint_epoch_{epoch + 1:03d}.pth.tar"
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "exp_dir": exp_folders["exp_dir"],
                    "is_eval_epoch": True,
                },
                False,
                save_dir=exp_folders["checkpoints_dir"],
                filename=eval_ckpt_filename,
            )

        # 只有 eval epoch 才写 metrics
        if do_eval:
            epoch_row = {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_metrics["train_loss"]),
                "train_acc": float(train_metrics["train_acc"]),
                "val_loss": float(val_metrics["val_loss"]),
                "val_acc": float(val_metrics["overall"]["accuracy"]),
                "val_balanced_acc": float(
                    val_metrics["overall"]["balanced_multiclass_accuracy"]
                ),
                "val_macro_recall": float(val_metrics["overall"]["macro_recall"]),
                "val_macro_f1": float(val_metrics["overall"]["macro_f1"]),
                "val_macro_precision": float(val_metrics["overall"]["macro_precision"]),
                "val_auc_macro_ovr": float(
                    val_metrics["overall"]["multiclass_macro_auc_ovr"]
                ),
                "val_mean_auc_all_diagnoses": float(
                    val_metrics["overall"]["mean_auc_all_diagnoses"]
                ),
                "val_mean_ap_all_diagnoses": float(
                    val_metrics["overall"]["mean_average_precision_all_diagnoses"]
                ),
                "val_mean_sensitivity": float(
                    val_metrics["overall"]["mean_sensitivity"]
                ),
                "val_mean_specificity": float(
                    val_metrics["overall"]["mean_specificity"]
                ),
                "val_mean_ppv": float(val_metrics["overall"]["mean_ppv"]),
                "val_mean_npv": float(val_metrics["overall"]["mean_npv"]),
                "val_melanoma_auc80": float(val_metrics["overall"]["melanoma_auc80"]),
                "val_malignant_vs_benign_auc": float(
                    val_metrics["overall"]["malignant_vs_benign_auc"]
                ),
                "roc_curve_path": val_metrics["roc_curve_path"],
                "confusion_matrix_path": val_metrics["confusion_matrix_png_path"],
                "val_predictions_path": val_metrics["val_predictions_csv_path"],
                "detailed_metrics_path": val_metrics["detailed_metrics_json_path"],
            }

            update_epoch_metrics_csv(metrics_csv_path, epoch_row)
            update_epoch_metrics_json(metrics_json_path, epoch_row)

        # 更新实验元信息
        experiment_metadata["best_result"]["best_epoch"] = best_epoch
        experiment_metadata["best_result"]["best_val_balanced_acc"] = float(best_acc1)
        experiment_metadata["best_result"]["best_model_path"] = (
            best_model_path if os.path.exists(best_model_path) else ""
        )
        experiment_metadata["last_epoch_finished"] = epoch + 1
        experiment_metadata["updated_time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        save_json(experiment_metadata, metadata_json_path)

        if do_eval:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_bal_acc={val_metrics['overall']['balanced_multiclass_accuracy']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
            )


# =========================================================
# 6. 训练函数
# =========================================================
def train(train_loader, model, criterion, optimizer, epoch, device, args):
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.NONE)

    model.train()

    # 使用 tqdm 包装 dataloader
    # desc：左侧显示当前阶段
    # total：总 batch 数
    # leave=False：一个 epoch 结束后不保留整条进度条，日志更干净
    progress_bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Train Epoch {epoch + 1}",
        leave=False,
    )

    for i, (images, target, sample_ids) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 在 tqdm 右侧动态显示平均 loss / acc
        progress_bar.set_postfix(
            {"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"}
        )

    return {
        "train_loss": float(losses.avg),
        "train_acc": float(top1.avg),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


# =========================================================
# 7. 验证函数
# =========================================================
def validate(
    val_loader,
    model,
    criterion,
    device,
    args,
    num_classes,
    class_names,
    epoch,
    roc_dir,
    cm_dir,
    predictions_dir,
    metrics_dir,
):
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)

    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_sample_ids = []

    with torch.no_grad():
        progress_bar = tqdm(
            val_loader, total=len(val_loader), desc=f"Validate {epoch}", leave=False
        )

        for i, (images, target, sample_ids) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_sample_ids.extend(sample_ids)

            acc1 = accuracy(output, target, topk=(1,))[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # 在 tqdm 右侧动态显示平均 loss / acc
            progress_bar.set_postfix(
                {"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"}
            )

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # 1) 保存混淆矩阵
    confusion_matrix_csv_path, confusion_matrix_png_path = (
        save_confusion_matrix_artifacts(
            y_true=all_targets,
            y_pred=all_preds,
            class_names=class_names,
            save_dir=cm_dir,
            epoch=epoch,
        )
    )

    # 2) 保存 ROC 曲线
    roc_curve_path, roc_points_json_path = save_multiclass_roc_artifacts(
        y_true=all_targets,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=roc_dir,
        epoch=epoch,
    )

    # 3) 保存样本级预测 CSV
    val_predictions_csv_path = save_val_predictions_csv(
        sample_ids=all_sample_ids,
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=predictions_dir,
        epoch=epoch,
    )

    # 4) 计算详细指标
    detailed_metrics = compute_detailed_classification_metrics(
        y_true=all_targets, y_pred=all_preds, y_prob=all_probs, class_names=class_names
    )
    detailed_metrics["val_loss"] = float(losses.avg)

    detailed_metrics_json_path = save_detailed_metrics_json(
        detailed_metrics=detailed_metrics, save_dir=metrics_dir, epoch=epoch
    )

    return {
        "val_loss": float(losses.avg),
        "overall": detailed_metrics["overall"],
        "per_class": detailed_metrics["per_class"],
        "roc_curve_path": roc_curve_path,
        "roc_points_json_path": roc_points_json_path,
        "confusion_matrix_csv_path": confusion_matrix_csv_path,
        "confusion_matrix_png_path": confusion_matrix_png_path,
        "val_predictions_csv_path": val_predictions_csv_path,
        "detailed_metrics_json_path": detailed_metrics_json_path,
    }


# =========================================================
# 8. 保存 checkpoint
# =========================================================
def save_checkpoint(
    state, is_best, save_dir="checkpoints", filename="checkpoint.pth.tar"
):
    """
    保存模型 checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    best_filepath = os.path.join(save_dir, "model_best.pth.tar")

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, best_filepath)


# =========================================================
# 9. 工具类：日志统计
# =========================================================
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """记录当前值、累计值、平均值"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            return ""
        elif self.summary_type is Summary.AVERAGE:
            return f"{self.name} {self.avg:.3f}"
        elif self.summary_type is Summary.SUM:
            return f"{self.name} {self.sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            return f"{self.name} {self.count:.3f}"
        else:
            raise ValueError("invalid summary type")


class ProgressMeter:
    """保留原类，但当前主流程已不再使用这个冗长打印方式"""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters if str(meter) != ""]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters if meter.summary() != ""]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# =========================================================
# 10. 准确率计算
# =========================================================
def accuracy(output, target, topk=(1,)):
    """
    计算 top-k 准确率
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        train_batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / train_batch_size))
        return res


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
