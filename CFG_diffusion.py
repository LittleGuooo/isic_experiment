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
from diffusers import (
    DDPMPipeline,
    DDIMPipeline,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


# =========================================================
# 1. 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM baseline for ISIC2018 dermoscopy images"
    )

    # ===== CFG 参数 =====
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="CFG guidance scale (only used during inference)",
    )

    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.15,
        help="Probability of dropping condition during training (CFG)",
    )

    # -------------------------
    # 断点续训
    # -------------------------
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="指定 .pth.tar checkpoint 路径以从上次训练中断处继续，"
        "会自动复用原实验文件夹（metrics/samples/fid 等目录接续写入）",
    )

    # -------------------------
    # 运行模式
    # -------------------------
    parser.add_argument(
        "--run_mode",
        type=str,
        default="train",
        choices=["train", "val_only", "infer_only"],
        help="train: 正常训练；val_only: 仅评估（同时评估 train/val）；infer_only: 仅推理生成图片",
    )

    # -------------------------
    # only-infer 专用参数
    # -------------------------
    parser.add_argument(
        "--infer_label",
        type=str,
        default=None,
        choices=[
            None,
            "MEL",
            "NV",
            "BCC",
            "AKIEC",
            "BKL",
            "DF",
            "VASC",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ],
        help="infer_only 模式下指定生成类别；若模型是 conditional，则建议必须指定",
    )
    parser.add_argument(
        "--infer_num_images",
        type=int,
        default=0,
        help="infer_only 模式下要生成的图片数量",
    )

    # -------------------------
    # 采样器选择
    # -------------------------
    parser.add_argument(
        "--use_ddim_sampling",
        action="store_true",
        help="推理/评估时使用 DDIM 采样器替代 DDPM，可大幅加快生成速度",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM 随机性系数：0.0 为完全确定性采样，1.0 退化为 DDPM",
    )

    # -------------------------
    # 数据路径
    # -------------------------
    parser.add_argument(
        "--train_gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        help="训练集 GroundTruth CSV 路径，列格式：image,MEL,NV,BCC,AKIEC,BKL,DF,VASC",
    )
    parser.add_argument(
        "--val_gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        help="验证集 GroundTruth CSV 路径，格式同训练集",
    )
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
        help="训练集图片目录，图片命名格式：<image_id>.jpg",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Validation_Input",
        help="验证集图片目录，图片命名格式：<image_id>.jpg",
    )

    # -------------------------
    # 数据过滤模式
    # -------------------------
    parser.add_argument(
        "--data_mode",
        type=str,
        default="all",
        choices=["all", "single_label"],
        help="all: 使用全部类别; single_label: 只使用一个类别（需配合 --target_label）",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        choices=[
            "MEL",
            "NV",
            "BCC",
            "AKIEC",
            "BKL",
            "DF",
            "VASC",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ],
        help="当 data_mode=single_label 时指定目标类别；"
        "可用类别名（MEL/NV/BCC/AKIEC/BKL/DF/VASC）"
        "或对应索引（0~6）",
    )

    # -------------------------
    # 类别条件开关
    # -------------------------
    parser.add_argument(
        "--use_class_conditioning",
        action="store_true",
        help="开启类别条件 DDPM。开启后，训练/采样时都会把类别标签作为条件输入模型。",
    )

    # -------------------------
    # 输出目录
    # -------------------------
    parser.add_argument(
        "--output_root",
        type=str,
        default="experiments",
        help="所有实验结果的根目录，每次新运行会在此下创建带时间戳的子文件夹",
    )

    # -------------------------
    # 图像参数
    # -------------------------
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="训练和生成图像的分辨率，同时决定 UNet 输入尺寸",
    )

    # -------------------------
    # 训练超参数
    # -------------------------
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="训练时每个 GPU 的 batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="评估/生成时的 batch size（同时用于 FID 采样和可视化样本生成）",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader 的并行读取进程数",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="总训练轮数（含 resume 续训时已完成的 epoch）",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数，等效 batch size = train_batch_size × gradient_accumulation_steps",
    )

    # -------------------------
    # 优化器参数（AdamW）
    # -------------------------
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="AdamW 基础学习率"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.95, help="AdamW 一阶矩估计的指数衰减率"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="AdamW 二阶矩估计的指数衰减率"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="AdamW 权重衰减系数"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="AdamW 数值稳定性 epsilon，防止除零",
    )

    # -------------------------
    # 学习率调度器
    # -------------------------
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="学习率调度策略，传入 diffusers get_scheduler 的 name 参数"
        "（如 cosine / linear / constant 等）",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率从 0 线性 warmup 到 learning_rate 所需的优化步数",
    )

    # -------------------------
    # 混合精度
    # -------------------------
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练：no=全 fp32；fp16=半精度（适合消费级 GPU）；"
        "bf16=BFloat16（适合 A100/H100）",
    )

    # -------------------------
    # DDPM 噪声调度参数
    # -------------------------
    parser.add_argument(
        "--ddpm_num_steps", type=int, default=1000, help="DDPM 训练时的总扩散步数 T"
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=1000,
        help="推理/采样时的去噪步数；使用 DDIM 时可设为 50~200 以加速",
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        help="噪声调度方案：linear（原始 DDPM）或 squaredcos_cap_v2（改进版）",
    )

    # -------------------------
    # 保存与评估频率
    # -------------------------
    parser.add_argument(
        "--enable_per_class_metrics",
        action="store_true",
        help="Whether to compute per-class FID/KID metrics (only effective for train split).",
    )
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=10,
        help="每隔多少 epoch 保存一批可视化生成样本",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=1,
        help="每隔多少 epoch 保存一次模型 checkpoint",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=10,
        help="每隔多少 epoch 计算一次 FID 和 Precision/Recall",
    )

    # -------------------------
    # FID 采样数量
    # -------------------------
    parser.add_argument(
        "--num_fid_samples_train",
        type=int,
        default=1024,
        help="用于计算训练集 FID 的生成图片数量；0 表示跳过训练集 FID",
    )
    parser.add_argument(
        "--num_fid_samples_val",
        "--num_fid_samples_valid",
        dest="num_fid_samples_val",
        type=int,
        default=193,
        help="用于计算验证集 FID 的生成图片数量；0 表示跳过验证集 FID。"
        "（--num_fid_samples_valid 为兼容旧命令的别名）",
    )

    # -------------------------
    # Manifold Improved Precision & Recall 参数
    # -------------------------
    parser.add_argument(
        "--ipr_k",
        type=int,
        default=3,
        help="流形估计的 k 近邻数；k=3 为论文默认值，"
        "值越大流形越宽松，precision 倾向偏高",
    )

    # -------------------------
    # 随机种子（复现）
    # -------------------------
    parser.add_argument(
        "--seed", type=int, default=42, help="全局随机种子，固定后可复现训练结果"
    )

    return parser.parse_args()


# =========================================================
# 2. 实验目录和日志工具
# =========================================================
def make_experiment_name(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = args.data_mode
    label_tag = (
        f"label_{args.target_label}"
        if args.data_mode == "single_label"
        else "all_labels"
    )
    cond_tag = "cond" if args.use_class_conditioning else "uncond"
    exp_name = (
        f"{timestamp}_ddpm_{cond_tag}_{mode_tag}_{label_tag}"
        f"_res{args.resolution}_bs{args.train_batch_size}_seed{args.seed}"
    )
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
        "fid_generated_dir": fid_generated_dir,
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


def format_count_ratio_dict(count_dict):
    total = sum(count_dict.values())
    formatted = {}
    for class_name, count in count_dict.items():
        ratio = (count / total * 100.0) if total > 0 else 0.0
        formatted[class_name] = f"{count} ({ratio:.2f}%)"
    return formatted


def print_class_distribution(title, count_dict):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    total_count = sum(count_dict.values())
    for class_name, count in count_dict.items():
        ratio = (count / total_count * 100.0) if total_count > 0 else 0.0
        print(f"{class_name}: {count} ({ratio:.2f}%)")
    print(f"Total: {total_count} (100.00%)")
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


def recover_exp_dir_from_checkpoint(checkpoint_path, checkpoint_data):
    if "exp_dir" in checkpoint_data:
        return checkpoint_data["exp_dir"]
    checkpoints_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    exp_dir = os.path.dirname(checkpoints_dir)
    return exp_dir


def normalize_label_to_index_and_name(target_label, class_names):
    """
    把类别名 / 类别索引统一转成 (idx, name)
    """
    if target_label is None:
        return None, None

    if str(target_label).isdigit():
        target_idx = int(target_label)
        if target_idx < 0 or target_idx >= len(class_names):
            raise ValueError(
                f"target_label index {target_idx} out of range [0, {len(class_names)-1}]"
            )
        return target_idx, class_names[target_idx]

    if target_label not in class_names:
        raise ValueError(
            f"target_label '{target_label}' not found in class names: {class_names}"
        )
    return class_names.index(target_label), target_label


def make_runtime_run_name(args):
    """
    为 only-val / only-infer 生成本次运行目录名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_tag = "no_ckpt"
    if args.resume_from_checkpoint is not None:
        ckpt_tag = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]

    sampler_tag = "ddim" if args.use_ddim_sampling else "ddpm"

    if args.run_mode == "val_only":
        return (
            f"{timestamp}_val_{ckpt_tag}_{sampler_tag}"
            f"_steps{args.ddpm_num_inference_steps}_seed{args.seed}"
        )

    if args.run_mode == "infer_only":
        label_tag = str(args.infer_label) if args.infer_label is not None else "none"
        return (
            f"{timestamp}_infer_{ckpt_tag}_{label_tag}_{sampler_tag}"
            f"_n{args.infer_num_images}_steps{args.ddpm_num_inference_steps}_seed{args.seed}"
        )

    return timestamp


def setup_runtime_run_folders(exp_dir, run_mode, run_name):
    """
    在 checkpoint 对应实验目录下创建：
    - run_vals/<run_name>/
    - run_infers/<run_name>/
    """
    if run_mode == "val_only":
        root_dir = os.path.join(exp_dir, "run_vals")
    elif run_mode == "infer_only":
        root_dir = os.path.join(exp_dir, "run_infers")
    else:
        raise ValueError(f"Unsupported run_mode for runtime folder: {run_mode}")

    run_dir = os.path.join(root_dir, run_name)
    metrics_dir = os.path.join(run_dir, "metrics")
    generated_dir = os.path.join(run_dir, "generated_images")
    metadata_dir = os.path.join(run_dir, "metadata")

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    return {
        "root_dir": root_dir,
        "run_dir": run_dir,
        "metrics_dir": metrics_dir,
        "generated_dir": generated_dir,
        "metadata_dir": metadata_dir,
        "run_config_json": os.path.join(run_dir, "run_config.json"),
        "run_summary_json": os.path.join(run_dir, "run_summary.json"),
    }


def sync_experiment_metadata_for_resume(
    experiment_metadata,
    args,
    start_epoch,
    global_step,
):
    """
    恢复训练时，把当前实际运行参数写回 experiment_metadata.json
    只更新“当前状态视图”，不篡改历史 checkpoint 本身。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    experiment_metadata["updated_time"] = now_str
    experiment_metadata["resume_from_checkpoint"] = args.resume_from_checkpoint
    experiment_metadata["last_runtime_args"] = vars(args)

    # 更新 data 配置
    if "data" not in experiment_metadata:
        experiment_metadata["data"] = {}
    experiment_metadata["data"].update(
        {
            "train_gt_csv_path": args.train_gt_csv_path,
            "val_gt_csv_path": args.val_gt_csv_path,
            "train_img_dir": args.train_img_dir,
            "val_img_dir": args.val_img_dir,
            "data_mode": args.data_mode,
            "target_label": args.target_label,
            "use_class_conditioning": args.use_class_conditioning,
        }
    )

    # 更新 model 配置
    if "model" not in experiment_metadata:
        experiment_metadata["model"] = {}
    experiment_metadata["model"].update(
        {
            "resolution": args.resolution,
            "ddpm_num_steps": args.ddpm_num_steps,
            "ddpm_num_inference_steps": args.ddpm_num_inference_steps,
            "ddpm_beta_schedule": args.ddpm_beta_schedule,
            "use_ddim_sampling": args.use_ddim_sampling,
            "ddim_eta": args.ddim_eta,
            "use_class_conditioning": args.use_class_conditioning,
        }
    )

    # 更新 training 配置
    if "training" not in experiment_metadata:
        experiment_metadata["training"] = {}
    experiment_metadata["training"].update(
        {
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "start_epoch": start_epoch,
            "initial_global_step": global_step,
        }
    )

    return experiment_metadata


# =========================================================
# 3. 数据集
# =========================================================
class ISIC2018DDPMDataset(Dataset):
    def __init__(
        self, gt_csv_path, img_dir, transform=None, data_mode="all", target_label=None
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.data_mode = data_mode
        self.target_label = target_label

        df = pd.read_csv(gt_csv_path)
        self.class_columns = [c for c in df.columns if c != "image"]
        df["label_int"] = df[self.class_columns].values.argmax(axis=1)

        if data_mode == "single_label":
            if target_label is None:
                raise ValueError(
                    "When data_mode='single_label', --target_label must be provided."
                )
            if str(target_label).isdigit():
                target_label_idx = int(target_label)
            else:
                if target_label not in self.class_columns:
                    raise ValueError(
                        f"target_label '{target_label}' not found in class columns: {self.class_columns}"
                    )
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
# 4. FID / 采样 / per-class 工具
# =========================================================
def tensor_to_uint8_for_fid(x):
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x


def uint8_tensor_to_pil(x_uint8):
    arr = x_uint8.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def build_sampling_scheduler(noise_scheduler, use_ddim_sampling=False):
    if use_ddim_sampling:
        return DDIMScheduler.from_config(noise_scheduler.config)
    return DDPMScheduler.from_config(noise_scheduler.config)


def allocate_samples_by_ratio(count_dict, total_samples):
    """
    按真实数据分布比例，把 total_samples 分配到每个类别。
    使用“最大余数法”，保证：
    1. 总和严格等于 total_samples
    2. 比例尽量接近真实分布
    """
    class_names = list(count_dict.keys())
    total_count = sum(count_dict.values())

    if total_samples <= 0:
        return {k: 0 for k in class_names}

    if total_count == 0:
        return {k: 0 for k in class_names}

    exact = {}
    floor_alloc = {}
    remainders = []

    for class_name in class_names:
        value = count_dict[class_name] / total_count * total_samples
        exact[class_name] = value
        floor_alloc[class_name] = int(math.floor(value))
        remainders.append((value - floor_alloc[class_name], class_name))

    current_sum = sum(floor_alloc.values())
    remaining = total_samples - current_sum

    remainders.sort(key=lambda x: x[0], reverse=True)
    for i in range(remaining):
        _, class_name = remainders[i]
        floor_alloc[class_name] += 1

    return floor_alloc


@torch.no_grad()
def collect_real_images_by_class(
    real_loader, device, class_names, target_counts_by_class
):
    """
    按类别收集真实图像，返回 uint8 张量字典。
    每个类别最多收集 target_counts_by_class[class_name] 张。
    """
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    need_by_idx = {
        class_to_idx[name]: int(cnt) for name, cnt in target_counts_by_class.items()
    }
    collected = {idx: [] for idx in range(len(class_names))}
    count_now = {idx: 0 for idx in range(len(class_names))}

    total_need = sum(need_by_idx.values())
    if total_need <= 0:
        return {
            name: torch.empty(0, 3, 0, 0, dtype=torch.uint8, device=device)
            for name in class_names
        }

    progress_bar = tqdm(
        total=total_need, desc="Collect real images by class", leave=True
    )
    for batch in real_loader:
        images = batch["input"].to(device)
        labels = batch["label"]

        images_uint8 = tensor_to_uint8_for_fid(images)

        for i in range(images_uint8.size(0)):
            label_idx = int(labels[i])
            if label_idx not in need_by_idx:
                continue
            if count_now[label_idx] >= need_by_idx[label_idx]:
                continue
            collected[label_idx].append(images_uint8[i : i + 1])
            count_now[label_idx] += 1
            progress_bar.update(1)

        if all(count_now[idx] >= need_by_idx[idx] for idx in need_by_idx):
            break

    progress_bar.close()

    result = {}
    for class_name, class_idx in class_to_idx.items():
        if len(collected[class_idx]) == 0:
            result[class_name] = torch.empty(
                0, 3, 0, 0, dtype=torch.uint8, device=device
            )
        else:
            result[class_name] = torch.cat(collected[class_idx], dim=0)
    return result


@torch.no_grad()
def sample_images_with_model(
    model,
    sampling_scheduler,
    device,
    resolution,
    batch_size,
    num_inference_steps,
    generator,
    use_class_conditioning=False,
    class_labels=None,
    ddim_eta=0.0,
):
    """
    用手写 denoising loop 采样。
    这样无论是否开启类别条件，都能统一控制。
    """
    try:
        sampling_scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        sampling_scheduler.set_timesteps(num_inference_steps)

    sample = torch.randn(
        (batch_size, model.config.in_channels, resolution, resolution),
        generator=generator,
        device=device,
    )

    null_class_id = 7  # ISIC2018 共 7 类，索引 0~6；使用 7 作为 null 条件的类别 ID
    for t in sampling_scheduler.timesteps:
        model_input = sample
        if hasattr(sampling_scheduler, "scale_model_input"):
            model_input = sampling_scheduler.scale_model_input(model_input, t)

        if use_class_conditioning and class_labels is not None:
            # =============================================
            # CFG采样过程
            # =============================================

            # 条件预测
            noise_pred_cond = model(model_input, t, class_labels=class_labels).sample

            # 无条件预测（全部设为 null）
            null_class_labels = torch.full_like(class_labels, fill_value=null_class_id)

            noise_pred_uncond = model(
                model_input, t, class_labels=null_class_labels
            ).sample

            # CFG 合成
            guidance_scale = args.guidance_scale  # 可调（2~5常用）

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred = model(model_input, t).sample

        if isinstance(sampling_scheduler, DDIMScheduler):
            step_output = sampling_scheduler.step(
                noise_pred, t, sample, eta=ddim_eta, generator=generator
            )
        else:
            step_output = sampling_scheduler.step(
                noise_pred, t, sample, generator=generator
            )

        sample = step_output.prev_sample

    return tensor_to_uint8_for_fid(sample)


@torch.no_grad()
def generate_images_by_class_for_metrics(
    accelerator,
    model,
    noise_scheduler,
    class_names,
    target_counts_by_class,
    fake_save_root,
    save_dir_name,
    resolution,
    eval_batch_size,
    num_inference_steps,
    use_ddim_sampling=False,
    ddim_eta=0.0,
    use_class_conditioning=False,
):
    """
    按类别生成假图。
    - overall 指标会把各类假图拼接起来
    - per-class 指标直接用各类各自的 fake tensor
    """
    device = accelerator.device
    generated_dir = os.path.join(fake_save_root, save_dir_name)
    os.makedirs(generated_dir, exist_ok=True)

    sampling_scheduler = build_sampling_scheduler(
        noise_scheduler=noise_scheduler, use_ddim_sampling=use_ddim_sampling
    )

    fake_by_class = {}
    total_need = sum(target_counts_by_class.values())
    progress_bar = tqdm(
        total=total_need, desc=f"Generate fake images ({save_dir_name})", leave=True
    )

    global_counter = 0
    for class_idx, class_name in enumerate(class_names):
        target_count = int(target_counts_by_class.get(class_name, 0))
        if target_count <= 0:
            fake_by_class[class_name] = torch.empty(
                0, 3, 0, 0, dtype=torch.uint8, device=device
            )
            continue

        class_save_dir = os.path.join(generated_dir, class_name)
        os.makedirs(class_save_dir, exist_ok=True)

        cur_fake_batches = []
        produced = 0
        batch_id = 0

        while produced < target_count:
            cur_bs = min(eval_batch_size, target_count - produced)

            generator = torch.Generator(device=device).manual_seed(
                1000 + class_idx * 100000 + batch_id
            )

            if use_class_conditioning:
                class_labels = torch.full(
                    (cur_bs,), fill_value=class_idx, device=device, dtype=torch.long
                )
            else:
                class_labels = None

            fake_uint8 = sample_images_with_model(
                model=model,
                sampling_scheduler=sampling_scheduler,
                device=device,
                resolution=resolution,
                batch_size=cur_bs,
                num_inference_steps=num_inference_steps,
                generator=generator,
                use_class_conditioning=use_class_conditioning,
                class_labels=class_labels,
                ddim_eta=ddim_eta,
            )

            cur_fake_batches.append(fake_uint8)

            for i in range(fake_uint8.size(0)):
                pil_img = uint8_tensor_to_pil(fake_uint8[i])
                pil_img.save(
                    os.path.join(class_save_dir, f"fid_sample_{global_counter:05d}.png")
                )
                global_counter += 1

            produced += cur_bs
            batch_id += 1
            progress_bar.update(cur_bs)

        fake_by_class[class_name] = torch.cat(cur_fake_batches, dim=0)

    progress_bar.close()
    return fake_by_class, generated_dir


def concat_class_tensors(
    class_tensor_dict, class_names, target_counts_by_class, device
):
    """
    按 class_names 的顺序把 per-class tensor 拼起来，形成 overall tensor。
    """
    tensors = []
    for class_name in class_names:
        count = int(target_counts_by_class.get(class_name, 0))
        if count <= 0:
            continue
        ten = class_tensor_dict[class_name]
        if ten.size(0) > 0:
            tensors.append(ten[:count])

    if len(tensors) == 0:
        return torch.empty(0, 3, 0, 0, dtype=torch.uint8, device=device)
    return torch.cat(tensors, dim=0)


@torch.no_grad()
def compute_fid_from_real_and_fake(real_images_uint8, fake_images_uint8, device):
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # 分 batch 计算fid，避免一次性把所有图像送入 Inception 模型导致 OOM
    batch_size = 32

    # --- real ---
    for i in range(0, real_images_uint8.size(0), batch_size):
        fid.update(real_images_uint8[i : i + batch_size], real=True)

    # --- fake ---
    fake_images_uint8 = fake_images_uint8[: real_images_uint8.size(0)]
    for i in range(0, fake_images_uint8.size(0), batch_size):
        fid.update(fake_images_uint8[i : i + batch_size], real=False)

    return float(fid.compute().item())


@torch.no_grad()
def compute_kid_from_real_and_fake(
    real_images_uint8,
    fake_images_uint8,
    device,
    subsets=50,
    subset_size=50,
):
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None, None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None, None

    valid_subset_size = min(
        subset_size, real_images_uint8.size(0), fake_images_uint8.size(0)
    )
    if valid_subset_size < 2:
        return None, None

    kid = KernelInceptionDistance(
        feature=2048,
        subsets=subsets,
        subset_size=valid_subset_size,
        normalize=False,
    ).to(device)

    batch_size = 32

    for i in range(0, real_images_uint8.size(0), batch_size):
        kid.update(real_images_uint8[i : i + batch_size], real=True)

    fake_images_uint8 = fake_images_uint8[: real_images_uint8.size(0)]
    for i in range(0, fake_images_uint8.size(0), batch_size):
        kid.update(fake_images_uint8[i : i + batch_size], real=False)

    kid_mean, kid_std = kid.compute()
    return float(kid_mean.item()), float(kid_std.item())


# =========================================================
# 5. Manifold Improved Precision & Recall
# =========================================================
Manifold = namedtuple("Manifold", ["features", "radii"])


def _build_vgg16_feature_extractor(device):
    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    vgg = vgg.to(device).eval()
    return vgg


@torch.no_grad()
def _extract_vgg16_features(images_uint8, vgg16, device, batch_size=64):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_feats = []
    n = images_uint8.shape[0]
    for start in range(0, n, batch_size):
        batch = images_uint8[start : start + batch_size].to(device)
        batch = batch.float() / 255.0
        batch = F.interpolate(
            batch, size=(224, 224), mode="bilinear", align_corners=False
        )
        batch = (batch - mean) / std
        conv_feat = vgg16.features(batch)
        conv_feat = vgg16.avgpool(conv_feat)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        fc_feat = vgg16.classifier[:4](conv_feat)
        all_feats.append(fc_feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def _compute_pairwise_distances(X, Y=None):
    X = X.astype(np.float64)
    X_sq = np.sum(X**2, axis=1, keepdims=True)

    if Y is None:
        Y = X
        Y_sq = X_sq
    else:
        Y = Y.astype(np.float64)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True)

    diff_sq = X_sq + Y_sq.T - 2.0 * X.dot(Y.T)
    diff_sq = np.clip(diff_sq, 0, None)
    return np.sqrt(diff_sq)


def _distances_to_radii(distances, k):
    n = distances.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        kth_idx = np.argpartition(distances[i], k + 1)[: k + 1]
        radii[i] = distances[i][kth_idx].max()
    return radii


def _build_manifold(features, k):
    distances = _compute_pairwise_distances(features)
    radii = _distances_to_radii(distances, k)
    return Manifold(features, radii)


def _compute_precision_or_recall(manifold_ref, feats_query):
    dist = _compute_pairwise_distances(manifold_ref.features, feats_query)
    in_manifold = (dist < manifold_ref.radii[:, None]).any(axis=0)
    return float(in_manifold.mean())


@torch.no_grad()
def compute_manifold_precision_recall(
    real_images_uint8, fake_images_uint8, device, k=3, vgg_batch_size=64
):
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None, None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None, None

    print("  [IPR] Loading VGG16 for Improved Precision/Recall...")
    vgg16 = _build_vgg16_feature_extractor(device)

    print("  [IPR] Extracting features for real images...")
    real_feats = _extract_vgg16_features(
        real_images_uint8, vgg16, device, batch_size=vgg_batch_size
    )

    print("  [IPR] Extracting features for fake images...")
    n_real = real_images_uint8.size(0)
    fake_feats = _extract_vgg16_features(
        fake_images_uint8[:n_real], vgg16, device, batch_size=vgg_batch_size
    )

    print("  [IPR] Building manifolds (k={})...".format(k))
    manifold_real = _build_manifold(real_feats, k)
    manifold_fake = _build_manifold(fake_feats, k)

    print("  [IPR] Computing precision and recall...")
    precision = _compute_precision_or_recall(manifold_real, fake_feats)
    recall = _compute_precision_or_recall(manifold_fake, real_feats)

    del vgg16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return precision, recall


# =========================================================
# 6. 每个 split 的整体 + per-class 评估
# =========================================================
@torch.no_grad()
def evaluate_split_with_overall_and_per_class_metrics(
    split_name,
    real_loader,
    accelerator,
    model,
    noise_scheduler,
    class_names,
    dataset_count_dict,
    num_total_samples,
    fid_dir,
    fid_generated_dir,
    epoch,
    resolution,
    eval_batch_size,
    num_inference_steps,
    use_ddim_sampling,
    ddim_eta,
    use_class_conditioning,
    ipr_k,
    kid_subsets=50,
    kid_subset_size=50,
    compute_per_class_metrics=False,
    per_class_max_real_samples=None,
):
    """
    1. 先按真实类别比例为每个类别分配要评估的样本数
    2. 收集各类别真实图像
    3. 按类别生成假图
    4. 先算 overall FID / precision / recall
    5. 再算 per-class FID / precision / recall
    """
    device = accelerator.device

    if num_total_samples <= 0:
        return {
            "overall_fid": None,
            "overall_kid_mean": None,
            "overall_kid_std": None,
            "overall_precision": None,
            "overall_recall": None,
            "overall_json_path": "",
            "per_class_json_path": "",
            "generated_dir": "",
            "per_class_generated_dir": "",
            "per_class_metrics": {},
            "allocated_counts_by_class": {name: 0 for name in class_names},
            "per_class_counts_by_class": {},
        }

    allocated_counts_by_class = allocate_samples_by_ratio(
        dataset_count_dict, num_total_samples
    )

    print(f"\n[{split_name.upper()}] allocated evaluation samples by class:")
    print_class_distribution(
        f"{split_name.upper()} Eval Allocation", allocated_counts_by_class
    )

    real_by_class = collect_real_images_by_class(
        real_loader=real_loader,
        device=device,
        class_names=class_names,
        target_counts_by_class=allocated_counts_by_class,
    )

    fake_by_class, generated_dir = generate_images_by_class_for_metrics(
        accelerator=accelerator,
        model=model,
        noise_scheduler=noise_scheduler,
        class_names=class_names,
        target_counts_by_class=allocated_counts_by_class,
        fake_save_root=fid_generated_dir,
        save_dir_name=f"epoch_{epoch:03d}_{split_name}_generated",
        resolution=resolution,
        eval_batch_size=eval_batch_size,
        num_inference_steps=num_inference_steps,
        use_ddim_sampling=use_ddim_sampling,
        ddim_eta=ddim_eta,
        use_class_conditioning=use_class_conditioning,
    )

    real_overall = concat_class_tensors(
        real_by_class, class_names, allocated_counts_by_class, device
    )
    fake_overall = concat_class_tensors(
        fake_by_class, class_names, allocated_counts_by_class, device
    )

    overall_fid = compute_fid_from_real_and_fake(real_overall, fake_overall, device)
    overall_kid_mean, overall_kid_std = compute_kid_from_real_and_fake(
        real_images_uint8=real_overall,
        fake_images_uint8=fake_overall,
        device=device,
        subsets=kid_subsets,
        subset_size=kid_subset_size,
    )
    overall_precision, overall_recall = compute_manifold_precision_recall(
        real_images_uint8=real_overall,
        fake_images_uint8=fake_overall,
        device=device,
        k=ipr_k,
    )

    overall_json_path = os.path.join(
        fid_dir, f"epoch_{epoch:03d}_{split_name}_fid.json"
    )
    save_json(
        {
            "epoch": epoch,
            "split": split_name,
            "num_real_images": int(real_overall.size(0)),
            "num_fake_images": int(fake_overall.size(0)),
            "num_images_by_class": format_count_ratio_dict(allocated_counts_by_class),
            "num_images_by_class_raw": allocated_counts_by_class,
            "fid": float(overall_fid) if overall_fid is not None else None,
            "kid_mean": (
                float(overall_kid_mean) if overall_kid_mean is not None else None
            ),
            "kid_std": float(overall_kid_std) if overall_kid_std is not None else None,
            "precision": (
                float(overall_precision) if overall_precision is not None else None
            ),
            "recall": float(overall_recall) if overall_recall is not None else None,
            "ipr_k": ipr_k,
            "kid_subsets": int(kid_subsets),
            "kid_subset_size": int(kid_subset_size),
            "generated_dir": generated_dir,
            "sampler": "ddim" if use_ddim_sampling else "ddpm",
            "num_inference_steps": int(num_inference_steps),
            "ddim_eta": float(ddim_eta) if use_ddim_sampling else None,
            "use_class_conditioning": bool(use_class_conditioning),
            "overall_generated_follow_real_distribution": True,
        },
        overall_json_path,
    )

    print(
        f"[{split_name.upper()}] Overall FID: " f"{overall_fid:.6f}"
        if overall_fid is not None
        else f"[{split_name.upper()}] Overall FID: None"
    )
    if overall_kid_mean is not None and overall_kid_std is not None:
        print(
            f"[{split_name.upper()}] Overall KID: mean={overall_kid_mean:.6f}, std={overall_kid_std:.6f}"
        )
    else:
        print(f"[{split_name.upper()}] Overall KID: None")
    print(
        f"[{split_name.upper()}] Overall Precision: " f"{overall_precision:.4f}"
        if overall_precision is not None
        else f"[{split_name.upper()}] Overall Precision: None"
    )
    print(
        f"[{split_name.upper()}] Overall Recall: " f"{overall_recall:.4f}"
        if overall_recall is not None
        else f"[{split_name.upper()}] Overall Recall: None"
    )

    per_class_metrics = {}
    per_class_json_path = ""
    per_class_generated_dir = ""
    per_class_counts_by_class = {}

    # -------------------------------------------------
    # 只有训练集才计算 per-class 指标
    # 并且每个类别单独使用：
    # min(该类真实样本数, per_class_max_real_samples)
    # -------------------------------------------------
    if compute_per_class_metrics and split_name == "train":
        print(f"\n[{split_name.upper()}] Per-class evaluation ENABLED")
        if per_class_max_real_samples is None:
            raise ValueError(
                "When compute_per_class_metrics=True, per_class_max_real_samples must be provided."
            )

        per_class_counts_by_class = {
            class_name: min(
                int(dataset_count_dict[class_name]), int(per_class_max_real_samples)
            )
            for class_name in class_names
        }

        print(f"\n[{split_name.upper()}] Per-class metrics sample counts")
        print_class_distribution(
            f"{split_name.upper()} Per-class Eval Counts", per_class_counts_by_class
        )

        # 单独收集 per-class real
        per_class_real_by_class = collect_real_images_by_class(
            real_loader=real_loader,
            device=device,
            class_names=class_names,
            target_counts_by_class=per_class_counts_by_class,
        )

        # -------------------------------------------------
        # 复用 overall 阶段已经生成好的 fake_by_class
        # 如果某个类别数量不够，再只补生成缺少的部分
        # -------------------------------------------------
        extra_counts_by_class = {}
        for class_name in class_names:
            already_have = int(fake_by_class[class_name].size(0))
            target_need = int(per_class_counts_by_class[class_name])
            extra_counts_by_class[class_name] = max(0, target_need - already_have)

        total_extra_needed = sum(extra_counts_by_class.values())

        # 默认先认为 per-class 复用 overall 的生成结果目录
        per_class_generated_dir = generated_dir

        # 只有确实不够时，才额外补生成
        if total_extra_needed > 0:
            print(
                f"\n[{split_name.upper()}] Reusing overall fake images and topping up for per-class..."
            )
            print_class_distribution(
                f"{split_name.upper()} Per-class Extra Fake Allocation",
                extra_counts_by_class,
            )

            extra_fake_by_class, per_class_generated_dir = (
                generate_images_by_class_for_metrics(
                    accelerator=accelerator,
                    model=model,
                    noise_scheduler=noise_scheduler,
                    class_names=class_names,
                    target_counts_by_class=extra_counts_by_class,
                    fake_save_root=fid_generated_dir,
                    save_dir_name=f"epoch_{epoch:03d}_{split_name}_per_class_generated_topup",
                    resolution=resolution,
                    eval_batch_size=eval_batch_size,
                    num_inference_steps=num_inference_steps,
                    use_ddim_sampling=use_ddim_sampling,
                    ddim_eta=ddim_eta,
                    use_class_conditioning=use_class_conditioning,
                )
            )
        else:
            # 不需要补生成时，给每个类别一个空 tensor，后面统一拼接
            extra_fake_by_class = {
                class_name: fake_by_class[class_name][:0] for class_name in class_names
            }

        # 把 overall 已有的 fake 和 top-up 的 fake 拼起来
        per_class_fake_by_class = {}
        for class_name in class_names:
            reused_fake = fake_by_class[class_name]
            extra_fake = extra_fake_by_class[class_name]

            per_class_fake_by_class[class_name] = torch.cat(
                [reused_fake, extra_fake], dim=0
            )[: per_class_counts_by_class[class_name]]

        total_per_class = sum(per_class_counts_by_class.values())

        print(f"\n[{split_name.upper()}] Per-class metrics")
        print("-" * 80)
        for class_name in class_names:
            real_c = per_class_real_by_class[class_name]
            fake_c = per_class_fake_by_class[class_name]
            class_count = int(per_class_counts_by_class.get(class_name, 0))
            ratio_c = (
                (class_count / total_per_class * 100.0) if total_per_class > 0 else 0.0
            )

            if class_count <= 0 or real_c.size(0) == 0 or fake_c.size(0) == 0:
                per_class_metrics[class_name] = {
                    "num_real_images": f"{class_count} ({ratio_c:.2f}%)",
                    "num_real_images_raw": class_count,
                    "num_fake_images": f"{class_count} ({ratio_c:.2f}%)",
                    "num_fake_images_raw": class_count,
                    "fid": None,
                    "kid_mean": None,
                    "kid_std": None,
                    "precision": None,
                    "recall": None,
                }
                print(f"{class_name}: skipped (0 samples)")
                continue

            fid_c = compute_fid_from_real_and_fake(real_c, fake_c, device)
            kid_mean_c, kid_std_c = compute_kid_from_real_and_fake(
                real_images_uint8=real_c,
                fake_images_uint8=fake_c,
                device=device,
                subsets=kid_subsets,
                subset_size=min(kid_subset_size, class_count),
            )

            # ⭐ 限制 IPR 使用的样本数（防止内存爆炸）
            ipr_limit = 300

            real_ipr = real_c[:ipr_limit]
            fake_ipr = fake_c[:ipr_limit]

            precision_c, recall_c = compute_manifold_precision_recall(
                real_images_uint8=real_ipr,
                fake_images_uint8=fake_ipr,
                device=device,
                k=ipr_k,
            )

            per_class_metrics[class_name] = {
                "num_real_images": f"{class_count} ({ratio_c:.2f}%)",
                "num_real_images_raw": class_count,
                "num_fake_images": f"{class_count} ({ratio_c:.2f}%)",
                "num_fake_images_raw": class_count,
                "fid": float(fid_c) if fid_c is not None else None,
                "kid_mean": float(kid_mean_c) if kid_mean_c is not None else None,
                "kid_std": float(kid_std_c) if kid_std_c is not None else None,
                "precision": float(precision_c) if precision_c is not None else None,
                "recall": float(recall_c) if recall_c is not None else None,
            }

            fid_str = f"{fid_c:.6f}" if fid_c is not None else "None"
            kid_mean_str = f"{kid_mean_c:.6f}" if kid_mean_c is not None else "None"
            kid_std_str = f"{kid_std_c:.6f}" if kid_std_c is not None else "None"
            precision_str = f"{precision_c:.4f}" if precision_c is not None else "None"
            recall_str = f"{recall_c:.4f}" if recall_c is not None else "None"

            print(
                f"{class_name}: count={class_count} ({ratio_c:.2f}%), "
                f"FID={fid_str}, KID(mean/std)=({kid_mean_str}/{kid_std_str}), "
                f"Precision={precision_str}, Recall={recall_str}"
            )

        per_class_json_path = os.path.join(
            fid_dir, f"epoch_{epoch:03d}_{split_name}_per_class_metrics.json"
        )
        save_json(
            {
                "epoch": epoch,
                "split": split_name,
                "num_images_by_class": format_count_ratio_dict(
                    per_class_counts_by_class
                ),
                "num_images_by_class_raw": per_class_counts_by_class,
                "metrics_by_class": per_class_metrics,
                "generated_dir": per_class_generated_dir,
                "sampler": "ddim" if use_ddim_sampling else "ddpm",
                "num_inference_steps": int(num_inference_steps),
                "ddim_eta": float(ddim_eta) if use_ddim_sampling else None,
                "use_class_conditioning": bool(use_class_conditioning),
                "ipr_k": ipr_k,
                "kid_subsets": int(kid_subsets),
                "kid_subset_size": int(kid_subset_size),
                "per_class_real_sample_rule": f"min(real_class_count, {int(per_class_max_real_samples)})",
            },
            per_class_json_path,
        )

    return {
        "overall_fid": overall_fid,
        "overall_kid_mean": overall_kid_mean,
        "overall_kid_std": overall_kid_std,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_json_path": overall_json_path,
        "per_class_json_path": per_class_json_path,
        "generated_dir": generated_dir,
        "per_class_generated_dir": per_class_generated_dir,
        "per_class_metrics": per_class_metrics,
        "allocated_counts_by_class": allocated_counts_by_class,
        "per_class_counts_by_class": per_class_counts_by_class,
    }


@torch.no_grad()
def run_validation_only(
    args,
    accelerator,
    model,
    noise_scheduler,
    train_eval_loader,
    val_eval_loader,
    class_names,
    train_class_distribution,
    val_class_distribution,
    exp_folders,
    checkpoint_epoch,
):
    """
    only-val 模式：
    1. 加载 checkpoint 后
    2. 同时评估 train / val
    3. 全部结果写到 exp_dir/run_vals/<run_name>/
    """
    model.eval()

    run_name = make_runtime_run_name(args)
    run_folders = setup_runtime_run_folders(
        exp_dir=exp_folders["exp_dir"],
        run_mode="val_only",
        run_name=run_name,
    )

    save_json(
        {
            "run_mode": args.run_mode,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path": args.resume_from_checkpoint,
            "experiment_dir": exp_folders["exp_dir"],
            "checkpoint_epoch": checkpoint_epoch,
            "args": vars(args),
        },
        run_folders["run_config_json"],
    )

    print(f"\n[VAL_ONLY] Run directory: {run_folders['run_dir']}")

    # -------------------------
    # 1) train split 评估
    # -------------------------
    enable_train_fid = args.num_fid_samples_train > 0
    enable_val_fid = args.num_fid_samples_val > 0

    train_eval_result = None
    val_eval_result = None

    if enable_train_fid:
        print("\n[VAL_ONLY] Evaluating TRAIN split ...")
        train_eval_result = evaluate_split_with_overall_and_per_class_metrics(
            split_name="train",
            real_loader=train_eval_loader,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            class_names=class_names,
            dataset_count_dict=train_class_distribution,
            num_total_samples=args.num_fid_samples_train,
            fid_dir=run_folders["metrics_dir"],
            fid_generated_dir=run_folders["generated_dir"],
            epoch=checkpoint_epoch,
            resolution=args.resolution,
            eval_batch_size=args.eval_batch_size,
            num_inference_steps=args.ddpm_num_inference_steps,
            use_ddim_sampling=args.use_ddim_sampling,
            ddim_eta=args.ddim_eta,
            use_class_conditioning=args.use_class_conditioning,
            ipr_k=args.ipr_k,
            kid_subsets=50,
            kid_subset_size=50,
            compute_per_class_metrics=args.enable_per_class_metrics,
            per_class_max_real_samples=(
                args.num_fid_samples_train if args.enable_per_class_metrics else None
            ),
        )
    else:
        print("[VAL_ONLY] Train evaluation skipped (--num_fid_samples_train 0).")

    # -------------------------
    # 2) val split 评估
    # -------------------------
    if enable_val_fid:
        print("\n[VAL_ONLY] Evaluating VAL split ...")
        val_eval_result = evaluate_split_with_overall_and_per_class_metrics(
            split_name="val",
            real_loader=val_eval_loader,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            class_names=class_names,
            dataset_count_dict=val_class_distribution,
            num_total_samples=args.num_fid_samples_val,
            fid_dir=run_folders["metrics_dir"],
            fid_generated_dir=run_folders["generated_dir"],
            epoch=checkpoint_epoch,
            resolution=args.resolution,
            eval_batch_size=args.eval_batch_size,
            num_inference_steps=args.ddpm_num_inference_steps,
            use_ddim_sampling=args.use_ddim_sampling,
            ddim_eta=args.ddim_eta,
            use_class_conditioning=args.use_class_conditioning,
            ipr_k=args.ipr_k,
            kid_subsets=50,
            kid_subset_size=50,
            compute_per_class_metrics=False,
            per_class_max_real_samples=None,
        )
    else:
        print("[VAL_ONLY] Val evaluation skipped (--num_fid_samples_val 0).")

    run_summary = {
        "run_mode": "val_only",
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": args.resume_from_checkpoint,
        "experiment_dir": exp_folders["exp_dir"],
        "checkpoint_epoch": checkpoint_epoch,
        "sampler": "ddim" if args.use_ddim_sampling else "ddpm",
        "num_inference_steps": int(args.ddpm_num_inference_steps),
        "ddim_eta": float(args.ddim_eta) if args.use_ddim_sampling else None,
        "use_class_conditioning": bool(args.use_class_conditioning),
        "train_result": train_eval_result,
        "val_result": val_eval_result,
    }
    save_json(run_summary, run_folders["run_summary_json"])

    print(f"[VAL_ONLY] Summary saved to: {run_folders['run_summary_json']}")


@torch.no_grad()
def run_inference_only(
    args,
    accelerator,
    model,
    noise_scheduler,
    class_names,
    exp_folders,
    checkpoint_epoch,
):
    """
    only-infer 模式：
    1. 加载 checkpoint
    2. 指定类别 / 数量生成图片
    3. 输出到 exp_dir/run_infers/<run_name>/
    """
    model.eval()

    if args.infer_num_images <= 0:
        raise ValueError("When run_mode='infer_only', --infer_num_images must be > 0.")

    infer_label_idx, infer_label_name = normalize_label_to_index_and_name(
        args.infer_label, class_names
    )

    # 无条件模型不允许指定类别
    if (not args.use_class_conditioning) and (args.infer_label is not None):
        raise ValueError(
            "Current run is unconditional (use_class_conditioning=False), "
            "so --infer_label should not be specified."
        )

    # 条件模型建议必须显式指定类别
    if args.use_class_conditioning and args.infer_label is None:
        raise ValueError(
            "When use_class_conditioning=True and run_mode='infer_only', "
            "--infer_label must be specified."
        )

    run_name = make_runtime_run_name(args)
    run_folders = setup_runtime_run_folders(
        exp_dir=exp_folders["exp_dir"],
        run_mode="infer_only",
        run_name=run_name,
    )

    if infer_label_name is not None:
        infer_image_dir = os.path.join(
            run_folders["generated_dir"], f"generated_{infer_label_name}"
        )
    else:
        infer_image_dir = os.path.join(
            run_folders["generated_dir"], "generated_unconditional"
        )
    os.makedirs(infer_image_dir, exist_ok=True)

    save_json(
        {
            "run_mode": args.run_mode,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path": args.resume_from_checkpoint,
            "experiment_dir": exp_folders["exp_dir"],
            "checkpoint_epoch": checkpoint_epoch,
            "infer_label": args.infer_label,
            "infer_label_name": infer_label_name,
            "infer_label_idx": infer_label_idx,
            "infer_num_images": args.infer_num_images,
            "args": vars(args),
        },
        run_folders["run_config_json"],
    )

    print(f"\n[INFER_ONLY] Run directory: {run_folders['run_dir']}")
    print(f"[INFER_ONLY] Image output directory: {infer_image_dir}")

    sampling_scheduler = build_sampling_scheduler(
        noise_scheduler=noise_scheduler,
        use_ddim_sampling=args.use_ddim_sampling,
    )

    saved_image_paths = []
    generated_count = 0
    batch_id = 0

    while generated_count < args.infer_num_images:
        cur_bs = min(args.eval_batch_size, args.infer_num_images - generated_count)

        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.seed + batch_id
        )

        if args.use_class_conditioning:
            class_labels = torch.full(
                (cur_bs,),
                fill_value=infer_label_idx,
                device=accelerator.device,
                dtype=torch.long,
            )
        else:
            class_labels = None

        samples_uint8 = sample_images_with_model(
            model=model,
            sampling_scheduler=sampling_scheduler,
            device=accelerator.device,
            resolution=args.resolution,
            batch_size=cur_bs,
            num_inference_steps=args.ddpm_num_inference_steps,
            generator=generator,
            use_class_conditioning=args.use_class_conditioning,
            class_labels=class_labels,
            ddim_eta=args.ddim_eta,
        )

        for i in range(samples_uint8.size(0)):
            pil_img = uint8_tensor_to_pil(samples_uint8[i])

            if infer_label_name is not None:
                file_name = f"sample_{generated_count:05d}_{infer_label_name}.png"
            else:
                file_name = f"sample_{generated_count:05d}.png"

            img_path = os.path.join(infer_image_dir, file_name)
            pil_img.save(img_path)
            saved_image_paths.append(img_path)
            generated_count += 1

        batch_id += 1

    run_summary = {
        "run_mode": "infer_only",
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": args.resume_from_checkpoint,
        "experiment_dir": exp_folders["exp_dir"],
        "checkpoint_epoch": checkpoint_epoch,
        "sampler": "ddim" if args.use_ddim_sampling else "ddpm",
        "num_inference_steps": int(args.ddpm_num_inference_steps),
        "ddim_eta": float(args.ddim_eta) if args.use_ddim_sampling else None,
        "use_class_conditioning": bool(args.use_class_conditioning),
        "infer_label": args.infer_label,
        "infer_label_name": infer_label_name,
        "infer_label_idx": infer_label_idx,
        "infer_num_images": int(args.infer_num_images),
        "image_output_dir": infer_image_dir,
        "num_saved_images": len(saved_image_paths),
    }
    save_json(run_summary, run_folders["run_summary_json"])

    print(f"[INFER_ONLY] Summary saved to: {run_folders['run_summary_json']}")


# =========================================================
# 7. 主函数
# =========================================================
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if (
        args.run_mode in ["val_only", "infer_only"]
        and args.resume_from_checkpoint is None
    ):
        raise ValueError(
            f"When run_mode='{args.run_mode}', --resume_from_checkpoint must be provided."
        )

    if args.resume_from_checkpoint is not None:
        print(f"Loading checkpoint from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        recovered_exp_dir = recover_exp_dir_from_checkpoint(
            args.resume_from_checkpoint, checkpoint
        )
        exp_name = os.path.basename(recovered_exp_dir)
        exp_folders = setup_experiment_folders(
            base_dir=os.path.dirname(recovered_exp_dir), exp_name=exp_name
        )
        print(f"Resuming experiment at: {recovered_exp_dir}")
    else:
        checkpoint = None
        exp_name = make_experiment_name(args)
        exp_folders = setup_experiment_folders(args.output_root, exp_name)

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(
        exp_folders["metadata_dir"], "experiment_metadata.json"
    )
    best_model_path = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

    # Accelerator 初始化
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # 数据准备
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = ISIC2018DDPMDataset(
        gt_csv_path=args.train_gt_csv_path,
        img_dir=args.train_img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label,
    )
    val_dataset = ISIC2018DDPMDataset(
        gt_csv_path=args.val_gt_csv_path,
        img_dir=args.val_img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label,
    )

    gt_df = pd.read_csv(args.train_gt_csv_path)
    class_names = [c for c in gt_df.columns if c != "image"]

    # CFG修改：类别数 = 实际类别数 + 1（因为 UNet2DModel 的 num_class_embeds 需要包含一个额外的 "无类别" 索引）
    num_classes = len(class_names) + 1

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_indices = np.arange(len(train_dataset))
    val_indices = np.arange(len(val_dataset))
    train_class_distribution = count_labels_from_indices(
        train_dataset.labels, train_indices, class_names
    )
    val_class_distribution = count_labels_from_indices(
        val_dataset.labels, val_indices, class_names
    )

    print_class_distribution(
        "Train Dataset Class Distribution", train_class_distribution
    )
    print_class_distribution(
        "Validation Dataset Class Distribution", val_class_distribution
    )

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
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # -------------------------
    # 模型：
    # 1. 如果开启类别条件，就给 UNet2DModel 加 num_class_embeds
    # 2. forward 时再传 class_labels
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
        num_class_embeds=num_classes if args.use_class_conditioning else None,
        resnet_time_scale_shift="scale_shift",
    )

    # 噪声调度器（DDPM Scheduler）
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
    )

    # -------------------------
    # only-val / only-infer：只加载模型权重，不进入训练
    # -------------------------
    if args.run_mode in ["val_only", "infer_only"]:
        if checkpoint is None:
            raise ValueError(
                "Checkpoint must be loaded for val_only / infer_only mode."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(accelerator.device)
        model.eval()

        checkpoint_epoch = checkpoint.get("epoch", 0)

        if args.run_mode == "val_only":
            if accelerator.is_main_process:
                run_validation_only(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    noise_scheduler=noise_scheduler,
                    train_eval_loader=train_eval_loader,
                    val_eval_loader=val_eval_loader,
                    class_names=class_names,
                    train_class_distribution=train_class_distribution,
                    val_class_distribution=val_class_distribution,
                    exp_folders=exp_folders,
                    checkpoint_epoch=checkpoint_epoch,
                )
            accelerator.wait_for_everyone()
            accelerator.end_training()
            return

        if args.run_mode == "infer_only":
            if accelerator.is_main_process:
                run_inference_only(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    noise_scheduler=noise_scheduler,
                    class_names=class_names,
                    exp_folders=exp_folders,
                    checkpoint_epoch=checkpoint_epoch,
                )
            accelerator.wait_for_everyone()
            accelerator.end_training()
            return

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # diffusers加速器学习率调度器
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    start_epoch = 0
    global_step = 0
    best_val_fid = float("inf")
    best_train_fid = float("inf")

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", 0)
        best_val_fid = checkpoint.get("best_val_fid", float("inf"))
        best_train_fid = checkpoint.get("best_train_fid", float("inf"))
        print(f"Resume training from epoch {start_epoch + 1}")
        print(f"Recovered global_step = {global_step}")
        print(f"Recovered best_val_fid = {best_val_fid}")
        print(f"Recovered best_train_fid = {best_train_fid}")

    # diffusers加速器
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.resume_from_checkpoint is not None and os.path.exists(metadata_json_path):
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            experiment_metadata = json.load(f)
        print(f"Loaded existing metadata from: {metadata_json_path}")

        if "paths" not in experiment_metadata:
            experiment_metadata["paths"] = {
                "metrics_csv": metrics_csv_path,
                "metrics_json": metrics_json_path,
                "metadata_json": metadata_json_path,
                "checkpoints_dir": exp_folders["checkpoints_dir"],
                "samples_dir": exp_folders["samples_dir"],
                "fid_dir": exp_folders["fid_dir"],
                "fid_generated_dir": exp_folders["fid_generated_dir"],
                "diffusers_model_index_copy": os.path.join(
                    exp_folders["metadata_dir"], "diffusers_pipeline_model_index.json"
                ),
            }
        if "best_result" not in experiment_metadata:
            experiment_metadata["best_result"] = {
                "best_epoch_by_val_fid": -1,
                "best_val_fid": None,
                "best_epoch_by_train_fid": -1,
                "best_train_fid": None,
                "best_model_path": "",
            }

        # 恢复训练时，把“当前实际使用参数”同步回 metadata
        if args.run_mode == "train":
            experiment_metadata = sync_experiment_metadata_for_resume(
                experiment_metadata=experiment_metadata,
                args=args,
                start_epoch=start_epoch,
                global_step=global_step,
            )
    else:
        experiment_metadata = {
            "experiment_name": exp_name,
            "run_mode": args.run_mode,
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
                "use_class_conditioning": args.use_class_conditioning,
                "num_classes": num_classes,
                "class_names": class_names,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "class_distribution": {
                    "train_dataset": format_count_ratio_dict(train_class_distribution),
                    "val_dataset": format_count_ratio_dict(val_class_distribution),
                },
            },
            "model": {
                "resolution": args.resolution,
                "ddpm_num_steps": args.ddpm_num_steps,
                "ddpm_num_inference_steps": args.ddpm_num_inference_steps,
                "ddpm_beta_schedule": args.ddpm_beta_schedule,
                "use_ddim_sampling": args.use_ddim_sampling,
                "ddim_eta": args.ddim_eta,
                "use_class_conditioning": args.use_class_conditioning,
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
                "diffusers_model_index_copy": os.path.join(
                    exp_folders["metadata_dir"], "diffusers_pipeline_model_index.json"
                ),
            },
            "best_result": {
                "best_epoch_by_val_fid": -1,
                "best_val_fid": None,
                "best_epoch_by_train_fid": -1,
                "best_train_fid": None,
                "best_model_path": "",
            },
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
            desc=f"Train Epoch [{epoch + 1}/{args.num_epochs}]",
        )

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            class_labels = batch["label"].to(clean_images.device).long()

            # =============== CFG dropout =================================
            # null class id = 最后一个类
            null_class_id = num_classes - 1

            # 设置 unconditional 概率（CFG论文常用 0.1~0.2）
            uncond_prob = args.uncond_prob

            # 生成 mask：哪些样本变成 unconditional
            mask = (
                torch.rand(class_labels.shape, device=class_labels.device) < uncond_prob
            )

            # 替换为 null class
            class_labels = class_labels.clone()
            class_labels[mask] = null_class_id
            # =================================================================

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # -------------------------
                # 开启类别条件时，把 class_labels 传进去
                # -------------------------
                if args.use_class_conditioning:
                    noise_pred = model(
                        noisy_images, timesteps, class_labels=class_labels
                    ).sample
                else:
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
                progress_bar.set_postfix(
                    {"loss": f"{loss_item:.6f}", "update": global_step}
                )

        progress_bar.close()
        accelerator.wait_for_everyone()

        train_loss_epoch = epoch_loss_sum / max(epoch_loss_count, 1)

        need_save_images = ((epoch + 1) % args.save_images_epochs == 0) or (
            epoch == args.num_epochs - 1
        )
        need_save_model = ((epoch + 1) % args.save_model_epochs == 0) or (
            epoch == args.num_epochs - 1
        )

        enable_train_fid = args.num_fid_samples_train > 0
        enable_val_fid = args.num_fid_samples_val > 0

        need_eval = (
            ((epoch + 1) % args.eval_epochs == 0) or (epoch == args.num_epochs - 1)
        ) and (enable_train_fid or enable_val_fid)

        fid_train_value = None
        fid_val_value = None
        train_kid_mean = None
        train_kid_std = None
        val_kid_mean = None
        val_kid_std = None
        train_precision = None
        train_recall = None
        val_precision = None
        val_recall = None
        train_fid_json_path = ""
        val_fid_json_path = ""
        train_per_class_json_path = ""
        val_per_class_json_path = ""
        train_generated_dir = ""
        val_generated_dir = ""
        train_per_class_generated_dir = ""
        val_per_class_generated_dir = ""
        sample_dir = ""
        diffusers_model_index_copy_path = ""

        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)

            # 这里只保留 pipeline 的保存能力
            if args.use_ddim_sampling:
                save_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
                pipeline = DDIMPipeline(unet=unet, scheduler=save_scheduler)
            else:
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

            pipeline = pipeline.to(accelerator.device)

            # 1) 保存可视化样本
            if need_save_images:
                epoch_dir = os.path.join(
                    exp_folders["samples_dir"], f"epoch_{epoch + 1:03d}"
                )
                os.makedirs(epoch_dir, exist_ok=True)

                # 为了让可视化样本在 conditional 时也有明确类别，这里按训练集比例分配
                sample_alloc = allocate_samples_by_ratio(
                    train_class_distribution, args.eval_batch_size
                )
                sampling_scheduler = build_sampling_scheduler(
                    noise_scheduler=noise_scheduler,
                    use_ddim_sampling=args.use_ddim_sampling,
                )

                sample_counter = 0
                for class_idx, class_name in enumerate(class_names):
                    cur_n = sample_alloc[class_name]
                    if cur_n <= 0:
                        continue

                    generator = torch.Generator(device=accelerator.device).manual_seed(
                        0 + class_idx
                    )
                    if args.use_class_conditioning:
                        class_labels = torch.full(
                            (cur_n,),
                            fill_value=class_idx,
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                    else:
                        class_labels = None

                    samples_uint8 = sample_images_with_model(
                        model=unet,
                        sampling_scheduler=sampling_scheduler,
                        device=accelerator.device,
                        resolution=args.resolution,
                        batch_size=cur_n,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        generator=generator,
                        use_class_conditioning=args.use_class_conditioning,
                        class_labels=class_labels,
                        ddim_eta=args.ddim_eta,
                    )

                    for i in range(samples_uint8.size(0)):
                        pil_img = uint8_tensor_to_pil(samples_uint8[i])
                        if args.use_class_conditioning:
                            file_name = f"sample_{sample_counter:03d}_{class_name}.png"
                        else:
                            file_name = f"sample_{sample_counter:03d}.png"
                        pil_img.save(os.path.join(epoch_dir, file_name))
                        sample_counter += 1

                sample_dir = epoch_dir
                print(f"Samples saved to: {sample_dir}")

            # 2) 计算 overall + per-class 指标
            if need_eval:
                print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
                print("-" * 60)

                if enable_train_fid:
                    train_eval_result = (
                        evaluate_split_with_overall_and_per_class_metrics(
                            split_name="train",
                            real_loader=train_eval_loader,
                            accelerator=accelerator,
                            model=unet,
                            noise_scheduler=noise_scheduler,
                            class_names=class_names,
                            dataset_count_dict=train_class_distribution,
                            num_total_samples=args.num_fid_samples_train,
                            fid_dir=exp_folders["fid_dir"],
                            fid_generated_dir=exp_folders["fid_generated_dir"],
                            epoch=epoch + 1,
                            resolution=args.resolution,
                            eval_batch_size=args.eval_batch_size,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            use_ddim_sampling=args.use_ddim_sampling,
                            ddim_eta=args.ddim_eta,
                            use_class_conditioning=args.use_class_conditioning,
                            ipr_k=args.ipr_k,
                            kid_subsets=50,
                            kid_subset_size=50,
                            compute_per_class_metrics=args.enable_per_class_metrics,
                            per_class_max_real_samples=(
                                args.num_fid_samples_train
                                if args.enable_per_class_metrics
                                else None
                            ),
                        )
                    )
                    fid_train_value = train_eval_result["overall_fid"]
                    train_kid_mean = train_eval_result["overall_kid_mean"]
                    train_kid_std = train_eval_result["overall_kid_std"]
                    train_precision = train_eval_result["overall_precision"]
                    train_recall = train_eval_result["overall_recall"]
                    train_fid_json_path = train_eval_result["overall_json_path"]
                    train_per_class_json_path = train_eval_result["per_class_json_path"]
                    train_generated_dir = train_eval_result["generated_dir"]
                    train_per_class_generated_dir = train_eval_result[
                        "per_class_generated_dir"
                    ]
                else:
                    print("Train FID skipped (--num_fid_samples_train 0).")

                if enable_val_fid:
                    val_eval_result = evaluate_split_with_overall_and_per_class_metrics(
                        split_name="val",
                        real_loader=val_eval_loader,
                        accelerator=accelerator,
                        model=unet,
                        noise_scheduler=noise_scheduler,
                        class_names=class_names,
                        dataset_count_dict=val_class_distribution,
                        num_total_samples=args.num_fid_samples_val,
                        fid_dir=exp_folders["fid_dir"],
                        fid_generated_dir=exp_folders["fid_generated_dir"],
                        epoch=epoch + 1,
                        resolution=args.resolution,
                        eval_batch_size=args.eval_batch_size,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        use_ddim_sampling=args.use_ddim_sampling,
                        ddim_eta=args.ddim_eta,
                        use_class_conditioning=args.use_class_conditioning,
                        ipr_k=args.ipr_k,
                        kid_subsets=50,
                        kid_subset_size=50,
                        compute_per_class_metrics=False,
                        per_class_max_real_samples=None,
                    )
                    fid_val_value = val_eval_result["overall_fid"]
                    val_kid_mean = val_eval_result["overall_kid_mean"]
                    val_kid_std = val_eval_result["overall_kid_std"]
                    val_precision = val_eval_result["overall_precision"]
                    val_recall = val_eval_result["overall_recall"]
                    val_fid_json_path = val_eval_result["overall_json_path"]
                    val_per_class_json_path = val_eval_result[
                        "per_class_json_path"
                    ]  # 这里会保持为空字符串
                    val_generated_dir = val_eval_result["generated_dir"]
                    val_per_class_generated_dir = val_eval_result[
                        "per_class_generated_dir"
                    ]
                else:
                    print("Val FID skipped (--num_fid_samples_val 0).")

            # 3) 保存模型
            # best checkpoint 依据 overall train FID
            is_best = False
            if fid_train_value is not None:
                if fid_train_value < best_train_fid:
                    best_train_fid = fid_train_value
                    is_best = True
                    experiment_metadata["best_result"]["best_epoch_by_train_fid"] = (
                        epoch + 1
                    )
                    experiment_metadata["best_result"]["best_train_fid"] = float(
                        fid_train_value
                    )

            if need_save_model or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_state_dict": unet.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "best_val_fid": best_val_fid,
                        "best_train_fid": best_train_fid,
                        "args": vars(args),
                        "exp_dir": exp_folders["exp_dir"],
                    },
                    is_best=is_best,
                    save_dir=exp_folders["checkpoints_dir"],
                    filename="last.pth.tar",
                )

                if is_best:
                    experiment_metadata["best_result"][
                        "best_model_path"
                    ] = best_model_path

                pipeline.save_pretrained(exp_folders["exp_dir"])
                diffusers_model_index_copy_path = save_diffusers_model_index_copy(
                    exp_dir=exp_folders["exp_dir"],
                    metadata_dir=exp_folders["metadata_dir"],
                )
                print(
                    f"Checkpoint saved to: {os.path.join(exp_folders['checkpoints_dir'], 'last.pth.tar')}"
                )
                print(
                    f"Diffusers model index copy saved to: {diffusers_model_index_copy_path}"
                )
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
                "train_fid": (
                    float(fid_train_value) if fid_train_value is not None else None
                ),
                "train_kid_mean": (
                    float(train_kid_mean) if train_kid_mean is not None else None
                ),
                "train_kid_std": (
                    float(train_kid_std) if train_kid_std is not None else None
                ),
                "val_fid": float(fid_val_value) if fid_val_value is not None else None,
                "val_kid_mean": (
                    float(val_kid_mean) if val_kid_mean is not None else None
                ),
                "val_kid_std": float(val_kid_std) if val_kid_std is not None else None,
                "train_precision": (
                    float(train_precision) if train_precision is not None else None
                ),
                "train_recall": (
                    float(train_recall) if train_recall is not None else None
                ),
                "val_precision": (
                    float(val_precision) if val_precision is not None else None
                ),
                "val_recall": float(val_recall) if val_recall is not None else None,
                "sample_dir": sample_dir,
                "train_fid_json_path": train_fid_json_path,
                "val_fid_json_path": val_fid_json_path,
                "train_per_class_json_path": train_per_class_json_path,
                "val_per_class_json_path": val_per_class_json_path,
                "train_generated_dir": train_generated_dir,
                "val_generated_dir": val_generated_dir,
                "train_per_class_generated_dir": train_per_class_generated_dir,
                "val_per_class_generated_dir": val_per_class_generated_dir,
                "checkpoint_path": os.path.join(
                    exp_folders["checkpoints_dir"], "last.pth.tar"
                ),
                "diffusers_model_index_copy_path": diffusers_model_index_copy_path,
            }
            update_epoch_metrics_csv(metrics_csv_path, epoch_row)
            update_epoch_metrics_json(metrics_json_path, epoch_row)

            experiment_metadata["last_epoch_finished"] = epoch + 1
            experiment_metadata["updated_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            experiment_metadata["paths"]["diffusers_model_index_copy"] = os.path.join(
                exp_folders["metadata_dir"], "diffusers_pipeline_model_index.json"
            )
            save_json(experiment_metadata, metadata_json_path)

    accelerator.end_training()


# =========================================================
# 8. 运行入口
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)
