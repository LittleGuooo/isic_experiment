import argparse
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
from torch.utils.data import Dataset, DataLoader, random_split

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
        description="Classifier Guidance Training and Guided Sampling for ISIC2018 Task3"
    )

    # -------------------------
    # 1. 基础运行参数
    # -------------------------
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="随机种子",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="使用的GPU编号，如0。未设置时自动选择cuda/cpu",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="从 classifier checkpoint 恢复训练，例如 classifier_last.pth.tar",
    )

    # -------------------------
    # 2. 数据路径
    # -------------------------
    parser.add_argument(
        "--train_gt_csv_path",
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        type=str,
        help="训练集标签CSV路径",
    )
    parser.add_argument(
        "--val_gt_csv_path",
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        type=str,
        help="验证集标签CSV路径",
    )
    parser.add_argument(
        "--train_img_dir",
        default="dataset/ISIC2018_Task3_Training_Input",
        type=str,
        help="训练集图片文件夹路径",
    )
    parser.add_argument(
        "--val_img_dir",
        default="dataset/ISIC2018_Task3_Validation_Input",
        type=str,
        help="验证集图片文件夹路径",
    )

    # 修复bug
    parser.add_argument(
        "--weights", default=None, type=str, help="path to classifier/model weights"
    )
    parser.add_argument(
        "--arch", default="ResNet50", type=str, help="classifier 架构名称"
    )

    # -------------------------
    # 3. classifier 训练参数
    # -------------------------
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=30,
        help="classifier 训练轮数",
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-4,
        help="classifier 学习率",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="classifier 训练 batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader 的 num_workers",
    )
    parser.add_argument(
        "--classifier_save_every",
        type=int,
        default=5,
        help="每隔多少个 epoch 额外保存一个 classifier checkpoint",
    )

    # -------------------------
    # 4. diffusion / classifier guidance 参数
    # -------------------------
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        required=True,
        help="外部已训练 diffusion model 的 checkpoint 路径",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="输入图像分辨率",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=7,
        help="类别数量，ISIC2018 Task3 为 7",
    )
    parser.add_argument(
        "--ddpm_num_steps",
        type=int,
        default=1000,
        help="DDPM 训练时的总扩散步数 T",
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        help="DDPM beta schedule",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="DDIM 采样步数",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM eta",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="classifier guidance 强度",
    )
    parser.add_argument(
        "--use_class_conditioning",
        action="store_true",
        help="扩散模型是否使用 class conditioning",
    )
    parser.add_argument(
        "--time_scale_shift",
        type=str,
        default="default",
        help="UNet 的 resnet_time_scale_shift 参数",
    )
    parser.add_argument(
        "--scheduler_config",
        type=str,
        default=None,
        help="可选：DDIMScheduler.from_pretrained 所需目录；不填则手动构造 scheduler",
    )

    # -------------------------
    # 5. classifier 结构参数
    # -------------------------
    parser.add_argument(
        "--classifier_feat_size",
        type=int,
        default=4,
        help="进入 AttentionPool2d 前特征图的空间尺寸",
    )
    parser.add_argument(
        "--classifier_num_heads",
        type=int,
        default=8,
        help="注意力池化 head 数",
    )
    parser.add_argument(
        "--classifier_use_rotary",
        action="store_true",
        help="是否使用 RotAttentionPool2d",
    )

    # -------------------------
    # 6. guided generation 参数
    # -------------------------
    parser.add_argument(
        "--num_generate_total",
        type=int,
        default=1024,
        help="总共生成多少张图像",
    )
    parser.add_argument(
        "--guided_gen_batch_size",
        type=int,
        default=16,
        help="guided sampling 时单次生成 batch size",
    )

    # -------------------------
    # 7. 评估参数
    # -------------------------
    parser.add_argument(
        "--ipr_k",
        type=int,
        default=3,
        help="IPR 评估时的 top-k 参数",
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
    exp_name = f"{timestamp}_{args.arch}_{weights_tag}_lr{args.classifier_lr}_bs{args.batch_size}_{seed_tag}"
    return exp_name


def setup_experiment_folders(base_dir="experiments", exp_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, exp_name or timestamp)

    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    metadata_dir = os.path.join(exp_dir, "metadata")
    samples_dir = os.path.join(exp_dir, "samples")
    fid_dir = os.path.join(exp_dir, "fid")
    fid_generated_dir = os.path.join(exp_dir, "fid_generated_images")
    confusion_matrices_dir = os.path.join(exp_dir, "confusion_matrices")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(fid_dir, exist_ok=True)
    os.makedirs(fid_generated_dir, exist_ok=True)
    os.makedirs(confusion_matrices_dir, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "metadata_dir": metadata_dir,
        "samples_dir": samples_dir,
        "fid_dir": fid_dir,
        "fid_generated_dir": fid_generated_dir,
        "confusion_matrices_dir": confusion_matrices_dir,
    }


def reuse_experiment_folders(exp_dir):
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    metadata_dir = os.path.join(exp_dir, "metadata")
    samples_dir = os.path.join(exp_dir, "samples")
    fid_dir = os.path.join(exp_dir, "fid")
    fid_generated_dir = os.path.join(exp_dir, "fid_generated_images")
    confusion_matrices_dir = os.path.join(exp_dir, "confusion_matrices")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(fid_dir, exist_ok=True)
    os.makedirs(fid_generated_dir, exist_ok=True)
    os.makedirs(confusion_matrices_dir, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "metadata_dir": metadata_dir,
        "samples_dir": samples_dir,
        "fid_dir": fid_dir,
        "fid_generated_dir": fid_generated_dir,
        "confusion_matrices_dir": confusion_matrices_dir,
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


import seaborn as sns


def save_confusion_matrix_artifacts(
    y_true,
    y_pred,
    class_names,
    save_dir,
    epoch,
):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm_csv_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch:03d}.csv")
    cm_png_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch:03d}.png")

    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=200)
    plt.close()

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


def print_class_distribution(title, count_dict):
    """
    打印类别统计结果
    """
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    total_count = 0
    for class_name, count in count_dict.items():
        print(f"{class_name}: {count}")
        total_count += count
    print(f"Total: {total_count}")
    print(f"{'=' * 60}\n")


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
# 5. 主函数（只给出你需要替换/新增的位置）
# =========================================================
def main():
    args = parse_args()
    global best_acc1

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
    train_gt_csv_path = args.train_gt_csv_path
    val_gt_csv_path = args.val_gt_csv_path
    train_img_dir = args.train_img_dir
    val_img_dir = args.val_img_dir

    # -----------------------------
    # 创建或复用实验目录
    # -----------------------------
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")

        if "exp_dir" in checkpoint:
            exp_dir = checkpoint["exp_dir"]
        else:
            exp_dir = os.path.dirname(os.path.dirname(args.resume))
            print(
                f"=> warning: checkpoint 中没有 'exp_dir'，使用路径反推实验目录: {exp_dir}"
            )

        exp_folders = reuse_experiment_folders(exp_dir)
        exp_name = os.path.basename(exp_dir)
        print(f"=> reusing experiment folder: {exp_folders['exp_dir']}")
    else:
        checkpoint = None
        exp_name = make_experiment_name(args)
        exp_folders = setup_experiment_folders(
            base_dir="experiments", exp_name=exp_name
        )

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(
        exp_folders["metadata_dir"], "experiment_metadata.json"
    )
    best_model_path = os.path.join(
        exp_folders["checkpoints_dir"], "classifier_best.pth.tar"
    )

    start_epoch = 0
    best_epoch = -1
    best_acc1 = 0.0

    # =====================================================
    # 数据加载部分
    # =====================================================
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = ISICResNetDataset(
        gt_csv_path=train_gt_csv_path,
        img_dir=train_img_dir,
        transform=train_transforms,
    )

    val_dataset = ISICResNetDataset(
        gt_csv_path=val_gt_csv_path,
        img_dir=val_img_dir,
        transform=val_transforms,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")

    gt_df = pd.read_csv(args.train_gt_csv_path)
    class_names = [c for c in gt_df.columns if c != "image"]
    num_classes = len(class_names)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    # =====================================================
    # 1) 加载外部已训练好的 diffusion model
    # =====================================================
    if args.diffusion_checkpoint is None:
        raise ValueError(
            "必须提供 --diffusion_checkpoint，当前场景下扩散模型来自外部已训练权重。"
        )

    print(f"=> loading external diffusion checkpoint from: {args.diffusion_checkpoint}")
    diffusion_checkpoint = torch.load(args.diffusion_checkpoint, map_location="cpu")
    if "model_state_dict" not in diffusion_checkpoint:
        raise ValueError(
            "diffusion checkpoint 中缺少 'model_state_dict'，当前代码无法加载。"
        )

    from diffusers import UNet2DModel, DDIMScheduler, DDPMScheduler

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
        resnet_time_scale_shift=args.time_scale_shift,
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        diffusion_checkpoint["model_state_dict"],
        strict=False,
    )

    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print("WARNING: diffusion checkpoint 与当前 UNet2DModel 结构可能不完全一致")
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)
    model = model.to(device)
    model.eval()

    # 训练 classifier 时用于加噪，必须和 diffusion 训练时的 scheduler 一致
    train_noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
    )

    # 采样时使用 DDIM
    sampling_scheduler = DDIMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
    )
    sampling_scheduler.set_timesteps(args.num_inference_steps, device=device)

    # =====================================================
    # 2) 从外部 diffusion UNet 构建 classifier
    # =====================================================
    classifier = build_classifier_from_trained_unet(
        unet=model,
        num_classes=args.num_classes,
        feat_size=args.classifier_feat_size,
        num_heads=args.classifier_num_heads,
        use_rotary=args.classifier_use_rotary,
        device=device,
        resolution=args.resolution,
    )

    # -----------------------------
    # 初始化实验元信息
    # -----------------------------
    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": "classifier_guidance",
        "run_stage": "train_classifier_then_guided_sampling",
        "seed": args.seed,
        "device": str(device),
        "data": {
            "train_gt_csv_path": train_gt_csv_path,
            "val_gt_csv_path": val_gt_csv_path,
            "train_img_dir": train_img_dir,
            "val_img_dir": val_img_dir,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "split_mode": "official_train_val_split",
            "num_classes": num_classes,
            "class_names": class_names,
            "class_distribution": {
                "train_dataset": train_class_distribution,
                "val_dataset": val_class_distribution,
            },
        },
        "diffusion_model": {
            "source": "external_pretrained_checkpoint",
            "checkpoint_path": args.diffusion_checkpoint,
            "architecture": "UNet2DModel",
            "resolution": args.resolution,
            "ddpm_num_steps": args.ddpm_num_steps,
            "ddpm_beta_schedule": args.ddpm_beta_schedule,
            "num_inference_steps": args.num_inference_steps,
            "use_class_conditioning": args.use_class_conditioning,
            "time_scale_shift": args.time_scale_shift,
            "status": "frozen_for_classifier_guidance",
        },
        "classifier": {
            "type": "ClassifierWithUNetDownsample",
            "backbone_source": "reuse_unet_down_blocks_and_mid_block_from_external_diffusion",
            "num_classes": args.num_classes,
            "feat_size": args.classifier_feat_size,
            "num_heads": args.classifier_num_heads,
            "use_rotary": args.classifier_use_rotary,
            "trainable_parts": ["attention_pool", "classifier_head"],
            "frozen_parts": ["unet_down_blocks", "unet_mid_block"],
            "epochs": args.classifier_epochs,
            "learning_rate": args.classifier_lr,
            "batch_size": args.batch_size,
        },
        "guided_sampling": {
            "sampler": "GuidedDDIMSampler",
            "guidance_scale": args.guidance_scale,
            "ddim_eta": args.ddim_eta,
            "num_inference_steps": args.num_inference_steps,
            "num_generate_total": args.num_generate_total,
            "allocation_rule": "follow_train_class_distribution",
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "last_classifier_checkpoint": os.path.join(
                exp_folders["checkpoints_dir"], "classifier_last.pth.tar"
            ),
            "best_classifier_checkpoint": best_model_path,
            "guided_samples_dir": exp_folders["samples_dir"],
            "fid_dir": exp_folders["fid_dir"],
            "fid_generated_dir": exp_folders["fid_generated_dir"],
        },
        "resume": {
            "resume_path": args.resume,
            "start_epoch": start_epoch,
        },
        "best_result": {
            "best_epoch": best_epoch,
            "best_val_balanced_acc": float(best_acc1),
            "best_model_path": "",
        },
    }

    save_json(experiment_metadata, metadata_json_path)

    # =====================================================
    # 3) 训练分类器
    # =====================================================
    # 根据训练集类别频数构造 class weights
    train_counts = np.array(
        [train_class_distribution[name] for name in class_names], dtype=np.float32
    )
    class_weights = train_counts.sum() / (len(train_counts) * train_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    train_classifier(
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        noise_scheduler=train_noise_scheduler,
        epochs=args.classifier_epochs,
        lr=args.classifier_lr,
        device=device,
        class_names=class_names,
        exp_folders=exp_folders,
        resume_path=args.resume,
        save_every=args.classifier_save_every,
        metrics_csv_path=metrics_csv_path,
        metrics_json_path=metrics_json_path,
        metadata_json_path=metadata_json_path,
        class_weights=class_weights,
    )

    # =====================================================
    # 4) 生成带引导的图像：按训练集比例分配总生成数
    #    然后复用你代码2中的评估逻辑
    # =====================================================
    guided_sampler = GuidedDDIMSampler(
        model=model,
        classifier=classifier,
        scheduler=sampling_scheduler,
        device=device,
        use_class_conditioning=args.use_class_conditioning,
    )

    # 按训练集类别比例分配总生成数量
    gen_alloc = diffusion.allocate_samples_by_ratio(
        dataset_count_dict=train_class_distribution,
        num_total_samples=args.num_generate_total,
    )

    guided_generated_root = os.path.join(
        exp_folders["samples_dir"], "guided_epoch_final"
    )
    os.makedirs(guided_generated_root, exist_ok=True)

    print("Guided generation allocation:")
    for class_name in class_names:
        print(f"  {class_name}: {gen_alloc[class_name]}")

    sample_counter = 0

    for class_idx, class_name in enumerate(class_names):
        cur_n = gen_alloc[class_name]
        if cur_n <= 0:
            continue

        print(f"[Guided Sampling] class={class_name}, total={cur_n}")

        # 这里新增：按 guided_gen_batch_size 分块生成，避免一次性爆显存
        remaining = cur_n
        inner_batch_idx = 0

        while remaining > 0:
            current_bs = min(args.guided_gen_batch_size, remaining)

            # 如果 diffusion 是 class-conditioned，这里传给 UNet 的类别标签
            if args.use_class_conditioning:
                diffusion_class_labels = torch.full(
                    (current_bs,),
                    fill_value=class_idx,
                    device=device,
                    dtype=torch.long,
                )
            else:
                diffusion_class_labels = None

            # classifier guidance 的目标类别
            target_labels = torch.full(
                (current_bs,),
                fill_value=class_idx,
                device=device,
                dtype=torch.long,
            )

            generated_images = guided_sampler.sample(
                shape=(current_bs, 3, args.resolution, args.resolution),
                target_labels=target_labels,
                class_labels=diffusion_class_labels,
                guidance_scale=args.guidance_scale,
                eta=args.ddim_eta,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + class_idx * 1000 + inner_batch_idx
                ),
                show_progress=True,
            )

            # [-1, 1] -> [0, 255]
            generated_uint8 = (
                ((generated_images.clamp(-1, 1) + 1.0) * 127.5)
                .round()
                .to(torch.uint8)
                .cpu()
            )

            for i in range(generated_uint8.size(0)):
                pil_img = diffusion.uint8_tensor_to_pil(generated_uint8[i])
                file_name = f"sample_{sample_counter:06d}_{class_name}.png"
                pil_img.save(os.path.join(guided_generated_root, file_name))
                sample_counter += 1

            remaining -= current_bs
            inner_batch_idx += 1

    print(f"Guided samples saved to: {guided_generated_root}")

    # =====================================================
    # 5) 复用你代码2中的评估逻辑
    #    这里不重写你的评估函数，直接调用你已有的 overall + per-class 评估入口
    # =====================================================
    # 说明：
    # - 如果你的 evaluate_split_with_overall_and_per_class_metrics 当前只能“边生成边评估”，
    #   你需要让它支持一个可选参数 generated_dir_override；
    # - 这里按“直接评估已生成目录”来调用。
    guided_eval_result = diffusion.evaluate_split_with_overall_and_per_class_metrics(
        split_name="guided_train_ratio",
        real_loader=val_loader,  # 你也可以换成 train_loader，看你希望对哪个 split 比
        accelerator=None,  # 这里如果你的函数内部不依赖 accelerator，可以传 None
        model=None,  # 因为这里不再现场生成
        noise_scheduler=None,
        class_names=class_names,
        dataset_count_dict=train_class_distribution,
        num_total_samples=args.num_generate_total,
        fid_dir=exp_folders["fid_dir"],
        fid_generated_dir=exp_folders["fid_generated_dir"],
        epoch=0,
        resolution=args.resolution,
        eval_batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        use_ddim_sampling=True,
        ddim_eta=args.ddim_eta,
        use_class_conditioning=args.use_class_conditioning,
        ipr_k=args.ipr_k,
        kid_subsets=50,
        kid_subset_size=50,
        compute_per_class_metrics=True,
        per_class_max_real_samples=args.num_generate_total,
        generated_dir_override=guided_generated_root,  # 需要你在评估函数里支持这个参数
    )

    print("Guided evaluation result:")
    print(guided_eval_result)


# =========================================================
# 6. classifier guidance 相关
# =========================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Union, Tuple
from diffusers import UNet2DModel, DDIMScheduler
from timm.layers import AttentionPool2d as AbsAttentionPool2d
from timm.layers import RotAttentionPool2d
import diffusion
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


def evaluate_guidance_classifier(
    classifier,
    data_loader,
    noise_scheduler,
    device,
    class_names,
):
    """
    评估用于 classifier guidance 的分类器 p(y | x_t, t)

    评估方式：
    - 对每张图随机采样一个 timestep t
    - 用 diffusion 的 forward process 加噪得到 x_t
    - 让 classifier 预测类别
    """
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_count = 0
    cpu_rng = torch.Generator(device="cpu").manual_seed(12345)
    gpu_rng = None
    if device.type == "cuda":
        gpu_rng = torch.Generator(device=device).manual_seed(12345)
    with torch.no_grad():
        for images, labels, sample_ids in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device).long()

            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
                generator=gpu_rng if device.type == "cuda" else None,
            ).long()

            if device.type == "cuda":
                noise = torch.randn(
                    images.shape,
                    device=device,
                    generator=gpu_rng,
                    dtype=images.dtype,
                )
            else:
                noise = torch.randn(
                    images.shape,
                    device=device,
                    generator=cpu_rng,
                    dtype=images.dtype,
                )
            noisy_images = noise_scheduler.add_noise(images, noise, t)

            logits = classifier(noisy_images, t)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / max(
        len(all_labels), 1
    )
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    per_class_recall = {
        class_name: float(report[class_name]["recall"]) for class_name in class_names
    }
    per_class_precision = {
        class_name: float(report[class_name]["precision"]) for class_name in class_names
    }
    per_class_f1 = {
        class_name: float(report[class_name]["f1-score"]) for class_name in class_names
    }

    result = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }
    return result


class ClassifierWithUNetDownsample(nn.Module):
    """
    基于已训练 UNet2DModel 的下采样路径 + 中间块做分类器
    """

    def __init__(
        self,
        unet: UNet2DModel,
        num_classes: int = 7,
        feat_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_heads: int = 8,
        use_rotary: bool = False,
    ):
        super().__init__()

        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks
        self.mid_block = unet.mid_block
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding
        self.num_features = unet.config.block_out_channels[-1]

        self.use_rotary = use_rotary
        self.feat_size = None

        if use_rotary:
            self.attention_pool = RotAttentionPool2d(
                in_features=self.num_features,
                out_features=self.num_features,
                num_heads=num_heads,
                qkv_bias=True,
            )
        else:
            assert feat_size is not None, "使用绝对位置编码时必须指定 feat_size"
            self.feat_size = (
                feat_size if isinstance(feat_size, tuple) else (feat_size, feat_size)
            )
            self.attention_pool = AbsAttentionPool2d(
                in_features=self.num_features,
                feat_size=self.feat_size,
                out_features=self.num_features,
                num_heads=num_heads,
                qkv_bias=True,
            )

        self.classifier = nn.Linear(self.num_features, num_classes)
        self._freeze_unet_backbone()

    def set_frozen_backbone_eval(self):
        """
        冻结的 UNet backbone 始终保持在 eval 模式
        只训练 attention_pool 和 classifier
        """
        self.conv_in.eval()
        self.down_blocks.eval()
        self.mid_block.eval()
        self.time_proj.eval()
        self.time_embedding.eval()

    def _freeze_unet_backbone(self):
        for param in self.conv_in.parameters():
            param.requires_grad = False
        for param in self.down_blocks.parameters():
            param.requires_grad = False
        for param in self.mid_block.parameters():
            param.requires_grad = False
        for param in self.time_proj.parameters():
            param.requires_grad = False
        for param in self.time_embedding.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 这里 timesteps 必传，因为 classifier guidance 训练的是 p(y | x_t, t)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.to(x.device).long()

        # 若只传了单个 timestep，扩展到 batch 大小
        if timesteps.shape[0] == 1 and x.shape[0] > 1:
            timesteps = timesteps.expand(x.shape[0])

        temb = self.time_proj(timesteps)
        temb = self.time_embedding(temb)

        # 关键：先经过 UNet 的输入卷积
        x = self.conv_in(x)

        # 按 UNet2DModel 的 down_blocks 调用方式来走
        skip_sample = None
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                x, _, skip_sample = downsample_block(
                    hidden_states=x, temb=temb, skip_sample=skip_sample
                )
            else:
                x, _ = downsample_block(hidden_states=x, temb=temb)

        x = self.mid_block(x, temb)

        pooled = self.attention_pool(x)
        logits = self.classifier(pooled)
        return logits


def build_classifier_from_trained_unet(
    unet: UNet2DModel,
    num_classes: int = 7,
    feat_size: Union[int, Tuple[int, int]] = (4, 4),
    num_heads: int = 8,
    use_rotary: bool = False,
    device: str = "cuda",
    resolution: int = 128,
) -> ClassifierWithUNetDownsample:
    """
    直接从“已加载权重的 UNet2DModel”构建 classifier
    如果使用绝对位置编码池化，则自动推断最后特征图大小
    """
    if not use_rotary:
        # 根据 down_blocks 中包含 downsampler 的数量，估算总下采样倍数
        downsample_factor = 1
        for block in unet.down_blocks:
            if hasattr(block, "downsamplers") and block.downsamplers is not None:
                downsample_factor *= 2

        inferred_feat_size = resolution // downsample_factor
        feat_size = (inferred_feat_size, inferred_feat_size)

    classifier = ClassifierWithUNetDownsample(
        unet=unet,
        num_classes=num_classes,
        feat_size=feat_size,
        num_heads=num_heads,
        use_rotary=use_rotary,
    ).to(device)
    return classifier


class GuidedDDIMSampler:
    """
    带 classifier gradient guidance 的 DDIM 采样器
    """

    def __init__(
        self,
        model: UNet2DModel,
        classifier: nn.Module,
        scheduler: DDIMScheduler,
        device: str = "cuda",
        use_class_conditioning: bool = False,
    ):
        self.model = model
        self.classifier = classifier
        self.scheduler = scheduler
        self.device = device
        self.use_class_conditioning = use_class_conditioning

    def sample(
        self,
        shape: tuple,
        target_labels: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        返回范围仍然是 diffusion 内部常用的 [-1, 1] 张量
        """
        self.model.eval()
        self.classifier.eval()

        x = torch.randn(shape, device=self.device, generator=generator)

        timesteps = self.scheduler.timesteps
        iterator = tqdm(
            timesteps, disable=not show_progress, desc="Guided DDIM Sampling"
        )

        for t in iterator:
            # classifier guidance 需要对 x 求梯度，所以这里不能整个套 torch.no_grad()
            x_in = x.detach().requires_grad_(True)

            # 1) UNet 预测 epsilon
            with torch.no_grad():
                if self.use_class_conditioning:
                    eps = self.model(x_in, t, class_labels=class_labels).sample
                else:
                    eps = self.model(x_in, t).sample

            # 2) classifier 计算 log p(y | x_t, t) 对 x_t 的梯度
            t_batch = torch.full(
                (x.shape[0],),
                fill_value=int(t),
                device=self.device,
                dtype=torch.long,
            )
            logits = self.classifier(x_in, t_batch)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[
                torch.arange(x.shape[0], device=self.device), target_labels
            ]
            grad = torch.autograd.grad(selected.sum(), x_in)[0]

            # 3) 按 classifier guidance 公式修正 epsilon
            alpha_bar_t = self.scheduler.alphas_cumprod[t].to(self.device)
            eps_guided = eps - guidance_scale * torch.sqrt(1.0 - alpha_bar_t) * grad

            # 4) DDIM 更新
            with torch.no_grad():
                step_output = self.scheduler.step(
                    model_output=eps_guided,
                    timestep=t,
                    sample=x,
                    eta=eta,
                    return_dict=True,
                )
                x = step_output.prev_sample

        return x


def train_classifier(
    classifier: ClassifierWithUNetDownsample,
    train_loader,
    val_loader,
    noise_scheduler,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
    class_names=None,
    exp_folders=None,
    resume_path: str = None,
    save_every: int = 5,
    metrics_csv_path: str = None,
    metrics_json_path: str = None,
    metadata_json_path: str = None,
    class_weights: torch.Tensor = None,
):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    start_epoch = 0
    best_val_bal_acc = -1.0
    best_epoch = -1

    last_ckpt_path = None
    best_ckpt_path = None
    if exp_folders is not None:
        last_ckpt_path = os.path.join(
            exp_folders["checkpoints_dir"], "classifier_last.pth.tar"
        )
        best_ckpt_path = os.path.join(
            exp_folders["checkpoints_dir"], "classifier_best.pth.tar"
        )

    # resume
    if resume_path is not None and os.path.isfile(resume_path):
        print(f"=> resume classifier training from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        best_val_bal_acc = checkpoint.get("best_val_balanced_acc", -1.0)
        best_epoch = checkpoint.get("best_epoch", -1)

        print(f"=> resumed start_epoch = {start_epoch}")
        print(f"=> resumed best_val_balanced_acc = {best_val_bal_acc:.4f}")
        print(f"=> resumed best_epoch = {best_epoch}")

    for epoch in range(start_epoch, epochs):
        classifier.train()
        total_loss = 0.0
        total_count = 0
        all_train_labels = []
        all_train_preds = []

        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Classifier Train Epoch [{epoch + 1}/{epochs}]",
            leave=True,
        )

        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].to(device)
            labels = batch[1].to(device).long()

            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
            ).long()

            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, t)

            logits = classifier(noisy_images, t)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

            # 保存训练阶段的标签和预测结果
            all_train_labels.extend(labels.cpu().tolist())
            all_train_preds.extend(logits.argmax(dim=1).cpu().tolist())

            progress_bar.update(1)
            progress_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        progress_bar.close()

        # 计算训练准确率
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        print(f"train_accuracy        : {train_acc:.4f}")

        train_loss = total_loss / max(total_count, 1)

        # 验证阶段
        val_result = evaluate_guidance_classifier(
            classifier=classifier,
            data_loader=val_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            class_names=class_names,
        )

        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_result['loss']:.4f}, "
            f"val_acc={val_result['accuracy']:.4f}, "
            f"val_bal_acc={val_result['balanced_accuracy']:.4f}"
        )

        # 保存混淆矩阵
        cm_csv_path, cm_png_path = save_confusion_matrix_artifacts(
            y_true=all_train_labels,
            y_pred=all_train_preds,
            class_names=class_names,
            save_dir=exp_folders["confusion_matrices_dir"],
            epoch=epoch + 1,
        )

        print(f"Confusion matrix saved to: {cm_png_path}")

        # 处理模型保存
        checkpoint_state = {
            "epoch": epoch + 1,
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_balanced_accuracy": float(val_result["balanced_accuracy"]),
            "best_val_balanced_acc": float(
                max(best_val_bal_acc, val_result["balanced_accuracy"])
            ),
            "best_epoch": int(best_epoch),
        }

        if last_ckpt_path is not None:
            torch.save(checkpoint_state, last_ckpt_path)
            print(f"Last classifier checkpoint saved to: {last_ckpt_path}")

        if val_result["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_result["balanced_accuracy"]
            best_epoch = epoch + 1

            checkpoint_state["best_val_balanced_acc"] = float(best_val_bal_acc)
            checkpoint_state["best_epoch"] = best_epoch

            if best_ckpt_path is not None:
                torch.save(checkpoint_state, best_ckpt_path)
                print(
                    f"New best classifier saved to: {best_ckpt_path} "
                    f"(epoch={best_epoch}, bal_acc={best_val_bal_acc:.4f})"
                )

        if metrics_csv_path is not None:
            append_classifier_metrics_csv(metrics_csv_path, checkpoint_state)

        if metrics_json_path is not None:
            append_classifier_metrics_json(metrics_json_path, checkpoint_state)

    print(
        f"\nBest classifier epoch: {best_epoch}, best val balanced accuracy: {best_val_bal_acc:.4f}"
    )


import csv
import json
from tqdm import tqdm


def append_classifier_metrics_csv(csv_path, row_dict):
    """
    把每个 epoch 的扁平指标追加写入 CSV
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    flat_row = {
        "epoch": row_dict.get("epoch"),
        "train_loss": row_dict.get("train_loss"),
        "train_accuracy": row_dict.get("train_accuracy"),
        "val_loss": row_dict.get("val_loss"),
        "val_accuracy": row_dict.get("val_accuracy"),
        "val_balanced_accuracy": row_dict.get("val_balanced_accuracy"),
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_row)


def append_classifier_metrics_json(json_path, row_dict):
    """
    把每个 epoch 的完整指标追加写入 JSON
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(row_dict)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_classifier_experiment_metadata(
    metadata_json_path,
    epoch,
    best_epoch,
    best_val_balanced_acc,
    last_checkpoint_path,
    best_checkpoint_path,
):
    """
    每个 epoch 结束后，把实验元数据里的训练进度和 best 结果同步更新
    """
    if os.path.exists(metadata_json_path):
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["last_epoch_finished"] = epoch
    metadata["updated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "paths" not in metadata:
        metadata["paths"] = {}
    metadata["paths"]["last_classifier_checkpoint"] = last_checkpoint_path
    metadata["paths"]["best_classifier_checkpoint"] = best_checkpoint_path

    if "best_result" not in metadata:
        metadata["best_result"] = {}

    metadata["best_result"]["best_epoch"] = best_epoch
    metadata["best_result"]["best_val_balanced_acc"] = float(best_val_balanced_acc)
    metadata["best_result"]["best_model_path"] = (
        best_checkpoint_path if os.path.exists(best_checkpoint_path) else ""
    )

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


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
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
