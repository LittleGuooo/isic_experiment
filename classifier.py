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
    precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 新增：使用 tqdm 改善训练/验证进度条

# =========================================================
# 1. 支持的模型名称
# =========================================================
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

# =========================================================
# 2. 命令行参数
# =========================================================
parser = argparse.ArgumentParser(description='PyTorch ResNet50 Training for ISIC2018 Task3')
parser.add_argument('--arch', default='resnet50', choices=model_names,
                    help='model architecture (default: resnet50)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    dest='lr', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    dest='weight_decay', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency (保留参数，但这里主要使用 tqdm 进度条)')
parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set only')
parser.add_argument('--weights', default=None, type=str,
                    help='pretrained weights name, e.g. DEFAULT / IMAGENET1K_V1 / IMAGENET1K_V2')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use, e.g. 0. If not set, automatically choose cuda/cpu')

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

        self.class_columns = [c for c in df.columns if c != 'image']
        self.df = df.reset_index(drop=True)

        # 提前计算整数标签，后面统计类别分布会用到
        self.labels = self.df[self.class_columns].values.argmax(axis=1).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row['image'])
        img_name = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

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
    exp_name = f"{timestamp}_{args.arch}_{weights_tag}_lr{args.lr}_bs{args.batch_size}_{seed_tag}"
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
        "predictions_dir": predictions_dir
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
        "predictions_dir": predictions_dir
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
        "macro_average_auc": None
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
            "auc": float(roc_auc)
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
                label=f"Macro-average ROC (AUC={macro_auc:.3f})"
            )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
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


def save_val_predictions_csv(sample_ids, y_true, y_pred, y_prob, class_names, save_dir, epoch):
    """
    保存样本级验证预测结果
    每个样本一行：
    case_id, y_true, y_pred, prob_MEL, prob_NV, ...
    """
    data = {
        "case_id": list(sample_ids),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
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

        sensitivity = safe_div(tp, tp + fn)       # recall
        specificity = safe_div(tn, tn + fp)
        accuracy_i = safe_div(tp + tn, total_samples)
        ppv = safe_div(tp, tp + fp)               # precision
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
            "sensitivity": float(sensitivity) if not np.isnan(sensitivity) else float("nan"),
            "specificity": float(specificity) if not np.isnan(specificity) else float("nan"),
            "accuracy": float(accuracy_i) if not np.isnan(accuracy_i) else float("nan"),
            "auc": float(auc_i) if not np.isnan(auc_i) else float("nan"),
            "mean_average_precision": float(ap_i) if not np.isnan(ap_i) else float("nan"),
            "f1_score": float(f1_i) if not np.isnan(f1_i) else float("nan"),
            "ppv": float(ppv) if not np.isnan(ppv) else float("nan"),
            "npv": float(npv) if not np.isnan(npv) else float("nan"),
            "auc80": float(melanoma_auc80) if not np.isnan(melanoma_auc80) else float("nan")
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

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
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
            roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
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

    melanoma_auc80 = per_class_metrics["MEL"]["auc80"] if "MEL" in per_class_metrics else float("nan")

    metrics = {
        "overall": {
            "accuracy": float(overall_accuracy),  # 保持和你原代码一致：百分比
            "balanced_multiclass_accuracy": float(balanced_macro_recall),   # macro recall
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
            "melanoma_auc80": float(melanoma_auc80) if not np.isnan(melanoma_auc80) else float("nan"),
            "malignant_vs_benign_auc": float(malignant_vs_benign_auc) if not np.isnan(malignant_vs_benign_auc) else float("nan")
        },
        "per_class": per_class_metrics,
        "malignant_definition_used": malignant_names
    }

    return metrics


# =========================================================
# 5. 主函数
# =========================================================
def main():
    global best_acc1
    args = parser.parse_args()

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
            'You have chosen to seed training. '
            'This may slow down training a bit, but improves reproducibility.'
        )

    # -----------------------------
    # 选择设备
    # -----------------------------
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # -----------------------------
    # ISIC2018 Task3 路径
    # -----------------------------
    train_gt_csv_path = r'dataset\ISIC2018_Task3_Training_GroundTruth.csv'
    val_gt_csv_path = r'dataset\ISIC2018_Task3_Validation_GroundTruth.csv'
    train_img_dir = r'dataset\ISIC2018_Task3_Training_Input'
    val_img_dir = r'dataset\ISIC2018_Task3_Validation_Input'

    # -----------------------------
    # 先准备 start_epoch / checkpoint 变量
    # 注意：删除了 --start-epoch 参数后，训练起点完全由 checkpoint 决定
    # -----------------------------
    start_epoch = 0
    best_epoch = -1
    checkpoint = None

    # -----------------------------
    # 创建或复用实验目录
    # resume 时：优先复用 checkpoint 里记录的旧实验目录
    # -----------------------------
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"=> loading checkpoint metadata from '{args.resume}' to recover experiment folder")
        checkpoint = torch.load(args.resume, map_location=device)

        if 'exp_dir' not in checkpoint:
            raise ValueError("resume 的 checkpoint 中没有 'exp_dir'，无法复用旧实验目录。")

        exp_folders = reuse_experiment_folders(checkpoint['exp_dir'])
        exp_name = os.path.basename(checkpoint['exp_dir'])
        print(f"=> reusing experiment folder: {exp_folders['exp_dir']}")
    else:
        exp_name = make_experiment_name(args)
        exp_folders = setup_experiment_folders(base_dir="experiments", exp_name=exp_name)

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(exp_folders["metadata_dir"], "experiment_metadata.json")

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
    class_columns = [c for c in gt_df.columns if c != 'image']
    num_classes = len(class_columns)
    class_names = class_columns

    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"当前代码只处理带有 model.fc 的模型，当前模型 {args.arch} 不满足该条件。")

    model = model.to(device)

    # -----------------------------
    # 损失函数、优化器、学习率调度器
    # -----------------------------
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
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
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']

        print(f"=> loaded checkpoint '{args.resume}' (finished epoch {checkpoint['epoch']})")
        print(f"=> training will continue from epoch {start_epoch + 1}")

    elif args.resume is not None:
        print(f"=> no checkpoint found at '{args.resume}'")

    # =====================================================
    # 数据加载部分
    # =====================================================
    resnet_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ISICResNetDataset(
        gt_csv_path=train_gt_csv_path,
        img_dir=train_img_dir,
        transform=resnet_transforms
    )

    val_dataset = ISICResNetDataset(
        gt_csv_path=val_gt_csv_path,
        img_dir=val_img_dir,
        transform=resnet_transforms
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")

    # 只打印 train / val，不再打印 full dataset
    train_class_distribution = count_labels_from_dataset(
        labels=train_dataset.labels,
        class_names=class_names
    )

    val_class_distribution = count_labels_from_dataset(
        labels=val_dataset.labels,
        class_names=class_names
    )

    print_class_distribution("Train Dataset Class Distribution", train_class_distribution)
    print_class_distribution("Validation Dataset Class Distribution", val_class_distribution)

    pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory
    )

    # -----------------------------
    # 初始化实验元信息
    # resume 时会写回原实验目录中的 metadata
    # -----------------------------
    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.arch,
        "weights": args.weights,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
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
            "split_ratio": "official train / official val",
            "num_classes": num_classes,
            "class_names": class_names,
            "class_distribution": {
                "train_dataset": train_class_distribution,
                "val_dataset": val_class_distribution
            }
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "last_checkpoint": os.path.join(exp_folders["checkpoints_dir"], "last.pth.tar"),
            "best_checkpoint": best_model_path,
            "roc_dir": exp_folders["roc_dir"],
            "confusion_matrix_dir": exp_folders["cm_dir"],
            "predictions_dir": exp_folders["predictions_dir"]
        },
        "resume": {
            "resume_path": args.resume,
            "start_epoch": start_epoch
        },
        "best_result": {
            "best_epoch": best_epoch,
            "best_val_balanced_acc": float(best_acc1),
            "best_model_path": best_model_path if os.path.exists(best_model_path) else ""
        }
    }
    save_json(experiment_metadata, metadata_json_path)

    # -----------------------------
    # 仅评估模式
    # 删除 start-epoch 参数后：
    # 如果是 resume + evaluate，则使用 checkpoint 恢复到的 start_epoch 作为评估标记轮次
    # 否则默认 epoch=0
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
            metrics_dir=exp_folders["metrics_dir"]
        )

        eval_row = {
            "epoch": eval_epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": None,
            "train_acc": None,
            "val_loss": float(val_metrics["val_loss"]),
            "val_acc": float(val_metrics["overall"]["accuracy"]),
            "val_balanced_acc": float(val_metrics["overall"]["balanced_multiclass_accuracy"]),
            "val_macro_recall": float(val_metrics["overall"]["macro_recall"]),
            "val_macro_f1": float(val_metrics["overall"]["macro_f1"]),
            "val_macro_precision": float(val_metrics["overall"]["macro_precision"]),
            "val_auc_macro_ovr": float(val_metrics["overall"]["multiclass_macro_auc_ovr"]),
            "val_mean_auc_all_diagnoses": float(val_metrics["overall"]["mean_auc_all_diagnoses"]),
            "val_mean_ap_all_diagnoses": float(val_metrics["overall"]["mean_average_precision_all_diagnoses"]),
            "val_mean_sensitivity": float(val_metrics["overall"]["mean_sensitivity"]),
            "val_mean_specificity": float(val_metrics["overall"]["mean_specificity"]),
            "val_mean_ppv": float(val_metrics["overall"]["mean_ppv"]),
            "val_mean_npv": float(val_metrics["overall"]["mean_npv"]),
            "val_melanoma_auc80": float(val_metrics["overall"]["melanoma_auc80"]),
            "val_malignant_vs_benign_auc": float(val_metrics["overall"]["malignant_vs_benign_auc"]),
            "roc_curve_path": val_metrics["roc_curve_path"],
            "confusion_matrix_path": val_metrics["confusion_matrix_png_path"],
            "val_predictions_path": val_metrics["val_predictions_csv_path"],
            "detailed_metrics_path": val_metrics["detailed_metrics_json_path"]
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

        train_metrics = train(train_loader, model, criterion, optimizer, epoch, device, args)

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
            metrics_dir=exp_folders["metrics_dir"]
        )

        scheduler.step()

        # 用 balanced multi-class accuracy 作为 best 指标
        current_score = val_metrics["overall"]["balanced_multiclass_accuracy"]
        is_best = current_score > best_acc1
        if is_best:
            best_acc1 = current_score
            best_epoch = epoch + 1

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'exp_dir': exp_folders["exp_dir"]  # 新增：保存实验目录，供 resume 时复用
        }, is_best, save_dir=exp_folders["checkpoints_dir"], filename='last.pth.tar')

        # 每轮指标保存（保存汇总指标）
        epoch_row = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["train_loss"]),
            "train_acc": float(train_metrics["train_acc"]),
            "val_loss": float(val_metrics["val_loss"]),
            "val_acc": float(val_metrics["overall"]["accuracy"]),
            "val_balanced_acc": float(val_metrics["overall"]["balanced_multiclass_accuracy"]),
            "val_macro_recall": float(val_metrics["overall"]["macro_recall"]),
            "val_macro_f1": float(val_metrics["overall"]["macro_f1"]),
            "val_macro_precision": float(val_metrics["overall"]["macro_precision"]),
            "val_auc_macro_ovr": float(val_metrics["overall"]["multiclass_macro_auc_ovr"]),
            "val_mean_auc_all_diagnoses": float(val_metrics["overall"]["mean_auc_all_diagnoses"]),
            "val_mean_ap_all_diagnoses": float(val_metrics["overall"]["mean_average_precision_all_diagnoses"]),
            "val_mean_sensitivity": float(val_metrics["overall"]["mean_sensitivity"]),
            "val_mean_specificity": float(val_metrics["overall"]["mean_specificity"]),
            "val_mean_ppv": float(val_metrics["overall"]["mean_ppv"]),
            "val_mean_npv": float(val_metrics["overall"]["mean_npv"]),
            "val_melanoma_auc80": float(val_metrics["overall"]["melanoma_auc80"]),
            "val_malignant_vs_benign_auc": float(val_metrics["overall"]["malignant_vs_benign_auc"]),
            "roc_curve_path": val_metrics["roc_curve_path"],
            "confusion_matrix_path": val_metrics["confusion_matrix_png_path"],
            "val_predictions_path": val_metrics["val_predictions_csv_path"],
            "detailed_metrics_path": val_metrics["detailed_metrics_json_path"]
        }
        update_epoch_metrics_csv(metrics_csv_path, epoch_row)
        update_epoch_metrics_json(metrics_json_path, epoch_row)

        # 更新实验元信息
        experiment_metadata["best_result"]["best_epoch"] = best_epoch
        experiment_metadata["best_result"]["best_val_balanced_acc"] = float(best_acc1)
        experiment_metadata["best_result"]["best_model_path"] = best_model_path if os.path.exists(best_model_path) else ""
        experiment_metadata["last_epoch_finished"] = epoch + 1
        experiment_metadata["updated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_json(experiment_metadata, metadata_json_path)

        # 简洁 epoch 汇总输出
        print(f"Train Summary | loss={train_metrics['train_loss']:.4f} | acc={train_metrics['train_acc']:.2f}%")
        print(
            f"Val Summary   | loss={val_metrics['val_loss']:.4f} "
            f"| acc={val_metrics['overall']['accuracy']:.2f}% "
            f"| bal_acc={val_metrics['overall']['balanced_multiclass_accuracy']:.4f} "
            f"| macro_f1={val_metrics['overall']['macro_f1']:.4f} "
            f"| auc_macro_ovr={val_metrics['overall']['multiclass_macro_auc_ovr']:.4f}"
        )
        print(
            f"Extra Metrics | mean_auc={val_metrics['overall']['mean_auc_all_diagnoses']:.4f} "
            f"| mean_ap={val_metrics['overall']['mean_average_precision_all_diagnoses']:.4f} "
            f"| mel_auc80={val_metrics['overall']['melanoma_auc80']:.4f} "
            f"| mal_vs_ben_auc={val_metrics['overall']['malignant_vs_benign_auc']:.4f}"
        )
        print(f"Artifacts      | ROC={val_metrics['roc_curve_path']}")
        print(f"Artifacts      | CM ={val_metrics['confusion_matrix_png_path']}")
        print(f"Artifacts      | PRED={val_metrics['val_predictions_csv_path']}")
        print(f"Artifacts      | DETAIL={val_metrics['detailed_metrics_json_path']}")


# =========================================================
# 6. 训练函数
# =========================================================
def train(train_loader, model, criterion, optimizer, epoch, device, args):
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.NONE)

    model.train()

    # 使用 tqdm 包装 dataloader
    # desc：左侧显示当前阶段
    # total：总 batch 数
    # leave=False：一个 epoch 结束后不保留整条进度条，日志更干净
    progress_bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Train Epoch {epoch + 1}",
        leave=False
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
        progress_bar.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "acc": f"{top1.avg:.2f}%"
        })

    return {
        "train_loss": float(losses.avg),
        "train_acc": float(top1.avg),
        "lr": float(optimizer.param_groups[0]["lr"])
    }


# =========================================================
# 7. 验证函数
# =========================================================
def validate(val_loader, model, criterion, device, args, num_classes, class_names, epoch,
             roc_dir, cm_dir, predictions_dir, metrics_dir):
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_sample_ids = []

    with torch.no_grad():
        progress_bar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Validate {epoch}",
            leave=False
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
            progress_bar.set_postfix({
                "loss": f"{losses.avg:.4f}",
                "acc": f"{top1.avg:.2f}%"
            })

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # 1) 保存混淆矩阵
    confusion_matrix_csv_path, confusion_matrix_png_path = save_confusion_matrix_artifacts(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        save_dir=cm_dir,
        epoch=epoch
    )

    # 2) 保存 ROC 曲线
    roc_curve_path, roc_points_json_path = save_multiclass_roc_artifacts(
        y_true=all_targets,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=roc_dir,
        epoch=epoch
    )

    # 3) 保存样本级预测 CSV
    val_predictions_csv_path = save_val_predictions_csv(
        sample_ids=all_sample_ids,
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=predictions_dir,
        epoch=epoch
    )

    # 4) 计算详细指标
    detailed_metrics = compute_detailed_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names
    )
    detailed_metrics["val_loss"] = float(losses.avg)

    detailed_metrics_json_path = save_detailed_metrics_json(
        detailed_metrics=detailed_metrics,
        save_dir=metrics_dir,
        epoch=epoch
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
        "detailed_metrics_json_path": detailed_metrics_json_path
    }


# =========================================================
# 8. 保存 checkpoint
# =========================================================
def save_checkpoint(state, is_best, save_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    保存模型 checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    best_filepath = os.path.join(save_dir, 'model_best.pth.tar')

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
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            return ''
        elif self.summary_type is Summary.AVERAGE:
            return f'{self.name} {self.avg:.3f}'
        elif self.summary_type is Summary.SUM:
            return f'{self.name} {self.sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            return f'{self.name} {self.count:.3f}'
        else:
            raise ValueError('invalid summary type')


class ProgressMeter:
    """保留原类，但当前主流程已不再使用这个冗长打印方式"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters if str(meter) != '']
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters if meter.summary() != '']
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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


if __name__ == '__main__':
    main()