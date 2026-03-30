import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
from collections import Counter  # 新增：用于统计每个类别的样本数

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
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

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
parser = argparse.ArgumentParser(description='PyTorch ResNet50 Training for MILK10k')
parser.add_argument('--arch', default='resnet50', choices=model_names,
                    help='model architecture (default: resnet50)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    dest='lr', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    dest='weight_decay', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set only')
parser.add_argument('--pretrained', action='store_true',
                    help='use ImageNet pretrained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use, e.g. 0. If not set, automatically choose cuda/cpu')

best_acc1 = 0.0


# =========================================================
# 3. 数据集定义
# =========================================================
class ISICResNetDataset(Dataset):
    def __init__(self, meta_csv_path, gt_csv_path, img_dir, transform=None):
        """
        初始化 ISIC / MILK10k 单模态分类数据集
        """
        self.img_dir = img_dir
        self.transform = transform

        meta_df = pd.read_csv(meta_csv_path)
        gt_df = pd.read_csv(gt_csv_path)

        df = pd.merge(meta_df, gt_df, on='lesion_id', how='inner')

        if 'image_type' in df.columns:
            derm_mask = df['image_type'].str.contains('dermoscopic', case=False, na=False)
            df = df[derm_mask].copy()

        self.df = df.drop_duplicates(subset=['lesion_id'], keep='first').reset_index(drop=True)
        self.class_columns = [c for c in gt_df.columns if c != 'lesion_id']

        # 提前把每个样本的整数标签算好，后面做分层划分会用到
        if 'label' in self.df.columns:
            self.labels = self.df['label'].astype(int).tolist()
        else:
            self.labels = self.df[self.class_columns].values.argmax(axis=1).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id_col = 'isic_id' if 'isic_id' in row.index else ('image_id' if 'image_id' in row.index else None)
        if img_id_col is None:
            raise KeyError("无法在 CSV 中找到 'isic_id' 或 'image_id' 列。")

        lesion_id = str(row['lesion_id']) if 'lesion_id' in row.index else None
        img_name = f"{row[img_id_col]}.jpg"

        if lesion_id:
            img_path = os.path.join(self.img_dir, lesion_id, img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if 'label' in row.index:
            label = torch.tensor(int(row['label']), dtype=torch.long)
        else:
            if all(c in row.index for c in self.class_columns):
                label_array = row[self.class_columns].values.astype(float)
                label = torch.tensor(label_array.argmax(), dtype=torch.long)
            else:
                raise KeyError("无法提取标签：既没有 'label' 列，也没有 one-hot 类别列。")

        # 这里返回的 sample_id 就作为后续保存 CSV 时的 case_id
        sample_id = str(row['lesion_id']) if 'lesion_id' in row.index else str(idx)

        return image, label, sample_id


# =========================================================
# 4. 实验目录与日志工具函数
# =========================================================
def make_experiment_name(args):
    """
    生成规范实验名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pretrained_tag = "pretrained" if args.pretrained else "scratch"
    seed_tag = f"seed{args.seed}" if args.seed is not None else "seedNone"
    exp_name = f"{timestamp}_{args.arch}_{pretrained_tag}_lr{args.lr}_bs{args.batch_size}_{seed_tag}"
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
    predictions_dir = os.path.join(exp_dir, "predictions")  # 新增：样本级预测保存目录

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
    追加保存每轮指标到 CSV
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
    追加保存每轮指标到 JSON
    """
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(row_dict)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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
            print(f"[Warning] Epoch {epoch}: 类别 '{class_name}' 在当前验证集缺少正样本或负样本，跳过该类 ROC。")
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


def save_val_predictions_csv(sample_ids, y_true, y_pred, y_prob, save_dir, epoch):
    """
    保存样本级验证预测结果
    文件名格式：
    val_predictions_epoch_XX.csv
    每个样本一行：
    case_id, y_true, y_pred, prob_0, prob_1, ...
    """
    data = {
        "case_id": list(sample_ids),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }

    num_classes = y_prob.shape[1]
    for i in range(num_classes):
        data[f"prob_{i}"] = y_prob[:, i].tolist()

    pred_df = pd.DataFrame(data)
    pred_csv_path = os.path.join(save_dir, f"val_predictions_epoch_{epoch:02d}.csv")
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    return pred_csv_path


def count_labels_from_indices(labels, indices, class_names):
    """
    根据给定下标统计每个类别的样本数量
    labels: 完整数据集的整数标签列表
    indices: 要统计的样本下标列表
    class_names: 类别名列表
    """
    counter = Counter([labels[i] for i in indices])

    # 按类别顺序生成一个更清晰的字典，方便打印和保存到 metadata
    count_dict = {}
    for class_idx, class_name in enumerate(class_names):
        count_dict[class_name] = int(counter.get(class_idx, 0))

    return count_dict


def print_class_distribution(title, count_dict):
    """
    把类别统计结果打印出来
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
    # 路径配置
    # -----------------------------
    meta_csv_path = r'dataset\MILK10k_Training_Metadata.csv'
    gt_csv_path = r'dataset\MILK10k_Training_GroundTruth.csv'
    img_dir = r'dataset\MILK10k_Training_Input\MILK10k_Training_Input'

    # -----------------------------
    # 创建实验目录
    # -----------------------------
    exp_name = make_experiment_name(args)
    exp_folders = setup_experiment_folders(base_dir="experiments", exp_name=exp_name)

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(exp_folders["metadata_dir"], "experiment_metadata.json")

    # -----------------------------
    # 创建模型
    # -----------------------------
    if args.pretrained:
        print(f"=> using pretrained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    # -----------------------------
    # 读取类别信息并修改最后分类层
    # -----------------------------
    gt_df = pd.read_csv(gt_csv_path)
    class_columns = [c for c in gt_df.columns if c != 'lesion_id']
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
    # -----------------------------
    best_epoch = -1
    best_model_path = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']

            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
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

    full_dataset = ISICResNetDataset(
        meta_csv_path=meta_csv_path,
        gt_csv_path=gt_csv_path,
        img_dir=img_dir,
        transform=resnet_transforms
    )

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # =====================================================
    # 分层划分训练集和验证集
    # 目的：尽量保证 train / val 中每个类别的比例接近整体分布，
    # 从而避免验证集里某些类别完全缺失，导致 ROC 无法计算
    # =====================================================
    all_indices = np.arange(total_size)
    all_labels = np.array(full_dataset.labels)

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=args.seed if args.seed is not None else 42,
        stratify=all_labels
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"Full dataset size : {len(full_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")

    # =====================================================
    # 新增：训练前统计 full / train / val 三部分每一类图片数量
    # =====================================================
    full_class_distribution = count_labels_from_indices(
        labels=full_dataset.labels,
        indices=all_indices,
        class_names=class_names
    )

    train_class_distribution = count_labels_from_indices(
        labels=full_dataset.labels,
        indices=train_indices,
        class_names=class_names
    )

    val_class_distribution = count_labels_from_indices(
        labels=full_dataset.labels,
        indices=val_indices,
        class_names=class_names
    )

    # 打印输出，方便你训练前直接检查类别分布
    print_class_distribution("Full Dataset Class Distribution", full_class_distribution)
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
    # -----------------------------
    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.arch,
        "pretrained": bool(args.pretrained),
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "start_epoch": args.start_epoch,
        "seed": args.seed,
        "device": str(device),
        "data": {
            "meta_csv_path": meta_csv_path,
            "gt_csv_path": gt_csv_path,
            "img_dir": img_dir,
            "full_dataset_size": total_size,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "split_ratio": "0.9 / 0.1",
            "num_classes": num_classes,
            "class_names": class_names,

            # 新增：把 full/train/val 的类别分布保存到元信息里
            "class_distribution": {
                "full_dataset": full_class_distribution,
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
        "best_result": {
            "best_epoch": best_epoch,
            "best_val_acc": float(best_acc1),
            "best_model_path": best_model_path if os.path.exists(best_model_path) else ""
        }
    }
    save_json(experiment_metadata, metadata_json_path)

    # -----------------------------
    # 仅评估模式
    # -----------------------------
    if args.evaluate:
        val_metrics = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            args=args,
            num_classes=num_classes,
            class_names=class_names,
            epoch=args.start_epoch,
            roc_dir=exp_folders["roc_dir"],
            cm_dir=exp_folders["cm_dir"],
            predictions_dir=exp_folders["predictions_dir"]
        )

        eval_row = {
            "epoch": args.start_epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": None,
            "train_acc": None,
            "val_loss": float(val_metrics["val_loss"]),
            "val_acc": float(val_metrics["val_acc"]),
            "val_macro_f1": float(val_metrics["val_macro_f1"]),
            "val_auroc": float(val_metrics["val_auroc"]),
            "roc_curve_path": val_metrics["roc_curve_path"],
            "confusion_matrix_path": val_metrics["confusion_matrix_png_path"],
            "val_predictions_path": val_metrics["val_predictions_csv_path"]
        }
        update_epoch_metrics_csv(metrics_csv_path, eval_row)
        update_epoch_metrics_json(metrics_json_path, eval_row)
        return

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(args.start_epoch, args.epochs):
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
            predictions_dir=exp_folders["predictions_dir"]
        )

        scheduler.step()

        acc1 = val_metrics["val_acc"]
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch + 1

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, save_dir=exp_folders["checkpoints_dir"], filename='last.pth.tar')

        # 每轮指标保存
        epoch_row = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["train_loss"]),
            "train_acc": float(train_metrics["train_acc"]),
            "val_loss": float(val_metrics["val_loss"]),
            "val_acc": float(val_metrics["val_acc"]),
            "val_macro_f1": float(val_metrics["val_macro_f1"]),
            "val_auroc": float(val_metrics["val_auroc"]),
            "roc_curve_path": val_metrics["roc_curve_path"],
            "confusion_matrix_path": val_metrics["confusion_matrix_png_path"],
            "val_predictions_path": val_metrics["val_predictions_csv_path"]
        }
        update_epoch_metrics_csv(metrics_csv_path, epoch_row)
        update_epoch_metrics_json(metrics_json_path, epoch_row)

        # 更新实验元信息
        experiment_metadata["best_result"]["best_epoch"] = best_epoch
        experiment_metadata["best_result"]["best_val_acc"] = float(best_acc1)
        experiment_metadata["best_result"]["best_model_path"] = best_model_path if os.path.exists(best_model_path) else ""
        experiment_metadata["last_epoch_finished"] = epoch + 1
        experiment_metadata["updated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_json(experiment_metadata, metadata_json_path)


# =========================================================
# 6. 训练函数
# =========================================================
def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.NONE)

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()

    end = time.time()
    for i, (images, target, sample_ids) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        num_classes = output.size(1)
        if num_classes >= 5:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top5.update(acc5[0].item(), images.size(0))
        else:
            acc1 = accuracy(output, target, topk=(1,))
            acc1 = [acc1[0]]
            top5.update(0.0, images.size(0))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return {
        "train_loss": float(losses.avg),
        "train_acc": float(top1.avg),
        "lr": float(optimizer.param_groups[0]["lr"])
    }


# =========================================================
# 7. 验证函数
# =========================================================
def validate(val_loader, model, criterion, device, args, num_classes, class_names, epoch, roc_dir, cm_dir, predictions_dir):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Val: '
    )

    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_sample_ids = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target, sample_ids) in enumerate(val_loader):
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

            if output.size(1) >= 5:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top5.update(acc5[0].item(), images.size(0))
            else:
                acc1 = accuracy(output, target, topk=(1,))
                acc1 = [acc1[0]]
                top5.update(0.0, images.size(0))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # 1) 分类指标
    val_acc = accuracy_score(all_targets, all_preds) * 100.0
    val_macro_f1 = f1_score(all_targets, all_preds, average='macro')

    # 2) AUROC
    try:
        y_true_bin = label_binarize(all_targets, classes=np.arange(num_classes))
        val_auroc = roc_auc_score(
            y_true_bin,
            all_probs,
            average='macro',
            multi_class='ovr'
        )
    except ValueError as e:
        print(f"[Warning] AUROC 计算失败，可能是验证集缺少某些类别。错误信息: {e}")
        val_auroc = float("nan")

    # 3) 保存混淆矩阵
    confusion_matrix_csv_path, confusion_matrix_png_path = save_confusion_matrix_artifacts(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        save_dir=cm_dir,
        epoch=epoch
    )

    # 4) 保存 ROC 曲线
    roc_curve_path, roc_points_json_path = save_multiclass_roc_artifacts(
        y_true=all_targets,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=roc_dir,
        epoch=epoch
    )

    # 5) 保存样本级预测 CSV（新增）
    val_predictions_csv_path = save_val_predictions_csv(
        sample_ids=all_sample_ids,
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        save_dir=predictions_dir,
        epoch=epoch
    )

    print(
        f" * Val Metrics => "
        f"Loss: {losses.avg:.4f}, "
        f"Acc: {val_acc:.4f}, "
        f"Macro-F1: {val_macro_f1:.4f}, "
        f"AUROC: {val_auroc:.4f}"
    )
    print(f" * ROC curve saved to: {roc_curve_path}")
    print(f" * Confusion matrix saved to: {confusion_matrix_png_path}")
    print(f" * Validation predictions saved to: {val_predictions_csv_path}")

    return {
        "val_loss": float(losses.avg),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_auroc": float(val_auroc),
        "roc_curve_path": roc_curve_path,
        "roc_points_json_path": roc_points_json_path,
        "confusion_matrix_csv_path": confusion_matrix_csv_path,
        "confusion_matrix_png_path": confusion_matrix_png_path,
        "val_predictions_csv_path": val_predictions_csv_path
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
    """打印训练/验证过程中的日志"""
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