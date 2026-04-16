import os
import random
import shutil
import warnings
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .augmentation import build_train_dataset, parse_ratios
from .dataset import (
    count_labels_from_dataset,
    print_class_distribution,
    format_count_ratio_dict,
)
from .metrics import (
    compute_detailed_classification_metrics,
    save_confusion_matrix_artifacts,
    save_detailed_metrics_json,
    save_json,
    save_multiclass_roc_artifacts,
    save_val_predictions_csv,
    update_epoch_metrics_csv,
    update_epoch_metrics_json,
)


# 全局变量：记录历史最佳 balanced accuracy
best_acc1 = 0.0


class Summary(Enum):
    """
    这个枚举类本来一般用于控制 AverageMeter 的汇总方式。
    当前代码里虽然没有复杂使用，但保留是合理的。
    """

    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """
    用于统计某个指标的当前值、累计和、平均值。
    常见于训练循环里记录 loss / accuracy。
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """把统计量清零。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计量。
        val: 当前 batch 的指标值
        n: 当前 batch 的样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_experiment_name(args):
    """
    构造实验名。
    这样保存出来的实验目录会包含时间、模型、学习率、batch size、seed 等信息。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_tag = args.weights if args.weights is not None else "scratch"
    aug_tag = "diffaug" if args.use_diffusion_augmentation else "noaug"
    return (
        f"{timestamp}_{args.arch}_{weights_tag}_{aug_tag}_"
        f"lr{args.lr}_bs{args.batch_size}_seed{args.seed}"
    )


def setup_experiment_folders(base_dir, exp_name):
    """
    创建实验目录及其子目录。
    所有训练输出都会保存到这些目录中。
    """
    exp_dir = os.path.join(base_dir, exp_name)
    folders = {
        "exp_dir": exp_dir,
        "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
        "metrics_dir": os.path.join(exp_dir, "metrics"),
        "metadata_dir": os.path.join(exp_dir, "metadata"),
        "roc_dir": os.path.join(exp_dir, "roc_curves"),
        "cm_dir": os.path.join(exp_dir, "confusion_matrices"),
        "predictions_dir": os.path.join(exp_dir, "predictions"),
    }

    os.makedirs(exp_dir, exist_ok=True)
    for key, path in folders.items():
        if key != "exp_dir":
            os.makedirs(path, exist_ok=True)
    return folders


def reuse_experiment_folders(exp_dir):
    """
    断点恢复训练时，复用旧实验目录。
    """
    return setup_experiment_folders(os.path.dirname(exp_dir), os.path.basename(exp_dir))


def save_checkpoint(
    state, is_best, save_dir="checkpoints", filename="checkpoint.pth.tar"
):
    """
    保存 checkpoint。
    如果 is_best=True，则额外复制一份为 model_best.pth.tar。
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    best_filepath = os.path.join(save_dir, "model_best.pth.tar")

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, best_filepath)


def accuracy(output, target, topk=(1,)):
    """
    计算 top-k 准确率。

    output: 模型输出 logits，shape [B, C]
    target: 真实标签，shape [B]

    返回值是一个 list，比如 topk=(1, 5) 时返回 [top1, top5]
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        # 取每个样本预测分数最高的前 k 个类别索引
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # 比较预测是否命中真实标签
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(
    train_loader, model, criterion, optimizer, epoch, device, scaler, use_amp
):
    """
    训练一个 epoch。

    新增：
    - scaler: AMP 混合精度的梯度缩放器
    - use_amp: 是否启用 AMP
    """
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.NONE)

    model.train()

    progress_bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Train Epoch {epoch + 1}",
        leave=False,
    )

    for images, target, _ in progress_bar:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 开启 AMP 后，前向和 loss 计算会自动用混合精度
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # AMP 训练三步：scale -> backward -> step -> update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix(
            {"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"}
        )

    return {
        "train_loss": float(losses.avg),
        "train_acc": float(top1.avg),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def validate(
    val_loader,
    model,
    criterion,
    device,
    num_classes,
    class_names,
    epoch,
    roc_dir,
    cm_dir,
    predictions_dir,
    metrics_dir,
):
    """
    在验证集上评估模型，并保存各种评估产物：
    - loss
    - acc
    - confusion matrix
    - ROC curve
    - 每个样本预测结果
    - detailed metrics JSON
    """
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)

    model.eval()

    # 用于汇总整个验证集的结果
    all_targets, all_preds, all_probs, all_sample_ids = [], [], [], []

    with torch.no_grad():
        progress_bar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Validate {epoch}",
            leave=False,
        )

        for images, target, sample_ids in progress_bar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 前向传播
            output = model(images)
            loss = criterion(output, target)

            # logits -> probabilities
            probs = torch.softmax(output, dim=1)

            # 取概率最大的类别作为预测结果
            preds = torch.argmax(output, dim=1)

            # 保存 batch 结果，后面统一拼接
            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_sample_ids.extend(sample_ids)

            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            progress_bar.set_postfix(
                {"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"}
            )

    # 把所有 batch 拼成整个验证集数组
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # 保存混淆矩阵
    confusion_matrix_csv_path, confusion_matrix_png_path = (
        save_confusion_matrix_artifacts(
            y_true=all_targets,
            y_pred=all_preds,
            class_names=class_names,
            save_dir=cm_dir,
            epoch=epoch,
        )
    )

    # 保存 ROC 曲线
    roc_curve_path, roc_points_json_path = save_multiclass_roc_artifacts(
        y_true=all_targets,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=roc_dir,
        epoch=epoch,
    )

    # 保存逐样本预测表
    val_predictions_csv_path = save_val_predictions_csv(
        sample_ids=all_sample_ids,
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=predictions_dir,
        epoch=epoch,
    )

    # 计算详细分类指标
    detailed_metrics = compute_detailed_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
    )
    detailed_metrics["val_loss"] = float(losses.avg)

    detailed_metrics_json_path = save_detailed_metrics_json(
        detailed_metrics, metrics_dir, epoch
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


def run_training(args, train_dataset, val_dataset, class_names, num_classes, device):
    """
    训练主流程。

    支持：
    1. 从头训练
    2. resume 继续训练
    3. 只做 evaluate
    4. 可选扩散数据增强（Diffusion Augmentation）
    """
    global best_acc1

    start_epoch = 0
    best_epoch = -1
    early_stop_counter = 0
    early_stopped = False
    checkpoint = None

    # =========================
    # 处理 resume
    # =========================
    if args.resume is not None and os.path.isfile(args.resume):
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
    best_model_path = os.path.join(exp_folders["checkpoints_dir"], "model_best.pth.tar")

    # =========================
    # 构建分类模型
    # =========================
    if args.weights is not None:
        # 加载 torchvision 预训练权重
        weights_enum = models.get_model_weights(args.arch)
        model = models.__dict__[args.arch](weights=weights_enum[args.weights])
    else:
        # 从头训练
        model = models.__dict__[args.arch](weights=None)

    # 当前代码假设分类头是 model.fc（适用于 ResNet 类模型）
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"当前代码只处理带有 model.fc 的模型，当前模型 {args.arch} 不满足该条件。"
        )

    model = model.to(device)

    # =========================
    # 损失函数、优化器、学习率调度器
    # =========================

    class_weights = None
    if args.use_class_weights:
        # 统计当前训练集每个类别的样本数
        class_counts = np.bincount(train_dataset.labels, minlength=num_classes)

        # 类别越少，权重越大
        class_weights = 1.0 / np.maximum(class_counts, 1)

        # 归一化到“平均权重约为 1”，这样更稳一点
        class_weights = class_weights / class_weights.mean()
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

        print(f"Using class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 用余弦退火替代原来的 StepLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    # AMP 梯度缩放器：只有 CUDA + --use-amp 时才真正启用
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and args.use_amp)
    )

    # =========================
    # 如果有 checkpoint，则恢复
    # =========================
    if checkpoint is not None:
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        best_epoch = checkpoint.get("best_epoch", -1)
        early_stop_counter = checkpoint.get("early_stop_counter", 0)
        early_stopped = checkpoint.get("early_stopped", False)

        print(
            f"=> loaded checkpoint '{args.resume}' (finished epoch {checkpoint['epoch']})"
        )
        print(f"=> training will continue from epoch {start_epoch + 1}")

    # =========================
    # 打印原始数据集类别分布
    # =========================
    train_class_distribution = count_labels_from_dataset(
        train_dataset.labels, class_names
    )
    val_class_distribution = count_labels_from_dataset(val_dataset.labels, class_names)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")
    print_class_distribution(
        "Train Dataset Class Distribution", train_class_distribution
    )
    print_class_distribution(
        "Validation Dataset Class Distribution", val_class_distribution
    )

    # =========================
    # 增强数据集，并构建最终训练集
    # =========================
    aug_output_dir = args.aug_output_dir or os.path.join(
        exp_folders["exp_dir"], "train_augmented_data"
    )
    final_train_dataset, synth_dataset, aug_output_dir = build_train_dataset(
        args=args,
        train_dataset=train_dataset,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
        output_dir=aug_output_dir,
    )

    synth_class_distribution = None
    augmented_train_class_distribution = train_class_distribution

    if synth_dataset is not None:
        # synth_dataset.samples 中每项为 (img_path, label, sample_id)
        synth_labels = [label for _, label, _ in synth_dataset.samples]
        synth_class_distribution = count_labels_from_dataset(synth_labels, class_names)

        # 原始训练集标签 + 合成数据标签
        augmented_train_labels = train_dataset.labels + synth_labels
        augmented_train_class_distribution = count_labels_from_dataset(
            augmented_train_labels, class_names
        )

        print_class_distribution(
            "Synthetic Dataset Class Distribution", synth_class_distribution
        )
        print_class_distribution(
            "Augmented Train Dataset Class Distribution",
            augmented_train_class_distribution,
        )

    pin_memory = device.type == "cuda"
    persistent_workers = args.workers > 0

    # =========================
    # DataLoader
    # =========================
    train_loader = DataLoader(
        final_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else 0,
    )

    # =========================
    # 整理当前启用的类别增强比例
    # 例如 {"MEL": 2.0, "BCC": 1.5}
    # =========================
    active_ratios_by_name = {}
    for class_idx, ratio in parse_ratios(args.ratios, num_classes).items():
        if ratio > 0:
            active_ratios_by_name[class_names[class_idx]] = ratio

    # =========================
    # 保存实验元数据
    # =========================

    formatted_class_distribution = {
        "train_dataset": format_count_ratio_dict(train_class_distribution),
        "val_dataset": format_count_ratio_dict(val_class_distribution),
        "synthetic_dataset": format_count_ratio_dict(synth_class_distribution),
        "augmented_train_dataset": format_count_ratio_dict(
            augmented_train_class_distribution
        ),
    }

    mode_specific_params = build_mode_specific_params(args)

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
            "train_gt_csv_path": args.train_gt_csv,
            "val_gt_csv_path": args.val_gt_csv,
            "train_img_dir": args.train_img_dir,
            "val_img_dir": args.val_img_dir,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "synthetic_dataset_size": (
                0 if synth_dataset is None else len(synth_dataset)
            ),
            "augmented_train_dataset_size": len(final_train_dataset),
            "split_ratio": "official train / official val",
            "num_classes": num_classes,
            "class_names": class_names,
            "class_distribution": formatted_class_distribution,
        },
        "diffusion_augmentation": {
            "enabled": bool(args.use_diffusion_augmentation),
            "mode": args.mode,
            "use_class_conditioning": bool(args.use_class_conditioning),
            "use_ddim_sampling": bool(args.use_ddim_sampling),
            "ddim_eta": args.ddim_eta,
            "resolution": args.resolution,
            "ddpm_num_steps": args.ddpm_num_steps,
            "ddpm_num_inference_steps": args.ddpm_num_inference_steps,
            "ddpm_beta_schedule": args.ddpm_beta_schedule,
            "diffusion_checkpoint": args.diffusion_checkpoint,
            "ratios_raw": args.ratios,
            "active_ratios_by_class_name": active_ratios_by_name,
            "aug_output_dir": aug_output_dir,
            "mode_specific_params": mode_specific_params,
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
        "early_stopping": {
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "monitor": "val_balanced_acc",
            "mode": "max",
            "counter": early_stop_counter,
            "stopped_early": early_stopped,
        },
        "best_result": {
            "best_epoch": best_epoch,
            "best_val_balanced_acc": float(best_acc1),
            "best_model_path": (
                best_model_path if os.path.exists(best_model_path) else ""
            ),
        },
    }
    save_json(experiment_metadata, metadata_json_path)

    # =========================
    # 只做评估，不训练
    # =========================
    if args.evaluate:
        eval_epoch = start_epoch if checkpoint is not None else 0

        val_metrics = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
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

    # =========================
    # 正常训练循环
    # =========================
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{args.epochs} {'=' * 25}")

        train_metrics = train_one_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            use_amp=(device.type == "cuda" and args.use_amp),
        )

        # 是否在这一轮做验证
        do_eval = ((epoch + 1) % args.eval_freq == 0) or ((epoch + 1) == args.epochs)
        val_metrics = None
        is_best = False

        if do_eval:
            val_metrics = validate(
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                class_names=class_names,
                epoch=epoch + 1,
                roc_dir=exp_folders["roc_dir"],
                cm_dir=exp_folders["cm_dir"],
                predictions_dir=exp_folders["predictions_dir"],
                metrics_dir=exp_folders["metrics_dir"],
            )

            current_score = val_metrics["overall"]["balanced_multiclass_accuracy"]
            is_best = current_score > (best_acc1 + args.early_stop_min_delta)

            if is_best:
                best_acc1 = current_score
                best_epoch = epoch + 1
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        # 更新学习率
        scheduler.step()

        # 每轮都保存 last checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "exp_dir": exp_folders["exp_dir"],
                "is_eval_epoch": do_eval,
                "early_stop_counter": early_stop_counter,
                "early_stopped": early_stopped,
            },
            is_best,
            save_dir=exp_folders["checkpoints_dir"],
            filename="last.pth.tar",
        )

        # 如果需要，每个评估轮额外单独存一份 checkpoint
        if do_eval and args.save_every_eval:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "exp_dir": exp_folders["exp_dir"],
                    "is_eval_epoch": True,
                    "early_stop_counter": early_stop_counter,
                    "early_stopped": early_stopped,
                },
                False,
                save_dir=exp_folders["checkpoints_dir"],
                filename=f"checkpoint_epoch_{epoch + 1:03d}.pth.tar",
            )

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

            # 更新 metadata 中记录的最佳结果
            experiment_metadata["best_result"]["best_epoch"] = best_epoch
            experiment_metadata["best_result"]["best_val_balanced_acc"] = float(
                best_acc1
            )
            experiment_metadata["best_result"]["best_model_path"] = (
                best_model_path if os.path.exists(best_model_path) else ""
            )
            experiment_metadata["last_epoch_finished"] = epoch + 1
            experiment_metadata["updated_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            save_json(experiment_metadata, metadata_json_path)

            print(
                f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_metrics['train_loss']:.4f} "
                f"| val_bal_acc={val_metrics['overall']['balanced_multiclass_accuracy']:.4f}"
            )

            experiment_metadata["early_stopping"]["counter"] = early_stop_counter
            experiment_metadata["early_stopping"]["stopped_early"] = early_stopped
            save_json(experiment_metadata, metadata_json_path)

            if (
                args.early_stop_patience > 0
                and early_stop_counter >= args.early_stop_patience
            ):
                early_stopped = True
                experiment_metadata["early_stopping"]["counter"] = early_stop_counter
                experiment_metadata["early_stopping"]["stopped_early"] = True
                experiment_metadata["last_epoch_finished"] = epoch + 1
                experiment_metadata["updated_time"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                save_json(experiment_metadata, metadata_json_path)

                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"No improvement in val_bal_acc for {early_stop_counter} eval rounds."
                )
                break
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_metrics['train_loss']:.4f}"
            )


def setup_seed_and_device(args):
    """
    设置随机种子与运行设备。

    随机种子会影响：
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA

    这样做有利于复现实验结果，但可能让训练速度稍慢。
    """
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        warnings.warn(
            "You have chosen to seed training. This may slow down training a bit, but improves reproducibility."
        )

    # 优先使用 GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    else:
        device = torch.device("cpu")

    return device


def build_mode_specific_params(args):
    """
    根据当前扩散增强模式，整理只属于该 mode 的关键参数。
    这样 metadata 不会混在一起，更利于后续复现实验。
    """
    if args.mode == "ddpm":
        return {}

    if args.mode == "cfg":
        return {
            "cfg_scale": args.cfg_scale,
            "cond_drop_prob": args.cond_drop_prob,
        }

    if args.mode == "cg":
        return {
            "classifier_ckpt_path": args.classifier_ckpt_path,
            "classifier_guidance_scale": args.classifier_guidance_scale,
            "classifier_num_heads": args.classifier_num_heads,
            "classifier_use_rotary": bool(args.classifier_use_rotary),
            "classifier_feat_size": args.classifier_feat_size,
        }

    return {}
