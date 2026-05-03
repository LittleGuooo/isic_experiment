import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .augmentation import build_train_dataset
from .utils import (
    AverageMeter,
    Summary,
    accuracy,
    count_labels_from_dataset,
    format_count_ratio_dict,
    parse_ratios,
    print_class_distribution,
    save_json,
)
from .metrics import (
    compute_detailed_classification_metrics,
    save_confusion_matrix_artifacts,
    save_detailed_metrics_json,
    save_multiclass_roc_artifacts,
    save_val_predictions_csv,
    update_epoch_metrics_csv,
    update_epoch_metrics_json,
)


# 全局变量：记录历史最佳 balanced accuracy
best_balanced_acc = 0.0


def make_experiment_name(args):
    """
    构造简洁实验名：
    日期时间 + 分类器骨架 + 真实增强状态。
    """
    timestamp = datetime.now().strftime("%y%m%d-%H%M")

    mode_name_map = {
        "ddpm": "ddpm",
        "cfg": "cfg",
        "cg": "cg",
        "latent_ddpm": "ldm",
    }

    if args.use_diffusion_augmentation:
        mode_tag = mode_name_map.get(args.mode, args.mode)
        aug_tag = f"diff-{mode_tag}"
    else:
        aug_tag = "noaug"

    return f"{timestamp}_{args.arch}_{aug_tag}"


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


def load_experiment_metadata(exp_dir):
    """从旧实验目录读取 experiment_metadata.json，用于 resume 时恢复增强数据路径。"""
    metadata_path = os.path.join(exp_dir, "metadata", "experiment_metadata.json")

    if not os.path.isfile(metadata_path):
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_aug_output_dir(args, checkpoint, exp_folders):
    """
    决定本次训练使用哪个合成数据目录。

    优先级：
    1. 用户显式传入 --aug-output-dir
    2. checkpoint 中保存的 augmentation.aug_output_dir
    3. 旧实验 metadata 中保存的 diffusion_augmentation.aug_output_dir
    4. 当前实验目录下 train_augmented_data
    """
    if args.aug_output_dir is not None:
        return args.aug_output_dir

    if checkpoint is not None:
        ckpt_aug = checkpoint.get("augmentation", {})
        ckpt_aug_dir = ckpt_aug.get("aug_output_dir")

        if ckpt_aug_dir:
            args.use_diffusion_augmentation = bool(
                ckpt_aug.get("enabled", args.use_diffusion_augmentation)
            )
            return ckpt_aug_dir

        metadata = load_experiment_metadata(checkpoint["exp_dir"])
        if metadata is not None:
            meta_aug = metadata.get("diffusion_augmentation", {})
            meta_aug_dir = meta_aug.get("aug_output_dir")

            if meta_aug_dir:
                args.use_diffusion_augmentation = bool(
                    meta_aug.get("enabled", args.use_diffusion_augmentation)
                )
                return meta_aug_dir

    return os.path.join(exp_folders["exp_dir"], "train_augmented_data")


def get_labels_from_dataset(dataset):
    """
    从 Dataset 或 ConcatDataset 中取出所有 label。

    普通 ISICResNetDataset / SavedSyntheticISICDataset 有 labels 属性。
    ConcatDataset 没有 labels 属性，需要递归取子数据集。
    """
    if hasattr(dataset, "labels"):
        return list(dataset.labels)

    if hasattr(dataset, "datasets"):
        labels = []
        for sub_dataset in dataset.datasets:
            labels.extend(get_labels_from_dataset(sub_dataset))
        return labels

    raise AttributeError("当前 dataset 无法提取 labels，不能计算 class weights。")


def build_classifier(args, num_classes, device):
    """构建 torchvision ResNet 分类器，并替换最后的分类头。"""
    if args.weights is not None:
        weights_enum = models.get_model_weights(args.arch)
        model = models.__dict__[args.arch](weights=weights_enum[args.weights])
    else:
        model = models.__dict__[args.arch](weights=None)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"当前代码只处理带有 model.fc 的模型，当前模型 {args.arch} 不满足该条件。"
        )

    return model.to(device)


def build_criterion(args, train_dataset_for_weights, num_classes, device):
    """
    构建 CrossEntropyLoss。

    关键点：
    class weights 必须基于最终训练集 final_train_dataset 计算，
    而不是只基于原始 train_dataset 计算。
    """
    class_weights = None

    if args.use_class_weights:
        train_labels = get_labels_from_dataset(train_dataset_for_weights)

        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.mean()

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )

        print(f"Using class weights from final train dataset: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)

    return criterion


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

        # 清空梯度
        optimizer.zero_grad()

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
    global best_balanced_acc

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
    model = build_classifier(
        args=args,
        num_classes=num_classes,
        device=device,
    )

    # =========================
    # 优化器、学习率调度器、AMP scaler
    # =========================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.use_amp),
    )

    # 指定了 resume 就加载 checkpoint 继续训练
    if checkpoint is not None:
        start_epoch = checkpoint["epoch"]
        best_balanced_acc = checkpoint.get(
            "best_balanced_acc",
            checkpoint.get("best_acc1", 0.0),
        )
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

    # 打印原始数据集类别分布
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
    # evaluate 模式
    # =========================
    if args.evaluate:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
        ).to(device)

        pin_memory = device.type == "cuda"
        persistent_workers = args.workers > 0

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if persistent_workers else None,
        )

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
    # 增强数据集，并构建最终训练集
    # =========================
    aug_output_dir = resolve_aug_output_dir(
        args=args,
        checkpoint=checkpoint,
        exp_folders=exp_folders,
    )
    final_train_dataset, synth_dataset, aug_output_dir = build_train_dataset(
        args=args,
        train_dataset=train_dataset,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
        output_dir=aug_output_dir,
    )
    # 注意：criterion 必须在 final_train_dataset 构建完成后再创建。
    # 因为如果启用了扩散增强，类别分布已经改变。
    criterion = build_criterion(
        args=args,
        train_dataset_for_weights=final_train_dataset,
        num_classes=num_classes,
        device=device,
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
        prefetch_factor=2 if persistent_workers else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
    )

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

    active_ratios_by_name = {}
    for class_idx, ratio in parse_ratios(args.ratios, num_classes).items():
        if ratio > 0:
            active_ratios_by_name[class_names[class_idx]] = ratio

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
            "best_val_balanced_acc": float(best_balanced_acc),
            "best_model_path": (
                best_model_path if os.path.exists(best_model_path) else ""
            ),
        },
    }
    save_json(experiment_metadata, metadata_json_path)

    # =========================
    # 训练模式循环
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

        # ==========================
        # 做验证的epoch
        # ==========================
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
            is_best = current_score > (best_balanced_acc + args.early_stop_min_delta)

            if is_best:
                best_balanced_acc = current_score
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
                "best_balanced_acc": best_balanced_acc,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "exp_dir": exp_folders["exp_dir"],
                "args_dict": vars(args),
                "class_names": class_names,
                "num_classes": num_classes,
                "primary_metric": "balanced_multiclass_accuracy",
                "augmentation": {
                    "enabled": bool(synth_dataset is not None),
                    "aug_output_dir": aug_output_dir,
                    "mode": args.mode,
                },
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
                    "best_balanced_acc": best_balanced_acc,
                    "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "exp_dir": exp_folders["exp_dir"],
                    "args_dict": vars(args),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "primary_metric": "balanced_multiclass_accuracy",
                    "augmentation": {
                        "enabled": bool(synth_dataset is not None),
                        "aug_output_dir": aug_output_dir,
                        "mode": args.mode,
                    },
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
                best_balanced_acc
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


def run_test(args, test_dataset, class_names, num_classes, device):
    """
    测试模式（test-only）：
    1. 加载训练好的分类器 checkpoint
    2. 对带 ground truth 的测试集做完整评估
    3. 保存 ROC、混淆矩阵、逐样本预测、detailed metrics 等结果
    """
    if args.test_checkpoint is None:
        raise ValueError("启用 test-only 时，必须提供 --test-checkpoint。")

    # =========================
    # 创建测试输出目录
    # =========================
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    exp_name = f"{timestamp}_{args.arch}_test-only"
    exp_folders = setup_experiment_folders(base_dir="experiments", exp_name=exp_name)

    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(
        exp_folders["metadata_dir"], "experiment_metadata.json"
    )

    # =========================
    # 构建分类模型
    # =========================
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
    # 加载 checkpoint
    # =========================
    checkpoint = torch.load(args.test_checkpoint, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # =========================
    # 损失函数
    # test-only 不需要 class weights，保持最小实现
    # =========================
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    ).to(device)

    # =========================
    # 打印测试集类别分布
    # =========================
    test_class_distribution = count_labels_from_dataset(
        test_dataset.labels, class_names
    )
    print(f"Test dataset size: {len(test_dataset)}")
    print_class_distribution("Test Dataset Class Distribution", test_class_distribution)

    # =========================
    # DataLoader
    # =========================
    pin_memory = device.type == "cuda"
    persistent_workers = args.workers > 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
    )

    # =========================
    # 保存测试元数据
    # =========================
    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "test_only",
        "model_name": args.arch,
        "weights": args.weights,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_path": args.test_checkpoint,
        "data": {
            "test_gt_csv_path": args.test_gt_csv,
            "test_img_dir": args.test_img_dir,
            "test_dataset_size": len(test_dataset),
            "num_classes": num_classes,
            "class_names": class_names,
            "class_distribution": format_count_ratio_dict(test_class_distribution),
        },
        "paths": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "metadata_json": metadata_json_path,
            "roc_dir": exp_folders["roc_dir"],
            "confusion_matrix_dir": exp_folders["cm_dir"],
            "predictions_dir": exp_folders["predictions_dir"],
        },
    }
    save_json(experiment_metadata, metadata_json_path)

    # =========================
    # 直接复用 validate 做一次完整测试
    # =========================
    test_metrics = validate(
        val_loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        epoch=1,
        roc_dir=exp_folders["roc_dir"],
        cm_dir=exp_folders["cm_dir"],
        predictions_dir=exp_folders["predictions_dir"],
        metrics_dir=exp_folders["metrics_dir"],
    )

    test_row = {
        "epoch": 1,
        "test_loss": float(test_metrics["val_loss"]),
        "test_acc": float(test_metrics["overall"]["accuracy"]),
        "test_balanced_acc": float(
            test_metrics["overall"]["balanced_multiclass_accuracy"]
        ),
        "test_macro_recall": float(test_metrics["overall"]["macro_recall"]),
        "test_macro_f1": float(test_metrics["overall"]["macro_f1"]),
        "test_macro_precision": float(test_metrics["overall"]["macro_precision"]),
        "test_auc_macro_ovr": float(
            test_metrics["overall"]["multiclass_macro_auc_ovr"]
        ),
        "test_mean_auc_all_diagnoses": float(
            test_metrics["overall"]["mean_auc_all_diagnoses"]
        ),
        "test_mean_ap_all_diagnoses": float(
            test_metrics["overall"]["mean_average_precision_all_diagnoses"]
        ),
        "test_mean_sensitivity": float(test_metrics["overall"]["mean_sensitivity"]),
        "test_mean_specificity": float(test_metrics["overall"]["mean_specificity"]),
        "test_mean_ppv": float(test_metrics["overall"]["mean_ppv"]),
        "test_mean_npv": float(test_metrics["overall"]["mean_npv"]),
        "test_melanoma_auc80": float(test_metrics["overall"]["melanoma_auc80"]),
        "test_malignant_vs_benign_auc": float(
            test_metrics["overall"]["malignant_vs_benign_auc"]
        ),
        "roc_curve_path": test_metrics["roc_curve_path"],
        "confusion_matrix_path": test_metrics["confusion_matrix_png_path"],
        "test_predictions_path": test_metrics["val_predictions_csv_path"],
        "detailed_metrics_path": test_metrics["detailed_metrics_json_path"],
    }

    update_epoch_metrics_csv(metrics_csv_path, test_row)
    update_epoch_metrics_json(metrics_json_path, test_row)

    print("\n========================= Test Result =========================")
    print(f"test_loss                 : {test_metrics['val_loss']:.6f}")
    print(f"test_acc                  : {test_metrics['overall']['accuracy']:.4f}")
    print(
        f"test_balanced_acc         : "
        f"{test_metrics['overall']['balanced_multiclass_accuracy']:.4f}"
    )
    print(f"test_macro_recall         : {test_metrics['overall']['macro_recall']:.4f}")
    print(
        f"test_macro_precision      : {test_metrics['overall']['macro_precision']:.4f}"
    )
    print(f"test_macro_f1             : {test_metrics['overall']['macro_f1']:.4f}")
    print(
        f"test_auc_macro_ovr        : "
        f"{test_metrics['overall']['multiclass_macro_auc_ovr']:.4f}"
    )
    print(
        f"test_mean_auc_all_diag    : "
        f"{test_metrics['overall']['mean_auc_all_diagnoses']:.4f}"
    )
    print(
        f"test_mean_ap_all_diag     : "
        f"{test_metrics['overall']['mean_average_precision_all_diagnoses']:.4f}"
    )
    print(
        f"test_melanoma_auc80       : "
        f"{test_metrics['overall']['melanoma_auc80']:.4f}"
    )
    print(
        f"test_malignant_vs_benign  : "
        f"{test_metrics['overall']['malignant_vs_benign_auc']:.4f}"
    )
    print("==============================================================")


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
