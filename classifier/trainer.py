import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .augmentation import build_augmented_train_dataset
from .dataset import ISICResNetDataset
from .evaluator import evaluate
from .experiment import (
    build_mode_specific_params,
    make_experiment_name,
    resolve_aug_output_dir,
    reuse_experiment_folders,
    save_checkpoint,
    setup_experiment_folders,
)
from .metrics import update_epoch_metrics_csv, update_epoch_metrics_json
from .utils import (
    AverageMeter,
    Summary,
    accuracy,
    count_labels_from_dataset,
    format_count_ratio_dict,
    parse_ratios,
    print_class_distribution,
    save_json,
    setup_seed_and_device,
)

# 训练过程中根据验证集 balanced accuracy 更新。
best_balanced_acc = 0.0


def build_transforms(args):
    """
    构建 train / eval transform。

    训练集保留随机裁剪和翻转；验证/测试集使用确定性预处理，避免评估结果抖动。
    """
    input_size = 224
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_train_val_datasets(args, train_transform, eval_transform):
    """构建原始 train/val dataset。扩散增强不在这里做。"""
    train_dataset = ISICResNetDataset(
        gt_csv_path=args.train_gt_csv,
        img_dir=args.train_img_dir,
        transform=train_transform,
    )
    val_dataset = ISICResNetDataset(
        gt_csv_path=args.val_gt_csv,
        img_dir=args.val_img_dir,
        transform=eval_transform,
    )

    class_names = list(train_dataset.class_columns)
    if class_names != list(val_dataset.class_columns):
        raise ValueError("train 和 val 的类别列不一致，请检查 ground truth CSV。")

    return train_dataset, val_dataset, class_names, len(class_names)


def build_test_dataset(args, eval_transform):
    """构建 test dataset。test-only 要求测试集有 ground truth。"""
    test_dataset = ISICResNetDataset(
        gt_csv_path=args.test_gt_csv,
        img_dir=args.test_img_dir,
        transform=eval_transform,
    )
    class_names = list(test_dataset.class_columns)
    return test_dataset, class_names, len(class_names)


def build_dataloaders(
    args, train_dataset=None, val_dataset=None, test_dataset=None, device=None
):
    """统一构建 DataLoader。传入哪个 dataset 就返回哪个 loader。"""
    pin_memory = device is not None and device.type == "cuda"
    persistent_workers = args.workers > 0
    common_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if persistent_workers:
        common_kwargs["prefetch_factor"] = 2

    loaders = {}
    if train_dataset is not None:
        loaders["train"] = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    if val_dataset is not None:
        loaders["val"] = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    if test_dataset is not None:
        loaders["test"] = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return loaders


def get_labels_from_dataset(dataset):
    """从普通 Dataset 或 ConcatDataset 中递归提取 labels，用于 class weights。"""
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if hasattr(dataset, "datasets"):
        labels = []
        for sub_dataset in dataset.datasets:
            labels.extend(get_labels_from_dataset(sub_dataset))
        return labels
    raise AttributeError("当前 dataset 无法提取 labels，不能计算 class weights。")


def build_classifier(args, num_classes, device, use_pretrained=True):
    """构建 torchvision ResNet 分类器，并替换最后的 fc 分类头。"""
    weights = args.weights if use_pretrained else None
    if weights is not None:
        weights_enum = models.get_model_weights(args.arch)
        model = models.__dict__[args.arch](weights=weights_enum[weights])
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


def build_criterion(args, train_dataset=None, num_classes=None, device=None):
    """
    构建 CrossEntropyLoss。

    如果启用 class weights，必须基于最终训练集 final_train_dataset 计算。
    """
    class_weights = None
    if args.use_class_weights:
        if train_dataset is None or num_classes is None:
            raise ValueError(
                "启用 class weights 时必须提供 train_dataset 和 num_classes。"
            )
        train_labels = get_labels_from_dataset(train_dataset)
        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.mean()
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"Using class weights from final train dataset: {class_weights.tolist()}")

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)


def build_optimizer(args, model):
    """保持原有 SGD 优化器逻辑。"""
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    """保持原有 CosineAnnealingLR 学习率调度。"""
    return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def train_one_epoch(
    train_loader, model, criterion, optimizer, epoch, device, scaler, use_amp
):
    """训练一个 epoch。"""
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.NONE)

    # 切换到训练模式，启用 BatchNorm/Dropout 的训练行为。
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

        # 每个 batch 先清空上一次反向传播留下的梯度。
        optimizer.zero_grad()

        # AMP 只在 CUDA 且 args.use_amp=True 时启用，用于减少显存占用并加速训练。
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # GradScaler 负责缩放 loss，降低混合精度训练中的梯度下溢风险。
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


def _metric_row(prefix, epoch, train_metrics, eval_metrics, optimizer):
    """把 evaluate() 返回的嵌套指标压平成一行，便于写入 CSV/JSON 日志。"""
    overall = eval_metrics["overall"]

    row = {
        "epoch": epoch,
    }

    if train_metrics is not None:
        row.update(
            {
                "train_loss": float(train_metrics["train_loss"]),
                "train_acc": float(train_metrics["train_acc"]),
            }
        )

    row.update(
        {
            "lr": (
                float(optimizer.param_groups[0]["lr"])
                if optimizer is not None
                else None
            ),
            f"{prefix}_loss": float(eval_metrics["loss"]),
            f"{prefix}_acc": float(overall["accuracy"]) / 100.0,
            f"{prefix}_balanced_acc": float(overall["balanced_multiclass_accuracy"]),
            f"{prefix}_macro_recall": float(overall["macro_recall"]),
            f"{prefix}_macro_f1": float(overall["macro_f1"]),
            f"{prefix}_macro_precision": float(overall["macro_precision"]),
            f"{prefix}_mcc": float(overall["mcc"]),
            f"{prefix}_cohen_kappa": float(overall["cohen_kappa"]),
            f"{prefix}_mean_youden_index": float(overall["mean_youden_index"]),
            f"{prefix}_auc_macro_ovr": float(overall["multiclass_macro_auc_ovr"]),
            f"{prefix}_mean_auc_all_diagnoses": float(
                overall["mean_auc_all_diagnoses"]
            ),
            f"{prefix}_mean_ap_all_diagnoses": float(
                overall["mean_average_precision_all_diagnoses"]
            ),
            f"{prefix}_mean_sensitivity": float(overall["mean_sensitivity"]),
            f"{prefix}_mean_specificity": float(overall["mean_specificity"]),
            f"{prefix}_mean_ppv": float(overall["mean_ppv"]),
            f"{prefix}_mean_npv": float(overall["mean_npv"]),
            f"{prefix}_melanoma_auc80": float(overall["melanoma_auc80"]),
            f"{prefix}_malignant_vs_benign_auc": float(
                overall["malignant_vs_benign_auc"]
            ),
            "roc_curve_path": eval_metrics["roc_curve_path"],
            "confusion_matrix_path": eval_metrics["confusion_matrix_png_path"],
            f"{prefix}_predictions_path": eval_metrics["predictions_csv_path"],
            "detailed_metrics_path": eval_metrics["detailed_metrics_json_path"],
        }
    )

    return row


def _build_checkpoint_state(
    args,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    exp_dir,
    class_names,
    num_classes,
    best_epoch,
    aug_output_dir,
    synth_dataset,
    early_stop_counter,
    early_stopped,
    is_eval_epoch,
):
    """集中整理 checkpoint 字段，避免保存 last/best/eval 时复制多份字典。"""
    return {
        "epoch": epoch,
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "best_balanced_acc": best_balanced_acc,
        "best_epoch": best_epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "exp_dir": exp_dir,
        "args_dict": vars(args),
        "class_names": class_names,
        "num_classes": num_classes,
        "primary_metric": "balanced_multiclass_accuracy",
        "augmentation": {
            "enabled": bool(synth_dataset is not None),
            "aug_output_dir": aug_output_dir,
            "mode": args.mode,
        },
        "is_eval_epoch": is_eval_epoch,
        "early_stop_counter": early_stop_counter,
        "early_stopped": early_stopped,
    }


def _build_training_metadata(
    args,
    exp_name,
    exp_folders,
    device,
    train_dataset,
    val_dataset,
    final_train_dataset,
    synth_dataset,
    class_names,
    num_classes,
    train_class_distribution,
    val_class_distribution,
    synth_class_distribution,
    augmented_train_class_distribution,
    aug_output_dir,
    start_epoch,
    best_epoch,
):
    """整理实验配置、数据分布、增强配置和早停配置，保存到 metadata JSON。"""
    active_ratios_by_name = {}
    for class_idx, ratio in parse_ratios(args.ratios, num_classes).items():
        if ratio > 0:
            active_ratios_by_name[class_names[class_idx]] = ratio

    return {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "arch": args.arch,
            "weights": args.weights,
            "num_classes": num_classes,
            "class_names": class_names,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "use_amp": bool(args.use_amp),
            "use_class_weights": bool(args.use_class_weights),
            "eval_freq": args.eval_freq,
            "seed": args.seed,
            "device": str(device),
        },
        "data_config": {
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
            "class_distribution": {
                "train_dataset": format_count_ratio_dict(train_class_distribution),
                "val_dataset": format_count_ratio_dict(val_class_distribution),
                "synthetic_dataset": format_count_ratio_dict(synth_class_distribution),
                "augmented_train_dataset": format_count_ratio_dict(
                    augmented_train_class_distribution
                ),
            },
        },
        "augmentation_config": {
            "enabled": bool(synth_dataset is not None),
            "requested": bool(args.use_diffusion_augmentation),
            "mode": args.mode,
            "use_class_conditioning": bool(args.use_class_conditioning),
            "use_cross_attention_conditioning": bool(
                args.use_cross_attention_conditioning
            ),
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
            "mode_specific_params": build_mode_specific_params(args),
        },
        "resume_config": {
            "resume_path": args.resume,
            "start_epoch": start_epoch,
        },
        "early_stopping_config": {
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "monitor": "val_balanced_acc",
            "mode": "max",
        },
        "best_result": {
            "best_epoch": best_epoch,
            "best_val_balanced_acc": float(best_balanced_acc),
        },
        "runtime_info": {},
    }


def _evaluate_if_needed(
    args,
    epoch,
    val_loader,
    model,
    criterion,
    device,
    class_names,
    exp_folders,
    best_balanced_acc,
    best_epoch,
    early_stop_counter,
):
    """按 eval_freq 执行验证，并更新 best 指标和 early-stop 计数。"""
    do_eval = ((epoch + 1) % args.eval_freq == 0) or ((epoch + 1) == args.epochs)
    val_metrics = None
    is_best = False

    if not do_eval:
        return (
            do_eval,
            val_metrics,
            is_best,
            best_balanced_acc,
            best_epoch,
            early_stop_counter,
        )

    val_metrics = evaluate(
        loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        class_names=class_names,
        epoch=epoch + 1,
        output_dirs=exp_folders,
        split_name="val",
    )

    current_score = val_metrics["overall"]["balanced_multiclass_accuracy"]
    is_best = current_score > (best_balanced_acc + args.early_stop_min_delta)

    if is_best:
        best_balanced_acc = current_score
        best_epoch = epoch + 1
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    return (
        do_eval,
        val_metrics,
        is_best,
        best_balanced_acc,
        best_epoch,
        early_stop_counter,
    )


def _record_eval_and_check_early_stop(
    args,
    epoch,
    train_metrics,
    val_metrics,
    optimizer,
    metrics_csv_path,
    metrics_json_path,
    experiment_metadata,
    metadata_json_path,
    best_epoch,
    best_balanced_acc,
    best_model_path,
    early_stop_counter,
    early_stopped,
):
    """保存验证轮次的指标和 metadata，并判断是否触发 early stopping。"""
    epoch_row = _metric_row("val", epoch + 1, train_metrics, val_metrics, optimizer)
    update_epoch_metrics_csv(metrics_csv_path, epoch_row)
    update_epoch_metrics_json(metrics_json_path, epoch_row)

    experiment_metadata["best_result"] = {
        "best_epoch": best_epoch,
        "best_val_balanced_acc": float(best_balanced_acc),
        "best_model_path": (best_model_path if os.path.exists(best_model_path) else ""),
    }
    experiment_metadata["runtime_info"].update(
        {
            "last_epoch_finished": epoch + 1,
            "updated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "early_stop_counter": early_stop_counter,
            "early_stopped": early_stopped,
        }
    )
    save_json(experiment_metadata, metadata_json_path)

    print(
        f"Epoch {epoch + 1}/{args.epochs} | "
        f"train_loss={train_metrics['train_loss']:.4f} | "
        f"val_bal_acc={val_metrics['overall']['balanced_multiclass_accuracy']:.4f}"
    )

    if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
        early_stopped = True
        experiment_metadata["runtime_info"].update(
            {
                "last_epoch_finished": epoch + 1,
                "updated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "early_stop_counter": early_stop_counter,
                "early_stopped": True,
            }
        )
        save_json(experiment_metadata, metadata_json_path)
        print(
            f"Early stopping triggered at epoch {epoch + 1}. "
            f"No improvement in val_bal_acc for {early_stop_counter} eval rounds."
        )

    return early_stopped


def run_training(args):
    """
    训练模式入口。

    main.py 只把 args 传进来；dataset、dataloader、model、optimizer、scheduler 都在这里构建。
    """
    global best_balanced_acc

    device = setup_seed_and_device(args)
    train_transform, eval_transform = build_transforms(args)
    train_dataset, val_dataset, class_names, num_classes = build_train_val_datasets(
        args, train_transform, eval_transform
    )

    start_epoch = 0
    best_epoch = -1
    early_stop_counter = 0
    early_stopped = False
    checkpoint = None
    best_balanced_acc = 0.0

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

    aug_output_dir = resolve_aug_output_dir(
        args=args, checkpoint=checkpoint, exp_folders=exp_folders
    )
    final_train_dataset, synth_dataset, aug_output_dir = build_augmented_train_dataset(
        args=args,
        train_dataset=train_dataset,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
        output_dir=aug_output_dir,
    )

    synth_class_distribution = None
    augmented_train_class_distribution = train_class_distribution
    # 只有实际产生合成数据时，才统计增强后的类别分布。
    if synth_dataset is not None:
        synth_labels = [label for _, label, _ in synth_dataset.samples]
        synth_class_distribution = count_labels_from_dataset(synth_labels, class_names)
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

    loaders = build_dataloaders(
        args, train_dataset=final_train_dataset, val_dataset=val_dataset, device=device
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = build_classifier(
        args=args, num_classes=num_classes, device=device, use_pretrained=True
    )
    criterion = build_criterion(
        args=args,
        train_dataset=final_train_dataset,
        num_classes=num_classes,
        device=device,
    )
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and args.use_amp)
    )

    # 恢复模型、优化器、调度器和 AMP scaler，保证训练能从中断处继续。
    if checkpoint is not None:
        start_epoch = checkpoint["epoch"]
        best_balanced_acc = checkpoint.get(
            "best_balanced_acc", checkpoint.get("best_acc1", 0.0)
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

    experiment_metadata = _build_training_metadata(
        args=args,
        exp_name=exp_name,
        exp_folders=exp_folders,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        final_train_dataset=final_train_dataset,
        synth_dataset=synth_dataset,
        class_names=class_names,
        num_classes=num_classes,
        train_class_distribution=train_class_distribution,
        val_class_distribution=val_class_distribution,
        synth_class_distribution=synth_class_distribution,
        augmented_train_class_distribution=augmented_train_class_distribution,
        aug_output_dir=aug_output_dir,
        start_epoch=start_epoch,
        best_epoch=best_epoch,
    )
    save_json(experiment_metadata, metadata_json_path)

    # 主训练循环：每个 epoch 训练一次，只在 eval_freq 指定的轮次做验证。
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

        # 按 eval_freq 验证，并更新 best / early-stop 计数。
        (
            do_eval,
            val_metrics,
            is_best,
            best_balanced_acc,
            best_epoch,
            early_stop_counter,
        ) = _evaluate_if_needed(
            args=args,
            epoch=epoch,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            class_names=class_names,
            exp_folders=exp_folders,
            best_balanced_acc=best_balanced_acc,
            best_epoch=best_epoch,
            early_stop_counter=early_stop_counter,
        )

        # 每个 epoch 结束后更新学习率。
        scheduler.step()

        # 每轮都保存 last checkpoint，用于中断后继续训练。
        state = _build_checkpoint_state(
            args=args,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            exp_dir=exp_folders["exp_dir"],
            class_names=class_names,
            num_classes=num_classes,
            best_epoch=best_epoch,
            aug_output_dir=aug_output_dir,
            synth_dataset=synth_dataset,
            early_stop_counter=early_stop_counter,
            early_stopped=early_stopped,
            is_eval_epoch=do_eval,
        )
        save_checkpoint(
            state,
            is_best,
            save_dir=exp_folders["checkpoints_dir"],
            filename="last.pth.tar",
        )

        # 按 save_freq 额外保存周期性 checkpoint。
        do_save = (epoch + 1) % args.save_freq == 0
        if do_save:
            save_checkpoint(
                state,
                False,
                save_dir=exp_folders["checkpoints_dir"],
                filename=f"checkpoint_epoch_{epoch + 1:03d}.pth.tar",
            )

        # 验证轮次写 val 指标、更新 metadata、检查 early stopping。
        if do_eval:
            early_stopped = _record_eval_and_check_early_stop(
                args=args,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                optimizer=optimizer,
                metrics_csv_path=metrics_csv_path,
                metrics_json_path=metrics_json_path,
                experiment_metadata=experiment_metadata,
                metadata_json_path=metadata_json_path,
                best_epoch=best_epoch,
                best_balanced_acc=best_balanced_acc,
                best_model_path=best_model_path,
                early_stop_counter=early_stop_counter,
                early_stopped=early_stopped,
            )

            if early_stopped:
                break
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f}"
            )


def _load_classifier_checkpoint(args, checkpoint_path, device, fallback_num_classes):
    """test-only 使用：读取 checkpoint，构建模型并加载权重。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        args.arch = checkpoint.get("arch", args.arch)
        class_names = checkpoint.get("class_names")
        num_classes = checkpoint.get("num_classes", fallback_num_classes)
        state_dict = checkpoint.get(
            "state_dict", checkpoint.get("model_state_dict", checkpoint)
        )
    else:
        class_names = None
        num_classes = fallback_num_classes
        state_dict = checkpoint

    model = build_classifier(
        args=args, num_classes=num_classes, device=device, use_pretrained=False
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint, class_names, num_classes


def run_test(args):
    """
    仅测试模式入口。

    加载 checkpoint 后，只在 test set 上评估。
    """
    if args.test_checkpoint is None:
        raise ValueError("启用 test-only 时，必须提供 --test-checkpoint。")

    device = setup_seed_and_device(args)
    _, eval_transform = build_transforms(args)
    test_dataset, test_class_names, fallback_num_classes = build_test_dataset(
        args, eval_transform
    )

    model, checkpoint, ckpt_class_names, num_classes = _load_classifier_checkpoint(
        args=args,
        checkpoint_path=args.test_checkpoint,
        device=device,
        fallback_num_classes=fallback_num_classes,
    )
    class_names = ckpt_class_names if ckpt_class_names is not None else test_class_names
    if class_names != test_class_names:
        print(
            "Warning: checkpoint 中的 class_names 与 test CSV 类别列不完全一致，将按 checkpoint 顺序评估。"
        )

    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    exp_name = f"{timestamp}_{args.arch}_test-only"
    exp_folders = setup_experiment_folders(base_dir="experiments", exp_name=exp_name)
    metrics_csv_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.csv")
    metrics_json_path = os.path.join(exp_folders["metrics_dir"], "epoch_metrics.json")
    metadata_json_path = os.path.join(
        exp_folders["metadata_dir"], "experiment_metadata.json"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    test_class_distribution = count_labels_from_dataset(
        test_dataset.labels, test_class_names
    )
    print(f"Test dataset size: {len(test_dataset)}")
    print_class_distribution("Test Dataset Class Distribution", test_class_distribution)

    test_loader = build_dataloaders(args, test_dataset=test_dataset, device=device)[
        "test"
    ]

    experiment_metadata = {
        "experiment_name": exp_name,
        "experiment_dir": exp_folders["exp_dir"],
        "mode": "test_only",
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "arch": args.arch,
            "checkpoint_path": args.test_checkpoint,
            "num_classes": num_classes,
            "class_names": class_names,
        },
        "data_config": {
            "test_gt_csv_path": args.test_gt_csv,
            "test_img_dir": args.test_img_dir,
            "test_dataset_size": len(test_dataset),
            "class_distribution": format_count_ratio_dict(test_class_distribution),
        },
        "runtime_info": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
        },
    }
    save_json(experiment_metadata, metadata_json_path)

    test_metrics = evaluate(
        loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        class_names=class_names,
        epoch=1,
        output_dirs=exp_folders,
        split_name="test",
    )

    test_row = _metric_row("test", 1, None, test_metrics, optimizer=None)
    update_epoch_metrics_csv(metrics_csv_path, test_row)
    update_epoch_metrics_json(metrics_json_path, test_row)

    experiment_metadata["best_result"] = {
        "test_balanced_acc": float(
            test_metrics["overall"]["balanced_multiclass_accuracy"]
        ),
        "test_acc": float(test_metrics["overall"]["accuracy"]),
    }
    experiment_metadata["runtime_info"]["updated_time"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    save_json(experiment_metadata, metadata_json_path)

    print("\n========================= Test Result =========================")
    print(f"test_loss                 : {test_metrics['loss']:.6f}")
    print(f"test_acc                  : {test_metrics['overall']['accuracy']:.4f}")
    print(
        f"test_balanced_acc         : {test_metrics['overall']['balanced_multiclass_accuracy']:.4f}"
    )
    print(f"test_macro_recall         : {test_metrics['overall']['macro_recall']:.4f}")
    print(
        f"test_macro_precision      : {test_metrics['overall']['macro_precision']:.4f}"
    )
    print(f"test_macro_f1             : {test_metrics['overall']['macro_f1']:.4f}")
    print(f"test_mcc                  : {test_metrics['overall']['mcc']:.4f}")
    print(f"test_cohen_kappa          : {test_metrics['overall']['cohen_kappa']:.4f}")
    print(
        f"test_mean_youden_index    : {test_metrics['overall']['mean_youden_index']:.4f}"
    )
    print(
        f"test_auc_macro_ovr        : {test_metrics['overall']['multiclass_macro_auc_ovr']:.4f}"
    )
    print(
        f"test_mean_auc_all_diag    : {test_metrics['overall']['mean_auc_all_diagnoses']:.4f}"
    )
    print(
        f"test_mean_ap_all_diag     : {test_metrics['overall']['mean_average_precision_all_diagnoses']:.4f}"
    )
    print(
        f"test_melanoma_auc80       : {test_metrics['overall']['melanoma_auc80']:.4f}"
    )
    print(
        f"test_malignant_vs_benign  : {test_metrics['overall']['malignant_vs_benign_auc']:.4f}"
    )
    print("==============================================================")
