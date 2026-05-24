import os
import torch


def load_training_checkpoint(
    args,
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    ema_model,
    modes,
    extra_components,
):
    """
    从 checkpoint 恢复训练状态。

    恢复内容包括：
        - model 权重
        - optimizer 状态
        - lr_scheduler 状态
        - EMA 权重，如果启用
        - mode-specific extra state，例如 CFG/CG/LDM 的额外状态
        - epoch / global_step / best_metric

    """
    start_epoch = 0
    global_step = 0
    best_metric = None

    ckpt_path = getattr(args, "resume_from_checkpoint", None)

    if ckpt_path is None:
        return start_epoch, global_step, best_metric

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if "extra_state" in checkpoint:
        modes["load_checkpoint_extra_state"](
            checkpoint=checkpoint["extra_state"],
            extra_components=extra_components,
            device=accelerator.device,
        )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_metric = checkpoint.get("best_metric", None)

    accelerator.print(
        f"Resumed checkpoint from {ckpt_path}, "
        f"start_epoch={start_epoch}, global_step={global_step}"
    )

    return start_epoch, global_step, best_metric


def save_training_checkpoint(
    path,
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    ema_model,
    epoch,
    global_step,
    best_metric,
    args,
    modes,
    extra_components,
):
    """
    保存训练 checkpoint。

    如果当前不是主进程，直接 return，
    """
    if not accelerator.is_main_process:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)

    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        # 保存 unwrap 后的模型权重
        "model_state_dict": unwrapped_model.state_dict(),
        # 保存优化器状态，恢复后动量、二阶矩等不会丢。
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        # 保存学习率调度器状态，恢复后 warmup / cosine 进度不会重置。
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "best_metric": best_metric,
        # 保存完整命令行参数。
        "args": vars(args),
        # 复用目录。
        "exp_dir": getattr(
            args,
            "exp_dir",
            os.path.dirname(os.path.dirname(os.path.abspath(path))),
        ),
        # 保存不同模式自己的额外状态。
        "extra_state": modes["checkpoint_extra_state"](extra_components),
    }

    if ema_model is not None:
        state["ema_state_dict"] = ema_model.state_dict()

    torch.save(state, path)
