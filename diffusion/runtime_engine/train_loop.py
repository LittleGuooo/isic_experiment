import os

import torch
from tqdm.auto import tqdm

from .checkpoint import save_training_checkpoint
from .evaluation import run_generation_evaluation, save_evaluation_summary
from ..metrics import save_visual_samples_during_training
from ..modes.ldm_ae import save_ldm_ae_pretrained_outputs
from ..utils import cleanup_after_generation


def _get_params_to_clip(args, model, extra_components):
    """
    返回需要做 gradient clipping 的参数。

    textual inversion 只训练 placeholder token embedding，
    所以不能直接 clip model.parameters()。
    """
    # Textual Inversion 只更新输入 embedding 中与占位符 token 相关的权重。
    # 这里返回 text encoder 的 embedding 参数，避免对未参与训练的 UNet 参数做裁剪。
    if args.mode == "sd_textual_inversion":
        return extra_components["text_encoder"].get_input_embeddings().parameters()

    # 普通扩散模型和 LoRA 模式直接从主模型中取参数。
    # LoRA 模式中未训练参数通常 requires_grad=False，clip_grad_norm_ 会自动忽略它们。
    return model.parameters()


def _run_textual_inversion_before_step(args, modes, extra_components, accelerator):
    """
    textual inversion 在 optimizer.step() 前，
    需要保护非 placeholder token embedding。
    """
    # 其他训练模式不需要修改 embedding 梯度。
    if args.mode == "sd_textual_inversion":
        modes["before_optimizer_step"](extra_components, accelerator)


def _run_textual_inversion_after_step(args, modes, extra_components, accelerator):
    """
    textual inversion 在 optimizer.step() 后，
    需要恢复非 placeholder token embedding。
    """
    # optimizer.step() 可能会改变整张 embedding 表。
    # Textual Inversion 模式需要把非占位符 token 的权重恢复回去。
    if args.mode == "sd_textual_inversion":
        modes["after_optimizer_step"](extra_components, accelerator)


@torch.no_grad()
def _update_ema_if_needed(args, accelerator, model, ema_model):
    """
    EMA 只应该在真正完成一次 optimizer update 后更新。

    accelerator.sync_gradients=False 时通常表示还处于梯度累积中间 step，
    这时不应该更新 EMA。
    """
    # 未启用 EMA 时直接跳过。
    if ema_model is None:
        return

    # 梯度累积中间 step 不是真正的参数更新点。
    # 如果这里也更新 EMA，会让 EMA 更新次数多于 optimizer 更新次数。
    if not accelerator.sync_gradients:
        return

    decay = float(getattr(args, "ema_decay", 0.9999))

    # accelerator.prepare() 之后 model 可能被 DDP 等包装。
    # EMA 应该读取解包后的真实模型参数。
    unwrapped_model = accelerator.unwrap_model(model)
    ema_model.to(accelerator.device)

    # 标准 EMA：ema = decay * ema + (1 - decay) * current_weight。
    for ema_param, model_param in zip(
        ema_model.parameters(),
        unwrapped_model.parameters(),
    ):
        ema_param.data.mul_(decay).add_(
            model_param.detach().data.to(ema_param.device),
            alpha=1.0 - decay,
        )


def _log_train_step_if_needed(
    args,
    accelerator,
    loss,
    aux,
    lr_scheduler,
    global_step,
):
    """
    记录 batch-level TensorBoard 日志。
    """
    # 没有启用 TensorBoard 时，不构造日志字典。
    if not args.use_tensorboard:
        return

    log_dict = {
        "train/loss": float(loss.detach().item()),
        "train/lr": float(lr_scheduler.get_last_lr()[0]),
    }

    # train_step 可以返回 mode-specific 指标，例如重建 loss、KL loss 或判别器 loss。
    # 这里只记录简单标量，避免把 tensor 或复杂对象直接传给 TensorBoard。
    if isinstance(aux, dict):
        for k, v in aux.items():
            if isinstance(v, (int, float)):
                log_dict[f"train/{k}"] = float(v)

    accelerator.log(log_dict, step=global_step)


def _train_one_batch(
    args,
    accelerator,
    model,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    ema_model,
    batch,
    modes,
    extra_components,
):
    """
    训练一个 batch。

    返回:
        loss: 当前 batch loss
        aux: mode-specific 额外日志
    """
    # accumulate() 由 Accelerate 管理梯度累积。
    # 在累积中间 step，accelerator.sync_gradients=False。
    with accelerator.accumulate(model):
        # 不同 mode 的前向传播和 loss 计算放在各自 train_step 中。
        # 例如 DDPM、LDM Autoencoder、LoRA、Textual Inversion 可以复用同一套外层循环。
        loss, aux = modes["train_step"](
            model=model,
            noise_scheduler=noise_scheduler,
            batch=batch,
            accelerator=accelerator,
            extra_components=extra_components,
        )

        # 使用 accelerator.backward()，让混合精度和分布式训练由 Accelerate 统一处理。
        accelerator.backward(loss)

        # Textual Inversion 在参数更新前需要屏蔽非占位符 token 的更新。
        _run_textual_inversion_before_step(
            args=args,
            modes=modes,
            extra_components=extra_components,
            accelerator=accelerator,
        )

        # 只在真正执行 optimizer update 的 step 做梯度裁剪。
        # 梯度累积尚未结束时裁剪，会改变累积结果。
        if accelerator.sync_gradients:
            max_grad_norm = getattr(args, "max_grad_norm", 1.0)
            params_to_clip = _get_params_to_clip(
                args=args,
                model=model,
                extra_components=extra_components,
            )
            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

        # Accelerate 包装后的 optimizer 会在累积中间 step 自动跳过真实更新。
        optimizer.step()

        # Textual Inversion 在参数更新后恢复非占位符 token embedding。
        _run_textual_inversion_after_step(
            args=args,
            modes=modes,
            extra_components=extra_components,
            accelerator=accelerator,
        )

        # EMA 必须放在 optimizer.step() 之后，读取本次更新后的模型参数。
        _update_ema_if_needed(
            args=args,
            accelerator=accelerator,
            model=model,
            ema_model=ema_model,
        )

        # scheduler 和 optimizer 一样，由 Accelerate 在累积中间 step 保持同步语义。
        lr_scheduler.step()

        # set_to_none=True 比写回全零更省显存，也能更容易暴露未产生梯度的参数。
        optimizer.zero_grad(set_to_none=True)

    return loss, aux


def _train_one_epoch(
    args,
    accelerator,
    model,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    ema_model,
    train_dataloader,
    modes,
    extra_components,
    epoch,
    global_step,
):
    """
    训练一个 epoch。

    返回:
        epoch_loss
        global_step
    """
    # 开启训练模式，确保 Dropout、BatchNorm 等模块使用训练行为。
    model.train()

    # total_loss / total_count 用于计算整个 epoch 的样本级平均 loss。
    total_loss = 0.0
    total_count = 0

    # 每个本地主进程显示一个进度条，避免多卡训练时重复打印。
    progress_bar = tqdm(
        total=len(train_dataloader),
        desc=f"Train epoch {epoch + 1}/{args.num_epochs}",
        disable=not accelerator.is_local_main_process,
        leave=True,
    )

    for batch in train_dataloader:
        # 完成一个 dataloader batch 的前向、反向和按需 optimizer update。
        loss, aux = _train_one_batch(
            args=args,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            batch=batch,
            modes=modes,
            extra_components=extra_components,
        )

        # 当前 loss 通常是 batch 平均值。先按样本数复制，再聚合所有进程，
        # 这样 total_loss / total_count 得到的是跨 GPU 的样本级平均 loss。
        gathered_loss = accelerator.gather_for_metrics(
            loss.detach().repeat(batch["input"].shape[0])
        )

        total_loss += gathered_loss.float().sum().item()
        total_count += gathered_loss.numel()

        # global_step 统计真实 optimizer update 次数，而不是 dataloader batch 数。
        if accelerator.sync_gradients:
            global_step += 1

            _log_train_step_if_needed(
                args=args,
                accelerator=accelerator,
                loss=loss,
                aux=aux,
                lr_scheduler=lr_scheduler,
                global_step=global_step,
            )

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "loss": total_loss / max(total_count, 1),
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )

    progress_bar.close()

    epoch_loss = total_loss / max(total_count, 1)

    return epoch_loss, global_step


def _log_epoch_loss_if_needed(args, accelerator, epoch_loss, global_step):
    """
    记录 epoch-level 训练 loss。
    """
    # epoch loss 使用真实 optimizer update 对应的 global_step 作为横轴。
    if args.use_tensorboard:
        accelerator.log(
            {
                "epoch/train_loss": float(epoch_loss),
            },
            step=global_step,
        )


def _should_save_visual_samples(args, epoch):
    """
    判断当前 epoch 是否保存可视化样本。

    最后一个 epoch 强制保存一次。
    """
    return args.save_images_epochs > 0 and (
        (epoch + 1) % args.save_images_epochs == 0 or (epoch + 1) == args.num_epochs
    )


def _save_visual_samples_if_needed(
    args,
    accelerator,
    model,
    noise_scheduler,
    train_eval_loader,
    class_names,
    modes,
    extra_components,
    exp_folders,
    epoch,
):
    """
    按需保存训练过程中的可视化样本。
    """
    if not _should_save_visual_samples(args, epoch):
        return

    # 生成少量样本用于肉眼检查训练是否崩溃、条件是否生效。
    save_visual_samples_during_training(
        args=args,
        accelerator=accelerator,
        model=model,
        noise_scheduler=noise_scheduler,
        train_eval_loader=train_eval_loader,
        class_names=class_names,
        modes=modes,
        extra_components=extra_components,
        exp_folders=exp_folders,
        epoch=epoch + 1,
    )

    # 生成过程通常会临时占用较多显存，完成后主动清理。
    cleanup_after_generation(accelerator)


def _should_run_generation_eval(args, epoch):
    """
    判断当前 epoch 是否运行 FID/KID/IPR 评估。

    ldm_ae 是 autoencoder 重建训练，不做生成质量评估。
    """
    if args.mode == "ldm_ae":
        return False

    if args.eval_epochs <= 0:
        return False

    return (epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == args.num_epochs


def _run_generation_eval_if_needed(
    args,
    accelerator,
    model,
    noise_scheduler,
    train_eval_loader,
    val_eval_loader,
    class_names,
    train_class_distribution,
    val_class_distribution,
    modes,
    extra_components,
    exp_folders,
    epoch,
    split_results,
):
    """
    按需运行 train / val 的生成质量评估。
    """
    if not _should_run_generation_eval(args, epoch):
        return split_results

    # train split 可选地计算整体指标和 per-class 指标。
    if args.num_fid_samples_train > 0:
        train_eval_result = run_generation_evaluation(
            args=args,
            split_name="train",
            real_loader=train_eval_loader,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            class_names=class_names,
            dataset_count_dict=train_class_distribution,
            num_total_samples=args.num_fid_samples_train,
            exp_folders=exp_folders,
            epoch=epoch + 1,
            modes=modes,
            extra_components=extra_components,
            compute_per_class_metrics=args.enable_per_class_metrics,
        )

        split_results["train_generation_eval"] = train_eval_result

    # val split 默认只计算整体指标，避免额外生成过多样本。
    if args.num_fid_samples_val > 0:
        val_eval_result = run_generation_evaluation(
            args=args,
            split_name="val",
            real_loader=val_eval_loader,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            class_names=class_names,
            dataset_count_dict=val_class_distribution,
            num_total_samples=args.num_fid_samples_val,
            exp_folders=exp_folders,
            epoch=epoch + 1,
            modes=modes,
            extra_components=extra_components,
            compute_per_class_metrics=False,
        )

        split_results["val_generation_eval"] = val_eval_result

    # 多卡训练时只允许主进程写 summary 文件。
    if accelerator.is_main_process:
        save_evaluation_summary(
            exp_folders=exp_folders,
            epoch=epoch + 1,
            split_results=split_results,
        )

    # 指标计算结束后释放生成和特征提取阶段的临时显存。
    cleanup_after_generation(accelerator)

    return split_results


def _save_checkpoint_pair_if_needed(
    args,
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    ema_model,
    epoch,
    global_step,
    best_metric,
    modes,
    extra_components,
    exp_folders,
):
    """
    保存 checkpoint。

    行为保持原逻辑：
        1. 每个 epoch 保存 last.pth.tar
        2. ldm_ae 额外保存 pretrained 格式
        3. 按 save_model_epochs 保存 epoch_xxx.pth.tar
    """
    # checkpoint 文件只能由主进程写入，避免多个进程同时覆盖同一路径。
    if not accelerator.is_main_process:
        return

    # last.pth.tar 每个 epoch 都覆盖一次，方便中断后优先恢复最新状态。
    last_ckpt_path = os.path.join(
        exp_folders["checkpoints_dir"],
        "last.pth.tar",
    )

    save_training_checkpoint(
        path=last_ckpt_path,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
        epoch=epoch + 1,
        global_step=global_step,
        best_metric=best_metric,
        args=args,
        modes=modes,
        extra_components=extra_components,
    )

    # ldm_ae 模式额外导出 Diffusers 可复用的 pretrained 权重；其他模式内部会跳过。
    save_ldm_ae_pretrained_outputs(
        args=args,
        accelerator=accelerator,
        model=model,
        ema_model=ema_model,
        exp_folders=exp_folders,
    )

    # 按固定间隔保留历史 checkpoint，避免只能拿到最后一个 epoch。
    should_save_epoch_ckpt = (
        args.save_model_epochs > 0 and (epoch + 1) % args.save_model_epochs == 0
    )

    if not should_save_epoch_ckpt:
        return

    epoch_ckpt_path = os.path.join(
        exp_folders["checkpoints_dir"],
        f"epoch_{epoch + 1:03d}.pth.tar",
    )

    save_training_checkpoint(
        path=epoch_ckpt_path,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
        epoch=epoch + 1,
        global_step=global_step,
        best_metric=best_metric,
        args=args,
        modes=modes,
        extra_components=extra_components,
    )

    save_ldm_ae_pretrained_outputs(
        args=args,
        accelerator=accelerator,
        model=model,
        ema_model=ema_model,
        exp_folders=exp_folders,
    )


def run_training_loop(
    args,
    accelerator,
    model,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    ema_model,
    train_dataloader,
    train_eval_loader,
    val_eval_loader,
    class_names,
    train_class_distribution,
    val_class_distribution,
    modes,
    extra_components,
    exp_folders,
    start_epoch=0,
    global_step=0,
    best_metric=None,
):
    """
    执行完整训练循环。

    这个函数现在只保留训练主线：
        1. train one epoch
        2. log epoch loss
        3. save visual samples if needed
        4. run generation eval if needed
        5. save checkpoints
    """

    for epoch in range(start_epoch, args.num_epochs):
        # 完成一个 epoch 的参数更新，并返回跨进程平均训练 loss。
        epoch_loss, global_step = _train_one_epoch(
            args=args,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            train_dataloader=train_dataloader,
            modes=modes,
            extra_components=extra_components,
            epoch=epoch,
            global_step=global_step,
        )

        # 统一收集当前 epoch 的训练和生成评估结果，供 summary 文件保存。
        split_results = {
            "train_loss": float(epoch_loss),
        }

        # 记录 epoch-level loss。
        _log_epoch_loss_if_needed(
            args=args,
            accelerator=accelerator,
            epoch_loss=epoch_loss,
            global_step=global_step,
        )

        if accelerator.is_main_process:
            accelerator.print(f"[Epoch {epoch + 1}] train_loss={epoch_loss:.6f}")

        # 多卡训练必须在进入生成阶段前同步，避免部分进程仍在训练。
        accelerator.wait_for_everyone()

        # 按配置保存可视化样本。
        _save_visual_samples_if_needed(
            args=args,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            train_eval_loader=train_eval_loader,
            class_names=class_names,
            modes=modes,
            extra_components=extra_components,
            exp_folders=exp_folders,
            epoch=epoch,
        )

        # 等待所有进程完成可视化生成，再进入指标评估。
        accelerator.wait_for_everyone()

        # 按配置运行 train / val 生成质量评估。
        split_results = _run_generation_eval_if_needed(
            args=args,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            train_eval_loader=train_eval_loader,
            val_eval_loader=val_eval_loader,
            class_names=class_names,
            train_class_distribution=train_class_distribution,
            val_class_distribution=val_class_distribution,
            modes=modes,
            extra_components=extra_components,
            exp_folders=exp_folders,
            epoch=epoch,
            split_results=split_results,
        )

        # 等待评估完成，确保 checkpoint 对应完整 epoch 状态。
        accelerator.wait_for_everyone()

        # 保存 last checkpoint，并按间隔额外保留 epoch checkpoint。
        _save_checkpoint_pair_if_needed(
            args=args,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            modes=modes,
            extra_components=extra_components,
            exp_folders=exp_folders,
        )

        # checkpoint 写完后再进入下一个 epoch。
        accelerator.wait_for_everyone()
