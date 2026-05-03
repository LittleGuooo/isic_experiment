import os
from tqdm.auto import tqdm
import torch
from .checkpointing import save_training_checkpoint
from .evaluation import run_generation_evaluation, save_evaluation_summary
from ..metrics import save_visual_samples_during_training
from ..modes.ldm_ae import save_ldm_ae_pretrained_outputs


def run_diffusion_training_loop(
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
    主要流程：
        1. 从 start_epoch 开始循环训练到 args.num_epochs。
        2. 每个 epoch 内遍历 train_dataloader。
        3. 调用 modes["train_step"] 计算当前 batch 的 loss。
        4. 使用 accelerator.backward(loss) 反向传播。
        5. 在梯度同步时进行 gradient clipping。
        6. 更新 optimizer、lr_scheduler，并在需要时更新 EMA 模型。
        7. 记录 batch 级 TensorBoard 日志。
        8. 每个 epoch 结束后计算平均训练损失。
        9. 按 save_images_epochs 保存生成图或重建图。
        10. 按 eval_epochs 计算 FID/KID/IPR 等生成质量指标。
        11. 按 save_model_epochs 保存 checkpoint。
        12. 保存 last.pth.tar，必要时保存 epoch_xxx.pth.tar。
    """

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        total_loss = 0.0
        total_count = 0

        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Train epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
            leave=True,
        )

        for step, batch in enumerate(train_dataloader):
            # 梯度累积（gradient accumulation）。
            # Accelerator 会自动控制什么时候真正同步梯度和更新 global_step。
            with accelerator.accumulate(model):
                # 前向过程
                loss, aux = modes["train_step"](
                    model=model,
                    noise_scheduler=noise_scheduler,
                    batch=batch,
                    accelerator=accelerator,
                    extra_components=extra_components,
                )

                accelerator.backward(loss)

                # 只有真正同步梯度时才做梯度裁剪；
                if accelerator.sync_gradients:
                    max_grad_norm = getattr(args, "max_grad_norm", 1.0)
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                # EMA 更新
                if ema_model is not None:
                    decay = float(getattr(args, "ema_decay", 0.9999))
                    unwrapped_model = accelerator.unwrap_model(model)
                    ema_model.to(accelerator.device)

                    with torch.no_grad():
                        for ema_param, model_param in zip(
                            ema_model.parameters(),
                            unwrapped_model.parameters(),
                        ):
                            ema_param.data.mul_(decay).add_(
                                model_param.detach().data.to(ema_param.device),
                                alpha=1.0 - decay,
                            )

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            gathered_loss = accelerator.gather_for_metrics(
                loss.detach().repeat(batch["input"].shape[0])
            )

            total_loss += gathered_loss.float().sum().item()
            total_count += gathered_loss.numel()

            # 当真正参数更新时，global_step 加 1，并记录一次 TensorBoard 日志。
            if accelerator.sync_gradients:
                global_step += 1

                # TensorBoard 日志记录
                if args.use_tensorboard:
                    log_dict = {
                        "train/loss": float(loss.detach().item()),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    }

                    # aux 是各模式 train_step 返回的额外指标：
                    # 例如 ldm_ae 会返回 recon_loss、kl_loss、perceptual_loss 等。
                    if isinstance(aux, dict):
                        for k, v in aux.items():
                            if isinstance(v, (int, float)):
                                log_dict[f"train/{k}"] = float(v)

                    accelerator.log(
                        log_dict,
                        step=global_step,
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

        split_results = {
            "train_loss": float(epoch_loss),
        }

        # 记录 epoch 级别训练损失。
        if args.use_tensorboard:
            accelerator.log(
                {
                    "epoch/train_loss": float(epoch_loss),
                },
                step=global_step,
            )

        if accelerator.is_main_process:
            accelerator.print(f"[Epoch {epoch + 1}] train_loss={epoch_loss:.6f}")

        accelerator.wait_for_everyone()

        # 保存可视化样本。
        should_save_images = args.save_images_epochs > 0 and (
            (epoch + 1) % args.save_images_epochs == 0 or (epoch + 1) == args.num_epochs
        )

        if should_save_images:
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

        accelerator.wait_for_everyone()

        # 进行评估
        should_eval = args.mode != "ldm_ae" and (
            (epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == args.num_epochs
        )

        if should_eval:
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

            if accelerator.is_main_process:
                save_evaluation_summary(
                    exp_folders=exp_folders,
                    epoch=epoch + 1,
                    split_results=split_results,
                )

        accelerator.wait_for_everyone()

        # 保存checkpoints
        if accelerator.is_main_process:
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

            save_ldm_ae_pretrained_outputs(
                args=args,
                accelerator=accelerator,
                model=model,
                ema_model=ema_model,
                exp_folders=exp_folders,
            )

            if (epoch + 1) % args.save_model_epochs == 0:
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

        accelerator.wait_for_everyone()
