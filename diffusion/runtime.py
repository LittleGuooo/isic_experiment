import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from .data import build_datasets_and_loaders, normalize_label_to_index_and_name
from .metrics import (
    allocate_samples_by_ratio,
    evaluate_split_with_overall_and_per_class_metrics,
    uint8_tensor_to_pil,
)
from .modeling import (
    build_model,
    build_noise_scheduler,
    build_sampling_scheduler,
    build_save_pipeline,
)
from .modes import get_modes
from .utils import (
    make_experiment_name,
    make_runtime_run_name,
    setup_experiment_folders,
    setup_runtime_run_folders,
    save_json,
    update_epoch_metrics_csv,
    update_epoch_metrics_json,
    print_class_distribution,
    save_checkpoint,
    save_diffusers_model_index_copy,
    recover_exp_dir_from_checkpoint,
    sync_experiment_metadata_for_resume,
    format_count_ratio_dict,
)

from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel


def set_seed(seed):
    # 固定 Python / NumPy / PyTorch 随机种子，便于复现实验
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_dataloader_with_new_batch_size(loader, new_batch_size):
    """
    基于已有 DataLoader 复用 dataset / sampler / collate_fn 等配置，
    只替换 batch_size。
    这样不改变外部接口，只让 classifier 训练单独使用自己的 batch size。
    """
    return DataLoader(
        dataset=loader.dataset,
        batch_size=new_batch_size,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=getattr(loader, "pin_memory_device", ""),
    )


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
    modes,
    extra_components,
):
    # 进入评估模式，关闭 Dropout / 使用 BatchNorm 的推理行为
    # 进入纯推理模式
    model.eval()
    # 为本次 val_only 运行创建单独的输出目录
    run_name = make_runtime_run_name(args)
    run_folders = setup_runtime_run_folders(
        exp_folders["exp_dir"], "val_only", run_name
    )

    # 保存本次验证运行的配置快照
    save_json(
        {
            "run_mode": args.run_mode,
            "mode": args.mode,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path": args.resume_from_checkpoint,
            "experiment_dir": exp_folders["exp_dir"],
            "checkpoint_epoch": checkpoint_epoch,
            "args": vars(args),
        },
        run_folders["run_config_json"],
    )

    enable_train_fid = args.num_fid_samples_train > 0
    enable_val_fid = args.num_fid_samples_val > 0
    train_eval_result = None
    val_eval_result = None

    # 如开启训练集评估，则先在 train split 上计算指标
    if enable_train_fid:
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
            modes=modes,
            extra_components=extra_components,
        )

    # 如开启验证集评估，则在 val split 上计算指标
    if enable_val_fid:
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
            modes=modes,
            extra_components=extra_components,
        )

    run_summary = {
        "run_mode": "val_only",
        "mode": args.mode,
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


@torch.no_grad()
def run_inference_only(
    args,
    accelerator,
    model,
    noise_scheduler,
    class_names,
    exp_folders,
    checkpoint_epoch,
    modes,
    extra_components,
):
    model.eval()
    infer_label_idx, infer_label_name = normalize_label_to_index_and_name(
        args.infer_label, class_names
    )

    run_name = make_runtime_run_name(args)
    run_folders = setup_runtime_run_folders(
        exp_folders["exp_dir"], "infer_only", run_name
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
            "mode": args.mode,
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

    # 为推理构造采样 scheduler（DDPM 或 DDIM）
    sampling_scheduler = build_sampling_scheduler(
        noise_scheduler,
        args.use_ddim_sampling,
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

        samples_uint8 = modes["sample_images"](
            model=model,
            sampling_scheduler=sampling_scheduler,
            device=accelerator.device,
            resolution=args.resolution,
            batch_size=cur_bs,
            num_inference_steps=args.ddpm_num_inference_steps,
            generator=generator,
            class_labels=class_labels,
            extra_components=extra_components,
            return_pil_safe_uint8=True,
        )

        for i in range(samples_uint8.size(0)):
            pil_img = uint8_tensor_to_pil(samples_uint8[i])
            file_name = (
                f"sample_{generated_count:05d}_{infer_label_name}.png"
                if infer_label_name is not None
                else f"sample_{generated_count:05d}.png"
            )
            img_path = os.path.join(infer_image_dir, file_name)
            pil_img.save(img_path)
            saved_image_paths.append(img_path)
            generated_count += 1

        batch_id += 1

    run_summary = {
        "run_mode": "infer_only",
        "mode": args.mode,
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


def run_train(args):
    # 训练主入口：负责数据、模型、优化器、评估、保存 checkpoint
    # 先固定随机种子
    set_seed(args.seed)

    # 如果指定了 checkpoint，则尝试从旧实验目录恢复
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        recovered_exp_dir = recover_exp_dir_from_checkpoint(
            args.resume_from_checkpoint,
            checkpoint,
        )
        exp_name = os.path.basename(recovered_exp_dir)
        exp_folders = setup_experiment_folders(
            os.path.dirname(recovered_exp_dir),
            exp_name,
        )
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

    # Accelerator 负责混合精度、分布式训练和梯度同步
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # 统一构造数据集、DataLoader、类别信息和类别分布
    bundle = build_datasets_and_loaders(args)
    train_dataset = bundle["train_dataset"]
    val_dataset = bundle["val_dataset"]
    train_dataloader = bundle["train_dataloader"]
    train_eval_loader = bundle["train_eval_loader"]
    val_eval_loader = bundle["val_eval_loader"]
    class_names = bundle["class_names"]
    num_classes = bundle["num_classes"]
    train_class_distribution = bundle["train_class_distribution"]
    val_class_distribution = bundle["val_class_distribution"]

    # cg 模式下classifier数据集构建
    classifier_train_dataloader = train_dataloader
    if (
        args.mode == "cg"
        and args.run_mode == "train"
        and args.classifier_train_batch_size is not None
        and args.classifier_train_batch_size > 0
        and args.classifier_train_batch_size != args.train_batch_size
    ):
        classifier_train_dataloader = _clone_dataloader_with_new_batch_size(
            train_dataloader,
            args.classifier_train_batch_size,
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print_class_distribution(
        "Train Dataset Class Distribution", train_class_distribution
    )
    print_class_distribution(
        "Validation Dataset Class Distribution", val_class_distribution
    )

    # 根据 args.mode 获取对应的训练 / 采样逻辑
    modes = get_modes(args)

    # 构建 UNet 扩散模型与训练用 scheduler
    model = build_model(args, num_classes)
    noise_scheduler = build_noise_scheduler(args)

    # ---------------------------
    # EMA：仅跟踪 diffusion UNet 参数
    # ---------------------------
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
        )

    # ------------------------------------------------------------------------------------
    # CG 分支：如果指定了 cg_diffusion_ckpt_path，则进入 CG 分类器训练流程，不再训练扩散模型
    # ------------------------------------------------------------------------------------
    if (
        args.mode == "cg"
        and args.run_mode == "train"
        and args.cg_diffusion_ckpt_path is not None
    ):
        from .modes.cg import load_diffusion_backbone_checkpoint

        model = load_diffusion_backbone_checkpoint(
            unet=model,
            ckpt_path=args.cg_diffusion_ckpt_path,
            device="cpu",
        )
        model = model.to(accelerator.device)
        model.eval()

        if accelerator.is_main_process:
            modes["train_classifier_only_from_diffusion"](
                unet=model,
                noise_scheduler=noise_scheduler,
                train_dataloader=classifier_train_dataloader,
                val_dataloader=val_eval_loader,
                num_classes=num_classes,
                device=accelerator.device,
                exp_folders=exp_folders,
            )

        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    extra_components = modes["build_extra_components"](
        num_classes=num_classes,
        device=accelerator.device,
    )

    # val_only / infer_only 分支：不进入训练循环，直接加载模型并执行
    if args.run_mode in ["val_only", "infer_only"]:
        if checkpoint is None:
            raise ValueError(
                "Checkpoint must be loaded for val_only / infer_only mode."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        modes["load_checkpoint_extra_state"](
            checkpoint=checkpoint,
            extra_components=extra_components,
            device=accelerator.device,
        )

        if ema_model is not None and "ema_state_dict" in checkpoint:
            ema_model.copy_to(model.parameters())

        model = model.to(accelerator.device)
        model.eval()
        checkpoint_epoch = checkpoint.get("epoch", 0)

        if args.run_mode == "val_only":
            # 只有主进程执行评估，其他进程等待评估完成后直接退出
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
                    modes=modes,
                    extra_components=extra_components,
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
                    modes=modes,
                    extra_components=extra_components,
                )
            accelerator.wait_for_everyone()
            accelerator.end_training()
            return

    # 训练模式下构造优化器
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

    # 学习率调度器由 diffusers 提供统一接口
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

    # 从断点继续训练
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        modes["load_checkpoint_extra_state"](
            checkpoint=checkpoint,
            extra_components=extra_components,
            device=accelerator.device,
        )

        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            modes["load_checkpoint_extra_state"](
                checkpoint=checkpoint,
                extra_components=extra_components,
                device=accelerator.device,
            )

            if ema_model is not None and "ema_state_dict" in checkpoint:
                ema_model.load_state_dict(checkpoint["ema_state_dict"])
                ema_model.to(accelerator.device)

            start_epoch = checkpoint["epoch"]
            global_step = checkpoint.get("global_step", 0)
            best_val_fid = checkpoint.get("best_val_fid", float("inf"))
            best_train_fid = checkpoint.get("best_train_fid", float("inf"))

        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", 0)
        best_val_fid = checkpoint.get("best_val_fid", float("inf"))
        best_train_fid = checkpoint.get("best_train_fid", float("inf"))

    # prepare(...) 会处理设备放置、分布式封装等细节
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    if ema_model is not None:
        ema_model.to(accelerator.device)

    # 保存或更新实验元数据，包含配置、路径、当前最佳结果等信息
    if args.resume_from_checkpoint is not None and os.path.exists(metadata_json_path):
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            experiment_metadata = __import__("json").load(f)

        experiment_metadata.setdefault(
            "paths",
            {
                "metrics_csv": metrics_csv_path,
                "metrics_json": metrics_json_path,
                "metadata_json": metadata_json_path,
                "checkpoints_dir": exp_folders["checkpoints_dir"],
                "samples_dir": exp_folders["samples_dir"],
                "fid_dir": exp_folders["fid_dir"],
                "fid_generated_dir": exp_folders["fid_generated_dir"],
                "diffusers_model_index_copy": os.path.join(
                    exp_folders["metadata_dir"],
                    "diffusers_pipeline_model_index.json",
                ),
            },
        )
        experiment_metadata.setdefault(
            "best_result",
            {
                "best_epoch_by_val_fid": -1,
                "best_val_fid": None,
                "best_epoch_by_train_fid": -1,
                "best_train_fid": None,
                "best_model_path": "",
            },
        )
        experiment_metadata = sync_experiment_metadata_for_resume(
            experiment_metadata,
            args,
            start_epoch,
            global_step,
        )
    else:
        experiment_metadata = {
            "experiment_name": exp_name,
            "run_mode": args.run_mode,
            "mode": args.mode,
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
            "mode": {
                "mode": args.mode,
                "cfg_scale": getattr(args, "cfg_scale", None),
                "cond_drop_prob": getattr(args, "cond_drop_prob", None),
                "classifier_ckpt_path": getattr(args, "classifier_ckpt_path", None),
                "classifier_guidance_scale": getattr(
                    args, "classifier_guidance_scale", None
                ),
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
                    exp_folders["metadata_dir"],
                    "diffusers_pipeline_model_index.json",
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

    # 进入 epoch 级训练循环
    for epoch in range(start_epoch, args.num_epochs):
        # 切回训练模式
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Train Epoch [{epoch + 1}/{args.num_epochs}]",
        )

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # 遍历一个 epoch 内的所有 batch
        for batch in train_dataloader:
            # accumulate(...) 用于支持梯度累积
            with accelerator.accumulate(model):
                loss, aux = modes["train_step"](
                    model=model,
                    noise_scheduler=noise_scheduler,
                    batch=batch,
                    accelerator=accelerator,
                    extra_components=extra_components,
                )

                # 反向传播由 accelerator 接管，兼容混合精度 / 分布式
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                # 标准训练三步：step -> scheduler.step -> zero_grad
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients and ema_model is not None:
                    ema_model.step(accelerator.unwrap_model(model).parameters())

            loss_item = loss.detach().item()
            batch_size_now = batch["input"].size(0)
            epoch_loss_sum += loss_item * batch_size_now
            epoch_loss_count += batch_size_now

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(
                    {"loss": f"{loss_item:.6f}", "update": global_step}
                )

        progress_bar.close()
        accelerator.wait_for_everyone()

        train_loss_epoch = epoch_loss_sum / max(epoch_loss_count, 1)

        # ------------------------------------------------------
        # 以下代码执行当前 epoch 的保存样图 / 模型 / 评估 /更新日志等操作
        # ------------------------------------------------------

        # 每隔 save_images_epochs 保存一次样图；
        need_save_images = ((epoch + 1) % args.save_images_epochs == 0) or (
            epoch == args.num_epochs - 1
        )

        # 这里保存的是 last.pth.tar，也就是“当前最新进度”
        need_save_model = ((epoch + 1) % args.save_model_epochs == 0) or (
            epoch == args.num_epochs - 1
        )

        enable_train_fid = args.num_fid_samples_train > 0
        enable_val_fid = args.num_fid_samples_val > 0

        # 是否需要在当前 epoch 执行评估
        need_eval = (
            ((epoch + 1) % args.eval_epochs == 0) or (epoch == args.num_epochs - 1)
        ) and (enable_train_fid or enable_val_fid)

        fid_train_value = None
        fid_val_value = None
        train_kid_mean = train_kid_std = val_kid_mean = val_kid_std = None
        train_precision = train_recall = val_precision = val_recall = None
        train_fid_json_path = val_fid_json_path = ""
        train_per_class_json_path = val_per_class_json_path = ""
        train_generated_dir = val_generated_dir = ""
        train_per_class_generated_dir = val_per_class_generated_dir = ""
        sample_dir = ""
        diffusers_model_index_copy_path = ""

        # 只在主进程执行“保存文件 / 跑评估 / 写日志”
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)

            # 保存 / 评估时临时切到 EMA 权重；结束后再恢复训练权重
            ema_applied = False
            if ema_model is not None:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
                ema_applied = True

            # 构建一个可保存的 diffusers pipeline
            # 后面会用 pipeline.save_pretrained(...) 保存到实验目录
            pipeline = build_save_pipeline(
                unet,
                noise_scheduler,
                args.use_ddim_sampling,
            )
            pipeline = pipeline.to(accelerator.device)

            # ---------------------------
            # 1. 保存当前 epoch 的样图
            # ---------------------------
            if need_save_images:
                # 为当前 epoch 创建样图目录，例如 samples/epoch_010/
                epoch_dir = os.path.join(
                    exp_folders["samples_dir"],
                    f"epoch_{epoch + 1:03d}",
                )
                os.makedirs(epoch_dir, exist_ok=True)

                # 按训练集真实类别分布，给每个类别分配要生成多少张样图
                # 这样保存出来的可视化样本在类别比例上更接近真实数据
                sample_alloc = allocate_samples_by_ratio(
                    train_class_distribution,
                    args.eval_batch_size,
                )

                # 为采样构建 scheduler
                # 可能是 DDPM scheduler，也可能是 DDIM scheduler
                sampling_scheduler = build_sampling_scheduler(
                    noise_scheduler,
                    args.use_ddim_sampling,
                )

                # 用来给保存出的样图编号
                sample_counter = 0

                # 逐类别生成样图
                for class_idx, class_name in enumerate(class_names):
                    # 当前类别需要生成的样图数量
                    cur_n = sample_alloc[class_name]

                    # 如果这个类别本轮不需要生成，就跳过
                    if cur_n <= 0:
                        continue

                    # 为当前类别构造一个固定随机种子
                    # 这样同一个 epoch 内不同类别的采样是可复现的
                    generator = torch.Generator(device=accelerator.device).manual_seed(
                        class_idx
                    )

                    # 如果启用了 class conditioning，就给当前批次构造类别标签
                    # 否则 class_labels=None，表示 unconditional sampling
                    class_labels = (
                        torch.full(
                            (cur_n,),
                            fill_value=class_idx,
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                        if args.use_class_conditioning
                        else None
                    )

                    # 调用当前模式（DDPM / CFG / CG）对应的采样函数生成图片
                    # 这里返回的是适合保存的 uint8 图像张量
                    samples_uint8 = modes["sample_images"](
                        model=unet,
                        sampling_scheduler=sampling_scheduler,
                        device=accelerator.device,
                        resolution=args.resolution,
                        batch_size=cur_n,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        generator=generator,
                        class_labels=class_labels,
                        extra_components=extra_components,
                        return_pil_safe_uint8=True,
                    )

                    # 把生成结果逐张保存到磁盘
                    for i in range(samples_uint8.size(0)):
                        pil_img = uint8_tensor_to_pil(samples_uint8[i])

                        # conditional 模式下文件名会带上类别名，便于查看
                        file_name = (
                            f"sample_{sample_counter:03d}_{class_name}.png"
                            if args.use_class_conditioning
                            else f"sample_{sample_counter:03d}.png"
                        )

                        pil_img.save(os.path.join(epoch_dir, file_name))
                        sample_counter += 1

                # 记录本轮样图目录，后面写入 metrics / metadata
                sample_dir = epoch_dir

            # ---------------------------
            # 2. 执行当前 epoch 的评估
            # ---------------------------
            if need_eval:
                # 如果启用了 train split 评估，就计算 train 的整体指标和可选逐类指标
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
                            modes=modes,
                            extra_components=extra_components,
                        )
                    )

                    # 从返回结果中提取 train split 的整体指标和路径信息
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

                # 如果启用了 val split 评估，就计算 val 的整体指标
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
                        modes=modes,
                        extra_components=extra_components,
                    )

                    # 从返回结果中提取 val split 的整体指标和路径信息
                    fid_val_value = val_eval_result["overall_fid"]
                    val_kid_mean = val_eval_result["overall_kid_mean"]
                    val_kid_std = val_eval_result["overall_kid_std"]
                    val_precision = val_eval_result["overall_precision"]
                    val_recall = val_eval_result["overall_recall"]
                    val_fid_json_path = val_eval_result["overall_json_path"]
                    val_generated_dir = val_eval_result["generated_dir"]
                    val_per_class_generated_dir = val_eval_result[
                        "per_class_generated_dir"
                    ]

            # ---------------------------
            # 3. 判断当前 epoch 是否刷新“最佳模型”
            # ---------------------------
            is_best = False

            # 当前代码是以 train_fid 作为 best 标准
            # 如果这一轮 train_fid 更小，就认为当前模型更好
            if fid_train_value is not None and fid_train_value < best_train_fid:
                best_train_fid = fid_train_value
                is_best = True

                # 更新 metadata 中的最佳 epoch 和最佳 train_fid
                experiment_metadata["best_result"]["best_epoch_by_train_fid"] = (
                    epoch + 1
                )
                experiment_metadata["best_result"]["best_train_fid"] = float(
                    fid_train_value
                )

            # ---------------------------
            # 4. 组装 checkpoint 内容
            # ---------------------------
            checkpoint_state = {
                # 当前完成到第几个 epoch
                "epoch": epoch + 1,
                # 当前全局 update step
                "global_step": global_step,
                # 模型参数
                "model_state_dict": unet.state_dict(),
                # 优化器状态
                "optimizer_state_dict": optimizer.state_dict(),
                # 学习率调度器状态
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                # 当前记录下来的最佳指标
                "best_val_fid": best_val_fid,
                "best_train_fid": best_train_fid,
                # 本次运行的全部命令行参数
                "args": vars(args),
                # 当前实验目录
                "exp_dir": exp_folders["exp_dir"],
                # 当前模式标识
                "mode": args.mode,
                # EMA 状态
                "ema_state_dict": (
                    ema_model.state_dict() if ema_model is not None else None
                ),
            }

            # 把 mode-specific 的附加状态也合并进 checkpoint
            # 例如 CFG 的 null_class_idx，或 CG 的额外组件信息
            checkpoint_state.update(modes["checkpoint_extra_state"](extra_components))

            # ---------------------------
            # 5. 保存“最新 checkpoint”
            # ---------------------------
            if need_save_model or is_best:
                # 保存到 last.pth.tar
                # 这个文件始终表示“最近一次保存的训练进度”
                save_checkpoint(
                    checkpoint_state,
                    is_best=is_best,
                    save_dir=exp_folders["checkpoints_dir"],
                    filename="last.pth.tar",
                )

                # 如果当前是最佳模型，还会额外复制成 model_best.pth.tar
                if is_best:
                    experiment_metadata["best_result"][
                        "best_model_path"
                    ] = best_model_path

                # 同时把 diffusers pipeline 形式的模型也保存到实验目录
                pipeline.save_pretrained(exp_folders["exp_dir"])

                # 并把 model_index.json 再复制一份到 metadata 目录中
                diffusers_model_index_copy_path = save_diffusers_model_index_copy(
                    exp_folders["exp_dir"],
                    exp_folders["metadata_dir"],
                )

            # ---------------------------
            # 6. 如果当前 epoch 做了评估，再额外留一个“按 epoch 命名”的 checkpoint
            # ---------------------------
            if need_eval:
                # 例如 epoch_010.pth.tar
                # 这类文件的作用是保留阶段性快照，而不是覆盖保存最新进度
                eval_ckpt_name = f"epoch_{epoch + 1:03d}.pth.tar"

                save_checkpoint(
                    checkpoint_state,
                    is_best=False,
                    save_dir=exp_folders["checkpoints_dir"],
                    filename=eval_ckpt_name,
                )

                # 保存 pipeline
                pipeline.save_pretrained(exp_folders["exp_dir"])

                # 保存一份 model_index.json 到 metadata 目录
                diffusers_model_index_copy_path = save_diffusers_model_index_copy(
                    exp_folders["exp_dir"],
                    exp_folders["metadata_dir"],
                )
            if ema_applied:
                ema_model.restore(unet.parameters())

        # 如果当前 epoch 做了评估，把指标与路径写入日志文件
        if accelerator.is_main_process and need_eval:
            epoch_row = {
                # 当前 epoch 编号
                "epoch": epoch + 1,
                # 当前 epoch 的平均训练损失
                "train_loss": float(train_loss_epoch),
                # train split 的 FID
                "train_fid": (
                    float(fid_train_value) if fid_train_value is not None else None
                ),
                # train split 的 KID
                "train_kid_mean": (
                    float(train_kid_mean) if train_kid_mean is not None else None
                ),
                "train_kid_std": (
                    float(train_kid_std) if train_kid_std is not None else None
                ),
                # val split 的 FID / KID
                "val_fid": float(fid_val_value) if fid_val_value is not None else None,
                "val_kid_mean": (
                    float(val_kid_mean) if val_kid_mean is not None else None
                ),
                "val_kid_std": (
                    float(val_kid_std) if val_kid_std is not None else None
                ),
                # train split 的 IPR 指标
                "train_precision": (
                    float(train_precision) if train_precision is not None else None
                ),
                "train_recall": (
                    float(train_recall) if train_recall is not None else None
                ),
                # val split 的 IPR 指标
                "val_precision": (
                    float(val_precision) if val_precision is not None else None
                ),
                "val_recall": (float(val_recall) if val_recall is not None else None),
                # 当前 epoch 的样图目录
                "sample_dir": sample_dir,
                # 各类评估 JSON 的路径
                "train_fid_json_path": train_fid_json_path,
                "val_fid_json_path": val_fid_json_path,
                "train_per_class_json_path": train_per_class_json_path,
                "val_per_class_json_path": val_per_class_json_path,
                # 生成图像目录
                "train_generated_dir": train_generated_dir,
                "val_generated_dir": val_generated_dir,
                "train_per_class_generated_dir": train_per_class_generated_dir,
                "val_per_class_generated_dir": val_per_class_generated_dir,
                # 当前默认的“最新 checkpoint”路径
                "checkpoint_path": os.path.join(
                    exp_folders["checkpoints_dir"],
                    "last.pth.tar",
                ),
                # metadata 中保存的 diffusers model_index 副本路径
                "diffusers_model_index_copy_path": diffusers_model_index_copy_path,
            }

            # 追加写入 CSV 日志
            update_epoch_metrics_csv(metrics_csv_path, epoch_row)

            # 追加写入 JSON 日志
            update_epoch_metrics_json(metrics_json_path, epoch_row)

            # 更新实验 metadata
            experiment_metadata["last_epoch_finished"] = epoch + 1
            experiment_metadata["updated_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            experiment_metadata["paths"]["diffusers_model_index_copy"] = os.path.join(
                exp_folders["metadata_dir"],
                "diffusers_pipeline_model_index.json",
            )

            # 把更新后的 metadata 持久化到 experiment_metadata.json
            save_json(experiment_metadata, metadata_json_path)

    # ---------------------------------------------------------------------------------
    # CG 模式：如果没指定 cg_diffusion_ckpt_path，则先训练扩散模型，再利用它训练 CG 分类器
    # ---------------------------------------------------------------------------------
    if (
        args.mode == "cg"
        and args.run_mode == "train"
        and args.cg_diffusion_ckpt_path is None
    ):
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            unet = unet.to(accelerator.device)
            unet.eval()

            modes["train_classifier_only_from_diffusion"](
                unet=unet,
                noise_scheduler=noise_scheduler,
                train_dataloader=classifier_train_dataloader,
                val_dataloader=val_eval_loader,
                num_classes=num_classes,
                device=accelerator.device,
                exp_folders=exp_folders,
            )

        # 等待所有进程到达这里，保证主进程的保存/评估先完成
        accelerator.wait_for_everyone()

    accelerator.end_training()
