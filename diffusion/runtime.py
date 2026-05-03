from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch

import os
from .modes.common import get_modes
from .data import build_datasets_and_loaders
from .modeling import build_model, build_noise_scheduler
from .utils import (
    set_seed,
    create_experiment_folders,
    save_json,
)

from .runtime_engine.classifier_training import (
    train_guidance_classifier_with_accelerator,
)
from .runtime_engine.checkpointing import load_training_checkpoint
from .runtime_engine.train_loop import run_diffusion_training_loop
from .runtime_engine.inference import run_inference_only


def run_train(args):
    set_seed(args.seed)

    # 创建/复用实验文件夹。
    exp_folders = create_experiment_folders(args)

    # TensorBoard 日志目录：当前实验目录/tensorboard
    tensorboard_dir = os.path.join(exp_folders["exp_dir"], "tensorboard")

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard" if args.use_tensorboard else None,
        project_dir=tensorboard_dir if args.use_tensorboard else None,
    )

    # TensorBoard tracker 初始化
    if args.use_tensorboard:
        tb_config = {}

        for key, value in vars(args).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                tb_config[key] = value
            else:
                tb_config[key] = str(value)

        accelerator.init_trackers(
            project_name=exp_folders["exp_name"],
            config=tb_config,
        )

    # 构建数据集和 dataloader
    data_bundle = build_datasets_and_loaders(args)
    train_dataloader = data_bundle["train_dataloader"]
    train_eval_loader = data_bundle["train_eval_loader"]
    val_eval_loader = data_bundle["val_eval_loader"]
    class_names = data_bundle["class_names"]
    train_class_distribution = data_bundle["train_class_distribution"]
    val_class_distribution = data_bundle["val_class_distribution"]
    # classifier 的训练复用 val_eval_loader
    val_dataloader = data_bundle["val_eval_loader"]
    num_classes = len(class_names)

    noise_scheduler = build_noise_scheduler(args)

    # classifier-only 训练分支
    if args.mode == "cg" and args.run_mode == "train_classifier":
        result = train_guidance_classifier_with_accelerator(
            args=args,
            noise_scheduler=noise_scheduler,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            num_classes=num_classes,
            accelerator=accelerator,
            exp_folders=exp_folders,
        )

        if accelerator.is_main_process:
            save_json(
                {
                    "mode": args.mode,
                    "run_mode": args.run_mode,
                    "classifier_training": result,
                },
                exp_folders["metadata_json_path"],
            )

        accelerator.end_training()
        return

    modes = get_modes(args)

    model = build_model(args, num_classes=num_classes)

    # EMA：Exponential Moving Average
    ema_model = None
    if getattr(args, "use_ema", False):
        import copy

        ema_model = copy.deepcopy(model)
        ema_model.to(accelerator.device)

        for p in ema_model.parameters():
            p.requires_grad = False

        ema_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = (
        len(train_dataloader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps

    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    extra_components = modes["build_extra_components"](
        num_classes=num_classes,
        device=accelerator.device,
    )

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )
    )

    # 将 ldm_ae 的 PatchGAN discriminator 也交给 accelerator.prepare()
    discriminator = extra_components.get("discriminator", None)
    d_optimizer = extra_components.get("d_optimizer", None)
    if discriminator is not None and d_optimizer is not None:
        discriminator, d_optimizer = accelerator.prepare(
            discriminator,
            d_optimizer,
        )

        # prepare 后的对象需要写回 extra_components，
        # 后续 train_step 才会使用被 Accelerator 包装过的版本。
        extra_components["discriminator"] = discriminator
        extra_components["d_optimizer"] = d_optimizer

    # 无 resume 参数则返回0值
    start_epoch, global_step, best_metric = load_training_checkpoint(
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
        modes=modes,
        extra_components=extra_components,
    )

    # 仅推理模式
    if args.run_mode == "infer_only":
        run_inference_only(
            args=args,
            accelerator=accelerator,
            model=model,
            noise_scheduler=noise_scheduler,
            class_names=class_names,
            modes=modes,
            extra_components=extra_components,
            output_dir=exp_folders["samples_dir"],
        )
        accelerator.end_training()
        return

    run_diffusion_training_loop(
        args=args,
        accelerator=accelerator,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
        train_dataloader=train_dataloader,
        train_eval_loader=train_eval_loader,
        val_eval_loader=val_eval_loader,
        class_names=class_names,
        train_class_distribution=train_class_distribution,
        val_class_distribution=val_class_distribution,
        modes=modes,
        extra_components=extra_components,
        exp_folders=exp_folders,
        start_epoch=start_epoch,
        global_step=global_step,
        best_metric=best_metric,
    )

    # 保存 metadata
    if accelerator.is_main_process:
        save_json(
            {
                # 当前运行模式，例如 ldm_ae / latent_ddpm / ddpm / cfg / cg
                "mode": args.mode,
                # 当前运行类型，例如 train / infer_only
                "run_mode": args.run_mode,
                # 实验目录，方便之后定位输出文件
                "exp_dir": exp_folders["exp_dir"],
                # 数据集类别名
                "class_names": class_names,
                # 训练集类别分布
                "train_class_distribution": train_class_distribution,
                # 验证集类别分布
                "val_class_distribution": val_class_distribution,
                # 命令行参数快照
                "args": {
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool)) or v is None
                        else str(v)
                    )
                    for k, v in vars(args).items()
                },
                # 模式相关配置。
                # 对 ldm_ae 来说，这里会包含 ae_config 和 ldm_ae_extra_state。
                "extra_state": modes["checkpoint_extra_state"](extra_components),
            },
            exp_folders["metadata_json_path"],
        )

    accelerator.end_training()
