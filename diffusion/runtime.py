import copy
import os

import torch
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

from .modes.common import get_modes
from .data import build_datasets_and_loaders
from .modeling import build_model, build_noise_scheduler
from .utils import set_seed
from .experiment import create_experiment_folders, save_json
from .runtime_engine.checkpoint import (
    resume_training_from_checkpoint_if_available,
    get_resume_exp_dir,
)
from .runtime_engine.train_loop import run_training_loop


def build_accelerator(args, exp_folders):
    """
    创建 Accelerator。

    Accelerator 负责：
    1. 混合精度；
    2. 分布式训练封装；
    3. 梯度累积；
    4. TensorBoard tracker。
    """
    tensorboard_dir = os.path.join(exp_folders["exp_dir"], "tensorboard")

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard" if args.use_tensorboard else None,
        project_dir=tensorboard_dir if args.use_tensorboard else None,
    )

    return accelerator


def init_tensorboard_if_needed(args, accelerator, exp_folders):
    """
    初始化 TensorBoard tracker。

    只在 use_tensorboard=True 时启用。
    """
    if not args.use_tensorboard:
        return

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


def build_ema_model_if_needed(args, model, accelerator):
    """
    构建 EMA 模型。

    EMA 是训练时的滑动平均权重。
    train_loop.py 中会在 sync_gradients=True 时更新它。
    """
    if not getattr(args, "use_ema", False):
        return None

    ema_model = copy.deepcopy(model)
    ema_model.to(accelerator.device)

    for p in ema_model.parameters():
        p.requires_grad = False

    ema_model.eval()
    return ema_model


def build_optimizer(args, model, extra_components):
    """
    根据 mode 构建优化器。

    这里保留最小必要分支：
    - sd_textual_inversion：只训练 text_encoder 的 input embedding；
    - sd_lora：只训练 requires_grad=True 的 LoRA 参数；
    - 其他模式：训练 model.parameters()。
    """
    if args.mode == "sd_textual_inversion":
        params = extra_components["text_encoder"].get_input_embeddings().parameters()

    elif args.mode == "sd_lora":
        params = [p for p in model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise ValueError(
                "No trainable parameters found in sd_lora mode. "
                "Check whether LoRA adapter was correctly added to UNet."
            )

    else:
        params = model.parameters()

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    return optimizer


def build_lr_scheduler(args, optimizer, train_dataloader):
    """
    构建学习率调度器。

    num_update_steps_per_epoch 按梯度累积后的真实 optimizer step 数计算。
    """
    num_update_steps_per_epoch = (
        len(train_dataloader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps

    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    return lr_scheduler


def prepare_core_components_with_accelerator(
    accelerator,
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    lr_scheduler,
):
    """
    把核心训练组件交给 accelerator.prepare()。

    prepare 后的 model/optimizer/dataloader/scheduler 可能会被包装，
    后续必须使用返回的新对象。
    """
    return accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )


def prepare_extra_components_if_needed(accelerator, extra_components):
    """
    处理 mode-specific 额外组件。

    当前主要用于 ldm_ae 的 PatchGAN discriminator 和 d_optimizer。
    """
    discriminator = extra_components.get("discriminator", None)
    d_optimizer = extra_components.get("d_optimizer", None)

    if discriminator is None or d_optimizer is None:
        return extra_components

    discriminator, d_optimizer = accelerator.prepare(
        discriminator,
        d_optimizer,
    )

    extra_components["discriminator"] = discriminator
    extra_components["d_optimizer"] = d_optimizer

    return extra_components


def _to_jsonable_value(value):
    """
    把 argparse 里的值转成 JSON 能稳定保存的格式。

    简单类型直接保存；
    复杂对象转成字符串，避免 json.dump 报错。
    """
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _args_to_plain_dict(args):
    """
    把 Namespace 转成普通 dict。
    """
    return {k: _to_jsonable_value(v) for k, v in vars(args).items()}


def _group_args_for_metadata(args):
    """
    把 args 按实验复盘时真正关心的类别分组。

    注意：
    这里只是改变 metadata 的可读性，
    不改变训练逻辑，不改变 checkpoint，不改变 argparse。
    """
    args_dict = _args_to_plain_dict(args)

    groups = {
        "experiment": [
            "mode",
            "run_mode",
            "output_root",
            "exp_dir",
            "resume_from_checkpoint",
            "seed",
        ],
        "data": [
            "data_root",
            "dataset",
            "data_mode",
            "target_label",
            "exclude_train_nv",
            "resolution",
            "center_crop",
            "random_flip",
            "train_batch_size",
            "eval_batch_size",
            "num_workers",
        ],
        "training": [
            "num_epochs",
            "gradient_accumulation_steps",
            "learning_rate",
            "lr_warmup_steps",
            "adam_beta1",
            "adam_beta2",
            "adam_weight_decay",
            "adam_epsilon",
            "max_grad_norm",
            "mixed_precision",
        ],
        "model": [
            "model_channels",
            "num_res_blocks",
            "channel_mult",
            "attention_resolutions",
            "dropout",
            "use_ema",
            "ema_decay",
        ],
        "diffusion": [
            "num_train_timesteps",
            "beta_schedule",
            "beta_start",
            "beta_end",
            "prediction_type",
            "clip_sample",
        ],
        "stable_diffusion": [
            "pretrained_model_name_or_path",
            "revision",
            "variant",
            "tokenizer_name",
            "placeholder_token",
            "initializer_token",
            "learnable_property",
            "num_vectors",
            "validation_prompt",
            "rank",
        ],
        "evaluation": [
            "eval_epochs",
            "num_fid_samples_train",
            "num_fid_samples_val",
            "enable_per_class_metrics",
            "save_images_epochs",
            "save_model_epochs",
        ],
        "logging": [
            "use_tensorboard",
        ],
    }

    grouped_args = {}

    used_keys = set()

    for group_name, keys in groups.items():
        group_values = {}

        for key in keys:
            if key in args_dict:
                group_values[key] = args_dict[key]
                used_keys.add(key)

        if len(group_values) > 0:
            grouped_args[group_name] = group_values

    # 防止你以后新增 argparse 参数后忘记分类。
    # 这些参数不会丢，只是暂时放进 other。
    other_args = {k: v for k, v in args_dict.items() if k not in used_keys}

    if len(other_args) > 0:
        grouped_args["other"] = other_args

    return grouped_args


def save_training_metadata_on_main_process(
    accelerator,
    args,
    exp_folders,
    class_names,
    train_class_distribution,
    val_class_distribution,
):
    """
    保存当前实验的 metadata。

    只让主进程写文件，避免分布式训练时多个进程同时写同一个 JSON。
    """
    if not accelerator.is_main_process:
        return

    save_json(
        {
            "mode": args.mode,
            "run_mode": args.run_mode,
            "exp_dir": exp_folders["exp_dir"],
            "class_names": class_names,
            "train_class_distribution": train_class_distribution,
            "val_class_distribution": val_class_distribution,
            "args": _group_args_for_metadata(args),
        },
        exp_folders["metadata_json_path"],
    )


def run_train(args):
    """
    主训练入口。

    这个函数本身不负责写具体的 loss / backward / evaluate 逻辑，
    它的作用是把一次训练需要的组件全部搭起来，然后交给 run_training_loop 执行。

    主要流程：
        1. 固定随机种子；
        2. 创建实验目录；
        3. 创建 Accelerator；
        4. 构建数据集和 dataloader；
        5. 构建 noise scheduler；
        6. 根据 mode 获取对应训练逻辑；
        7. 构建模型、EMA、额外组件；
        8. 构建 optimizer 和 lr scheduler；
        9. 交给 accelerator.prepare 包装；
        10. 尝试从 checkpoint 恢复训练；
        11. 进入真正的训练循环；
        12. 保存实验 metadata；
        13. 结束训练。
    """

    # 固定随机种子，尽量提高实验可复现性。
    set_seed(args.seed)

    # 复用旧实验目录或者创建新实验目录
    resume_exp_dir = get_resume_exp_dir(args)
    if resume_exp_dir is not None:
        args.exp_dir = resume_exp_dir
    exp_folders = create_experiment_folders(args)

    # 创建 Hugging Face Accelerate 的 Accelerator。
    # 它负责混合精度、梯度累积、分布式训练、TensorBoard tracker 等。
    accelerator = build_accelerator(args, exp_folders)

    # 如果 args.use_tensorboard=True，则初始化 TensorBoard。
    init_tensorboard_if_needed(args, accelerator, exp_folders)

    # 构建数据相关组件。
    data_bundle = build_datasets_and_loaders(args)

    # 训练用 dataloader。
    # 这个 dataloader 会参与 optimizer 更新。
    train_dataloader = data_bundle["train_dataloader"]

    # 训练集评估 dataloader。
    # 注意：它不是训练用的，不应该 shuffle，也不应该 weighted sampler。
    train_eval_loader = data_bundle["train_eval_loader"]

    # 验证集评估 dataloader。
    val_eval_loader = data_bundle["val_eval_loader"]

    # 类别名，例如 ISIC 2018 的 MEL / NV / BCC 等。
    class_names = data_bundle["class_names"]

    # 训练集和验证集的类别分布，用于记录实验信息和分析数据不平衡。
    train_class_distribution = data_bundle["train_class_distribution"]
    val_class_distribution = data_bundle["val_class_distribution"]

    # 这里单独命名 val_dataloader，是因为 accelerator.prepare 需要包装它。
    # 实际上它和 val_eval_loader 指向的是同一个对象。
    val_dataloader = data_bundle["val_eval_loader"]

    # 类别数量，用于 class-conditional diffusion 或分类条件 embedding。
    num_classes = len(class_names)

    # args.mode 构建 scheduler。
    noise_scheduler = build_noise_scheduler(args)

    # 根据 args.mode 获取当前模式需要的函数集合。
    modes = get_modes(args)

    # 根据 args.mode 构建模型。
    model = build_model(args, num_classes=num_classes)

    # 如果启用 EMA，则复制一份模型作为 EMA 权重。
    ema_model = build_ema_model_if_needed(args, model, accelerator)

    # 构建当前 mode 需要的额外组件。
    # 普通 DDPM 可能没有额外组件；
    # ldm_ae 可能有 discriminator；
    # textual inversion 可能有 tokenizer/text_encoder/vae 等。
    extra_components = modes["build_extra_components"](
        num_classes=num_classes,
        device=accelerator.device,
    )

    # 根据 mode 构建优化器。
    # sd_textual_inversion 只优化 token embedding；
    # sd_lora 只优化 LoRA 参数；
    # 其他模式默认优化 model.parameters()。
    optimizer = build_optimizer(args, model, extra_components)

    # 构建 cosine learning rate scheduler。
    # 注意这里的 max_train_steps 会考虑 gradient_accumulation_steps。
    lr_scheduler = build_lr_scheduler(
        args=args,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
    )

    # 交给 accelerator.prepare 包装核心训练组件。
    # 重要：prepare 之后必须使用返回的新对象，
    # 不能继续使用 prepare 之前的 model / optimizer / dataloader。
    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = prepare_core_components_with_accelerator(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
    )

    # 如果某些 mode 有额外组件，例如 discriminator 和 d_optimizer，
    # 也需要交给 accelerator.prepare 包装。
    extra_components = prepare_extra_components_if_needed(
        accelerator=accelerator,
        extra_components=extra_components,
    )

    # 尝试从 checkpoint 恢复训练。
    start_epoch, global_step, best_metric = (
        resume_training_from_checkpoint_if_available(
            args=args,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            modes=modes,
            extra_components=extra_components,
        )
    )

    # 保存实验 metadata。
    save_training_metadata_on_main_process(
        accelerator=accelerator,
        args=args,
        exp_folders=exp_folders,
        class_names=class_names,
        train_class_distribution=train_class_distribution,
        val_class_distribution=val_class_distribution,
    )

    # 真正进入训练循环。
    # loss、backward、optimizer.step、eval、save checkpoint 等核心逻辑
    # 应该主要在 run_training_loop 里面。
    run_training_loop(
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

    # 通知 accelerator 当前训练结束。
    accelerator.end_training()
