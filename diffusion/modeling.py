import torch
import torch.nn as nn
from diffusers import (
    DDPMPipeline,
    DDIMPipeline,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DModel,
    AutoencoderKL,
    UNet2DConditionModel,
)
from peft import LoraConfig


class ClassConditionedUNet2DConditionModel(nn.Module):
    """
    用 class label 构造 cross-attention 条件的最小 wrapper。

    输入:
        sample: noisy latents, shape = [B, C, H, W]
        timesteps: diffusion timesteps
        class_labels: shape = [B]

    内部流程:
        class_labels
        -> nn.Embedding(num_classes, cross_attention_dim)
        -> unsqueeze(1)
        -> encoder_hidden_states, shape = [B, 1, cross_attention_dim]
        -> UNet2DConditionModel(..., encoder_hidden_states=...)

    这样做的好处:
        1. class_condition_embedding 会包含在 model.parameters() 里；
        2. optimizer 会自动优化它；
        3. checkpoint 保存 model.state_dict() 时会自动保存它；
        4. EMA 模型也会自动包含它。
    """

    def __init__(
        self,
        unet,
        num_classes,
        cross_attention_dim,
    ):
        super().__init__()
        self.unet = unet
        self.class_condition_embedding = nn.Embedding(
            num_classes,
            cross_attention_dim,
        )

        # 暴露 config，兼容你现有代码里 model.config.in_channels 等访问方式。
        self.config = unet.config

    @property
    def dtype(self):
        return self.unet.dtype

    def forward(
        self,
        sample,
        timesteps,
        class_labels=None,
        encoder_hidden_states=None,
        **kwargs,
    ):
        # 如果外部没有直接传 encoder_hidden_states，就用 class_labels 构造。
        if encoder_hidden_states is None:
            if class_labels is None:
                raise ValueError(
                    "cross_attention conditioning requires class_labels or encoder_hidden_states."
                )

            # class_labels: [B]
            # condition: [B, cross_attention_dim]
            condition = self.class_condition_embedding(class_labels.long())

            # encoder_hidden_states: [B, 1, cross_attention_dim]
            # sequence_length=1 表示每张图只有一个类别条件 token。
            encoder_hidden_states = condition.unsqueeze(1)

        return self.unet(
            sample,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


def _build_sd_textual_inversion_placeholder():
    """
    Stable Diffusion textual inversion 模式的占位模型。

    注意：
        textual inversion 真正训练的是 text_encoder 的 input embedding，
        不是这里返回的模型。

    为什么返回 nn.Identity():
        你的主训练流程要求 build_model(args, num_classes) 必须返回一个 model，
        所以这里用 Identity 作为占位，保持外部训练流程统一。
    """
    return nn.Identity()


def _build_sd_lora_unet(args):
    """
    构建 Stable Diffusion LoRA fine-tuning 使用的 UNet。

    训练逻辑：
        1. 从预训练 Stable Diffusion 中加载 UNet；
        2. 冻结原始 UNet 参数；
        3. 注入 LoRA adapter；
        4. 只训练 LoRA 参数。

    这部分不负责 optimizer 构建。
    optimizer 会在 runtime.py 的 build_optimizer(...) 里只取 requires_grad=True 的参数。
    """
    if LoraConfig is None:
        raise ImportError(
            "mode='sd_lora' requires peft. Please install it with: pip install peft"
        )

    model = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # LoRA 的核心：冻结 base UNet。
    # 否则你就不是 LoRA 微调，而是在训练整个 UNet。
    model.requires_grad_(False)

    # Diffusers / PEFT 常见写法：
    # init_lora_weights 可以是 True，也可以是 "gaussian"。
    init_lora_weights = (
        "gaussian" if getattr(args, "lora_init", "gaussian") == "gaussian" else True
    )

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
    )

    # 给 UNet 注入 LoRA adapter。
    # 注入后，LoRA 参数会自动变成 requires_grad=True。
    model.add_adapter(unet_lora_config)

    model.train()

    # 梯度检查点：省显存，但会稍微增加计算时间。
    if getattr(args, "sd_enable_gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    # xFormers memory efficient attention：进一步省显存。
    # 需要你的环境正确安装 xformers。
    if getattr(args, "sd_enable_xformers", False):
        model.enable_xformers_memory_efficient_attention()

    # 打印可训练参数比例，防止误把整个 UNet 都训练了。
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"[sd_lora] trainable params: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / total_params:.4f}%)"
    )

    return model


def _build_sd_full_unet(args):
    """
    构建 Stable Diffusion Full UNet fine-tuning 使用的 UNet。

    训练逻辑：
        1. 从预训练 Stable Diffusion 中加载 UNet；
        2. 不冻结参数；
        3. 训练整个 UNet。

    注意：
        这个模式显存压力明显大于 sd_lora。
    """
    model = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # Full fine-tuning：训练整个 UNet。
    model.train()

    if getattr(args, "sd_enable_gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    if getattr(args, "sd_enable_xformers", False):
        model.enable_xformers_memory_efficient_attention()

    return model


def _build_autoencoder_kl(args):
    """
    构建 ldm_ae 模式使用的 AutoencoderKL。

    这个模式不是 diffusion denoising 训练，
    而是先训练一个 autoencoder，把 RGB 图像压缩到 latent space。

    后续 latent_ddpm 会在这个 latent space 里训练扩散模型。
    """
    model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=tuple(
            ["DownEncoderBlock2D"] * len(args.ae_block_out_channels)
        ),
        up_block_types=tuple(["UpDecoderBlock2D"] * len(args.ae_block_out_channels)),
        block_out_channels=tuple(args.ae_block_out_channels),
        layers_per_block=args.ae_layers_per_block,
        act_fn="silu",
        latent_channels=args.ae_latent_channels,
        norm_num_groups=args.ae_norm_num_groups,
        sample_size=args.resolution,
        scaling_factor=args.ae_scaling_factor,
        force_upcast=True,
        use_quant_conv=True,
        use_post_quant_conv=True,
        mid_block_add_attention=args.ae_mid_block_add_attention,
    )

    # slicing / tiling 是 AutoencoderKL 的显存优化选项。
    if args.ae_use_slicing:
        model.enable_slicing()

    if args.ae_use_tiling:
        model.enable_tiling()

    return model


def _get_num_class_embeds(args, num_classes):
    """
    根据是否使用类别条件，决定 UNet2DModel 的 num_class_embeds。

    返回：
        None:
            不使用类别条件。

        num_classes:
            普通 class-conditional diffusion。

        num_classes + 1:
            CFG 模式。
            额外的 1 个类别用于 null condition，也就是“空条件”。
    """
    if not args.use_class_conditioning:
        return None

    if args.mode == "cfg":
        return num_classes + 1

    return num_classes


def _build_latent_ddpm_unet(args, num_classes):
    """
    构建 latent_ddpm 模式使用的 UNet。

    latent_ddpm 和普通 pixel-space DDPM 的区别：
        普通 DDPM:
            输入输出是 RGB 图像，通道数通常是 3。

        latent_ddpm:
            输入输出是 AutoencoderKL 的 latent，
            通道数应该等于 args.ae_latent_channels。
    """
    in_out_channels = args.ae_latent_channels
    latent_sample_size = args.resolution // args.ae_downsample_factor

    # 方式 1：使用 cross-attention 做类别条件注入。
    # 这里不能直接把 class_labels 传给 UNet2DConditionModel，
    # 所以外面包了一层 ClassConditionedUNet2DConditionModel。
    if args.use_cross_attention_conditioning:
        unet = UNet2DConditionModel(
            sample_size=latent_sample_size,
            in_channels=in_out_channels,
            out_channels=in_out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
            cross_attention_dim=args.cross_attention_dim,
            attention_head_dim=args.attention_head_dim,
            num_class_embeds=None,
            resnet_time_scale_shift=args.resnet_time_scale_shift,
        )

        return ClassConditionedUNet2DConditionModel(
            unet=unet,
            num_classes=num_classes,
            cross_attention_dim=args.cross_attention_dim,
        )

    # 方式 2：使用 UNet2DModel 自带的 class embedding。
    num_class_embeds = _get_num_class_embeds(args, num_classes)

    return UNet2DModel(
        sample_size=latent_sample_size,
        in_channels=in_out_channels,
        out_channels=in_out_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=num_class_embeds,
        resnet_time_scale_shift=args.resnet_time_scale_shift,
    )


def _build_pixel_ddpm_unet(args, num_classes):
    """
    构建普通 pixel-space DDPM / CFG / CG 使用的 UNet。

    输入输出：
        sample shape = [B, 3, H, W]

    注意：
        这里仍然保留你原来的 resolution 分支：
            - resolution > 128 使用更深的 UNet；
            - resolution <= 128 使用默认 UNet。
    """
    num_class_embeds = _get_num_class_embeds(args, num_classes)

    return UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=num_class_embeds,
        resnet_time_scale_shift=args.resnet_time_scale_shift,
    )


def build_model(args, num_classes):
    """
    根据 args.mode 构建当前实验的主模型。

    这个函数只负责：
        1. 判断当前 mode；
        2. 调用对应的模型构建函数；
        3. 返回主模型。

    这个函数不负责：
        1. 构建 optimizer；
        2. 构建 noise scheduler；
        3. 加载 checkpoint；
        4. accelerator.prepare；
        5. 训练循环；
        6. loss 计算。

    这样拆分的好处：
        build_model 本身变成一个清晰的路由函数。
        真正复杂的模型构建细节被拆到 _build_xxx 函数里，
        以后你排查某个 mode 的 bug，不需要在一个巨大函数里来回跳。
    """

    if args.mode == "sd_textual_inversion":
        return _build_sd_textual_inversion_placeholder()

    if args.mode == "sd_lora":
        return _build_sd_lora_unet(args)

    if args.mode == "sd_full":
        return _build_sd_full_unet(args)

    if args.mode == "ldm_ae":
        return _build_autoencoder_kl(args)

    if args.mode == "latent_ddpm":
        return _build_latent_ddpm_unet(args, num_classes)

    # 其他模式默认走 pixel-space UNet。
    # 例如普通 ddpm / cfg / cg 等。
    return _build_pixel_ddpm_unet(args, num_classes)


def build_noise_scheduler(args):
    """
    根据训练 mode 构建 noise scheduler。

    返回值：
        - sd_textual_inversion / sd_full / sd_lora:
            使用对应 Stable Diffusion mode 自己的 scheduler 构建函数。
            这些模式通常依赖预训练模型配置，不应该手写普通 DDPM scheduler。

        - ldm_ae:
            返回 None。
            因为 autoencoder 训练不是 diffusion denoising 训练，不需要 noise scheduler。

        - 其他 DDPM / CFG / CG / latent_ddpm 模式:
            使用 DDPMScheduler。
            当前代码采用 epsilon prediction，也就是让模型预测加入的噪声。
    """

    # Stable Diffusion textual inversion：
    # scheduler 配置应尽量跟预训练 Stable Diffusion 保持一致。
    if args.mode == "sd_textual_inversion":
        from .modes.sd_textual_inversion import build_sd_full_noise_scheduler

        return build_sd_full_noise_scheduler(args)

    # Stable Diffusion full UNet fine-tuning：
    # 同样复用 SD 相关 mode 的 scheduler 构建逻辑。
    if args.mode == "sd_full":
        from .modes.sd_full import build_sd_full_noise_scheduler

        return build_sd_full_noise_scheduler(args)

    # Stable Diffusion LoRA：
    # LoRA 只改变可训练参数范围，不改变 diffusion scheduler 的基本需求。
    if args.mode == "sd_lora":
        from .modes.sd_lora import build_sd_lora_noise_scheduler

        return build_sd_lora_noise_scheduler(args)

    # Autoencoder 模式不是扩散去噪训练，因此不需要 noise scheduler。
    if args.mode == "ldm_ae":
        return None

    # 普通 DDPM / CFG / CG / latent_ddpm 等模式：
    # 训练时用 DDPMScheduler 负责前向加噪时间表。
    return DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type="epsilon",
    )


def build_sampling_scheduler(noise_scheduler, use_ddim_sampling=False):
    # 采样阶段可以在 DDPM / DDIM 间切换
    # from_config(...) 能直接复用已有 scheduler 的配置
    if use_ddim_sampling:
        return DDIMScheduler.from_config(noise_scheduler.config)
    return DDPMScheduler.from_config(noise_scheduler.config)


@torch.no_grad()
def run_sampling_loop(
    model,
    sampling_scheduler,
    device,
    resolution,
    batch_size,
    num_inference_steps,
    generator,
    predict_fn,
    ddim_eta=0.0,
    return_pil_safe_uint8=True,
):
    # 先让 scheduler 知道本次推理要跑多少步
    try:
        sampling_scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        # 某些版本的 scheduler.set_timesteps 不接收 device 参数
        sampling_scheduler.set_timesteps(num_inference_steps)

    # 从标准高斯噪声开始采样
    sample = torch.randn(
        (batch_size, model.config.in_channels, resolution, resolution),
        generator=generator,
        device=device,
    )

    # 按照 scheduler 给出的 timesteps 逐步去噪
    for t in sampling_scheduler.timesteps:
        # predict_fn 是外部传进来的“预测噪声函数”
        # 不同模式（DDPM / CFG / CG）会传入不同逻辑
        model_output = predict_fn(sample, t)

        if isinstance(sampling_scheduler, DDIMScheduler):
            # DDIM 支持 eta 参数控制随机性
            step_output = sampling_scheduler.step(
                model_output,
                t,
                sample,
                eta=ddim_eta,
                generator=generator,
            )
        else:
            step_output = sampling_scheduler.step(
                model_output,
                t,
                sample,
                generator=generator,
            )

        # prev_sample 表示从 x_t 更新到 x_{t-1}
        sample = step_output.prev_sample

    if return_pil_safe_uint8:
        # 将 [-1, 1] 范围的张量映射到 [0, 255] 的 uint8
        # 这样更适合保存成 PNG / JPG
        x = ((sample.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return x

    return sample


def build_save_pipeline(unet, noise_scheduler, use_ddim_sampling):
    # 这里不是训练用 pipeline，而是为了 save_pretrained(...) 方便保存
    if use_ddim_sampling:
        save_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        return DDIMPipeline(unet=unet, scheduler=save_scheduler)

    return DDPMPipeline(unet=unet, scheduler=noise_scheduler)
