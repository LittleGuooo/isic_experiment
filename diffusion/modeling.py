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


def build_model(args, num_classes):
    if args.mode == "sd_textual_inversion":
        return nn.Identity()

    # Stable Diffusion Full UNet fine-tuning
    if args.mode == "sd_full":
        model = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
        )

        # Full fine-tuning：训练 UNet 全部参数
        model.train()

        if getattr(args, "sd_enable_gradient_checkpointing", True):
            model.enable_gradient_checkpointing()

        if getattr(args, "sd_enable_xformers", False):
            model.enable_xformers_memory_efficient_attention()

        return model

    # ldm_ae 模式
    if args.mode == "ldm_ae":
        model = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=tuple(
                ["DownEncoderBlock2D"] * len(args.ae_block_out_channels)
            ),
            up_block_types=tuple(
                ["UpDecoderBlock2D"] * len(args.ae_block_out_channels)
            ),
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

        # 可选：节省显存
        if args.ae_use_slicing:
            model.enable_slicing()

        if args.ae_use_tiling:
            model.enable_tiling()

        return model

    # num_class_embeds 控制类别嵌入表大小
    # unconditional 时传 None
    num_class_embeds = None
    if args.use_class_conditioning:
        if args.mode == "cfg":
            # CFG 会额外预留 1 个“空条件类别”（null class）
            # 训练时有一部分样本会把真实标签替换成它
            num_class_embeds = num_classes + 1
        else:
            num_class_embeds = num_classes

    if args.mode == "latent_ddpm":
        # latent_ddpm 模式下，UNet 的输入输出通道数应该和 AE 的 latent_channels 一致。
        in_out_channels = args.ae_latent_channels
        latent_sample_size = args.resolution // args.ae_downsample_factor

        # 方式1：cross-attention 类别条件注入
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

        # 方式2：保持原来的类别条件注入方式
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

    # 构造其他扩散模型的unet
    if args.resolution > 128:
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

    # 默认分别率128 * 128时
    return UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=(
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
        ),
        num_class_embeds=num_class_embeds,
        resnet_time_scale_shift=args.resnet_time_scale_shift,
    )


def build_noise_scheduler(args):
    # Stable Diffusion full fine-tuning 使用预训练模型自带 scheduler 配置
    if args.mode == "sd_full":
        from .modes.sd_full import build_sd_full_noise_scheduler

        return build_sd_full_noise_scheduler(args)

    # ldm_ae 不需要 diffusion noise scheduler
    if args.mode == "ldm_ae":
        return None

    # 训练阶段使用 DDPMScheduler 管理前向加噪和反向去噪的时间表
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
