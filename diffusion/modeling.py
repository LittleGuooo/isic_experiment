import torch
from diffusers import (
    DDPMPipeline,
    DDIMPipeline,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DModel,
    AutoencoderKL,
)


def build_model(args, num_classes):
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
        # latent_ddpm 模式下，UNet 的输入输出通道数应该和 AE 的 latent_channels 一致
        in_out_channels = args.ae_latent_channels
        return UNet2DModel(
            sample_size=args.resolution // args.ae_downsample_factor,
            in_channels=in_out_channels,
            out_channels=in_out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            num_class_embeds=num_class_embeds,
            resnet_time_scale_shift=args.resnet_time_scale_shift,
        )

    # 这里构造的是 diffusers 的 UNet2DModel
    # 输入输出都是 3 通道图像噪声 / 预测噪声
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
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=num_class_embeds,
        resnet_time_scale_shift=args.resnet_time_scale_shift,
    )


def build_noise_scheduler(args):
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
