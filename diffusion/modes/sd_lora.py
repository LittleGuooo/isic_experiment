import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ISIC 2018 的固定类别顺序。
# batch["label"] 中的整数标签必须与这个顺序一致。
ISIC_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# 使用类别级 prompt 作为 Stable Diffusion 的文本条件。
# 这里不是图像级 caption，因此同一类别的图像会共享同一个文本描述。
ISIC_PROMPTS = {
    "MEL": "a dermoscopic image of melanoma",
    "NV": "a dermoscopic image of melanocytic nevus",
    "BCC": "a dermoscopic image of basal cell carcinoma",
    "AKIEC": "a dermoscopic image of actinic keratosis or intraepithelial carcinoma",
    "BKL": "a dermoscopic image of benign keratosis-like lesion",
    "DF": "a dermoscopic image of dermatofibroma",
    "VASC": "a dermoscopic image of vascular lesion",
}


def _get_weight_dtype(obj):
    """
    根据 mixed_precision 返回冻结模块使用的 dtype。

    """
    mixed_precision = getattr(obj, "mixed_precision", "no")

    if mixed_precision == "fp16":
        return torch.float16

    if mixed_precision == "bf16":
        return torch.bfloat16

    return torch.float32


def _get_frozen_components(extra_components):
    """
    从 extra_components 中取出冻结的 Stable Diffusion 组件。

    单独封装这个重复操作，避免 train_step 和 sample_images 中散落多次字典索引。
    """
    return (
        extra_components["vae"],
        extra_components["text_encoder"],
        extra_components["tokenizer"],
        extra_components["class_names"],
    )


def _labels_to_prompts(labels, class_names):
    """
    把类别 id 转换为 Stable Diffusion 文本 prompt。

    labels 会先移动到 CPU，再转成 Python list。这样做便于索引类别名和 prompt 字典；
    代价很小，因为这里只处理每个 batch 的少量整数标签。
    """
    prompts = []

    for label in labels.detach().cpu().tolist():
        class_name = class_names[int(label)]
        prompts.append(ISIC_PROMPTS[class_name])

    return prompts


@torch.no_grad()
def _encode_prompts(tokenizer, text_encoder, prompts, device):
    """
    将 prompt 编码为 CLIP hidden states。

    Text Encoder 已被冻结，因此不需要保存反向传播图。
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    encoder_hidden_states = text_encoder(input_ids)[0]

    return encoder_hidden_states


@torch.no_grad()
def _encode_images_to_latents(vae, images):
    """
    将归一化到 [-1, 1] 的图像编码为 Stable Diffusion latent。

    VAE 已被冻结，因此这里只做前向计算。scaling_factor 必须保留，
    因为预训练 UNet 接收的是缩放后的 latent 分布。
    """
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


@torch.no_grad()
def _decode_latents_to_images(vae, latents):
    """
    将 Stable Diffusion latent 解码为 [-1, 1] 范围内的图像。
    """
    # 编码时乘过 scaling_factor，解码前必须恢复原始尺度。
    latents = latents / vae.config.scaling_factor

    # VAE 可能已被转成 fp16 或 bf16，输入 latent 要与其 dtype 对齐。
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=vae_dtype)

    images = vae.decode(latents).sample
    return images.clamp(-1.0, 1.0)


def _build_default_class_labels(batch_size, device, num_classes):
    """
    为可视化采样构造默认类别标签。

    默认按类别循环取值，保证一个 batch 中尽量覆盖不同 ISIC 类别。
    """
    return (
        torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        )
        % num_classes
    )


def _build_sd_lora_config(args):
    """
    整理写入 checkpoint 的 LoRA 配置。

    这里只保存模式配置，不负责保存或恢复 adapter 权重；
    权重是否完整保存取决于外层 checkpoint 实现。
    """
    return {
        "pretrained_model_name_or_path": str(
            getattr(args, "pretrained_model_name_or_path", "")
        ),
        "train_unet_lora": True,
        "train_unet_full": False,
        "train_text_encoder": False,
        "train_vae": False,
        "class_names": ISIC_CLASS_NAMES,
        "lora_rank": int(getattr(args, "lora_rank", 16)),
        "lora_alpha": int(getattr(args, "lora_alpha", 16)),
        "lora_dropout": float(getattr(args, "lora_dropout", 0.0)),
        "lora_target_modules": list(
            getattr(
                args,
                "lora_target_modules",
                ["to_q", "to_k", "to_v", "to_out.0"],
            )
        ),
    }


def build_sd_lora(args):
    """
    构建 sd_lora 模式需要的函数集合。

    """

    def build_extra_components(num_classes, device):
        """
        加载并冻结 Stable Diffusion 的 tokenizer、Text Encoder 和 VAE。

        num_classes 由统一接口传入。本模式当前固定使用 ISIC_CLASS_NAMES，
        因此不会根据 num_classes 动态构造类别名。
        """
        pretrained_path = getattr(args, "pretrained_model_name_or_path", None)
        if pretrained_path is None:
            raise ValueError("mode='sd_lora' requires --pretrained_model_name_or_path.")

        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_path,
            subfolder="tokenizer",
        )

        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_path,
            subfolder="text_encoder",
        )

        vae = AutoencoderKL.from_pretrained(
            pretrained_path,
            subfolder="vae",
        )

        # LoRA 只训练 UNet 中的 adapter。
        # Text Encoder 和 VAE 只负责提供条件与 latent，不参与参数更新。
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        text_encoder.eval()
        vae.eval()

        # 冻结模块转为训练所需精度，减少显存占用。
        weight_dtype = _get_weight_dtype(args)
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "class_names": ISIC_CLASS_NAMES,
        }

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        """
        执行一个 Stable Diffusion LoRA 训练 step。

        loss 与全量微调 sd_full 相同：UNet 预测加入 latent 的噪声，
        再与真实噪声计算 MSE。LoRA 模式的区别不在 loss，而在于 optimizer
        只更新 UNet 中 requires_grad=True 的 adapter 参数。
        """
        vae, text_encoder, tokenizer, class_names = _get_frozen_components(
            extra_components
        )

        weight_dtype = _get_weight_dtype(accelerator)

        # 数据集应提前把图像归一化到 [-1, 1]。
        clean_images = batch["input"].to(
            accelerator.device,
            dtype=weight_dtype,
        )
        labels = batch["label"].to(accelerator.device).long()

        # 当前使用类别级文本条件，而不是每张图像单独提供 caption。
        prompts = _labels_to_prompts(labels, class_names)

        # 这两个组件已经冻结。显式 no_grad 可以减少显存占用，
        # 也避免误把 VAE 或 Text Encoder 纳入反向传播图。
        with torch.no_grad():
            encoder_hidden_states = _encode_prompts(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompts=prompts,
                device=accelerator.device,
            )

            latents = _encode_images_to_latents(
                vae=vae,
                images=clean_images,
            )

        # 为每个 latent 采样独立高斯噪声和扩散时间步。
        noise = torch.randn_like(latents)

        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
            dtype=torch.long,
        )

        # 前向扩散：根据时间步把噪声加入干净 latent。
        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        # UNet 接收 noisy latent、时间步和 CLIP 文本条件，预测噪声残差。
        noise_pred = model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        loss = F.mse_loss(
            noise_pred.float(),
            noise.float(),
            reduction="mean",
        )

        # aux 只用于日志，不参与反向传播。
        aux = {
            "total_loss": float(loss.detach().item()),
            "sd_lora_loss": float(loss.detach().item()),
            "latent_abs_mean": float(latents.detach().abs().mean().item()),
            "latent_std": float(latents.detach().std().item()),
        }

        return loss, aux

    @torch.no_grad()
    def sample_images(
        model,
        sampling_scheduler,
        device,
        resolution,
        batch_size,
        num_inference_steps,
        generator,
        class_labels=None,
        extra_components=None,
        return_pil_safe_uint8=False,
        **kwargs,
    ):
        """
        训练过程中保存 text-to-image 可视化样本。

        这是简化采样器：根据类别 prompt 从随机 latent 开始逐步去噪。
        当前没有实现 img2img、SDEdit 或 classifier-free guidance。
        """
        vae, text_encoder, tokenizer, class_names = _get_frozen_components(
            extra_components
        )

        # 未显式指定类别时，让一个 batch 尽量覆盖多个类别。
        if class_labels is None:
            class_labels = _build_default_class_labels(
                batch_size=batch_size,
                device=device,
                num_classes=len(class_names),
            )
        else:
            class_labels = class_labels.to(device).long()

        prompts = _labels_to_prompts(class_labels, class_names)

        encoder_hidden_states = _encode_prompts(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            device=device,
        )

        # Stable Diffusion 常见 VAE 下采样倍率为 8，因此 latent 空间尺寸为 resolution // 8。
        latent_channels = int(model.config.in_channels)
        latent_resolution = int(resolution // 8)

        # 初始噪声与 UNet 参数使用同一 dtype，避免无意义的类型转换。
        model_dtype = next(model.parameters()).dtype
        latents = torch.randn(
            (batch_size, latent_channels, latent_resolution, latent_resolution),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        sampling_scheduler.set_timesteps(num_inference_steps, device=device)

        # 反向扩散：每一步预测噪声，再由 scheduler 计算前一个 latent。
        for timestep in sampling_scheduler.timesteps:
            noise_pred = model(
                latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            latents = sampling_scheduler.step(
                noise_pred,
                timestep,
                latents,
                generator=generator,
            ).prev_sample

        images = _decode_latents_to_images(vae, latents)

        # 某些保存函数希望直接拿到 [0, 255] 的 uint8 tensor。
        if return_pil_safe_uint8:
            return ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)

        images = images.detach().cpu()

        del latents
        del encoder_hidden_states

        return images

    def checkpoint_extra_state(extra_components):
        """
        返回写入 checkpoint 的 LoRA 模式配置。

        extra_components 保留在统一接口中，但当前配置生成不需要读取它。
        """
        return {
            "sd_lora_config": _build_sd_lora_config(args),
        }

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        """
        加载 LoRA 模式额外状态。

        当前没有额外状态需要恢复，因此保留空实现以兼容统一 checkpoint 接口。
        """
        return None

    return {
        "name": "sd_lora",
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }


def build_sd_lora_noise_scheduler(args):
    """
    加载预训练 Stable Diffusion checkpoint 自带的训练 scheduler 配置。
    """
    pretrained_path = getattr(args, "pretrained_model_name_or_path", None)
    if pretrained_path is None:
        raise ValueError("sd_lora requires --pretrained_model_name_or_path")

    return DDPMScheduler.from_pretrained(
        pretrained_path,
        subfolder="scheduler",
    )
