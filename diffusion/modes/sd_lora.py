# Stable Diffusion LoRA fine-tuning：
#   - 冻结 VAE
#   - 冻结 CLIP Text Encoder
#   - 冻结 UNet 原始参数
#   - 只训练 UNet 里的 LoRA adapter 参数

import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

ISIC_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

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
    根据 mixed_precision 设置冻结模块 dtype。

    obj 可以是 args，也可以是 accelerator。
    """
    mixed_precision = getattr(obj, "mixed_precision", "no")

    if mixed_precision == "fp16":
        return torch.float16

    if mixed_precision == "bf16":
        return torch.bfloat16

    return torch.float32


def _labels_to_prompts(labels, class_names):
    """
    把类别 id 转成 Stable Diffusion 的文本 prompt。
    """
    prompts = []

    for label in labels.detach().cpu().tolist():
        class_name = class_names[int(label)]
        prompts.append(ISIC_PROMPTS[class_name])

    return prompts


@torch.no_grad()
def _encode_prompts(tokenizer, text_encoder, prompts, device):
    """
    prompt -> CLIP text encoder hidden states
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
    image [-1, 1] -> SD latent
    """
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


@torch.no_grad()
def _decode_latents_to_images(vae, latents):
    """
    SD latent -> image [-1, 1]
    """
    latents = latents / vae.config.scaling_factor

    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=vae_dtype)

    images = vae.decode(latents).sample
    return images.clamp(-1.0, 1.0)


def build_sd_lora(args):
    def build_extra_components(num_classes, device):
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

        # LoRA 只训练 UNet adapter。
        # VAE 和 Text Encoder 都冻结。
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        text_encoder.eval()
        vae.eval()

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
        Stable Diffusion LoRA 单步训练。

        注意：
        loss 和 sd_full 一样。
        区别在于 optimizer 只会更新 LoRA 参数。
        """
        vae = extra_components["vae"]
        text_encoder = extra_components["text_encoder"]
        tokenizer = extra_components["tokenizer"]
        class_names = extra_components["class_names"]

        weight_dtype = _get_weight_dtype(accelerator)

        clean_images = batch["input"].to(
            accelerator.device,
            dtype=weight_dtype,
        )
        labels = batch["label"].to(accelerator.device).long()

        prompts = _labels_to_prompts(labels, class_names)

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

        noise = torch.randn_like(latents)

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            dtype=torch.long,
        )

        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

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
        训练过程中保存可视化样本。

        当前仍然是 text-to-image。
        img2img / SDEdit 下一步再单独加。
        """
        vae = extra_components["vae"]
        text_encoder = extra_components["text_encoder"]
        tokenizer = extra_components["tokenizer"]
        class_names = extra_components["class_names"]

        if class_labels is None:
            class_labels = torch.arange(
                batch_size,
                device=device,
                dtype=torch.long,
            ) % len(class_names)
        else:
            class_labels = class_labels.to(device).long()

        prompts = _labels_to_prompts(class_labels, class_names)

        encoder_hidden_states = _encode_prompts(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            device=device,
        )

        latent_channels = int(model.config.in_channels)
        latent_resolution = int(resolution // 8)

        model_dtype = next(model.parameters()).dtype

        latents = torch.randn(
            (batch_size, latent_channels, latent_resolution, latent_resolution),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        sampling_scheduler.set_timesteps(num_inference_steps, device=device)

        for t in sampling_scheduler.timesteps:
            noise_pred = model(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            latents = sampling_scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            ).prev_sample

        images = _decode_latents_to_images(vae, latents)

        if return_pil_safe_uint8:
            return ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)

        images = images.detach().cpu()

        del latents
        del encoder_hidden_states

        return images

    def checkpoint_extra_state(extra_components):
        return {
            "sd_lora_config": {
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
        }

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
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
    使用预训练 Stable Diffusion checkpoint 自带 scheduler 配置。
    """
    pretrained_path = getattr(args, "pretrained_model_name_or_path", None)
    if pretrained_path is None:
        raise ValueError("sd_lora requires --pretrained_model_name_or_path")

    return DDPMScheduler.from_pretrained(
        pretrained_path,
        subfolder="scheduler",
    )
