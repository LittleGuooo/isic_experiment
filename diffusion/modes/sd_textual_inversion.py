import os

import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

ISIC_CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]


def build_sd_textual_inversion(args):
    def before_optimizer_step(extra_components, accelerator):
        text_encoder = extra_components["text_encoder"]
        placeholder_token_ids = extra_components["placeholder_token_ids"]

        embedding_weight = text_encoder.get_input_embeddings().weight
        grads = embedding_weight.grad

        if grads is None:
            return

        mask = torch.ones(
            grads.shape[0],
            dtype=torch.bool,
            device=grads.device,
        )
        mask[placeholder_token_ids] = False
        grads[mask] = 0

    def after_optimizer_step(extra_components, accelerator):
        text_encoder = extra_components["text_encoder"]
        placeholder_token_ids = extra_components["placeholder_token_ids"]
        orig_embeds_params = extra_components["orig_embeds_params"]

        embedding_weight = text_encoder.get_input_embeddings().weight

        index_no_updates = torch.ones(
            embedding_weight.shape[0],
            dtype=torch.bool,
            device=embedding_weight.device,
        )

        placeholder_token_ids = torch.as_tensor(
            placeholder_token_ids,
            dtype=torch.long,
            device=embedding_weight.device,
        )

        index_no_updates[placeholder_token_ids] = False

        orig_embeds_params = orig_embeds_params.to(
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )

        with torch.no_grad():
            embedding_weight[index_no_updates] = orig_embeds_params[index_no_updates]

    def build_extra_components(num_classes, device):

        pretrained_path = args.pretrained_model_name_or_path

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

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_path,
            subfolder="unet",
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
        )

        placeholder_tokens = args.ti_placeholder_tokens
        initializer_tokens = args.ti_initializer_tokens

        tokenizer.add_tokens(placeholder_tokens)

        # 新增 placeholder token 后，需要扩展 CLIP text encoder 的 embedding 表。
        # mean_resizing=False 可以关闭 transformers 的均值/协方差初始化提示；
        # 后面我们会手动用 initializer_token 的 embedding 覆盖 placeholder embedding。
        text_encoder.resize_token_embeddings(
            len(tokenizer),
            mean_resizing=False,
        )

        token_embeds = text_encoder.get_input_embeddings().weight.data

        # 保存原始 embedding 用于恢复非placeholder，避免原词表被污染
        orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

        placeholder_token_ids = []

        for placeholder_token, initializer_token in zip(
            placeholder_tokens,
            initializer_tokens,
        ):

            placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

            initializer_token_id = tokenizer.encode(
                initializer_token,
                add_special_tokens=False,
            )[0]

            token_embeds[placeholder_token_id] = token_embeds[
                initializer_token_id
            ].clone()

            placeholder_token_ids.append(placeholder_token_id)

        vae.requires_grad_(False)
        unet.requires_grad_(False)

        text_encoder.requires_grad_(False)

        text_encoder.get_input_embeddings().weight.requires_grad = True

        vae.eval()
        unet.eval()

        text_encoder.train()

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder.to(device),
            "vae": vae.to(device),
            "unet": unet.to(device),
            "noise_scheduler": noise_scheduler,
            "placeholder_token_ids": placeholder_token_ids,
            "placeholder_tokens": placeholder_tokens,
            "orig_embeds_params": orig_embeds_params,
        }

    def train_step(
        model,
        noise_scheduler,
        batch,
        accelerator,
        extra_components,
    ):

        tokenizer = extra_components["tokenizer"]
        text_encoder = extra_components["text_encoder"]
        vae = extra_components["vae"]
        unet = extra_components["unet"]

        placeholder_tokens = extra_components["placeholder_tokens"]
        placeholder_token_ids = extra_components["placeholder_token_ids"]

        images = batch["input"].to(accelerator.device)
        labels = batch["label"].to(accelerator.device)

        prompts = []

        for label in labels.detach().cpu().tolist():
            token = placeholder_tokens[int(label)]
            prompts.append(f"a dermoscopic image of {token}")

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(accelerator.device)

        encoder_hidden_states = text_encoder(input_ids)[0]

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()

        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        loss = F.mse_loss(model_pred.float(), noise.float())

        aux = {
            "ti_loss": float(loss.detach().item()),
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
        class_labels,
        extra_components,
        return_pil_safe_uint8=True,
        **kwargs,
    ):

        tokenizer = extra_components["tokenizer"]
        text_encoder = extra_components["text_encoder"]
        vae = extra_components["vae"]
        unet = extra_components["unet"]

        placeholder_tokens = extra_components["placeholder_tokens"]

        # 准备类别标签
        if class_labels is None:
            num_classes = len(placeholder_tokens)
            class_labels = (
                torch.arange(
                    batch_size,
                    device=device,
                    dtype=torch.long,
                )
                % num_classes
            )
        else:
            class_labels = class_labels.to(device).long()

        prompts = []

        for label in class_labels.detach().cpu().tolist():
            token = placeholder_tokens[int(label)]
            prompts.append(f"a dermoscopic image of {token}")

        # 修正 Stable Diffusion pipeline 期望的 scheduler 配置，避免 warning
        from diffusers.configuration_utils import FrozenDict

        scheduler_config = dict(sampling_scheduler.config)
        scheduler_config["steps_offset"] = 1
        scheduler_config["clip_sample"] = False
        sampling_scheduler._internal_dict = FrozenDict(scheduler_config)

        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=sampling_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        pipe = pipe.to(device)

        output = pipe(
            prompt=prompts,
            height=resolution,
            width=resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        )

        images = output.images

        return images

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        """
        从 checkpoint 恢复 textual inversion 学到的 placeholder embedding 权重。

        """
        learned_embeds = checkpoint.get("learned_embeds", None)
        if learned_embeds is None:
            # 兼容旧 checkpoint 或异常情况，不崩溃
            return None

        tokenizer = extra_components["tokenizer"]
        text_encoder = extra_components["text_encoder"]

        # 获取整个 embedding 矩阵的引用，准备原地修改
        embedding_weight = text_encoder.get_input_embeddings().weight

        for token_str, saved_embed in learned_embeds.items():
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id == tokenizer.unk_token_id:
                # 理论不应发生，但保留检查
                continue

            # 确保 saved_embed 在同一设备和 dtype
            saved_embed = saved_embed.to(
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )
            embedding_weight.data[token_id] = saved_embed

        return None

    def checkpoint_extra_state(extra_components):

        tokenizer = extra_components["tokenizer"]
        text_encoder = extra_components["text_encoder"]

        placeholder_tokens = extra_components["placeholder_tokens"]

        embeds = text_encoder.get_input_embeddings().weight.detach().cpu()

        learned_embeds = {}

        for token in placeholder_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            learned_embeds[token] = embeds[token_id]

        return {
            "learned_embeds": learned_embeds,
        }

    return {
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "before_optimizer_step": before_optimizer_step,
        "after_optimizer_step": after_optimizer_step,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }


def build_sd_full_noise_scheduler(args):
    """
    Stable Diffusion sd_textual_inversion 训练阶段使用预训练 checkpoint 自带 scheduler 配置。
    """
    pretrained_path = getattr(args, "pretrained_model_name_or_path", None)
    if pretrained_path is None:
        raise ValueError(
            "sd_textual_inversion requires --pretrained_model_name_or_path"
        )

    return DDPMScheduler.from_pretrained(
        pretrained_path,
        subfolder="scheduler",
    )
