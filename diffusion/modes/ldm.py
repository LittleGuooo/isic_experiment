# =========================
# modes/latent_ddpm.py
# 第二阶段：Latent-space DDPM
# =========================

import os
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL


def _tensor_to_uint8(x):
    """
    [-1, 1] -> uint8
    """
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def _load_frozen_autoencoder(args, device):
    """
    加载第一阶段训练好的 AutoencoderKL，并冻结。

    统一约定：
        args.autoencoder_ckpt_path 必须指向 Diffusers save_pretrained 目录。

    目录结构示例：
        autoencoder_ema_last/
            config.json
            diffusion_pytorch_model.safetensors

    不再直接支持 last.pth.tar：
        last.pth.tar 是训练恢复 checkpoint，
        不建议作为第二阶段 LDM 的 frozen VAE 输入。
    """
    ae_path = getattr(args, "autoencoder_ckpt_path", None)

    if ae_path is None:
        raise ValueError(
            "latent_ddpm 模式下必须提供 args.autoencoder_ckpt_path，"
            "并且该路径应指向 AutoencoderKL.save_pretrained(...) 保存的目录。"
        )

    if not os.path.isdir(ae_path):
        raise ValueError(
            "args.autoencoder_ckpt_path 现在必须是 Diffusers save_pretrained 目录，"
            f"但当前传入的是：{ae_path}"
        )

    # Diffusers 会从目录中的 config.json 和权重文件恢复完整 AutoencoderKL。
    vae = AutoencoderKL.from_pretrained(ae_path)

    # 移动到当前设备，并固定为 eval 模式。
    vae = vae.to(device)
    vae.eval()

    # 冻结 AE，第二阶段 LDM 只训练 latent-space UNet。
    for p in vae.parameters():
        p.requires_grad = False

    # 可选显存优化。
    if bool(getattr(args, "ae_use_slicing", False)):
        vae.enable_slicing()

    if bool(getattr(args, "ae_use_tiling", False)):
        vae.enable_tiling()

    return vae


@torch.no_grad()
def _encode_to_latents(vae, clean_images, sample_posterior=False):
    """
    使用冻结的 AutoencoderKL 把图像编码到 latent space。

    clean_images:
        输入图像，shape 通常是 [B, 3, H, W]，范围为 [-1, 1]。

    vae.encode(clean_images).latent_dist:
        AutoencoderKL 的 encoder 不直接返回一个固定 latent tensor，
        而是返回一个后验分布 q(z|x)，即 DiagonalGaussianDistribution。

    sample_posterior=True:
        从 q(z|x) 中随机采样 z，写作 z ~ q(z|x)。
        随机性更强，更接近 VAE / LDM 的概率建模形式。

    sample_posterior=False:
        直接取 q(z|x) 的均值 posterior.mean。
        更稳定，适合医学图像这类希望保留结构细节的任务。
    """
    posterior = vae.encode(clean_images).latent_dist
    if sample_posterior:
        z = posterior.sample()
    else:
        z = posterior.mean
    return posterior, z


@torch.no_grad()
def _decode_from_scaled_latents(vae, scaled_latents):
    """
    diffusion UNet 训练 / 采样使用的是 z_scaled = z * scaling_factor
    decode 前要除回去。
    """
    scaling_factor = float(vae.config.scaling_factor)
    latents = scaled_latents / scaling_factor
    images = vae.decode(latents).sample
    return images


def build_latent_ddpm(args):

    def build_extra_components(num_classes, device):
        vae = _load_frozen_autoencoder(args, device=device)

        # 推断 latent 分辨率：
        downsample_factor = int(getattr(args, "ae_downsample_factor", 8))
        latent_resolution = int(args.resolution // downsample_factor)

        extra = {
            "vae": vae,
            "latent_channels": int(vae.config.latent_channels),
            "latent_resolution": latent_resolution,
            "latent_scaling_factor": float(vae.config.scaling_factor),
            "sample_posterior_for_latent_train": bool(
                getattr(args, "latent_train_sample_posterior", True)
            ),
        }
        return extra

    def prepare_batch_labels(batch, device):
        # 只有开启类别条件时才返回标签
        if args.use_class_conditioning:
            return batch["label"].to(device).long()
        return None

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        """
        第二阶段训练：
            x -> frozen VAE -> z
            z_scaled = z * scaling_factor
            在 z_scaled 上训练 diffusion model 预测噪声
        """
        vae = extra_components["vae"]
        sample_posterior = extra_components["sample_posterior_for_latent_train"]

        clean_images = batch["input"]

        with torch.no_grad():
            posterior, latents = _encode_to_latents(
                vae=vae,
                clean_images=clean_images,
                sample_posterior=sample_posterior,
            )

            scaled_latents = latents * extra_components["latent_scaling_factor"]

        noise = torch.randn_like(scaled_latents)

        bsz = scaled_latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=scaled_latents.device,
            dtype=torch.long,
        )

        noisy_latents = noise_scheduler.add_noise(scaled_latents, noise, timesteps)

        class_labels = prepare_batch_labels(batch, scaled_latents.device)
        if args.use_class_conditioning and class_labels is not None:
            noise_pred = model(
                noisy_latents, timesteps, class_labels=class_labels
            ).sample
        else:
            noise_pred = model(noisy_latents, timesteps).sample

        diffusion_loss = F.mse_loss(noise_pred, noise)

        aux = {
            "total_loss": float(diffusion_loss.detach().item()),
            "diffusion_loss": float(diffusion_loss.detach().item()),
            "latent_abs_mean": float(scaled_latents.detach().abs().mean().item()),
            "latent_std": float(scaled_latents.detach().std().item()),
            "posterior_mean_abs": float(posterior.mean.detach().abs().mean().item()),
        }

        return diffusion_loss, aux

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
        第二阶段采样流程：
            Gaussian noise in latent space
            -> denoise in latent space
            -> decode by frozen VAE
        """
        vae = extra_components["vae"]
        latent_channels = extra_components["latent_channels"]
        latent_resolution = extra_components["latent_resolution"]

        # latent diffusion 的初始噪声形状
        latents = torch.randn(
            (batch_size, latent_channels, latent_resolution, latent_resolution),
            generator=generator,
            device=device,
            dtype=model.dtype,
        )

        sampling_scheduler.set_timesteps(num_inference_steps)

        for t in sampling_scheduler.timesteps:
            if args.use_class_conditioning and class_labels is not None:
                noise_pred = model(latents, t, class_labels=class_labels).sample
            else:
                noise_pred = model(latents, t).sample

            latents = sampling_scheduler.step(
                noise_pred,
                t,
                latents,
            ).prev_sample

        # latent -> image
        images = _decode_from_scaled_latents(vae, latents)
        images = images.clamp(-1.0, 1.0)

        if return_pil_safe_uint8:
            return _tensor_to_uint8(images)

        return images

    def checkpoint_extra_state(extra_components):
        return {
            "latent_ddpm_config": {
                "autoencoder_ckpt_path": str(
                    getattr(args, "autoencoder_ckpt_path", "")
                ),
                "latent_channels": int(extra_components["latent_channels"]),
                "latent_resolution": int(extra_components["latent_resolution"]),
                "latent_scaling_factor": float(
                    extra_components["latent_scaling_factor"]
                ),
                "latent_train_sample_posterior": bool(
                    extra_components["sample_posterior_for_latent_train"]
                ),
            }
        }

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        return None

    return {
        "name": "latent_ddpm",
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }
