# =========================
# modes/ldm_ae.py
# 第一阶段：KL-regularized Autoencoder 训练
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class VGGPerceptualLoss(nn.Module):
    """
    感知损失模块，基于预训练的 VGG16。

    输入张量约定：
        x_rec, x_target: [-1, 1], shape = [B, 3, H, W]
    """

    def __init__(self, resize=224):
        super().__init__()

        from torchvision import models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # 取几段中低层特征
        self.blocks = nn.ModuleList(
            [
                vgg[:4].eval(),
                vgg[4:9].eval(),
                vgg[9:16].eval(),
            ]
        )

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.resize = resize

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _preprocess(self, x):
        # [-1, 1] -> [0, 1]
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)

        if self.resize is not None:
            x = F.interpolate(
                x,
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet 标准化
        x = (x - self.mean) / self.std
        return x

    def forward(self, x_rec, x_target):
        x_rec = self._preprocess(x_rec)
        x_target = self._preprocess(x_target)

        loss = x_rec.new_zeros(())
        for block in self.blocks:
            x_rec = block(x_rec)
            x_target = block(x_target)
            loss = loss + F.l1_loss(x_rec, x_target)

        return loss


class PatchDiscriminator(nn.Module):
    """
    PatchGAN 判别器（PatchGAN discriminator）。

    输入:
        image: [-1, 1], shape = [B, 3, H, W]

    输出:
        logits: shape = [B, 1, h, w]

    含义:
        每个空间位置对应一个局部 patch 的真/假判断。
        这里不加 Sigmoid，因为后面使用 BCEWithLogitsLoss。
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def block(in_ch, out_ch, use_norm=True):
            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ]

            if use_norm:
                layers.append(nn.BatchNorm2d(out_ch))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            # 第 1 层不使用 BatchNorm，这是 PatchGAN 常见写法
            *block(in_channels, base_channels, use_norm=False),
            *block(base_channels, base_channels * 2, use_norm=True),
            *block(base_channels * 2, base_channels * 4, use_norm=True),
            # stride=1 保留更多 patch 位置
            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 patch logits，不加 sigmoid
            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.net(x)


def _set_requires_grad(module, requires_grad):
    """
    临时冻结或解冻判别器参数。
    生成器更新时冻结 D，避免把 G loss 的梯度累积到 D 上。
    """
    for p in module.parameters():
        p.requires_grad = requires_grad


def _compute_adversarial_losses(
    discriminator,
    d_optimizer,
    accelerator,
    real_images,
    fake_images,
):
    """
    更新一次判别器，并返回生成器需要的 adversarial loss。

    real_images:
        真实图像 x

    fake_images:
        Autoencoder 重建图像 x_rec

    返回:
        g_adv_loss: 用于更新 Autoencoder 的生成器对抗损失
        d_loss: 判别器自身损失，仅用于日志
    """

    bce = nn.BCEWithLogitsLoss()

    # =========================
    # 1) 更新判别器 D
    # =========================
    _set_requires_grad(discriminator, True)
    discriminator.train()

    d_optimizer.zero_grad(set_to_none=True)

    real_logits = discriminator(real_images.detach())
    fake_logits = discriminator(fake_images.detach())

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)

    d_real_loss = bce(real_logits, real_targets)
    d_fake_loss = bce(fake_logits, fake_targets)

    d_loss = 0.5 * (d_real_loss + d_fake_loss)

    accelerator.backward(d_loss)
    d_optimizer.step()

    # =========================
    # 2) 计算生成器 G 的对抗损失
    # =========================
    _set_requires_grad(discriminator, False)

    fake_logits_for_g = discriminator(fake_images)
    g_targets = torch.ones_like(fake_logits_for_g)

    # G 希望 D 把重建图像判断为真
    g_adv_loss = bce(fake_logits_for_g, g_targets)

    return g_adv_loss, d_loss


def _compute_kl_loss_from_posterior(posterior):
    """
    标准 VAE KL 项：
        KL(q(z|x) || N(0, I))
    """
    mu = posterior.mean
    logvar = posterior.logvar

    kl_per_sample = 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - 1.0 - logvar,
        dim=[1, 2, 3],
    )
    return kl_per_sample.mean()


def _compute_recon_loss(x_rec, x_target, loss_type="l1"):
    """
    像素重建项（pixel reconstruction loss）
    """
    if loss_type == "l1":
        return F.l1_loss(x_rec, x_target)
    if loss_type == "mse":
        return F.mse_loss(x_rec, x_target)

    raise ValueError(f"Unsupported recon loss type: {loss_type}")


def _compute_patch_based_loss_placeholder(x_rec, x_target, args):
    """
    patch-based loss 还没实现

    """
    return x_rec.new_zeros(())


def _encode_decode(model, x, sample_posterior=False):
    """
    统一做：
        x -> posterior -> z -> x_rec
    """
    posterior = model.encode(x).latent_dist

    if sample_posterior:
        z = posterior.sample()
    else:
        z = posterior.mean

    # 这里  sample 不是采样的意思，是返回 DecoderOutput 对象中的 tensor
    x_rec = model.decode(z).sample
    return posterior, z, x_rec


@torch.no_grad()
def _reconstruct_for_visualization(model, clean_images, args):
    """
    用于样图保存的重建函数。
    返回范围保持在 [-1, 1]
    """
    was_training = model.training
    model.eval()

    _, _, x_rec = _encode_decode(
        model=model,
        x=clean_images,
        sample_posterior=getattr(args, "ae_sample_posterior", False),
    )

    if was_training:
        model.train()

    return x_rec.clamp(-1.0, 1.0)


def save_ldm_ae_pretrained_outputs(
    args,
    accelerator,
    model,
    ema_model,
    exp_folders,
):
    """
    保存 ldm_ae 训练得到的 AutoencoderKL 为 Diffusers save_pretrained 格式。

    用途：
        第二阶段 latent_ddpm 直接通过 AutoencoderKL.from_pretrained(...) 加载。

    保存结果：
        checkpoints/autoencoder_ema_last/
            config.json
            diffusion_pytorch_model.safetensors

        checkpoints/autoencoder_raw_last/
            config.json
            diffusion_pytorch_model.safetensors
    """
    import os

    # 只在主进程保存，避免多卡训练时多个进程同时写文件。
    if not accelerator.is_main_process:
        return

    # 只在 ldm_ae 模式下生效。
    if getattr(args, "mode", None) != "ldm_ae":
        return

    checkpoints_dir = exp_folders["checkpoints_dir"]

    # 1) 保存 EMA 版本，推荐给第二阶段 latent_ddpm 使用
    if ema_model is not None:
        ae_ema_save_dir = os.path.join(
            checkpoints_dir,
            "autoencoder_ema_last",
        )

        ema_model.save_pretrained(
            ae_ema_save_dir,
            safe_serialization=True,
        )

    # 2) 保存当前原始模型版本，主要用于排查对比
    ae_raw_save_dir = os.path.join(
        checkpoints_dir,
        "autoencoder_raw_last",
    )

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        ae_raw_save_dir,
        safe_serialization=True,
    )


def build_ldm_ae(args):

    def _pack_extra_components(extra_components):
        """
        打包 ldm_ae 需要跨断点恢复的 extra_components 状态。
        """
        state = {
            "step_counter": int(extra_components.get("step_counter", 0)),
            "log_interval": int(extra_components.get("log_interval", 50)),
        }

        discriminator = extra_components.get("discriminator", None)
        d_optimizer = extra_components.get("d_optimizer", None)

        if discriminator is not None:
            state["discriminator_state_dict"] = discriminator.state_dict()

        if d_optimizer is not None:
            state["d_optimizer_state_dict"] = d_optimizer.state_dict()

        return state

    def _restore_extra_components(extra_components, saved_state):
        """
        从 checkpoint 里恢复 extra_components 的运行时状态。
        """
        if saved_state is None:
            return

        extra_components["step_counter"] = int(
            saved_state.get("step_counter", extra_components.get("step_counter", 0))
        )
        extra_components["log_interval"] = int(
            saved_state.get("log_interval", extra_components.get("log_interval", 50))
        )

        discriminator = extra_components.get("discriminator", None)
        if discriminator is not None and "discriminator_state_dict" in saved_state:
            discriminator.load_state_dict(saved_state["discriminator_state_dict"])

        d_optimizer = extra_components.get("d_optimizer", None)
        if d_optimizer is not None and "d_optimizer_state_dict" in saved_state:
            d_optimizer.load_state_dict(saved_state["d_optimizer_state_dict"])

    def build_extra_components(num_classes, device):
        extra = {}

        # 记录 train_step 内部打印频率
        extra["log_interval"] = int(getattr(args, "ae_log_interval", 300))
        extra["step_counter"] = 0

        # 感知损失模块
        perceptual_weight = float(getattr(args, "ae_perceptual_loss_weight", 0.0))
        if perceptual_weight > 0:
            perceptual_model = VGGPerceptualLoss(
                resize=int(getattr(args, "ae_perceptual_resize", 224))
            ).to(device)
            perceptual_model.eval()
            extra["perceptual_model"] = perceptual_model
        else:
            extra["perceptual_model"] = None

        # PatchGAN 判别器
        adv_weight = float(getattr(args, "ae_adv_loss_weight", 0.0))
        if adv_weight > 0:
            discriminator = PatchDiscriminator(
                in_channels=3,
                base_channels=int(getattr(args, "ae_discriminator_base_channels", 64)),
            ).to(device)

            d_optimizer = torch.optim.AdamW(
                discriminator.parameters(),
                lr=float(getattr(args, "ae_discriminator_lr", 1e-4)),
                betas=(
                    float(getattr(args, "ae_discriminator_beta1", 0.5)),
                    float(getattr(args, "ae_discriminator_beta2", 0.999)),
                ),
                weight_decay=float(getattr(args, "ae_discriminator_weight_decay", 0.0)),
            )

            extra["discriminator"] = discriminator
            extra["d_optimizer"] = d_optimizer
        else:
            extra["discriminator"] = None
            extra["d_optimizer"] = None

        return extra

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        """
        第一阶段 AutoencoderKL 的训练 step。

        当前 total_loss =
            w_recon * recon_loss
          + w_kl    * kl_loss
          + w_patch * patch_loss
          + w_perc  * perceptual_loss
          + w_adv   * g_adv_loss

        注意:
            - recon_loss 是像素级重建损失
            - perceptual_loss 是 VGG 感知损失
            - kl_loss 是 latent KL 正则
            - g_adv_loss 是生成器方向的对抗损失
            - d_loss 是判别器损失，只用于更新 discriminator 和日志
        """
        clean_images = batch["input"]

        posterior, z, recon_images = _encode_decode(
            model=model,
            x=clean_images,
            sample_posterior=getattr(args, "ae_sample_posterior", False),
        )

        recon_loss = _compute_recon_loss(
            x_rec=recon_images,
            x_target=clean_images,
            loss_type=getattr(args, "ae_recon_loss_type", "l1"),
        )

        kl_loss = _compute_kl_loss_from_posterior(posterior)

        patch_loss = _compute_patch_based_loss_placeholder(
            x_rec=recon_images,
            x_target=clean_images,
            args=args,
        )

        perceptual_model = extra_components.get("perceptual_model", None)
        if perceptual_model is not None:
            perceptual_loss = perceptual_model(recon_images, clean_images)
        else:
            perceptual_loss = recon_images.new_zeros(())

        discriminator = extra_components.get("discriminator", None)
        d_optimizer = extra_components.get("d_optimizer", None)

        adv_weight = float(getattr(args, "ae_adv_loss_weight", 0.0))
        adv_start_step = int(getattr(args, "ae_adv_start_step", 0))

        use_adv = (
            adv_weight > 0
            and discriminator is not None
            and d_optimizer is not None
            and extra_components["step_counter"] >= adv_start_step
        )

        if use_adv:
            g_adv_loss, d_loss = _compute_adversarial_losses(
                discriminator=discriminator,
                d_optimizer=d_optimizer,
                accelerator=accelerator,
                real_images=clean_images,
                fake_images=recon_images,
            )
        else:
            g_adv_loss = recon_images.new_zeros(())
            d_loss = recon_images.new_zeros(())

        total_loss = (
            float(getattr(args, "ae_recon_loss_weight", 1.0)) * recon_loss
            + float(getattr(args, "ae_kl_loss_weight", 1e-6)) * kl_loss
            + float(getattr(args, "ae_patch_loss_weight", 0.0)) * patch_loss
            + float(getattr(args, "ae_perceptual_loss_weight", 0.0)) * perceptual_loss
            + adv_weight * g_adv_loss
        )

        aux = {
            "total_loss": float(total_loss.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "patch_loss": float(patch_loss.detach().item()),
            "perceptual_loss": float(perceptual_loss.detach().item()),
            "g_adv_loss": float(g_adv_loss.detach().item()),
            "d_loss": float(d_loss.detach().item()),
            "latent_abs_mean": float(z.detach().abs().mean().item()),
            "latent_std": float(z.detach().std().item()),
        }

        extra_components["step_counter"] += 1
        log_interval = extra_components["log_interval"]

        if (
            accelerator.sync_gradients
            and accelerator.is_local_main_process
            and log_interval > 0
            and (extra_components["step_counter"] % log_interval == 0)
        ):
            tqdm.write(
                "[ldm_ae] "
                f"step={extra_components['step_counter']} | "
                f"total={aux['total_loss']:.6f} | "
                f"recon={aux['recon_loss'] * float(getattr(args, 'ae_recon_loss_weight', 1.0)):.6f} | "
                f"kl={aux['kl_loss'] * float(getattr(args, 'ae_kl_loss_weight', 1e-6)):.6f} | "
                f"patch={aux['patch_loss'] * float(getattr(args, 'ae_patch_loss_weight', 0.0)):.6f} | "
                f"perc={aux['perceptual_loss'] * float(getattr(args, 'ae_perceptual_loss_weight', 0.0)):.6f} | "
                f"g_adv={aux['g_adv_loss'] * adv_weight:.6f} | "
                f"d={aux['d_loss']:.6f} | "
                f"|z|mean={aux['latent_abs_mean']:.6f} | "
                f"z_std={aux['latent_std']:.6f}"
            )

        return total_loss, aux

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
        source_batch=None,
        **kwargs,
    ):
        """
        ldm_ae 下，这不是 diffusion sampling，
        而是 reconstruction 可视化。
        """
        if source_batch is None:
            raise ValueError(
                "ldm_ae.sample_images(...) 需要 source_batch 作为重建输入。"
            )

        clean_images = source_batch["input"].to(device)
        recon_images = _reconstruct_for_visualization(model, clean_images, args)

        if return_pil_safe_uint8:
            x_uint8 = (
                ((recon_images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
            )
            return x_uint8

        return recon_images

    def checkpoint_extra_state(extra_components):
        """
        把 AE 配置和 extra_components 的运行时状态一起写进 checkpoint。
        """
        return {
            "ae_config": {
                "ae_latent_channels": int(getattr(args, "ae_latent_channels", 4)),
                "ae_block_out_channels": list(
                    getattr(args, "ae_block_out_channels", [64, 128, 256, 512])
                ),
                "ae_layers_per_block": int(getattr(args, "ae_layers_per_block", 2)),
                "ae_norm_num_groups": int(getattr(args, "ae_norm_num_groups", 32)),
                "ae_mid_block_add_attention": bool(
                    getattr(args, "ae_mid_block_add_attention", False)
                ),
                "ae_scaling_factor": float(getattr(args, "ae_scaling_factor", 0.18215)),
                "ae_recon_loss_type": str(getattr(args, "ae_recon_loss_type", "l1")),
                "ae_recon_loss_weight": float(
                    getattr(args, "ae_recon_loss_weight", 1.0)
                ),
                "ae_kl_loss_weight": float(getattr(args, "ae_kl_loss_weight", 1e-6)),
                "ae_patch_loss_weight": float(
                    getattr(args, "ae_patch_loss_weight", 0.0)
                ),
                "ae_perceptual_loss_weight": float(
                    getattr(args, "ae_perceptual_loss_weight", 0.0)
                ),
                "ae_sample_posterior": bool(
                    getattr(args, "ae_sample_posterior", False)
                ),
                "ae_adv_loss_weight": float(getattr(args, "ae_adv_loss_weight", 0.0)),
                "ae_adv_start_step": int(getattr(args, "ae_adv_start_step", 0)),
                "ae_discriminator_base_channels": int(
                    getattr(args, "ae_discriminator_base_channels", 64)
                ),
                "ae_discriminator_lr": float(
                    getattr(args, "ae_discriminator_lr", 1e-4)
                ),
            },
            "ldm_ae_extra_state": _pack_extra_components(extra_components),
        }

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        """
        恢复 ldm_ae 的 extra_components 运行时状态。
        """
        saved_state = checkpoint.get("ldm_ae_extra_state", None)
        _restore_extra_components(extra_components, saved_state)
        return None

    return {
        "name": "ldm_ae",
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }
