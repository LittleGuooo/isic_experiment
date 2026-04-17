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


def build_ldm_ae(args):

    def _pack_extra_components(extra_components):
        """
        只打包真正需要跨断点恢复的 extra_components 状态。
        """
        return {
            "step_counter": int(extra_components.get("step_counter", 0)),
            "log_interval": int(extra_components.get("log_interval", 50)),
        }

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

    def build_extra_components(num_classes, device):
        extra = {}

        # 记录 train_step 内部打印频率
        extra["log_interval"] = int(getattr(args, "ae_log_interval", 50))
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

        return extra

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        """
        第一阶段 AutoencoderKL 的训练 step。

        total_loss =
            w_recon * recon_loss
          + w_kl    * kl_loss
          + w_patch * patch_loss
          + w_perc  * perceptual_loss
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

        total_loss = (
            float(getattr(args, "ae_recon_loss_weight", 1.0)) * recon_loss
            + float(getattr(args, "ae_kl_loss_weight", 1e-6)) * kl_loss
            + float(getattr(args, "ae_patch_loss_weight", 0.0)) * patch_loss
            + float(getattr(args, "ae_perceptual_loss_weight", 0.0)) * perceptual_loss
        )

        aux = {
            "total_loss": float(total_loss.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "patch_loss": float(patch_loss.detach().item()),
            "perceptual_loss": float(perceptual_loss.detach().item()),
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
