import torch
import torch.nn.functional as F

from ..modeling import run_sampling_loop


def build_ddpm(args):
    def build_extra_components(num_classes, device):
        # DDPM 模式不需要额外组件
        return {}

    def prepare_batch_labels(batch, device):
        # 只有开启类别条件时才返回标签
        if args.use_class_conditioning:
            return batch["label"].to(device).long()
        return None

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        # clean_images 的形状通常是 [B, C, H, W]
        clean_images = batch["input"]
        class_labels = prepare_batch_labels(batch, clean_images.device)

        # 为每张图采样一份噪声
        noise = torch.randn_like(clean_images)

        # 为 batch 中每个样本随机采一个扩散时间步 t
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        # q(x_t | x_0)：把干净图加噪得到 noisy_images
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # 条件模型和无条件模型调用方式略有不同
        if args.use_class_conditioning and class_labels is not None:
            noise_pred = model(
                noisy_images, timesteps, class_labels=class_labels
            ).sample
        else:
            noise_pred = model(noisy_images, timesteps).sample

        # 扩散模型基础目标：预测噪声 epsilon
        loss = F.mse_loss(noise_pred.float(), noise.float())

        aux = {
            "class_labels": class_labels,
            "timesteps": timesteps,
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
        return_pil_safe_uint8=True,
    ):
        def predict_fn(sample, t):
            model_input = sample

            # 某些 scheduler 会要求在推理前对输入做缩放
            if hasattr(sampling_scheduler, "scale_model_input"):
                model_input = sampling_scheduler.scale_model_input(model_input, t)

            if args.use_class_conditioning and class_labels is not None:
                return model(model_input, t, class_labels=class_labels).sample

            return model(model_input, t).sample

        return run_sampling_loop(
            model=model,
            sampling_scheduler=sampling_scheduler,
            device=device,
            resolution=resolution,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            generator=generator,
            predict_fn=predict_fn,
            ddim_eta=args.ddim_eta,
            return_pil_safe_uint8=return_pil_safe_uint8,
        )

    def checkpoint_extra_state(extra_components):
        # DDPM 没有额外状态需要存
        return {}

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        return None

    return {
        "name": "ddpm",
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }
