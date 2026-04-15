import torch
import torch.nn.functional as F

from ..modeling import run_sampling_loop


def build_cfg(args):
    def _get_null_class_idx(num_classes):
        # CFG 训练时需要一个 null class 作为“无条件”标签
        return num_classes

    def build_extra_components(num_classes, device):
        return {
            "num_classes": num_classes,
            "null_class_idx": _get_null_class_idx(num_classes),
        }

    def _maybe_drop_labels(class_labels, null_class_idx):
        """
        classifier-free guidance (CFG) 训练：

        以 cond_drop_prob 的概率把真实标签替换为 null label。
        这样模型会同时学到：
        1) 有条件预测
        2) 无条件预测
        采样时再用两者组合实现 guidance。
        """
        if class_labels is None:
            raise ValueError(
                "CFG requires class conditioning, but class_labels is None."
            )

        if args.cond_drop_prob <= 0.0:
            return class_labels

        # keep_mask=True 表示保留真实标签
        keep_mask = (
            torch.rand(
                class_labels.shape,
                device=class_labels.device,
            )
            >= args.cond_drop_prob
        )

        dropped = class_labels.clone()
        dropped[~keep_mask] = null_class_idx
        return dropped

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        clean_images = batch["input"]
        true_labels = batch["label"].to(clean_images.device).long()
        null_class_idx = extra_components["null_class_idx"]

        # 常规扩散训练：随机采样噪声和时间步
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # cfg：训练时随机把一部分标签替换成 null class
        dropped_labels = _maybe_drop_labels(true_labels, null_class_idx)

        noise_pred = model(noisy_images, timesteps, class_labels=dropped_labels).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        aux = {
            "class_labels": true_labels,
            "dropped_labels": dropped_labels,
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
        if class_labels is None:
            raise ValueError("CFG sampling requires class_labels.")

        null_class_idx = extra_components["null_class_idx"]
        cfg_scale = float(args.cfg_scale)

        # 为同一个 batch 构造“无条件标签”
        null_labels = torch.full(
            (batch_size,),
            fill_value=null_class_idx,
            device=device,
            dtype=torch.long,
        )

        def predict_fn(sample, t):
            model_input = sample
            if hasattr(sampling_scheduler, "scale_model_input"):
                model_input = sampling_scheduler.scale_model_input(model_input, t)

            # 无条件预测 epsilon_uncond
            eps_uncond = model(model_input, t, class_labels=null_labels).sample

            # 有条件预测 epsilon_cond
            eps_cond = model(model_input, t, class_labels=class_labels).sample

            # CFG 公式：eps = eps_uncond + s * (eps_cond - eps_uncond)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            return eps

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
        return {
            "cfg_null_class_idx": int(extra_components["null_class_idx"]),
        }

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        return None

    return {
        "name": "cfg",
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }
