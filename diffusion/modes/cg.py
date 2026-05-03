import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling import run_sampling_loop


class ClassifierGuidanceAdapter:
    """
    Classifier Guidance（分类器引导）适配器。

    作用：
        在采样阶段，用一个额外训练好的分类器 p(y | x_t, t)
        计算目标类别 y 对当前 noisy image x_t 的梯度：
            ∇_{x_t} log p(y | x_t, t)

        这个梯度会被加入扩散模型的去噪方向中，
        让生成图更倾向于指定类别。

    """

    def __init__(self, classifier, guidance_scale: float):
        self.classifier = classifier
        self.guidance_scale = float(guidance_scale)

    def _build_timestep_batch(self, x_t: torch.Tensor, t):
        if not torch.is_tensor(t):
            return torch.full(
                (x_t.shape[0],),
                fill_value=int(t),
                device=x_t.device,
                dtype=torch.long,
            )

        if t.ndim == 0:
            return torch.full(
                (x_t.shape[0],),
                fill_value=int(t.item()),
                device=x_t.device,
                dtype=torch.long,
            )

        if t.shape[0] == 1 and x_t.shape[0] > 1:
            return t.to(x_t.device).long().expand(x_t.shape[0])

        return t.to(x_t.device).long()

    def grad_log_prob(self, x_t: torch.Tensor, t, class_labels: torch.Tensor):
        """
        计算 ∇_{x_t} log p(y | x_t, t)。

        步骤：
            1. 让 x_t 开启梯度；
            2. classifier 预测每个类别的 logits；
            3. log_softmax 得到 log probability；
            4. 取出目标类别 class_labels 对应的 log probability；
            5. 对 x_t 求梯度。

        返回：
            grad，shape 与 x_t 相同，用于修正 diffusion model 的噪声预测。
        """
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            t_batch = self._build_timestep_batch(x_t, t)

            logits = self.classifier(x_t, t_batch)
            log_probs = F.log_softmax(logits, dim=1)

            selected = log_probs[
                torch.arange(x_t.shape[0], device=x_t.device),
                class_labels.to(x_t.device).long(),
            ]

            grad = torch.autograd.grad(
                selected.sum(),
                x_t,
                retain_graph=False,
                create_graph=False,
            )[0]

        return grad


def _build_classifier_from_checkpoint(args, num_classes, device):
    classifier = build_noisy_timestep_classifier(
        args=args,
        num_classes=num_classes,
        device=device,
    )

    checkpoint = torch.load(args.classifier_ckpt_path, map_location=device)

    if "classifier_state_dict" in checkpoint:
        state_dict = checkpoint["classifier_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()

    return classifier


def build_cg(args):
    def build_extra_components(num_classes, device):
        if not args.use_class_conditioning:
            raise ValueError(
                "CG mode requires class conditioning labels for guided sampling."
            )

        if args.run_mode == "train" and args.classifier_ckpt_path is None:
            return {
                "classifier": None,
                "cg_adapter": None,
            }

        if args.classifier_ckpt_path is None:
            raise ValueError("CG sampling requires --classifier_ckpt_path.")

        classifier = _build_classifier_from_checkpoint(
            args=args,
            num_classes=num_classes,
            device=device,
        )

        adapter = ClassifierGuidanceAdapter(
            classifier=classifier,
            guidance_scale=args.classifier_guidance_scale,
        )

        return {
            "classifier": classifier,
            "cg_adapter": adapter,
        }

    def train_step(model, noise_scheduler, batch, accelerator, extra_components):
        clean_images = batch["input"]
        class_labels = batch["label"].to(clean_images.device).long()

        noise = torch.randn_like(clean_images)

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(
            noisy_images,
            timesteps,
            class_labels=class_labels,
        ).sample

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
        if class_labels is None:
            raise ValueError("CG sampling requires class_labels.")

        adapter = None
        if extra_components is not None:
            adapter = extra_components.get("cg_adapter", None)

        def predict_fn(sample, t):
            """
            给 run_sampling_loop 使用的噪声预测函数。

            输入：
                sample: 当前时间步的 noisy image，也就是 x_t
                t: 当前 diffusion timestep

            输出：
                eps_guided: 被 classifier guidance 修正后的噪声预测

            """
            model_input = sample

            if hasattr(sampling_scheduler, "scale_model_input"):
                model_input = sampling_scheduler.scale_model_input(model_input, t)

            eps = model(
                model_input,
                t,
                class_labels=class_labels,
            ).sample

            if adapter is None:
                return eps

            grad = adapter.grad_log_prob(sample, t, class_labels)

            alpha_bar_t = sampling_scheduler.alphas_cumprod[t].to(device)
            alpha_bar_t = alpha_bar_t.reshape(1, 1, 1, 1)

            eps_guided = (
                eps - adapter.guidance_scale * torch.sqrt(1.0 - alpha_bar_t) * grad
            )

            return eps_guided

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
            "classifier_ckpt_path": args.classifier_ckpt_path,
            "classifier_guidance_scale": float(args.classifier_guidance_scale),
            "classifier_base_channels": int(
                getattr(args, "classifier_base_channels", 128)
            ),
            "classifier_time_dim": int(getattr(args, "classifier_time_dim", 512)),
            "classifier_dropout": float(getattr(args, "classifier_dropout", 0.1)),
        }

    def load_checkpoint_extra_state(checkpoint, extra_components, device):
        return None

    return {
        "name": "cg",
        "build_extra_components": build_extra_components,
        "train_step": train_step,
        "sample_images": sample_images,
        "checkpoint_extra_state": checkpoint_extra_state,
        "load_checkpoint_extra_state": load_checkpoint_extra_state,
    }


# ========================
# 独立的classifier
# ========================
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()

        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)

        freqs = torch.exp(exponent)
        args = timesteps[:, None] * freqs[None, :]

        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=min(32, in_channels),
            num_channels=in_channels,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=min(32, out_channels),
            num_channels=out_channels,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.time_to_film(time_emb)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class NoisyTimestepClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults=(1, 2, 4, 4),
        time_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.time_dim = int(time_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        blocks = []
        downsamples = []

        prev_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            blocks.append(FiLMResidualBlock(prev_channels, out_channels, time_dim))
            blocks.append(FiLMResidualBlock(out_channels, out_channels, time_dim))

            if i != len(channel_mults) - 1:
                downsamples.append(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
            else:
                downsamples.append(nn.Identity())

            prev_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList(downsamples)

        self.head = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(32, prev_channels),
                num_channels=prev_channels,
            ),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev_channels, num_classes),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]

        timesteps = timesteps.to(device=x.device).long()

        if timesteps.shape[0] == 1 and x.shape[0] > 1:
            timesteps = timesteps.expand(x.shape[0])

        if timesteps.shape[0] != x.shape[0]:
            raise ValueError(
                f"timesteps batch size {timesteps.shape[0]} "
                f"does not match image batch size {x.shape[0]}"
            )

        time_emb = self.time_embedding(timesteps)

        h = self.stem(x)

        block_idx = 0
        for downsample in self.downsamples:
            h = self.blocks[block_idx](h, time_emb)
            block_idx += 1

            h = self.blocks[block_idx](h, time_emb)
            block_idx += 1

            h = downsample(h)

        return self.head(h)


def build_noisy_timestep_classifier(args, num_classes: int, device: torch.device):
    base_channels = getattr(args, "classifier_base_channels", 128)
    time_dim = getattr(args, "classifier_time_dim", 512)
    dropout = getattr(args, "classifier_dropout", 0.1)

    classifier = NoisyTimestepClassifier(
        num_classes=num_classes,
        in_channels=3,
        base_channels=base_channels,
        time_dim=time_dim,
        dropout=dropout,
    )

    return classifier.to(device)
