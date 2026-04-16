import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling import build_model, run_sampling_loop

from timm.layers import AttentionPool2d as AbsAttentionPool2d
from timm.layers import RotAttentionPool2d

from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

import json
import os


class ClassifierWithUNetDownsample(nn.Module):
    """
    基于已训练 UNet2DModel 的下采样路径 + 中间块构建分类器。
    - 复用 diffusion UNet 的 conv_in / down_blocks / mid_block
    - 只在末端接 attention pooling + linear classifier
    - classifier 学的是 p(y | x_t, t)
    """

    def __init__(
        self,
        unet,
        num_classes=7,
        feat_size=None,
        num_heads=8,
        use_rotary=False,
    ):
        super().__init__()

        # 直接复用 UNet 的前半部分
        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks
        self.mid_block = unet.mid_block
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding

        self.num_features = unet.config.block_out_channels[-1]
        self.use_rotary = use_rotary
        self.feat_size = None

        if use_rotary:
            self.attention_pool = RotAttentionPool2d(
                in_features=self.num_features,
                out_features=self.num_features,
                num_heads=num_heads,
                qkv_bias=True,
            )
        else:
            if feat_size is None:
                raise ValueError(
                    "When classifier_use_rotary=False, feat_size must be provided."
                )
            self.feat_size = (
                feat_size if isinstance(feat_size, tuple) else (feat_size, feat_size)
            )
            self.attention_pool = AbsAttentionPool2d(
                in_features=self.num_features,
                feat_size=self.feat_size,
                out_features=self.num_features,
                num_heads=num_heads,
                qkv_bias=True,
            )

        self.classifier = nn.Linear(self.num_features, num_classes)

        # 冻结复用的 UNet backbone
        self._freeze_unet_backbone()

    def _freeze_unet_backbone(self):
        for param in self.conv_in.parameters():
            param.requires_grad = False
        for param in self.down_blocks.parameters():
            param.requires_grad = False
        for param in self.mid_block.parameters():
            param.requires_grad = False
        for param in self.time_proj.parameters():
            param.requires_grad = False
        for param in self.time_embedding.parameters():
            param.requires_grad = False

    def forward(self, x, timesteps):
        """
        输入:
        - x: x_t, 形状 [B, C, H, W]
        - timesteps: [B] 或标量 timestep

        输出:
        - logits: [B, num_classes]
        """
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.to(x.device).long()

        # 如果只给了一个 timestep，就扩展到 batch 维度
        if timesteps.shape[0] == 1 and x.shape[0] > 1:
            timesteps = timesteps.expand(x.shape[0])

        # 先走 UNet 的时间嵌入
        temb = self.time_proj(timesteps)
        temb = self.time_embedding(temb)

        # 再走 UNet 的前半部分
        x = self.conv_in(x)

        skip_sample = None
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                x, _, skip_sample = downsample_block(
                    hidden_states=x,
                    temb=temb,
                    skip_sample=skip_sample,
                )
            else:
                x, _ = downsample_block(hidden_states=x, temb=temb)

        x = self.mid_block(x, temb)

        # 池化 + 分类头
        pooled = self.attention_pool(x)
        logits = self.classifier(pooled)
        return logits


def build_classifier_from_trained_unet(
    unet,
    num_classes=7,
    feat_size=(4, 4),
    num_heads=8,
    use_rotary=False,
    device="cuda",
    resolution=128,
):
    """
    从“已按当前项目结构构建好的 UNet”包装出 classifier。

    """
    if not use_rotary:
        # 自动推断最后特征图大小，保持你原始代码行为
        downsample_factor = 1
        for block in unet.down_blocks:
            if hasattr(block, "downsamplers") and block.downsamplers is not None:
                downsample_factor *= 2

        inferred_feat_size = resolution // downsample_factor
        feat_size = (inferred_feat_size, inferred_feat_size)

    classifier = ClassifierWithUNetDownsample(
        unet=unet,
        num_classes=num_classes,
        feat_size=feat_size,
        num_heads=num_heads,
        use_rotary=use_rotary,
    ).to(device)

    return classifier


class ClassifierGuidanceAdapter:
    """
    把 classifier guidance 所需的“log p(y | x_t, t) 对 x_t 的梯度”抽成统一接口。

    """

    def __init__(self, classifier, guidance_scale):
        self.classifier = classifier
        self.guidance_scale = float(guidance_scale)

    def grad_log_prob(self, x_t, t, class_labels):
        """
        计算:
            ∇_{x_t} log p(y | x_t, t)

        注意：
        sample_images(...) 外层是 @torch.no_grad()，
        所以这里必须手动用 torch.enable_grad() 打开梯度。
        """
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)

            # 把单个 timestep 扩展成 batch 大小
            if not torch.is_tensor(t):
                t_batch = torch.full(
                    (x_t.shape[0],),
                    fill_value=int(t),
                    device=x_t.device,
                    dtype=torch.long,
                )
            else:
                if t.ndim == 0:
                    t_batch = torch.full(
                        (x_t.shape[0],),
                        fill_value=int(t.item()),
                        device=x_t.device,
                        dtype=torch.long,
                    )
                elif t.shape[0] == 1 and x_t.shape[0] > 1:
                    t_batch = t.to(x_t.device).long().expand(x_t.shape[0])
                else:
                    t_batch = t.to(x_t.device).long()

            logits = self.classifier(x_t, t_batch)
            log_probs = F.log_softmax(logits, dim=1)

            selected = log_probs[
                torch.arange(x_t.size(0), device=x_t.device),
                class_labels.long(),
            ]

            grad = torch.autograd.grad(
                selected.sum(),
                x_t,
                retain_graph=False,
                create_graph=False,
            )[0]

        return grad


def _build_noisy_classifier_batch(batch, noise_scheduler, device):
    """
    把一个 clean batch 转成 classifier 训练/验证所需的 noisy batch。
    返回:
    - noisy_images: x_t
    - labels: y
    - timesteps: t
    """
    images = batch["input"].to(device)
    labels = batch["label"].to(device).long()

    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (images.shape[0],),
        device=device,
    ).long()

    noise = torch.randn_like(images)
    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

    return noisy_images, labels, timesteps


def load_diffusion_backbone_checkpoint(unet, ckpt_path, device):
    """
    从 diffusion checkpoint 恢复 UNet 权重。
    这里只加载 diffusion backbone，本函数供 runtime.py 复用。
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Diffusion checkpoint must contain 'model_state_dict' to build CG classifier."
        )

    missing_keys, unexpected_keys = unet.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print("WARNING: diffusion checkpoint and current UNet may not fully match.")
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)

    return unet


@torch.no_grad()
def evaluate_guidance_classifier(
    classifier,
    data_loader,
    noise_scheduler,
    device,
):
    """
    在 noisy image (x_t, t) 上评估 classifier。
    data_loader 使用你当前项目的数据格式：
    batch["input"], batch["label"]
    """
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_count = 0

    for batch in data_loader:
        noisy_images, labels, timesteps = _build_noisy_classifier_batch(
            batch=batch,
            noise_scheduler=noise_scheduler,
            device=device,
        )

        logits = classifier(noisy_images, timesteps)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * noisy_images.size(0)
        total_count += noisy_images.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / max(
        len(all_labels), 1
    )
    bal_acc = (
        balanced_accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    )

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
    }


def train_guidance_classifier(
    classifier,
    train_loader,
    val_loader,
    noise_scheduler,
    epochs,
    lr,
    device,
    exp_folders,
    resume_path=None,
):
    """
    只训练 CG classifier，不训练 diffusion model。
    返回训练历史、最优指标和最优权重，供外层统一保存。
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    last_ckpt_path = None
    best_ckpt_path = None
    if exp_folders is not None:
        last_ckpt_path = os.path.join(
            exp_folders["checkpoints_dir"], "classifier_last.pth.tar"
        )
        best_ckpt_path = os.path.join(
            exp_folders["checkpoints_dir"], "classifier_best.pth.tar"
        )

    start_epoch = 0
    best_val_bal_acc = -1.0
    best_classifier_state_dict = None
    history = []

    # 从已有 classifier checkpoint 恢复
    if resume_path is not None and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if "classifier_state_dict" in checkpoint:
            classifier.load_state_dict(checkpoint["classifier_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_val_bal_acc = checkpoint.get(
                "best_val_balanced_accuracy",
                checkpoint.get("best_val_balanced_acc", -1.0),
            )

    for epoch in range(start_epoch, epochs):
        classifier.train()

        all_train_labels = []
        all_train_preds = []
        total_loss = 0.0
        total_count = 0

        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"CG Classifier Train [{epoch + 1}/{epochs}]",
            leave=True,
        )

        for batch in train_loader:
            noisy_images, labels, timesteps = _build_noisy_classifier_batch(
                batch=batch,
                noise_scheduler=noise_scheduler,
                device=device,
            )

            logits = classifier(noisy_images, timesteps)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * noisy_images.size(0)
            total_count += noisy_images.size(0)

            all_train_labels.extend(labels.detach().cpu().tolist())
            all_train_preds.extend(preds.detach().cpu().tolist())

            progress_bar.update(1)

        progress_bar.close()

        train_loss = total_loss / max(total_count, 1)
        train_acc = sum(
            int(p == y) for p, y in zip(all_train_preds, all_train_labels)
        ) / max(len(all_train_labels), 1)
        train_bal_acc = (
            balanced_accuracy_score(all_train_labels, all_train_preds)
            if len(all_train_labels) > 0
            else 0.0
        )

        val_result = evaluate_guidance_classifier(
            classifier=classifier,
            data_loader=val_loader,
            noise_scheduler=noise_scheduler,
            device=device,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_balanced_accuracy": float(train_bal_acc),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_balanced_accuracy": float(val_result["balanced_accuracy"]),
        }
        history.append(epoch_result)

        checkpoint_state = {
            "epoch": epoch + 1,
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_balanced_accuracy": float(train_bal_acc),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_balanced_accuracy": float(val_result["balanced_accuracy"]),
            "best_val_balanced_accuracy": float(
                max(best_val_bal_acc, val_result["balanced_accuracy"])
            ),
            "history": history,
            "exp_dir": exp_folders["exp_dir"] if exp_folders is not None else None,
        }

        if last_ckpt_path is not None:
            torch.save(checkpoint_state, last_ckpt_path)

        if val_result["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_result["balanced_accuracy"]
            best_classifier_state_dict = {
                k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()
            }
            if best_ckpt_path is not None:
                torch.save(checkpoint_state, best_ckpt_path)

        print(
            f"[CG classifier epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"train_bal_acc={train_bal_acc:.4f}, "
            f"val_loss={val_result['loss']:.4f}, "
            f"val_acc={val_result['accuracy']:.4f}, "
            f"val_bal_acc={val_result['balanced_accuracy']:.4f}"
        )

    # 如果整个训练过程中一次都没刷新 best，就保存最后一个状态
    if best_classifier_state_dict is None:
        best_classifier_state_dict = {
            k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()
        }

    return {
        "history": history,
        "best_val_balanced_accuracy": float(best_val_bal_acc),
        "best_classifier_state_dict": best_classifier_state_dict,
    }


def _build_classifier_from_your_code(args, num_classes, device):

    # 先按你当前项目的统一模型配置，构建一个“结构一致”的 UNet
    backbone_unet = build_model(args, num_classes)

    # 这些参数在你当前 config.py 里还没有，所以用 getattr 保持兼容
    classifier_num_heads = getattr(args, "classifier_num_heads", 8)
    classifier_use_rotary = getattr(args, "classifier_use_rotary", False)
    classifier_feat_size = getattr(args, "classifier_feat_size", 4)

    classifier = build_classifier_from_trained_unet(
        unet=backbone_unet,
        num_classes=num_classes,
        feat_size=classifier_feat_size,
        num_heads=classifier_num_heads,
        use_rotary=classifier_use_rotary,
        device=device,
        resolution=args.resolution,
    )

    checkpoint = torch.load(args.classifier_ckpt_path, map_location=device)

    # 兼容你原始 CG_diffusion.py 保存格式
    if "classifier_state_dict" in checkpoint:
        state_dict = checkpoint["classifier_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # 最后兜底：把整个 checkpoint 视作 state_dict
        state_dict = checkpoint

    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()

    return classifier


def _save_classifier_training_outputs(exp_folders, train_result):
    """
    保存 classifier 训练结果：
    1) 训练历史 JSON
    2) 最优 classifier checkpoint
    """
    metrics_dir = exp_folders["metrics_dir"]
    ckpt_dir = exp_folders["checkpoints_dir"]

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    history_json_path = os.path.join(metrics_dir, "cg_classifier_history.json")
    best_ckpt_path = os.path.join(ckpt_dir, "cg_classifier_best.pth.tar")

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": train_result["history"],
                "best_val_balanced_accuracy": train_result[
                    "best_val_balanced_accuracy"
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    torch.save(
        {
            "classifier_state_dict": train_result["best_classifier_state_dict"],
            "history": train_result["history"],
            "best_val_balanced_accuracy": train_result["best_val_balanced_accuracy"],
            "exp_dir": exp_folders["exp_dir"],
        },
        best_ckpt_path,
    )

    return {
        "history_json_path": history_json_path,
        "best_classifier_ckpt_path": best_ckpt_path,
    }


def build_cg(args):
    def build_extra_components(num_classes, device):
        if not args.use_class_conditioning:
            raise ValueError(
                "CG mode requires class conditioning labels for guided sampling."
            )

        # 训练阶段如果没有 classifier checkpoint，
        # 说明当前先训练 diffusion，classifier 之后再训。
        run_mode = getattr(args, "run_mode", None)

        # 只有在“明确是 diffusion 训练阶段”且还没有 classifier checkpoint 时，
        # 才允许先跳过 classifier，退化成普通 class-conditional diffusion
        if run_mode == "train" and args.classifier_ckpt_path is None:
            return {
                "classifier": None,
                "cg_adapter": None,
            }

        # 其它场景（包括 classifier.main 调用的增强采样）都要求提供 classifier ckpt
        if args.classifier_ckpt_path is None:
            raise ValueError("CG mode requires --classifier_ckpt_path.")

        classifier = _build_classifier_from_your_code(
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
        # CG 模式下，diffusion backbone 的训练目标仍然是预测噪声 epsilon
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

        noise_pred = model(noisy_images, timesteps, class_labels=class_labels).sample
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
            model_input = sample
            if hasattr(sampling_scheduler, "scale_model_input"):
                model_input = sampling_scheduler.scale_model_input(model_input, t)

            eps = model(model_input, t, class_labels=class_labels).sample

            # 还没有 classifier 的阶段，退化成普通 class-conditional diffusion 采样
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

    def build_classifier_for_training_from_unet(unet, num_classes, device):
        classifier_num_heads = getattr(args, "classifier_num_heads", 8)
        classifier_use_rotary = getattr(args, "classifier_use_rotary", False)
        classifier_feat_size = getattr(args, "classifier_feat_size", 4)

        classifier = build_classifier_from_trained_unet(
            unet=unet,
            num_classes=num_classes,
            feat_size=classifier_feat_size,
            num_heads=classifier_num_heads,
            use_rotary=classifier_use_rotary,
            device=device,
            resolution=args.resolution,
        )
        return classifier

    def train_classifier_only_from_diffusion(
        unet,
        noise_scheduler,
        train_dataloader,
        val_dataloader,
        num_classes,
        device,
        exp_folders,
    ):
        classifier = build_classifier_for_training_from_unet(
            unet=unet,
            num_classes=num_classes,
            device=device,
        )

        train_result = train_guidance_classifier(
            classifier=classifier,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            noise_scheduler=noise_scheduler,
            epochs=args.classifier_train_epochs,
            lr=args.classifier_train_lr,
            device=device,
            exp_folders=exp_folders,
            resume_path=args.classifier_ckpt_path,
        )

        save_paths = _save_classifier_training_outputs(
            exp_folders=exp_folders,
            train_result=train_result,
        )

        return {
            **train_result,
            **save_paths,
        }

    def checkpoint_extra_state(extra_components):
        return {
            "classifier_ckpt_path": args.classifier_ckpt_path,
            "classifier_guidance_scale": float(args.classifier_guidance_scale),
            "classifier_num_heads": getattr(args, "classifier_num_heads", 8),
            "classifier_use_rotary": bool(
                getattr(args, "classifier_use_rotary", False)
            ),
            "classifier_feat_size": getattr(args, "classifier_feat_size", 4),
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
        # 下面两个是 runtime.py 的 CG 特判分支所依赖的 hook
        "build_classifier_for_training_from_unet": build_classifier_for_training_from_unet,
        "train_classifier_only_from_diffusion": train_classifier_only_from_diffusion,
    }
