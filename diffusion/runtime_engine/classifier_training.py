import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

from ..modes.cg import build_noisy_timestep_classifier


def build_noisy_classifier_batch(batch, noise_scheduler, device):
    """
    为 CG guidance classifier 构造训练 batch。

    输入是真实图像 x_0 和类别标签 y。
    这里随机采样 timestep t 和噪声 epsilon，
    用 noise_scheduler.add_noise 得到 x_t。

    classifier 的训练目标是：
        输入: x_t, t
        输出: y

    也就是学习 p(y | x_t, t)。
    """
    images = batch["input"].to(device)
    labels = batch["label"].to(device).long()

    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (images.shape[0],),
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(images)
    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

    return noisy_images, labels, timesteps


@torch.no_grad()
def evaluate_guidance_classifier_accelerate(
    classifier,
    data_loader,
    noise_scheduler,
    accelerator,
):
    classifier.eval()

    criterion = nn.CrossEntropyLoss()
    device = accelerator.device

    total_loss = 0.0
    total_count = 0

    all_labels = []
    all_preds = []

    for batch in data_loader:
        noisy_images, labels, timesteps = build_noisy_classifier_batch(
            batch=batch,
            noise_scheduler=noise_scheduler,
            device=device,
        )

        logits = classifier(noisy_images, timesteps)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        gathered_loss = accelerator.gather_for_metrics(
            loss.detach().repeat(noisy_images.shape[0])
        )
        gathered_labels = accelerator.gather_for_metrics(labels.detach())
        gathered_preds = accelerator.gather_for_metrics(preds.detach())

        total_loss += gathered_loss.float().sum().item()
        total_count += gathered_loss.numel()

        all_labels.extend(gathered_labels.cpu().tolist())
        all_preds.extend(gathered_preds.cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)

    accuracy = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / max(
        len(all_labels), 1
    )

    if len(all_labels) > 0:
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    else:
        balanced_acc = 0.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
    }


def save_classifier_outputs(exp_folders, train_result):
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


def train_guidance_classifier_with_accelerator(
    args,
    noise_scheduler,
    train_loader,
    val_loader,
    num_classes,
    accelerator,
    exp_folders,
):
    """
    独立训练 p(y | x_t, t) classifier。

    """

    device = accelerator.device

    classifier = build_noisy_timestep_classifier(
        args=args,
        num_classes=num_classes,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.classifier_train_lr,
        weight_decay=getattr(args, "classifier_weight_decay", 0.01),
    )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_bal_acc = -1.0
    best_classifier_state_dict = None
    history = []

    resume_path = getattr(args, "classifier_ckpt_path", None)
    if resume_path is not None and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")

        if "classifier_state_dict" in checkpoint:
            classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        elif "model_state_dict" in checkpoint:
            classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            classifier.load_state_dict(checkpoint, strict=True)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_bal_acc = float(
            checkpoint.get(
                "best_val_balanced_accuracy",
                checkpoint.get("best_val_balanced_acc", -1.0),
            )
        )

        history = checkpoint.get("history", [])

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier,
        optimizer,
        train_loader,
        val_loader,
    )

    last_ckpt_path = os.path.join(
        exp_folders["checkpoints_dir"],
        "classifier_last.pth.tar",
    )
    best_ckpt_path = os.path.join(
        exp_folders["checkpoints_dir"],
        "classifier_best.pth.tar",
    )

    for epoch in range(start_epoch, args.classifier_train_epochs):
        classifier.train()

        total_loss = 0.0
        total_count = 0
        all_train_labels = []
        all_train_preds = []

        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"CG Classifier Train [{epoch + 1}/{args.classifier_train_epochs}]",
            disable=not accelerator.is_local_main_process,
            leave=True,
        )

        for batch in train_loader:
            noisy_images, labels, timesteps = build_noisy_classifier_batch(
                batch=batch,
                noise_scheduler=noise_scheduler,
                device=device,
            )

            logits = classifier(noisy_images, timesteps)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()

            preds = logits.argmax(dim=1)

            gathered_loss = accelerator.gather_for_metrics(
                loss.detach().repeat(noisy_images.shape[0])
            )
            gathered_labels = accelerator.gather_for_metrics(labels.detach())
            gathered_preds = accelerator.gather_for_metrics(preds.detach())

            total_loss += gathered_loss.float().sum().item()
            total_count += gathered_loss.numel()

            all_train_labels.extend(gathered_labels.cpu().tolist())
            all_train_preds.extend(gathered_preds.cpu().tolist())

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

        val_result = evaluate_guidance_classifier_accelerate(
            classifier=classifier,
            data_loader=val_loader,
            noise_scheduler=noise_scheduler,
            accelerator=accelerator,
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

        accelerator.wait_for_everyone()

        unwrapped_classifier = accelerator.unwrap_model(classifier)

        if val_result["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = float(val_result["balanced_accuracy"])
            best_classifier_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in unwrapped_classifier.state_dict().items()
            }

        if accelerator.is_main_process:
            checkpoint_state = {
                "epoch": epoch + 1,
                "classifier_state_dict": unwrapped_classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "train_balanced_accuracy": float(train_bal_acc),
                "val_loss": float(val_result["loss"]),
                "val_accuracy": float(val_result["accuracy"]),
                "val_balanced_accuracy": float(val_result["balanced_accuracy"]),
                "best_val_balanced_accuracy": float(best_val_bal_acc),
                "history": history,
                "exp_dir": exp_folders["exp_dir"],
            }

            torch.save(checkpoint_state, last_ckpt_path)

            if val_result["balanced_accuracy"] >= best_val_bal_acc:
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

    accelerator.wait_for_everyone()

    if best_classifier_state_dict is None:
        unwrapped_classifier = accelerator.unwrap_model(classifier)
        best_classifier_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in unwrapped_classifier.state_dict().items()
        }

    train_result = {
        "history": history,
        "best_val_balanced_accuracy": float(best_val_bal_acc),
        "best_classifier_state_dict": best_classifier_state_dict,
    }

    if accelerator.is_main_process:
        save_paths = save_classifier_outputs(
            exp_folders=exp_folders,
            train_result=train_result,
        )
    else:
        save_paths = {
            "history_json_path": "",
            "best_classifier_ckpt_path": "",
        }

    accelerator.wait_for_everyone()

    return {
        **train_result,
        **save_paths,
    }
