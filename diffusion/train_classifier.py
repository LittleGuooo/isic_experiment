import os
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from .config import parse_args
from .data import build_datasets_and_loaders
from .experiment import create_experiment_folders, save_json
from .modeling import build_noise_scheduler
from .modes.cg import NoisyTimestepClassifier
from .utils import set_seed


def build_classifier(args, num_classes):
    classifier = NoisyTimestepClassifier(
        image_channels=3,
        num_classes=num_classes,
        base_channels=args.classifier_base_channels,
        time_dim=args.classifier_time_dim,
        dropout=args.classifier_dropout,
    )
    return classifier


def train_classifier_one_epoch(
    args,
    classifier,
    noise_scheduler,
    train_loader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
):
    classifier.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in train_loader:
        clean_images = batch["pixel_values"]
        labels = batch["labels"]

        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps,
        )

        with accelerator.accumulate(classifier):
            logits = classifier(noisy_images, timesteps)
            loss = F.cross_entropy(logits, labels)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    classifier.parameters(),
                    args.max_grad_norm,
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        preds = logits.detach().argmax(dim=1)
        total_correct += (
            accelerator.gather_for_metrics((preds == labels).long()).sum().item()
        )
        total_seen += accelerator.gather_for_metrics(labels).numel()
        total_loss += accelerator.gather_for_metrics(loss.detach()).mean().item()

    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = total_correct / max(total_seen, 1)

    return {
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_accuracy": accuracy,
    }


@torch.no_grad()
def evaluate_classifier(
    classifier,
    noise_scheduler,
    val_loader,
    accelerator,
):
    classifier.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in val_loader:
        clean_images = batch["pixel_values"]
        labels = batch["labels"]

        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps,
        )

        logits = classifier(noisy_images, timesteps)
        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=1)

        total_correct += (
            accelerator.gather_for_metrics((preds == labels).long()).sum().item()
        )
        total_seen += accelerator.gather_for_metrics(labels).numel()
        total_loss += accelerator.gather_for_metrics(loss.detach()).mean().item()

    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = total_correct / max(total_seen, 1)

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
    }


def save_classifier_checkpoint(
    path,
    classifier,
    optimizer,
    lr_scheduler,
    epoch,
    best_val_accuracy,
    accelerator,
):
    if not accelerator.is_main_process:
        return

    unwrapped_classifier = accelerator.unwrap_model(classifier)

    checkpoint = {
        "epoch": epoch,
        "classifier": unwrapped_classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def train_guidance_classifier(
    args,
    noise_scheduler,
    train_loader,
    val_loader,
    num_classes,
    accelerator,
    exp_folders,
):
    classifier = build_classifier(args, num_classes)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.classifier_train_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.classifier_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = (
        len(train_loader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps

    max_train_steps = args.classifier_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    classifier, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        classifier,
        optimizer,
        train_loader,
        val_loader,
        lr_scheduler,
    )

    best_val_accuracy = 0.0

    for epoch in range(args.classifier_train_epochs):
        train_metrics = train_classifier_one_epoch(
            args=args,
            classifier=classifier,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            epoch=epoch,
        )

        val_metrics = evaluate_classifier(
            classifier=classifier,
            noise_scheduler=noise_scheduler,
            val_loader=val_loader,
            accelerator=accelerator,
        )

        current_val_accuracy = val_metrics["val_accuracy"]

        if accelerator.is_main_process:
            print(
                f"[Classifier] Epoch {epoch + 1}/{args.classifier_train_epochs} "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"train_acc={train_metrics['train_accuracy']:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_acc={val_metrics['val_accuracy']:.4f}"
            )

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy

            save_classifier_checkpoint(
                path=os.path.join(
                    exp_folders["checkpoints_dir"],
                    "best_classifier.pth.tar",
                ),
                classifier=classifier,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                accelerator=accelerator,
            )

    save_classifier_checkpoint(
        path=os.path.join(
            exp_folders["checkpoints_dir"],
            "last_classifier.pth.tar",
        ),
        classifier=classifier,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=args.classifier_train_epochs - 1,
        best_val_accuracy=best_val_accuracy,
        accelerator=accelerator,
    )

    return {
        "best_val_accuracy": best_val_accuracy,
        "best_checkpoint": os.path.join(
            exp_folders["checkpoints_dir"],
            "best_classifier.pth.tar",
        ),
        "last_checkpoint": os.path.join(
            exp_folders["checkpoints_dir"],
            "last_classifier.pth.tar",
        ),
    }


def main():
    args = parse_args()

    if args.mode != "cg":
        raise ValueError(
            "train_classifier.py 只用于 CG classifier 训练，请使用 --mode cg。"
        )

    set_seed(args.seed)

    exp_folders = create_experiment_folders(args)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard" if args.use_tensorboard else None,
        project_dir=(
            os.path.join(exp_folders["exp_dir"], "tensorboard")
            if args.use_tensorboard
            else None
        ),
    )

    if args.use_tensorboard:
        accelerator.init_trackers(
            project_name=exp_folders["exp_name"],
            config={
                k: v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                for k, v in vars(args).items()
            },
        )

    data_bundle = build_datasets_and_loaders(args)

    train_loader = data_bundle["train_dataloader"]
    val_loader = data_bundle["val_eval_loader"]
    class_names = data_bundle["class_names"]
    num_classes = len(class_names)

    noise_scheduler = build_noise_scheduler(args)

    result = train_guidance_classifier(
        args=args,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        accelerator=accelerator,
        exp_folders=exp_folders,
    )

    if accelerator.is_main_process:
        save_json(
            {
                "mode": args.mode,
                "script": "scripts/train_classifier.py",
                "class_names": class_names,
                "classifier_training": result,
                "args": {
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool)) or v is None
                        else str(v)
                    )
                    for k, v in vars(args).items()
                },
            },
            exp_folders["metadata_json_path"],
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
