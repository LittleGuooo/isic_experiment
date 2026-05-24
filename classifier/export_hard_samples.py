import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import ISICResNetDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export hard samples from a trained ISIC baseline classifier"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="baseline 分类器 checkpoint 路径，例如 experiments/xxx/checkpoints/model_best.pth.tar",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="分类器结构，必须和训练 baseline 时一致。",
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        required=True,
        help="要筛选困难样本的数据集 CSV，通常用训练集 GroundTruth CSV。",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        required=True,
        help="对应图片目录，通常是训练集图片目录。",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="输出 hard_samples.csv 路径。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="分类器输入分辨率。必须和 baseline 分类器训练/验证时一致。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.2,
        help="每类保留真实类别置信度最低的比例，默认 0.2，即每类最低 20%。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    return parser.parse_args()


def build_classifier(arch, num_classes, device):
    """
    和你 trainer.py 里的 build_classifier 逻辑保持一致：
    构建 ResNet，并替换最后的 fc 层。
    """
    model = models.__dict__[arch](weights=None)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"当前脚本只支持带 model.fc 的 ResNet，当前模型={arch}")

    return model.to(device)


def load_classifier_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


@torch.no_grad()
def export_sample_confidences(model, loader, class_names, device):
    """
    输出每个样本：
        image
        label
        label_idx
        confidence: 模型对真实类别的 softmax 概率
        pred
        pred_idx
        pred_confidence: 模型预测类别的最大概率

    这里的 hard sample 必须用 confidence 排序，
    不是用 pred_confidence 排序。
    """
    rows = []

    model.eval()

    for images, labels, image_ids in tqdm(loader, desc="Exporting confidences"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        pred_conf, preds = probs.max(dim=1)

        # 真实类别置信度
        true_conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)

        for i in range(images.size(0)):
            label_idx = int(labels[i].item())
            pred_idx = int(preds[i].item())

            rows.append(
                {
                    "image": str(image_ids[i]),
                    "label": class_names[label_idx],
                    "label_idx": label_idx,
                    "confidence": float(true_conf[i].item()),
                    "pred": class_names[pred_idx],
                    "pred_idx": pred_idx,
                    "pred_confidence": float(pred_conf[i].item()),
                    "correct": int(pred_idx == label_idx),
                }
            )

    return pd.DataFrame(rows)


def select_hard_per_class(conf_df, class_names, hard_ratio):
    """
    每个类别内部，按真实类别 confidence 从低到高排序，
    取最低 hard_ratio 的样本。
    """
    hard_rows = []

    for class_name in class_names:
        class_df = conf_df[conf_df["label"] == class_name].copy()

        if len(class_df) == 0:
            print(f"[WARN] class {class_name} has no samples, skipped.")
            continue

        class_df = class_df.sort_values("confidence", ascending=True)

        k = max(1, int(len(class_df) * hard_ratio))
        hard_rows.append(class_df.iloc[:k])

        print(
            f"[{class_name}] total={len(class_df)}, "
            f"hard={k}, "
            f"max_hard_conf={class_df.iloc[:k]['confidence'].max():.4f}"
        )

    if len(hard_rows) == 0:
        raise ValueError("No hard samples selected.")

    return pd.concat(hard_rows, axis=0).reset_index(drop=True)


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 必须使用 eval transform，不要用 RandomResizedCrop / ColorJitter。
    # 否则同一张图每次置信度会变，困难样本列表不稳定。
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ISICResNetDataset(
        gt_csv_path=args.gt_csv,
        img_dir=args.img_dir,
        transform=eval_transform,
    )

    class_names = dataset.class_columns
    num_classes = len(class_names)

    print(f"class_names: {class_names}")
    print(f"dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(
        arch=args.arch,
        num_classes=num_classes,
        device=device,
    )

    model = load_classifier_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    conf_df = export_sample_confidences(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
    )

    hard_df = select_hard_per_class(
        conf_df=conf_df,
        class_names=class_names,
        hard_ratio=args.hard_ratio,
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    hard_df.to_csv(args.output_csv, index=False)

    # 额外保存一份完整置信度表，方便你排查。
    all_csv = args.output_csv.replace(".csv", "_all_confidences.csv")
    conf_df.to_csv(all_csv, index=False)

    print(f"\n[DONE] hard samples saved to: {args.output_csv}")
    print(f"[DONE] all confidences saved to: {all_csv}")
    print(f"[DONE] hard samples count: {len(hard_df)}")


if __name__ == "__main__":
    main()
