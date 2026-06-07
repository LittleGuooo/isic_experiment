import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 复用分类器训练时的 transform，避免“训练时一种预处理、筛图时另一种预处理”。
from classifier.trainer import build_transforms

REQUIRED_META_COLS = ["output_path", "label"]
RESNET_CHOICES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Metadata CSV saved by the image generation script.",
    )
    parser.add_argument("--classifier_checkpoint", type=str, required=True)
    parser.add_argument(
        "--classifier_arch", type=str, default="resnet50", choices=RESNET_CHOICES
    )
    parser.add_argument(
        "--gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
    )
    parser.add_argument("--output_dataset_dir", type=str, required=True)
    parser.add_argument("--classifier_resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_ext", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--min_target_confidence", type=float, default=0.0)
    return parser.parse_args()


class GeneratedImageDataset(Dataset):
    """从生成脚本的 metadata CSV 中读取图片路径和生成目标类别。"""

    def __init__(self, metadata_csv: str, transform=None):
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)
        self.transform = transform
        self._check_columns()

    def _check_columns(self):
        missing = [c for c in REQUIRED_META_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"metadata csv missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(str(row["output_path"]))
        label = str(row["label"])

        if not img_path.is_file():
            raise FileNotFoundError(f"Generated image not found: {img_path}")

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        # idx 用来把预测结果稳定合并回原始 metadata 行。
        return image, label, idx


def read_class_names_from_gt(gt_csv_path: str):
    """读取 ISIC one-hot ground truth 中的类别列顺序。类别顺序必须和训练分类器时一致。"""
    df = pd.read_csv(gt_csv_path)
    if "image" not in df.columns:
        raise ValueError(f"GT csv must contain an 'image' column: {gt_csv_path}")
    return [c for c in df.columns if c != "image"]


def build_classifier(arch: str, num_classes: int, device: torch.device):
    """构建和训练脚本一致的 ResNet 分类器骨架。"""
    model = models.__dict__[arch](weights=None)

    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError(
            f"Only ResNet-like models with model.fc are supported, got {arch}"
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_classifier_checkpoint(
    model: nn.Module, checkpoint_path: str, device: torch.device
):
    """加载常见 checkpoint 格式，并兼容 DataParallel 产生的 module. 前缀。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict", checkpoint.get("model_state_dict", checkpoint)
        )
    else:
        state_dict = checkpoint

    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_generated_images(model, loader, class_names, device):
    """对生成图做推理，返回每张图的预测类别和置信度。"""
    rows = []
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for images, labels, row_indices in tqdm(loader, desc="Predict generated images"):
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1)
        pred_conf, pred_idx = probs.max(dim=1)

        for i, target_label in enumerate(labels):
            target_label = str(target_label)
            if target_label not in class_to_idx:
                raise ValueError(
                    f"Unknown label in metadata: {target_label}. Expected one of {class_names}"
                )

            pred_i = int(pred_idx[i].item())
            target_i = class_to_idx[target_label]

            rows.append(
                {
                    "row_idx": int(row_indices[i]),
                    "pred": class_names[pred_i],
                    "pred_idx": pred_i,
                    "pred_confidence": float(pred_conf[i].item()),
                    "target_confidence": float(probs[i, target_i].item()),
                }
            )

    return pd.DataFrame(rows).sort_values("row_idx").reset_index(drop=True)


def make_image_id(row: pd.Series, fallback_idx: int):
    """生成新数据集里的图片 ID：优先使用 source_image + aug_idx，缺失时从 output_path 兜底。"""
    source_image = str(row.get("source_image", "")).strip()
    if source_image == "" or source_image.lower() == "nan":
        source_image = Path(str(row["output_path"])).stem.split("_aug")[0]

    if "aug_idx" in row and pd.notna(row["aug_idx"]):
        aug_idx = int(row["aug_idx"])
    else:
        stem = Path(str(row["output_path"])).stem
        aug_idx = int(stem.split("_aug")[-1]) if "_aug" in stem else fallback_idx

    image_id = f"{source_image}_aug{aug_idx:03d}"
    for bad_char in ["/", "\\", ":", " "]:
        image_id = image_id.replace(bad_char, "_")
    return image_id


def copy_or_convert_image(src_path: str, dst_path: Path, output_ext: str):
    """复制到新数据集目录；jpg/png 都统一转 RGB，避免 alpha 通道或灰度图影响后续 Dataset。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as image:
        image = image.convert("RGB")
        if output_ext == "jpg":
            image.save(dst_path, quality=95)
        elif output_ext == "png":
            image.save(dst_path)
        else:
            raise ValueError(f"Unsupported output_ext: {output_ext}")


def build_new_aug_dataset(filtered_df, class_names, output_dataset_dir, output_ext):
    """把通过过滤的图片整理成 images/<class>/<image_id>.<ext> 和 ISIC 风格 one-hot CSV。"""
    output_dataset_dir = Path(output_dataset_dir)
    image_root = output_dataset_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    gt_rows, meta_rows = [], []

    for new_idx, (_, row) in enumerate(
        tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Build new dataset")
    ):
        label = str(row["label"])
        if label not in class_names:
            raise ValueError(f"Unknown label={label}, expected one of {class_names}")

        image_id = make_image_id(row, new_idx)
        dst_filename = f"{image_id}.{output_ext}"
        dst_path = image_root / label / dst_filename
        copy_or_convert_image(str(row["output_path"]), dst_path, output_ext)

        # image 保存为“类别子目录/图片ID”，因为图片实际在 images/<class>/ 下。
        gt_row = {"image": f"{label}/{image_id}"}
        gt_row.update({c: 1.0 if c == label else 0.0 for c in class_names})
        gt_rows.append(gt_row)

        meta_row = row.to_dict()
        meta_row.update(
            {
                "new_image": image_id,
                "new_label_dir": label,
                "new_image_relpath": f"{label}/{dst_filename}",
                "new_image_path": str(dst_path),
            }
        )
        meta_rows.append(meta_row)

    gt_csv = output_dataset_dir / "groundtruth_filtered_aug.csv"
    meta_csv = output_dataset_dir / "metadata_filtered_aug_dataset.csv"
    pd.DataFrame(gt_rows).to_csv(gt_csv, index=False)
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    return gt_csv, meta_csv, image_root


def merge_predictions(
    metadata_csv: str, pred_df: pd.DataFrame, min_target_confidence: float
):
    """把预测结果合并回 metadata，并生成最终保留标记 filter_keep，同时计算 generated_correct 和 delta_confidence"""
    meta_df = pd.read_csv(metadata_csv).reset_index(drop=True)
    if len(meta_df) != len(pred_df):
        raise RuntimeError(
            f"Prediction count mismatch: metadata={len(meta_df)}, pred={len(pred_df)}"
        )

    merged_df = meta_df.copy()
    for col in ["pred", "pred_idx", "pred_confidence", "target_confidence"]:
        merged_df[col] = pred_df[col]

    if "correct" in merged_df.columns:
        merged_df.rename(columns={"correct": "source_correct"}, inplace=True)
    else:
        merged_df["source_correct"] = 0

    merged_df["generated_correct"] = (merged_df["pred"] == merged_df["label"]).astype(
        int
    )

    if "source_confidence" not in merged_df.columns:
        merged_df["source_confidence"] = 0.0
    merged_df["delta_confidence"] = (
        merged_df["pred_confidence"] - merged_df["source_confidence"]
    )

    merged_df["filter_keep"] = (
        (merged_df["pred"] == merged_df["label"])
        & (merged_df["target_confidence"] >= min_target_confidence)
    ).astype(int)

    return merged_df


def save_filter_tables(merged_df: pd.DataFrame, output_dataset_dir: str):
    """保存完整预测表、保留表、拒绝表。"""
    output_dataset_dir = Path(output_dataset_dir)
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "with_pred": output_dataset_dir / "metadata_with_pred.csv",
        "filtered": output_dataset_dir / "metadata_filtered.csv",
        "rejected": output_dataset_dir / "metadata_rejected.csv",
    }

    filtered_df = merged_df[merged_df["filter_keep"] == 1].copy()
    rejected_df = merged_df[merged_df["filter_keep"] == 0].copy()

    merged_df.to_csv(paths["with_pred"], index=False)
    filtered_df.to_csv(paths["filtered"], index=False)
    rejected_df.to_csv(paths["rejected"], index=False)
    return filtered_df, rejected_df, paths


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    class_names = read_class_names_from_gt(args.gt_csv_path)
    print(f"[INFO] class_names: {class_names}")

    # build_transforms 通常读 args.resolution；这里让筛图分辨率和分类器 eval transform 对齐。
    args.resolution = args.classifier_resolution
    _, eval_transform = build_transforms(args)

    loader = DataLoader(
        GeneratedImageDataset(args.metadata_csv, transform=eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(args.classifier_arch, len(class_names), device)
    model = load_classifier_checkpoint(model, args.classifier_checkpoint, device)

    pred_df = predict_generated_images(model, loader, class_names, device)
    merged_df = merge_predictions(
        args.metadata_csv, pred_df, args.min_target_confidence
    )
    filtered_df, rejected_df, paths = save_filter_tables(
        merged_df, args.output_dataset_dir
    )

    print(f"[INFO] all predictions saved to: {paths['with_pred']}")
    print(f"[INFO] filtered metadata saved to: {paths['filtered']}")
    print(f"[INFO] rejected metadata saved to: {paths['rejected']}")

    if filtered_df.empty:
        raise ValueError(
            "No generated images passed filtering. Check classifier quality, class order, "
            "preprocessing, or min_target_confidence."
        )

    gt_csv, new_meta_csv, image_dir = build_new_aug_dataset(
        filtered_df=filtered_df,
        class_names=class_names,
        output_dataset_dir=args.output_dataset_dir,
        output_ext=args.output_ext,
    )

    print("\n[DONE] New filtered augmented dataset created.")
    print(f"[DONE] image_dir: {image_dir}")
    print(f"[DONE] groundtruth_csv: {gt_csv}")
    print(f"[DONE] metadata_csv: {new_meta_csv}")
    print(f"[DONE] kept: {len(filtered_df)} / {len(merged_df)}")
    print("\n[DONE] kept images per class:")
    print(filtered_df.groupby("label").size())


if __name__ == "__main__":
    main()
