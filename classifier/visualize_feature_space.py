#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_feature_space.py

用途：
1. 加载已经训练好的 ResNet 分类器 checkpoint；
2. 用确定性的 eval transform 提取每张图的倒数第二层 feature embedding；
3. 计算每个类别的 feature center；
4. 计算样本到真实类别中心、最近错误类别中心的距离；
5. 用 margin 找困难样本；
6. 生成 PCA / t-SNE 可视化图和困难样本 CSV。

推荐放置位置：
- 放在与你的 trainer.py、dataset.py、config.py 同一层目录下；
- 如果你的分类代码是一个 package，例如 classifier/trainer.py，
  推荐从项目根目录运行：
    python -m classifier.visualize_feature_space ...

"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from .trainer import build_transforms, build_classifier
    from .dataset import ISICResNetDataset
except ImportError:
    from trainer import build_transforms, build_classifier
    from dataset import ISICResNetDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize classifier feature space and mine hard samples."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gt-csv", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--split-name", type=str, default="feature_space_outputs")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    parser.add_argument("--weights", type=str, default=None)

    parser.add_argument(
        "--reducer",
        type=str,
        default="tsne",
        choices=["pca", "tsne"],
    )
    parser.add_argument("--pca-before-tsne-dim", type=int, default=50)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=50)

    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--neighbor-class", type=str, default=None)

    parser.add_argument("--no-normalize-features", action="store_true")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def parse_class_arg(class_arg, class_names):
    if class_arg is None:
        return None

    if class_arg.isdigit():
        idx = int(class_arg)
        if idx < 0 or idx >= len(class_names):
            raise ValueError(f"类别索引越界: {idx}, num_classes={len(class_names)}")
        return idx

    if class_arg not in class_names:
        raise ValueError(f"找不到类别名: {class_arg}. 当前 class_names={class_names}")

    return class_names.index(class_arg)


def try_get_path_from_dataset(dataset, index: int):
    candidate_attrs = ["samples", "image_paths", "img_paths", "paths"]

    for attr in candidate_attrs:
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            try:
                item = value[index]
                if isinstance(item, (tuple, list)):
                    return str(item[0])
                return str(item)
            except Exception:
                pass

    return ""


def build_eval_dataset(args):
    _, eval_transform = build_transforms(args)

    dataset = ISICResNetDataset(
        gt_csv_path=args.gt_csv,
        img_dir=args.img_dir,
        transform=eval_transform,
    )

    return dataset


def build_eval_loader(args, dataset, device):
    pin_memory = device.type == "cuda"
    persistent_workers = args.workers > 0

    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if persistent_workers:
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)


def load_classifier_from_checkpoint(args, device, fallback_num_classes):
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("当前脚本假设 checkpoint 是 dict，并且至少包含 state_dict。")

    args.arch = checkpoint.get("arch", args.arch)
    class_names = checkpoint.get("class_names", None)
    num_classes = checkpoint.get("num_classes", fallback_num_classes)

    if class_names is None:
        raise ValueError("checkpoint 中没有 class_names。")

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint.get("model_state_dict", None),
    )
    if state_dict is None:
        raise ValueError("checkpoint 中找不到 state_dict / model_state_dict。")

    model = build_classifier(
        args=args,
        num_classes=num_classes,
        device=device,
        use_pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, class_names, num_classes, checkpoint


@torch.no_grad()
def extract_features(model, loader, dataset, device):
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError("当前脚本只支持带有 Linear fc 的 ResNet 类模型。")

    classifier_head = model.fc
    model.fc = nn.Identity()
    model.eval()

    all_features = []
    all_logits = []
    all_labels = []
    all_sample_ids = []
    all_paths = []

    global_index = 0

    for batch in tqdm(loader, desc="Extracting features", dynamic_ncols=True):
        if len(batch) == 3:
            images, labels, sample_ids = batch
        elif len(batch) == 2:
            images, labels = batch
            sample_ids = [
                str(i) for i in range(global_index, global_index + len(labels))
            ]
        else:
            raise ValueError(
                "Dataset __getitem__ 应返回 (image, label) 或 (image, label, sample_id)。"
            )

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = model(images)
        logits = classifier_head(features)

        batch_size = images.size(0)

        all_features.append(features.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        for sid in sample_ids:
            all_sample_ids.append(str(sid))

        for i in range(batch_size):
            all_paths.append(try_get_path_from_dataset(dataset, global_index + i))

        global_index += batch_size

    model.fc = classifier_head

    features = np.concatenate(all_features, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    return features, logits, labels, all_sample_ids, all_paths


def compute_class_centers(features, labels, num_classes):
    feature_dim = features.shape[1]
    centers = np.full((num_classes, feature_dim), np.nan, dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        mask = labels == c
        counts[c] = int(mask.sum())
        if counts[c] > 0:
            centers[c] = features[mask].mean(axis=0)

    return centers, counts


def compute_distances(features, labels, centers):
    valid_centers = ~np.isnan(centers).any(axis=1)
    if not valid_centers.all():
        missing = np.where(~valid_centers)[0].tolist()
        print(f"Warning: 这些类别没有 center，距离会是 NaN: {missing}")

    diff = features[:, None, :] - centers[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    return dist_matrix


def build_difficulty_dataframe(
    labels,
    logits,
    probs,
    dist_matrix,
    sample_ids,
    image_paths,
    class_names,
    target_class_idx=None,
    neighbor_class_idx=None,
):
    num_samples, num_classes = dist_matrix.shape

    pred_idx = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    rows = []

    for i in range(num_samples):
        y = int(labels[i])
        pred = int(pred_idx[i])

        d_true = float(dist_matrix[i, y])

        dist_to_wrong = dist_matrix[i].copy()
        dist_to_wrong[y] = np.inf

        nearest_wrong_idx = int(np.nanargmin(dist_to_wrong))
        d_nearest_wrong = float(dist_to_wrong[nearest_wrong_idx])

        margin = d_nearest_wrong - d_true

        row = {
            "index": i,
            "sample_id": sample_ids[i],
            "image_path": image_paths[i],
            "true_label_idx": y,
            "true_label_name": class_names[y],
            "pred_label_idx": pred,
            "pred_label_name": class_names[pred],
            "is_correct": bool(pred == y),
            "confidence": float(confidence[i]),
            "distance_to_true_center": d_true,
            "nearest_wrong_class_idx": nearest_wrong_idx,
            "nearest_wrong_class_name": class_names[nearest_wrong_idx],
            "distance_to_nearest_wrong_center": d_nearest_wrong,
            "distance_margin": float(margin),
        }

        for c in range(num_classes):
            row[f"distance_to_{class_names[c]}"] = float(dist_matrix[i, c])

        if target_class_idx is not None and neighbor_class_idx is not None:
            d_target = float(dist_matrix[i, target_class_idx])
            d_neighbor = float(dist_matrix[i, neighbor_class_idx])
            row["target_class_name"] = class_names[target_class_idx]
            row["neighbor_class_name"] = class_names[neighbor_class_idx]
            row["distance_to_target_class"] = d_target
            row["distance_to_neighbor_class"] = d_neighbor
            row["target_neighbor_margin"] = d_neighbor - d_target

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values("distance_margin", ascending=True).reset_index(drop=True)
    df["rank_global"] = np.arange(1, len(df) + 1)

    df["rank_in_true_class"] = (
        df.groupby("true_label_name")["distance_margin"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    return df


def reduce_to_2d(features, args):
    n_samples = features.shape[0]

    if n_samples < 3:
        raise ValueError("样本数太少，无法做稳定二维可视化。")

    if args.reducer == "pca":
        reducer = PCA(n_components=2, random_state=args.seed)
        coords = reducer.fit_transform(features)
        return coords

    pca_dim = min(args.pca_before_tsne_dim, features.shape[1], n_samples - 1)
    if pca_dim > 2:
        pca = PCA(n_components=pca_dim, random_state=args.seed)
        features_for_tsne = pca.fit_transform(features)
    else:
        features_for_tsne = features

    perplexity = min(args.tsne_perplexity, max(1.0, (n_samples - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    coords = tsne.fit_transform(features_for_tsne)
    return coords


def scatter_by_label(coords, df, class_names, output_path, title):
    plt.figure(figsize=(10, 8))

    for class_name in class_names:
        mask = df["true_label_name"].values == class_name
        if mask.sum() == 0:
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.75,
            label=class_name,
        )

    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def scatter_by_correctness(coords, df, output_path, title):
    plt.figure(figsize=(10, 8))

    correct_mask = df["is_correct"].values.astype(bool)
    wrong_mask = ~correct_mask

    plt.scatter(
        coords[correct_mask, 0],
        coords[correct_mask, 1],
        s=10,
        alpha=0.55,
        label="correct",
    )
    plt.scatter(
        coords[wrong_mask, 0],
        coords[wrong_mask, 1],
        s=25,
        alpha=0.9,
        marker="x",
        label="wrong",
    )

    plt.legend()
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def scatter_by_margin(coords, df, output_path, title):
    margins = df["distance_margin"].values

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=margins,
        s=12,
        alpha=0.8,
    )
    plt.colorbar(sc, label="distance_margin = d_nearest_wrong - d_true")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_topk_hard_samples(df, class_names, output_dir, top_k):
    global_topk = df.sort_values("distance_margin", ascending=True).head(top_k)
    global_topk.to_csv(
        os.path.join(output_dir, "hard_samples_topk_global.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    per_class_rows = []
    for class_name in class_names:
        sub = df[df["true_label_name"] == class_name]
        sub = sub.sort_values("distance_margin", ascending=True).head(top_k)
        per_class_rows.append(sub)

    if len(per_class_rows) > 0:
        per_class_df = pd.concat(per_class_rows, axis=0)
    else:
        per_class_df = pd.DataFrame()

    per_class_df.to_csv(
        os.path.join(output_dir, "hard_samples_topk_per_class.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_target_neighbor_samples(
    df,
    class_names,
    target_class_idx,
    neighbor_class_idx,
    output_dir,
    top_k,
):
    if target_class_idx is None or neighbor_class_idx is None:
        return

    target_name = class_names[target_class_idx]
    neighbor_name = class_names[neighbor_class_idx]

    sub = df[df["true_label_idx"] == target_class_idx].copy()
    sub = sub.sort_values("target_neighbor_margin", ascending=True).head(top_k)

    filename = f"hard_samples_{target_name}_near_{neighbor_name}.csv"
    sub.to_csv(
        os.path.join(output_dir, filename),
        index=False,
        encoding="utf-8-sig",
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.split_name)
    ensure_dir(output_dir)

    device = get_device(args.gpu)
    print(f"Using device: {device}")

    dataset = build_eval_dataset(args)
    loader = build_eval_loader(args, dataset, device)

    dataset_class_names = list(dataset.class_columns)
    fallback_num_classes = len(dataset_class_names)

    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset class_names: {dataset_class_names}")

    model, class_names, num_classes, checkpoint = load_classifier_from_checkpoint(
        args=args,
        device=device,
        fallback_num_classes=fallback_num_classes,
    )

    print(f"Checkpoint arch: {args.arch}")
    print(f"Checkpoint class_names: {class_names}")

    if class_names != dataset_class_names:
        print(
            "Warning: checkpoint class_names 与 dataset class_columns 不一致。\n"
            "后续分析将以 checkpoint class_names 为准。请确认类别顺序没有错。"
        )

    features, logits, labels, sample_ids, image_paths = extract_features(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
    )

    probs = softmax_numpy(logits)

    print(f"features shape: {features.shape}")
    print(f"logits shape  : {logits.shape}")

    if not args.no_normalize_features:
        features_for_distance = l2_normalize(features)
        print("Feature normalization: ON")
    else:
        features_for_distance = features
        print("Feature normalization: OFF")

    centers, center_counts = compute_class_centers(
        features=features_for_distance,
        labels=labels,
        num_classes=num_classes,
    )

    if not args.no_normalize_features:
        centers = l2_normalize(centers)

    print("Samples used for each class center:")
    for idx, count in enumerate(center_counts):
        print(f"  {idx:02d} {class_names[idx]}: {count}")

    dist_matrix = compute_distances(
        features=features_for_distance,
        labels=labels,
        centers=centers,
    )

    target_class_idx = parse_class_arg(args.target_class, class_names)
    neighbor_class_idx = parse_class_arg(args.neighbor_class, class_names)

    if (target_class_idx is None) ^ (neighbor_class_idx is None):
        raise ValueError(
            "--target-class 和 --neighbor-class 必须同时提供，或者都不提供。"
        )

    difficulty_df = build_difficulty_dataframe(
        labels=labels,
        logits=logits,
        probs=probs,
        dist_matrix=dist_matrix,
        sample_ids=sample_ids,
        image_paths=image_paths,
        class_names=class_names,
        target_class_idx=target_class_idx,
        neighbor_class_idx=neighbor_class_idx,
    )

    coords = reduce_to_2d(features_for_distance, args)

    df_by_original_order = difficulty_df.sort_values(
        "index", ascending=True
    ).reset_index(drop=True)

    coords_df = pd.DataFrame(
        {
            "index": np.arange(len(coords)),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    coords_with_meta = pd.concat([df_by_original_order, coords_df[["x", "y"]]], axis=1)

    np.save(os.path.join(output_dir, "features_raw.npy"), features)
    np.save(
        os.path.join(output_dir, "features_for_distance.npy"), features_for_distance
    )
    np.save(os.path.join(output_dir, "logits.npy"), logits)
    np.save(os.path.join(output_dir, "probs.npy"), probs)
    np.save(os.path.join(output_dir, "class_centers.npy"), centers)
    np.save(os.path.join(output_dir, "distance_matrix.npy"), dist_matrix)
    np.save(os.path.join(output_dir, "coords_2d.npy"), coords)

    difficulty_df.to_csv(
        os.path.join(output_dir, "sample_difficulty.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    coords_with_meta.to_csv(
        os.path.join(output_dir, "coords_2d_with_metadata.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_topk_hard_samples(
        df=difficulty_df,
        class_names=class_names,
        output_dir=output_dir,
        top_k=args.top_k,
    )

    save_target_neighbor_samples(
        df=difficulty_df,
        class_names=class_names,
        target_class_idx=target_class_idx,
        neighbor_class_idx=neighbor_class_idx,
        output_dir=output_dir,
        top_k=args.top_k,
    )

    scatter_by_label(
        coords=coords,
        df=df_by_original_order,
        class_names=class_names,
        output_path=os.path.join(output_dir, f"{args.reducer}_by_true_label.png"),
        title=f"{args.split_name} feature space by true label",
    )

    scatter_by_correctness(
        coords=coords,
        df=df_by_original_order,
        output_path=os.path.join(output_dir, f"{args.reducer}_by_correctness.png"),
        title=f"{args.split_name} feature space by correctness",
    )

    scatter_by_margin(
        coords=coords,
        df=df_by_original_order,
        output_path=os.path.join(output_dir, f"{args.reducer}_by_distance_margin.png"),
        title=f"{args.split_name} feature space by distance margin",
    )

    print("\nDone.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
