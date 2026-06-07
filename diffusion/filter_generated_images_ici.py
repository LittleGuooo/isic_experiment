#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
filter_generated_by_classifier_and_feature_distance.py

用途：
    用已经训练好的 ISIC 分类器过滤扩散模型生成样本。

核心过滤逻辑：
    1) 分类器置信度过滤：生成图必须被分类器预测为 metadata 里的 true_label_name，且 true_prob 足够高。
    2) 特征距离过滤：生成图的分类器倒数第二层特征不能离同类别真实参考样本太远。

默认参考样本来源：
    metadata.csv 里的 source_a_image_path / source_b_image_path。
    这样不需要额外准备真实训练集 CSV。

重要提醒：
    图像预处理必须和你训练分类器时尽量一致，尤其是 img_size、mean、std。
    如果预处理不一致，置信度和特征距离都会不可信。

示例：
    python filter_generated_by_classifier_and_feature_distance.py \
        --metadata_csv experiments/sd_lora_ici_outputs/metadata.csv \
        --classifier_ckpt checkpoints/best_classifier.pt \
        --arch efficientnet_b0 \
        --output_dir experiments/sd_lora_ici_filtered \
        --min_true_prob 0.70 \
        --suspect_true_prob 0.50 \
        --enable_feature_distance \
        --copy_mode copy
"""

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DEFAULT_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


# -----------------------------
# 1. 参数
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter generated ISIC images by classifier confidence and feature distance."
    )

    # 输入输出
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="生成脚本输出的 metadata.csv。必须包含 output_path 和 true_label_name。",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="过滤结果输出目录。"
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        default=None,
        help="如果 metadata 里的 output_path 是相对路径，可用该目录拼接。",
    )
    parser.add_argument(
        "--copy_mode",
        type=str,
        default="copy",
        choices=["copy", "move", "none"],
        help="copy: 复制图片到 keep/suspect/reject；move: 移动；none: 只生成 CSV 报告。",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="如果 output_dir 已存在，先删除。"
    )

    # 分类器
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        required=True,
        help="训练好的分类器 checkpoint。支持 ckpt['model_state_dict'] / ckpt['state_dict'] / 纯 state_dict。",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b3",
        ],
        help="分类器结构。若你用的是自定义模型，需要改 build_classifier()。",
    )
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="类别名顺序必须和分类器训练时的 label id 顺序一致。",
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="严格加载 checkpoint。默认 strict=False，兼容常见键名差异。",
    )

    # 图像预处理：必须尽量和训练分类器时一致
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="归一化 mean。默认 ImageNet mean。若训练时不是这个，需要改。",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="归一化 std。默认 ImageNet std。若训练时不是这个，需要改。",
    )

    # 置信度过滤
    parser.add_argument(
        "--min_true_prob",
        type=float,
        default=0.40,
        help="keep 的最低 true-label softmax 概率。",
    )
    parser.add_argument(
        "--suspect_true_prob",
        type=float,
        default=0.20,
        help="低于 min_true_prob 但高于这个值，且预测类别正确时，标记 suspect。",
    )
    parser.add_argument(
        "--reject_wrong_pred",
        action="store_true",
        default=True,
        help="预测类别不是 true_label_name 时直接 reject。默认启用。",
    )

    # 特征距离过滤
    parser.add_argument(
        "--enable_feature_distance",
        action="store_true",
        help="开启特征距离过滤。默认关闭，只做分类器置信度过滤。",
    )
    parser.add_argument(
        "--reference_csv",
        type=str,
        default=None,
        help="可选：真实参考图像 CSV。若不提供，默认使用 metadata 中 source_a/source_b。",
    )
    parser.add_argument(
        "--reference_image_col",
        type=str,
        default="image_path",
        help="reference_csv 中真实图像路径列名。",
    )
    parser.add_argument(
        "--reference_label_col",
        type=str,
        default="true_label_name",
        help="reference_csv 中类别列名。",
    )
    parser.add_argument(
        "--reference_image_root",
        type=str,
        default=None,
        help="reference_csv 中路径为相对路径时，用该 root 拼接。",
    )
    parser.add_argument(
        "--dist_metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="特征距离类型。cosine 更稳定，l2 对特征尺度更敏感。",
    )
    parser.add_argument(
        "--suspect_dist_quantile",
        type=float,
        default=0.95,
        help="真实参考样本到本类中心距离的分位数，超过则 suspect。",
    )
    parser.add_argument(
        "--reject_dist_quantile",
        type=float,
        default=0.99,
        help="真实参考样本到本类中心距离的分位数，超过则 reject。",
    )
    parser.add_argument(
        "--min_refs_per_class",
        type=int,
        default=5,
        help="每类参考样本少于该数量时，不对该类应用特征距离过滤。",
    )

    # Runtime
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def validate_args(args):
    if len(args.class_names) != args.num_classes:
        raise ValueError(
            f"class_names 数量({len(args.class_names)})必须等于 num_classes({args.num_classes})。"
        )
    if not (0 <= args.suspect_true_prob <= args.min_true_prob <= 1):
        raise ValueError("需要满足 0 <= suspect_true_prob <= min_true_prob <= 1。")
    if args.enable_feature_distance:
        if not (0 < args.suspect_dist_quantile < args.reject_dist_quantile < 1):
            raise ValueError(
                "需要满足 0 < suspect_dist_quantile < reject_dist_quantile < 1。"
            )


# -----------------------------
# 2. 模型构建和特征提取
# -----------------------------


class ClassifierWithFeatures(nn.Module):
    """包装 torchvision 分类器，同时返回 logits 和倒数第二层特征。"""

    def __init__(self, arch: str, num_classes: int):
        super().__init__()
        self.arch = arch

        if arch.startswith("resnet"):
            backbone = getattr(models, arch)(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
            self.model = backbone
            self.feature_dim = in_features

        elif arch.startswith("efficientnet"):
            backbone = getattr(models, arch)(weights=None)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            self.model = backbone
            self.feature_dim = in_features
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch.startswith("resnet"):
            m = self.model
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        if self.arch.startswith("efficientnet"):
            m = self.model
            x = m.features(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        raise RuntimeError("Unknown architecture.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        if self.arch.startswith("resnet"):
            logits = self.model.fc(features)
        else:
            logits = self.model.classifier(features)
        return logits, features


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str):
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def load_classifier(args, device):
    model = ClassifierWithFeatures(args.arch, args.num_classes)

    ckpt = torch.load(args.classifier_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 兼容 DataParallel / Lightning / 自己封装的常见前缀
    for prefix in ["module.", "model.", "net.", "classifier."]:
        state_dict = strip_prefix_if_present(state_dict, prefix)

    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)
    if not args.strict_load:
        print(
            f"[INFO] Loaded classifier with strict=False. missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(missing) > 0:
            print(f"[WARN] First missing keys: {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[WARN] First unexpected keys: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


# -----------------------------
# 3. Dataset 和路径处理
# -----------------------------


def resolve_path(path_value, root: Optional[str] = None) -> str:
    if pd.isna(path_value):
        return ""
    path = str(path_value)
    if path == "":
        return ""
    if os.path.isabs(path):
        return path
    if root is not None:
        return os.path.normpath(os.path.join(root, path))
    return path


class ImagePathDataset(Dataset):
    def __init__(self, records: List[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec["image_path"]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            ok = True
            err = ""
        except Exception as e:
            image = torch.zeros(3, rec["img_size"], rec["img_size"])
            ok = False
            err = repr(e)
        return image, idx, ok, err


def make_transform(args):
    return transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
        ]
    )


def load_generated_records(
    args, class_to_idx: Dict[str, int]
) -> Tuple[pd.DataFrame, List[dict]]:
    df = pd.read_csv(args.metadata_csv)
    required = ["output_path", "true_label_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metadata_csv 缺少必要列: {missing}")

    records = []
    for row_idx, row in df.iterrows():
        label_name = str(row["true_label_name"])
        if label_name not in class_to_idx:
            raise ValueError(f"metadata 第 {row_idx} 行出现未知类别: {label_name}")

        records.append(
            {
                "row_idx": row_idx,
                "image_path": resolve_path(row["output_path"], args.generated_root),
                "label_name": label_name,
                "label_idx": class_to_idx[label_name],
                "img_size": args.img_size,
            }
        )
    return df, records


def build_reference_records_from_metadata(
    df: pd.DataFrame, args, class_to_idx: Dict[str, int]
) -> List[dict]:
    needed = ["true_label_name", "source_a_image_path", "source_b_image_path"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            "没有提供 --reference_csv 时，需要 metadata.csv 包含这些列: " + str(missing)
        )

    seen = set()
    records = []
    for _, row in df.iterrows():
        label_name = str(row["true_label_name"])
        if label_name not in class_to_idx:
            continue
        for col in ["source_a_image_path", "source_b_image_path"]:
            path = resolve_path(row[col], None)
            key = (path, label_name)
            if path and key not in seen:
                seen.add(key)
                records.append(
                    {
                        "image_path": path,
                        "label_name": label_name,
                        "label_idx": class_to_idx[label_name],
                        "img_size": args.img_size,
                    }
                )
    return records


def build_reference_records_from_csv(args, class_to_idx: Dict[str, int]) -> List[dict]:
    ref_df = pd.read_csv(args.reference_csv)
    for col in [args.reference_image_col, args.reference_label_col]:
        if col not in ref_df.columns:
            raise ValueError(f"reference_csv 缺少列: {col}")

    records = []
    for _, row in ref_df.iterrows():
        label_name = str(row[args.reference_label_col])
        if label_name not in class_to_idx:
            continue
        records.append(
            {
                "image_path": resolve_path(
                    row[args.reference_image_col], args.reference_image_root
                ),
                "label_name": label_name,
                "label_idx": class_to_idx[label_name],
                "img_size": args.img_size,
            }
        )
    return records


# -----------------------------
# 4. 批量推理
# -----------------------------


@torch.inference_mode()
def infer_records(model, records: List[dict], transform, args, device):
    dataset = ImagePathDataset(records, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = [None] * len(records)
    all_features = [None] * len(records)
    all_errors = [""] * len(records)

    for images, idxs, oks, errs in tqdm(loader, desc="Classifier inference"):
        images = images.to(device, non_blocking=True)
        logits, features = model(images)
        logits = logits.detach().cpu()
        features = features.detach().cpu()

        for b, idx in enumerate(idxs.tolist()):
            if bool(oks[b]):
                all_logits[idx] = logits[b]
                all_features[idx] = features[b]
            else:
                all_errors[idx] = str(errs[b])

    return all_logits, all_features, all_errors


# -----------------------------
# 5. 特征距离阈值
# -----------------------------


def feature_distance(x: torch.Tensor, center: torch.Tensor, metric: str) -> float:
    if metric == "cosine":
        x = F.normalize(x.float(), dim=0)
        center = F.normalize(center.float(), dim=0)
        # cosine distance = 1 - cosine similarity，越大越远
        return float(1.0 - torch.dot(x, center).item())
    if metric == "l2":
        return float(torch.norm(x.float() - center.float(), p=2).item())
    raise ValueError(metric)


def build_class_feature_stats(
    reference_records, ref_features, ref_errors, args, idx_to_class
):
    by_class = defaultdict(list)
    for rec, feat, err in zip(reference_records, ref_features, ref_errors):
        if feat is None:
            continue
        by_class[rec["label_idx"]].append(feat)

    stats = {}
    rows = []
    for class_idx, feats in by_class.items():
        class_name = idx_to_class[class_idx]
        n = len(feats)
        if n < args.min_refs_per_class:
            rows.append(
                {
                    "class_name": class_name,
                    "num_refs": n,
                    "enabled": False,
                    "suspect_threshold": np.nan,
                    "reject_threshold": np.nan,
                    "note": f"num_refs < {args.min_refs_per_class}",
                }
            )
            continue

        feat_mat = torch.stack(feats, dim=0).float()
        center = feat_mat.mean(dim=0)
        dists = np.array(
            [feature_distance(f, center, args.dist_metric) for f in feat_mat]
        )
        suspect_th = float(np.quantile(dists, args.suspect_dist_quantile))
        reject_th = float(np.quantile(dists, args.reject_dist_quantile))

        stats[class_idx] = {
            "center": center,
            "suspect_threshold": suspect_th,
            "reject_threshold": reject_th,
            "num_refs": n,
        }
        rows.append(
            {
                "class_name": class_name,
                "num_refs": n,
                "enabled": True,
                "suspect_threshold": suspect_th,
                "reject_threshold": reject_th,
                "ref_dist_mean": float(dists.mean()),
                "ref_dist_std": float(dists.std()),
                "ref_dist_min": float(dists.min()),
                "ref_dist_max": float(dists.max()),
                "note": "",
            }
        )

    return stats, pd.DataFrame(rows)


# -----------------------------
# 6. 判定逻辑
# -----------------------------


def decide_one_sample(rec, logits, feat, err, args, idx_to_class, feature_stats):
    result = {
        "image_read_ok": err == "",
        "read_error": err,
        "pred_label_name": "",
        "pred_label_idx": np.nan,
        "pred_prob": np.nan,
        "true_prob": np.nan,
        "confidence_decision": "",
        "feature_distance": np.nan,
        "feature_suspect_threshold": np.nan,
        "feature_reject_threshold": np.nan,
        "feature_decision": "not_used",
        "decision": "reject",
        "reject_reasons": [],
        "suspect_reasons": [],
    }

    if logits is None or feat is None:
        result["reject_reasons"].append("image_read_failed")
        result["confidence_decision"] = "reject"
        result["decision"] = "reject"
        return result

    probs = torch.softmax(logits.float(), dim=0)
    pred_idx = int(torch.argmax(probs).item())
    pred_prob = float(probs[pred_idx].item())
    true_idx = rec["label_idx"]
    true_prob = float(probs[true_idx].item())

    result.update(
        {
            "pred_label_name": idx_to_class[pred_idx],
            "pred_label_idx": pred_idx,
            "pred_prob": pred_prob,
            "true_prob": true_prob,
        }
    )

    # 分类器置信度过滤：这是主过滤器。
    if pred_idx != true_idx:
        result["confidence_decision"] = "reject"
        result["reject_reasons"].append("wrong_pred")
    elif true_prob >= args.min_true_prob:
        result["confidence_decision"] = "keep"
    elif true_prob >= args.suspect_true_prob:
        result["confidence_decision"] = "suspect"
        result["suspect_reasons"].append("low_true_prob")
    else:
        result["confidence_decision"] = "reject"
        result["reject_reasons"].append("very_low_true_prob")

    # 特征距离过滤：辅助过滤器。只在有足够参考样本的类别上启用。
    if args.enable_feature_distance and true_idx in feature_stats:
        stat = feature_stats[true_idx]
        dist = feature_distance(feat, stat["center"], args.dist_metric)
        result["feature_distance"] = dist
        result["feature_suspect_threshold"] = stat["suspect_threshold"]
        result["feature_reject_threshold"] = stat["reject_threshold"]

        if dist > stat["reject_threshold"]:
            result["feature_decision"] = "reject"
            result["reject_reasons"].append("feature_distance_outlier")
        elif dist > stat["suspect_threshold"]:
            result["feature_decision"] = "suspect"
            result["suspect_reasons"].append("feature_distance_high")
        else:
            result["feature_decision"] = "keep"

    elif args.enable_feature_distance:
        result["feature_decision"] = "not_enough_refs"

    # 最终决策：reject 优先级最高，其次 suspect，最后 keep。
    if result["reject_reasons"]:
        result["decision"] = "reject"
    elif result["suspect_reasons"]:
        result["decision"] = "suspect"
    else:
        result["decision"] = "keep"

    result["reject_reasons"] = ";".join(result["reject_reasons"])
    result["suspect_reasons"] = ";".join(result["suspect_reasons"])
    return result


# -----------------------------
# 7. 文件复制/移动和报告
# -----------------------------


def prepare_output_dir(args):
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["keep", "suspect", "reject"]:
        (out / split).mkdir(parents=True, exist_ok=True)
    return out


def copy_or_move_image(src: str, dst: Path, mode: str):
    if mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(src):
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    else:
        raise ValueError(mode)


def export_images(report_df: pd.DataFrame, output_dir: Path, args):
    if args.copy_mode == "none":
        return
    for _, row in tqdm(
        report_df.iterrows(), total=len(report_df), desc=f"{args.copy_mode} images"
    ):
        src = row["resolved_output_path"]
        decision = row["decision"]
        cls = row["true_label_name"]
        name = os.path.basename(src)
        dst = output_dir / decision / cls / name
        copy_or_move_image(src, dst, args.copy_mode)


def build_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        report_df.groupby(
            ["true_label_name", "nearest_wrong_class_name", "decision"], dropna=False
        )
        .size()
        .reset_index(name="count")
    )
    total = (
        report_df.groupby(["true_label_name", "nearest_wrong_class_name"], dropna=False)
        .size()
        .reset_index(name="total")
    )
    summary = summary.merge(
        total, on=["true_label_name", "nearest_wrong_class_name"], how="left"
    )
    summary["ratio"] = summary["count"] / summary["total"]
    return summary


# -----------------------------
# 8. 主流程
# -----------------------------


def main():
    args = parse_args()
    validate_args(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    class_to_idx = {name: i for i, name in enumerate(args.class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    output_dir = prepare_output_dir(args)
    print(f"[INFO] device={device}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] class order={args.class_names}")
    print(
        "[WARN] Make sure img_size/mean/std are exactly the same as classifier training."
    )

    transform = make_transform(args)
    model = load_classifier(args, device)

    metadata_df, generated_records = load_generated_records(args, class_to_idx)
    print(f"[INFO] generated samples: {len(generated_records)}")

    feature_stats = {}
    if args.enable_feature_distance:
        if args.reference_csv is not None:
            reference_records = build_reference_records_from_csv(args, class_to_idx)
            print(
                f"[INFO] reference samples from reference_csv: {len(reference_records)}"
            )
        else:
            reference_records = build_reference_records_from_metadata(
                metadata_df, args, class_to_idx
            )
            print(
                f"[INFO] reference samples from metadata source_a/source_b: {len(reference_records)}"
            )

        ref_logits, ref_features, ref_errors = infer_records(
            model, reference_records, transform, args, device
        )
        feature_stats, stats_df = build_class_feature_stats(
            reference_records, ref_features, ref_errors, args, idx_to_class
        )
        stats_path = output_dir / "feature_distance_thresholds.csv"
        stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] feature distance thresholds saved to: {stats_path}")

    gen_logits, gen_features, gen_errors = infer_records(
        model, generated_records, transform, args, device
    )

    decision_rows = []
    for rec, logits, feat, err in tqdm(
        zip(generated_records, gen_logits, gen_features, gen_errors),
        total=len(generated_records),
        desc="Deciding samples",
    ):
        decision = decide_one_sample(
            rec, logits, feat, err, args, idx_to_class, feature_stats
        )
        decision_rows.append(decision)

    decision_df = pd.DataFrame(decision_rows)
    report_df = pd.concat([metadata_df.reset_index(drop=True), decision_df], axis=1)
    report_df["resolved_output_path"] = [r["image_path"] for r in generated_records]

    report_path = output_dir / "filter_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary_df = build_summary(report_df)
    summary_path = output_dir / "filter_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    export_images(report_df, output_dir, args)

    print("[DONE]")
    print(f"[DONE] report:  {report_path}")
    print(f"[DONE] summary: {summary_path}")
    print(report_df["decision"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
