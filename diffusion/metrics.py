import math
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm.auto import tqdm

from .modeling import build_sampling_scheduler
from .utils import format_count_ratio_dict, print_class_distribution, save_json

# Manifold 用于 IPR (Improved Precision/Recall) 评估
Manifold = namedtuple("Manifold", ["features", "radii"])


def tensor_to_uint8_for_fid(x):
    # 把 [-1, 1] 张量映射到 [0, 255] 的 uint8
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def uint8_tensor_to_pil(x_uint8):
    # 从 [C, H, W] 的 uint8 张量恢复成 PIL.Image
    arr = x_uint8.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def allocate_samples_by_ratio(count_dict, total_samples):
    # 按真实数据类别比例分配每个类别要生成 / 评估的样本数
    class_names = list(count_dict.keys())
    total_count = sum(count_dict.values())
    if total_samples <= 0 or total_count == 0:
        return {k: 0 for k in class_names}

    floor_alloc = {}
    remainders = []

    # 先取每类的 floor 分配
    for class_name in class_names:
        value = count_dict[class_name] / total_count * total_samples
        floor_alloc[class_name] = int(math.floor(value))
        remainders.append((value - floor_alloc[class_name], class_name))

    # 剩余样本按小数部分从大到小补齐
    remaining = total_samples - sum(floor_alloc.values())
    remainders.sort(key=lambda x: x[0], reverse=True)
    for i in range(remaining):
        _, class_name = remainders[i]
        floor_alloc[class_name] += 1

    return floor_alloc


@torch.no_grad()
def collect_real_images_by_class(
    real_loader, device, class_names, target_counts_by_class
):
    # 把类别名映射成类别索引，便于和 batch["label"] 对齐
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    need_by_idx = {
        class_to_idx[name]: int(cnt) for name, cnt in target_counts_by_class.items()
    }

    collected = {idx: [] for idx in range(len(class_names))}
    count_now = {idx: 0 for idx in range(len(class_names))}
    total_need = sum(need_by_idx.values())

    if total_need <= 0:
        return {
            name: torch.empty(0, 3, 0, 0, dtype=torch.uint8, device=device)
            for name in class_names
        }

    progress_bar = tqdm(
        total=total_need, desc="Collect real images by class", leave=True
    )

    # 从真实数据集中按类别收集足够数量的图像
    for batch in real_loader:
        images = batch["input"].to(device)
        labels = batch["label"]

        # FID/KID 这里统一使用 uint8 图像
        images_uint8 = tensor_to_uint8_for_fid(images)

        for i in range(images_uint8.size(0)):
            label_idx = int(labels[i])

            if label_idx not in need_by_idx:
                continue
            if count_now[label_idx] >= need_by_idx[label_idx]:
                continue

            collected[label_idx].append(images_uint8[i : i + 1])
            count_now[label_idx] += 1
            progress_bar.update(1)

        if all(count_now[idx] >= need_by_idx[idx] for idx in need_by_idx):
            break

    progress_bar.close()

    result = {}
    for class_name, class_idx in class_to_idx.items():
        if len(collected[class_idx]) == 0:
            result[class_name] = torch.empty(
                0, 3, 0, 0, dtype=torch.uint8, device=device
            )
        else:
            result[class_name] = torch.cat(collected[class_idx], dim=0)
    return result


@torch.no_grad()
def generate_images_by_class_for_metrics(
    accelerator,
    model,
    noise_scheduler,
    class_names,
    target_counts_by_class,
    fake_save_root,
    save_dir_name,
    resolution,
    eval_batch_size,
    num_inference_steps,
    use_ddim_sampling=False,
    ddim_eta=0.0,
    use_class_conditioning=False,
    modes=None,
    extra_components=None,
):
    device = accelerator.device
    generated_dir = os.path.join(fake_save_root, save_dir_name)
    os.makedirs(generated_dir, exist_ok=True)

    # 根据设置切换 DDPM / DDIM 采样器
    sampling_scheduler = build_sampling_scheduler(
        noise_scheduler=noise_scheduler,
        use_ddim_sampling=use_ddim_sampling,
    )

    fake_by_class = {}
    total_need = sum(target_counts_by_class.values())
    progress_bar = tqdm(
        total=total_need, desc=f"Generate fake images ({save_dir_name})", leave=True
    )

    global_counter = 0
    for class_idx, class_name in enumerate(class_names):
        target_count = int(target_counts_by_class.get(class_name, 0))

        if target_count <= 0:
            fake_by_class[class_name] = torch.empty(
                0, 3, 0, 0, dtype=torch.uint8, device=device
            )
            continue

        class_save_dir = os.path.join(generated_dir, class_name)
        os.makedirs(class_save_dir, exist_ok=True)

        cur_fake_batches = []
        produced = 0
        batch_id = 0

        while produced < target_count:
            cur_bs = min(eval_batch_size, target_count - produced)

            # 为每个类别 / batch 固定一个可复现的随机种子
            generator = torch.Generator(device=device).manual_seed(
                1000 + class_idx * 100000 + batch_id
            )

            if use_class_conditioning:
                class_labels = torch.full(
                    (cur_bs,),
                    fill_value=class_idx,
                    device=device,
                    dtype=torch.long,
                )
            else:
                class_labels = None

            # 真正的采样逻辑由 modes["sample_images"] 提供
            fake_uint8 = modes["sample_images"](
                model=model,
                sampling_scheduler=sampling_scheduler,
                device=device,
                resolution=resolution,
                batch_size=cur_bs,
                num_inference_steps=num_inference_steps,
                generator=generator,
                class_labels=class_labels,
                extra_components=extra_components,
                return_pil_safe_uint8=True,
            )

            cur_fake_batches.append(fake_uint8)

            # 保存生成图，便于后续人工查看
            for i in range(fake_uint8.size(0)):
                pil_img = uint8_tensor_to_pil(fake_uint8[i])
                pil_img.save(
                    os.path.join(class_save_dir, f"fid_sample_{global_counter:05d}.png")
                )
                global_counter += 1

            produced += cur_bs
            batch_id += 1
            progress_bar.update(cur_bs)

        fake_by_class[class_name] = torch.cat(cur_fake_batches, dim=0)

    progress_bar.close()
    return fake_by_class, generated_dir


def concat_class_tensors(
    class_tensor_dict, class_names, target_counts_by_class, device
):
    # 按 class_names 顺序把各类别张量拼起来
    tensors = []
    for class_name in class_names:
        count = int(target_counts_by_class.get(class_name, 0))
        if count <= 0:
            continue

        ten = class_tensor_dict[class_name]
        if ten.size(0) > 0:
            tensors.append(ten[:count])

    if len(tensors) == 0:
        return torch.empty(0, 3, 0, 0, dtype=torch.uint8, device=device)

    return torch.cat(tensors, dim=0)


@torch.no_grad()
def compute_fid_from_real_and_fake(real_images_uint8, fake_images_uint8, device):
    # FID 需要 real / fake 两边都非空
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    batch_size = 32

    for i in range(0, real_images_uint8.size(0), batch_size):
        fid.update(real_images_uint8[i : i + batch_size], real=True)

    # fake 数量多于 real 时，截断到一致长度
    fake_images_uint8 = fake_images_uint8[: real_images_uint8.size(0)]
    for i in range(0, fake_images_uint8.size(0), batch_size):
        fid.update(fake_images_uint8[i : i + batch_size], real=False)

    return float(fid.compute().item())


@torch.no_grad()
def compute_kid_from_real_and_fake(
    real_images_uint8, fake_images_uint8, device, subsets=50, subset_size=50
):
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None, None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None, None

    # KID 的 subset_size 不能大于样本数
    valid_subset_size = min(
        subset_size, real_images_uint8.size(0), fake_images_uint8.size(0)
    )
    if valid_subset_size < 2:
        return None, None

    kid = KernelInceptionDistance(
        feature=2048, subsets=subsets, subset_size=valid_subset_size, normalize=False
    ).to(device)
    batch_size = 32

    for i in range(0, real_images_uint8.size(0), batch_size):
        kid.update(real_images_uint8[i : i + batch_size], real=True)

    fake_images_uint8 = fake_images_uint8[: real_images_uint8.size(0)]
    for i in range(0, fake_images_uint8.size(0), batch_size):
        kid.update(fake_images_uint8[i : i + batch_size], real=False)

    kid_mean, kid_std = kid.compute()
    return float(kid_mean.item()), float(kid_std.item())


def _build_vgg16_feature_extractor(device):
    # IPR 使用 VGG16 提取特征
    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    return vgg.to(device).eval()


@torch.no_grad()
def _extract_vgg16_features(images_uint8, vgg16, device, batch_size=64):
    # ImageNet 预训练模型对应的均值和方差
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_feats = []
    n = images_uint8.shape[0]
    for start in range(0, n, batch_size):
        batch = images_uint8[start : start + batch_size].to(device)
        batch = batch.float() / 255.0
        batch = F.interpolate(
            batch, size=(224, 224), mode="bilinear", align_corners=False
        )
        batch = (batch - mean) / std

        # 取 VGG16 中间层特征
        conv_feat = vgg16.features(batch)
        conv_feat = vgg16.avgpool(conv_feat)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        fc_feat = vgg16.classifier[:4](conv_feat)
        all_feats.append(fc_feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def _compute_pairwise_distances(X, Y=None):
    # 计算欧氏距离矩阵
    X = X.astype(np.float64)
    X_sq = np.sum(X**2, axis=1, keepdims=True)

    if Y is None:
        Y = X
        Y_sq = X_sq
    else:
        Y = Y.astype(np.float64)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True)

    diff_sq = X_sq + Y_sq.T - 2.0 * X.dot(Y.T)
    diff_sq = np.clip(diff_sq, 0, None)
    return np.sqrt(diff_sq)


def _distances_to_radii(distances, k):
    # 对每个样本，取其第 k 个近邻距离作为流形半径
    n = distances.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        kth_idx = np.argpartition(distances[i], k + 1)[: k + 1]
        radii[i] = distances[i][kth_idx].max()
    return radii


def _build_manifold(features, k):
    distances = _compute_pairwise_distances(features)
    radii = _distances_to_radii(distances, k)
    return Manifold(features, radii)


def _compute_precision_or_recall(manifold_ref, feats_query):
    # 若 query 特征落入参考流形任一点的半径内，则视为“命中”
    dist = _compute_pairwise_distances(manifold_ref.features, feats_query)
    in_manifold = (dist < manifold_ref.radii[:, None]).any(axis=0)
    return float(in_manifold.mean())


@torch.no_grad()
def compute_manifold_precision_recall(
    real_images_uint8, fake_images_uint8, device, k=3, vgg_batch_size=64
):
    if real_images_uint8 is None or fake_images_uint8 is None:
        return None, None
    if real_images_uint8.size(0) == 0 or fake_images_uint8.size(0) == 0:
        return None, None

    print("  [IPR] Loading VGG16 for Improved Precision/Recall...")
    vgg16 = _build_vgg16_feature_extractor(device)

    print("  [IPR] Extracting features for real images...")
    real_feats = _extract_vgg16_features(
        real_images_uint8, vgg16, device, batch_size=vgg_batch_size
    )

    print("  [IPR] Extracting features for fake images...")
    n_real = real_images_uint8.size(0)
    fake_feats = _extract_vgg16_features(
        fake_images_uint8[:n_real], vgg16, device, batch_size=vgg_batch_size
    )

    print(f"  [IPR] Building manifolds (k={k})...")
    manifold_real = _build_manifold(real_feats, k)
    manifold_fake = _build_manifold(fake_feats, k)

    print("  [IPR] Computing precision and recall...")
    precision = _compute_precision_or_recall(manifold_real, fake_feats)
    recall = _compute_precision_or_recall(manifold_fake, real_feats)

    del vgg16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return precision, recall


@torch.no_grad()
def evaluate_split_with_overall_and_per_class_metrics(
    split_name,
    real_loader,
    accelerator,
    model,
    noise_scheduler,
    class_names,
    dataset_count_dict,
    num_total_samples,
    fid_dir,
    fid_generated_dir,
    epoch,
    resolution,
    eval_batch_size,
    num_inference_steps,
    use_ddim_sampling,
    ddim_eta,
    use_class_conditioning,
    ipr_k,
    kid_subsets=50,
    kid_subset_size=50,
    compute_per_class_metrics=False,
    per_class_max_real_samples=None,
    modes=None,
    extra_components=None,
):
    device = accelerator.device

    # 没有评估样本需求时直接返回空结果
    if num_total_samples <= 0:
        return {
            "overall_fid": None,
            "overall_kid_mean": None,
            "overall_kid_std": None,
            "overall_precision": None,
            "overall_recall": None,
            "overall_json_path": "",
            "per_class_json_path": "",
            "generated_dir": "",
            "per_class_generated_dir": "",
            "per_class_metrics": {},
            "allocated_counts_by_class": {name: 0 for name in class_names},
            "per_class_counts_by_class": {},
        }

    allocated_counts_by_class = allocate_samples_by_ratio(
        dataset_count_dict, num_total_samples
    )
    print(f"\n[{split_name.upper()}] allocated evaluation samples by class:")
    print_class_distribution(
        f"{split_name.upper()} Eval Allocation", allocated_counts_by_class
    )

    # 先收集真实图像，再生成对应数量的假图像
    real_by_class = collect_real_images_by_class(
        real_loader, device, class_names, allocated_counts_by_class
    )
    fake_by_class, generated_dir = generate_images_by_class_for_metrics(
        accelerator=accelerator,
        model=model,
        noise_scheduler=noise_scheduler,
        class_names=class_names,
        target_counts_by_class=allocated_counts_by_class,
        fake_save_root=fid_generated_dir,
        save_dir_name=f"epoch_{epoch:03d}_{split_name}_generated",
        resolution=resolution,
        eval_batch_size=eval_batch_size,
        num_inference_steps=num_inference_steps,
        use_ddim_sampling=use_ddim_sampling,
        ddim_eta=ddim_eta,
        use_class_conditioning=use_class_conditioning,
        modes=modes,
        extra_components=extra_components,
    )

    # 汇总所有类别，计算 overall 指标
    real_overall = concat_class_tensors(
        real_by_class, class_names, allocated_counts_by_class, device
    )
    fake_overall = concat_class_tensors(
        fake_by_class, class_names, allocated_counts_by_class, device
    )

    overall_fid = compute_fid_from_real_and_fake(real_overall, fake_overall, device)
    overall_kid_mean, overall_kid_std = compute_kid_from_real_and_fake(
        real_overall,
        fake_overall,
        device,
        subsets=kid_subsets,
        subset_size=kid_subset_size,
    )
    overall_precision, overall_recall = compute_manifold_precision_recall(
        real_overall, fake_overall, device, k=ipr_k
    )

    # 保存 overall 指标到 JSON
    overall_json_path = os.path.join(
        fid_dir, f"epoch_{epoch:03d}_{split_name}_fid.json"
    )
    save_json(
        {
            "epoch": epoch,
            "split": split_name,
            "num_real_images": int(real_overall.size(0)),
            "num_fake_images": int(fake_overall.size(0)),
            "num_images_by_class": format_count_ratio_dict(allocated_counts_by_class),
            "num_images_by_class_raw": allocated_counts_by_class,
            "fid": float(overall_fid) if overall_fid is not None else None,
            "kid_mean": (
                float(overall_kid_mean) if overall_kid_mean is not None else None
            ),
            "kid_std": float(overall_kid_std) if overall_kid_std is not None else None,
            "precision": (
                float(overall_precision) if overall_precision is not None else None
            ),
            "recall": float(overall_recall) if overall_recall is not None else None,
            "ipr_k": ipr_k,
            "kid_subsets": int(kid_subsets),
            "kid_subset_size": int(kid_subset_size),
            "generated_dir": generated_dir,
            "sampler": "ddim" if use_ddim_sampling else "ddpm",
            "num_inference_steps": int(num_inference_steps),
            "ddim_eta": float(ddim_eta) if use_ddim_sampling else None,
            "use_class_conditioning": bool(use_class_conditioning),
            "overall_generated_follow_real_distribution": True,
        },
        overall_json_path,
    )

    per_class_metrics = {}
    per_class_json_path = ""
    per_class_generated_dir = ""
    per_class_counts_by_class = {}

    # 当前代码只对 train split 计算 per-class 指标
    if compute_per_class_metrics and split_name == "train":
        if per_class_max_real_samples is None:
            raise ValueError(
                "When compute_per_class_metrics=True, per_class_max_real_samples must be provided."
            )

        # 对每个类别单独限制真实样本上限
        per_class_counts_by_class = {
            class_name: min(
                int(dataset_count_dict[class_name]), int(per_class_max_real_samples)
            )
            for class_name in class_names
        }

        per_class_real_by_class = collect_real_images_by_class(
            real_loader, device, class_names, per_class_counts_by_class
        )

        # 如果 overall 阶段生成的假图还不够，就在这里补齐
        extra_counts_by_class = {}
        for class_name in class_names:
            already_have = int(fake_by_class[class_name].size(0))
            target_need = int(per_class_counts_by_class[class_name])
            extra_counts_by_class[class_name] = max(0, target_need - already_have)

        total_extra_needed = sum(extra_counts_by_class.values())
        per_class_generated_dir = generated_dir
        if total_extra_needed > 0:
            extra_fake_by_class, per_class_generated_dir = (
                generate_images_by_class_for_metrics(
                    accelerator=accelerator,
                    model=model,
                    noise_scheduler=noise_scheduler,
                    class_names=class_names,
                    target_counts_by_class=extra_counts_by_class,
                    fake_save_root=fid_generated_dir,
                    save_dir_name=f"epoch_{epoch:03d}_{split_name}_per_class_generated_topup",
                    resolution=resolution,
                    eval_batch_size=eval_batch_size,
                    num_inference_steps=num_inference_steps,
                    use_ddim_sampling=use_ddim_sampling,
                    ddim_eta=ddim_eta,
                    use_class_conditioning=use_class_conditioning,
                    modes=modes,
                    extra_components=extra_components,
                )
            )
        else:
            extra_fake_by_class = {
                class_name: fake_by_class[class_name][:0] for class_name in class_names
            }

        # 合并已有假图和补充假图
        per_class_fake_by_class = {}
        for class_name in class_names:
            per_class_fake_by_class[class_name] = torch.cat(
                [fake_by_class[class_name], extra_fake_by_class[class_name]], dim=0
            )[: per_class_counts_by_class[class_name]]

        total_per_class = sum(per_class_counts_by_class.values())
        for class_name in class_names:
            real_c = per_class_real_by_class[class_name]
            fake_c = per_class_fake_by_class[class_name]
            class_count = int(per_class_counts_by_class.get(class_name, 0))
            ratio_c = (
                (class_count / total_per_class * 100.0) if total_per_class > 0 else 0.0
            )

            if class_count <= 0 or real_c.size(0) == 0 or fake_c.size(0) == 0:
                per_class_metrics[class_name] = {
                    "num_real_images": f"{class_count} ({ratio_c:.2f}%)",
                    "num_real_images_raw": class_count,
                    "num_fake_images": f"{class_count} ({ratio_c:.2f}%)",
                    "num_fake_images_raw": class_count,
                    "fid": None,
                    "kid_mean": None,
                    "kid_std": None,
                    "precision": None,
                    "recall": None,
                }
                continue

            # 每个类别单独计算指标
            fid_c = compute_fid_from_real_and_fake(real_c, fake_c, device)
            kid_mean_c, kid_std_c = compute_kid_from_real_and_fake(
                real_c,
                fake_c,
                device,
                subsets=kid_subsets,
                subset_size=min(kid_subset_size, class_count),
            )

            # IPR 计算比较重，这里手动限制到最多 300 张
            ipr_limit = 300
            precision_c, recall_c = compute_manifold_precision_recall(
                real_c[:ipr_limit], fake_c[:ipr_limit], device, k=ipr_k
            )

            per_class_metrics[class_name] = {
                "num_real_images": f"{class_count} ({ratio_c:.2f}%)",
                "num_real_images_raw": class_count,
                "num_fake_images": f"{class_count} ({ratio_c:.2f}%)",
                "num_fake_images_raw": class_count,
                "fid": float(fid_c) if fid_c is not None else None,
                "kid_mean": float(kid_mean_c) if kid_mean_c is not None else None,
                "kid_std": float(kid_std_c) if kid_std_c is not None else None,
                "precision": float(precision_c) if precision_c is not None else None,
                "recall": float(recall_c) if recall_c is not None else None,
            }

        per_class_json_path = os.path.join(
            fid_dir, f"epoch_{epoch:03d}_{split_name}_per_class_metrics.json"
        )
        save_json(
            {
                "epoch": epoch,
                "split": split_name,
                "num_images_by_class": format_count_ratio_dict(
                    per_class_counts_by_class
                ),
                "num_images_by_class_raw": per_class_counts_by_class,
                "metrics_by_class": per_class_metrics,
                "generated_dir": per_class_generated_dir,
                "sampler": "ddim" if use_ddim_sampling else "ddpm",
                "num_inference_steps": int(num_inference_steps),
                "ddim_eta": float(ddim_eta) if use_ddim_sampling else None,
                "use_class_conditioning": bool(use_class_conditioning),
                "ipr_k": ipr_k,
                "kid_subsets": int(kid_subsets),
                "kid_subset_size": int(kid_subset_size),
                "per_class_real_sample_rule": f"min(real_class_count, {int(per_class_max_real_samples)})",
            },
            per_class_json_path,
        )

    return {
        "overall_fid": overall_fid,
        "overall_kid_mean": overall_kid_mean,
        "overall_kid_std": overall_kid_std,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_json_path": overall_json_path,
        "per_class_json_path": per_class_json_path,
        "generated_dir": generated_dir,
        "per_class_generated_dir": per_class_generated_dir,
        "per_class_metrics": per_class_metrics,
        "allocated_counts_by_class": allocated_counts_by_class,
        "per_class_counts_by_class": per_class_counts_by_class,
    }
