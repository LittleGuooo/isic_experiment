import json
import os
import random

import numpy as np
import torch
from PIL import Image


def set_seed(seed):
    """
    设置随机种子。

    作用：
    让 Python random、NumPy、PyTorch 的随机行为尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_after_generation(accelerator=None):
    """
    生成图像或评估后清理显存缓存。

    注意：
    这不会释放仍被 Python 变量引用的 tensor。
    它只清理 PyTorch CUDA cache 和 CUDA IPC cache。
    """
    if accelerator is not None:
        accelerator.wait_for_everyone()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def tensor_to_uint8_image(x):
    """
    把 [-1, 1] 的图像张量转换成 [0, 255] 的 uint8 张量。

    输入:
        x: Tensor, shape 可以是 [C,H,W] 或 [B,C,H,W]

    输出:
        uint8 Tensor，数值范围 [0, 255]
    """
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def uint8_tensor_to_pil(x):
    """
    把 [C,H,W] 的 uint8 Tensor 转换成 PIL.Image。
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

    arr = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def save_image_grid(images, path, nrow=8):
    """
    保存图像网格。

    输入:
        images: Tensor, shape [B,C,H,W]，数值范围建议为 [-1,1]
        path: 输出图片路径
        nrow: 每行图片数量
    """
    from torchvision.utils import make_grid

    os.makedirs(os.path.dirname(path), exist_ok=True)

    images_uint8 = tensor_to_uint8_image(images)
    grid = make_grid(images_uint8, nrow=nrow)
    pil = uint8_tensor_to_pil(grid)
    pil.save(path)


def save_json(obj, path):
    """
    保存 JSON 文件。

    注意：
    这里只负责基础 JSON 写入。
    实验 metadata 的组织逻辑放在 experiment.py。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    """
    读取 JSON 文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_labels_from_indices(label_indices, class_names):
    """
    根据整数 label 统计类别数量。

    输入:
        label_indices: list[int] 或 ndarray
        class_names: list[str]

    输出:
        dict，例如 {"MEL": 100, "NV": 200}
    """
    counts = {name: 0 for name in class_names}

    for idx in label_indices:
        idx = int(idx)
        counts[class_names[idx]] += 1

    return counts


def format_count_ratio_dict(count_dict):
    """
    把类别计数字典转换成带比例的字典。
    """
    total = sum(count_dict.values())

    result = {}

    for name, count in count_dict.items():
        ratio = 0.0 if total == 0 else count / total
        result[name] = {
            "count": int(count),
            "ratio": float(ratio),
        }

    return result


def print_class_distribution(title, count_dict):
    """
    打印类别分布。
    """
    print(f"\n{title}")

    total = sum(count_dict.values())

    for name, count in count_dict.items():
        ratio = 0.0 if total == 0 else count / total
        print(f"  {name}: {count} ({ratio:.4f})")
