import os
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISICResNetDataset(Dataset):
    """ISIC 2018 分类数据集，用于训练 ResNet 分类器。

    数据来源 CSV 文件，包含图像 ID 和 one-hot 类别标签。
    """

    def __init__(self, gt_csv_path, img_dir, transform=None):
        """
        Args:
            gt_csv_path: CSV 标注文件路径
            img_dir: 图像文件夹路径
            transform: torchvision.transforms 对象，用于预处理
        """
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(gt_csv_path)
        self.class_columns = [c for c in df.columns if c != "image"]
        self.df = df.reset_index(drop=True)
        # 将 one-hot 标签转换为整数索引
        self.labels = self.df[self.class_columns].values.argmax(axis=1).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """返回单个样本: image tensor, label, image_id"""
        row = self.df.iloc[idx]
        image_id = str(row["image"])
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # 将 one-hot 标签转换为整数 tensor
        label = torch.tensor(
            row[self.class_columns].values.astype(float).argmax(),
            dtype=torch.long,
        )
        return image, label, image_id


# ===== 文件位置：dataset.py =====
# 1) 在 SavedSyntheticISICDataset.__init__ 里新增 self.labels
# 2) 新增 format_count_ratio_dict
# 3) print_class_distribution 复用 format_count_ratio_dict，避免两套格式逻辑分裂


class SavedSyntheticISICDataset(Dataset):
    """从磁盘读取已经生成好的合成图。"""

    def __init__(self, root_dir, class_names, transform=None):
        """
        Args:
            root_dir: 合成图根目录，每个类别一个子文件夹
            class_names: 类别名称列表
            transform: torchvision.transforms 对象
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.samples = []
        self.labels = []  # 供统计类别分布时直接使用

        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, fname)
                    sample_id = os.path.splitext(fname)[0]
                    self.samples.append((img_path, label, sample_id))
                    self.labels.append(label)

    def __len__(self):
        """返回合成数据集样本总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """返回单个样本: image tensor, label, sample_id"""
        img_path, label, sample_id = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label, sample_id


def format_count_ratio_dict(count_dict):
    """
    把类别计数字典格式化成：
    {
        "MEL": "1113 (19.67%)",
        "NV": "6705 (81.23%)"
    }

    这个格式适合写入 experiment_metadata.json，
    和控制台输出保持一致。
    """
    if count_dict is None:
        return None

    total = sum(count_dict.values())
    formatted = {}

    for class_name, count in count_dict.items():
        ratio = (count / total * 100.0) if total > 0 else 0.0
        formatted[class_name] = f"{int(count)} ({ratio:.2f}%)"

    formatted["Total"] = f"{int(total)} (100.00%)"
    return formatted


def print_class_distribution(title, count_dict):
    """打印类别分布"""
    formatted = format_count_ratio_dict(count_dict)

    print(f"\n{title}")
    print("-" * 60)
    for class_name, value in formatted.items():
        print(f"{class_name:<20}: {value:>16}")


def count_labels_from_dataset(labels, class_names):
    """统计每个类别的样本数量"""
    counter = Counter(labels)
    return {
        class_name: int(counter.get(i, 0)) for i, class_name in enumerate(class_names)
    }
