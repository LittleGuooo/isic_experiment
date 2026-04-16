import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import count_labels_from_indices


class ISIC2018DDPMDataset(Dataset):
    def __init__(
        self,
        gt_csv_path,
        img_dir,
        transform=None,
        data_mode="all",
        target_label=None,
        exclude_label_name=None,
    ):
        # 保存基础配置，后续 __getitem__ 会直接使用
        self.img_dir = img_dir
        self.transform = transform
        self.data_mode = data_mode
        self.target_label = target_label
        self.exclude_label_name = exclude_label_name

        # 读取官方 GroundTruth CSV
        # CSV 里通常包含 image 列 + 多个 one-hot 类别列
        df = pd.read_csv(gt_csv_path)

        # 所有不是 image 的列都视为类别列
        self.class_columns = [c for c in df.columns if c != "image"]

        # one-hot -> 整数类别索引
        df["label_int"] = df[self.class_columns].values.argmax(axis=1)

        # 如果指定了需要剔除的类别名，就在这里做过滤
        # 这里只按类别名过滤，便于精确控制只剔除 NV
        if exclude_label_name is not None:
            if exclude_label_name not in self.class_columns:
                raise ValueError(
                    f"exclude_label_name '{exclude_label_name}' not found in class columns: {self.class_columns}"
                )

            exclude_label_idx = self.class_columns.index(exclude_label_name)

            # 过滤掉该类别的所有样本
            df = df[df["label_int"] != exclude_label_idx].copy().reset_index(drop=True)

        # 如果只保留某一类，就在这里做过滤
        if data_mode == "single_label":
            if target_label is None:
                raise ValueError(
                    "When data_mode='single_label', --target_label must be provided."
                )

            # 既支持传类别名，也支持直接传数字字符串
            if str(target_label).isdigit():
                target_label_idx = int(target_label)
            else:
                if target_label not in self.class_columns:
                    raise ValueError(
                        f"target_label '{target_label}' not found in class columns: {self.class_columns}"
                    )
                target_label_idx = self.class_columns.index(target_label)

            # 只保留目标类别样本
            df = df[df["label_int"] == target_label_idx].copy().reset_index(drop=True)

            # 记录当前筛选到的类别，方便后续打印或调试
            self.selected_label_idx = target_label_idx
            self.selected_label_name = self.class_columns[target_label_idx]
        else:
            self.selected_label_idx = None
            self.selected_label_name = None

        # 重新整理索引，避免过滤后索引不连续
        self.df = df.reset_index(drop=True)

        # labels 列表常用于统计类别分布
        self.labels = self.df["label_int"].astype(int).tolist()

    def __len__(self):
        # Dataset 长度 = 样本条数
        return len(self.df)

    def __getitem__(self, idx):
        # 取出第 idx 行元信息
        row = self.df.iloc[idx]
        image_id = str(row["image"])

        # 根据 image_id 拼出图片路径
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")

        # ISIC 图像按 RGB 读入
        image = Image.open(img_path).convert("RGB")

        # 若提供了 transform，则把 PIL Image 转成模型可用张量
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_int"])

        # 返回字典，方便训练阶段按键名取数据
        return {"input": image, "label": label, "sample_id": image_id}


def build_image_transforms(resolution):
    # 这里的归一化会把像素从 [0, 1] 映射到 [-1, 1]
    # 这与扩散模型常见输入范围一致
    return transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def normalize_label_to_index_and_name(target_label, class_names):
    # 把用户传入的类别名 / 类别索引，统一转换成 (index, name)
    if target_label is None:
        return None, None

    if str(target_label).isdigit():
        target_idx = int(target_label)
        if target_idx < 0 or target_idx >= len(class_names):
            raise ValueError(
                f"target_label index {target_idx} out of range [0, {len(class_names)-1}]"
            )
        return target_idx, class_names[target_idx]

    if target_label not in class_names:
        raise ValueError(
            f"target_label '{target_label}' not found in class names: {class_names}"
        )

    return class_names.index(target_label), target_label


def build_datasets_and_loaders(args):
    # 先构造图像预处理
    image_transforms = build_image_transforms(args.resolution)

    # 分别构造 train / val 数据集
    train_dataset = ISIC2018DDPMDataset(
        gt_csv_path=args.train_gt_csv_path,
        img_dir=args.train_img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label,
        exclude_label_name="NV" if args.exclude_train_nv else None,
    )

    val_dataset = ISIC2018DDPMDataset(
        gt_csv_path=args.val_gt_csv_path,
        img_dir=args.val_img_dir,
        transform=image_transforms,
        data_mode=args.data_mode,
        target_label=args.target_label,
    )

    # 从训练集 GT CSV 中读取类别名
    gt_df = pd.read_csv(args.train_gt_csv_path)
    class_names = [c for c in gt_df.columns if c != "image"]
    num_classes = len(class_names)

    # 这里构造索引数组，供类别统计函数使用
    train_indices = np.arange(len(train_dataset))
    val_indices = np.arange(len(val_dataset))

    # 统计各类别样本数，后面会用于分配采样数量
    train_class_distribution = count_labels_from_indices(
        train_dataset.labels, train_indices, class_names
    )
    val_class_distribution = count_labels_from_indices(
        val_dataset.labels, val_indices, class_names
    )

    # 如果有 CUDA，则开启 pin_memory 往往更利于主机到 GPU 传输
    pin_memory = False
    try:
        import torch

        pin_memory = torch.cuda.is_available()
    except Exception:
        pass

    # 训练 DataLoader：shuffle=True，drop_last=True
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    # train_eval_loader：用于在训练集上做评估
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=2,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    # val_eval_loader：验证集评估不需要打乱顺序
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=2,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    # 统一打包返回，减少主流程里的局部变量数量
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_dataloader": train_dataloader,
        "train_eval_loader": train_eval_loader,
        "val_eval_loader": val_eval_loader,
        "class_names": class_names,
        "num_classes": num_classes,
        "train_class_distribution": train_class_distribution,
        "val_class_distribution": val_class_distribution,
    }
