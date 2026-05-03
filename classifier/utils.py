# utils.py
import json
import random
import warnings
from collections import Counter
from enum import Enum

import numpy as np
import torch


def setup_seed_and_device(args):
    """
    设置随机种子与运行设备。

    随机种子会影响：
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA

    这样做有利于复现实验结果，但可能让训练速度稍慢。
    """
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        warnings.warn(
            "You have chosen to seed training. This may slow down training a bit, but improves reproducibility."
        )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    else:
        device = torch.device("cpu")

    return device


def save_json(data, json_path):
    """
    把 Python 对象保存成 JSON 文件。
    ensure_ascii=False 表示保存中文时不转义。
    indent=4 表示格式化缩进，便于阅读。
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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
    """打印类别分布。"""
    formatted = format_count_ratio_dict(count_dict)

    print(f"\n{title}")
    print("-" * 60)
    for class_name, value in formatted.items():
        print(f"{class_name:<20}: {value:>16}")


def count_labels_from_dataset(labels, class_names):
    """根据 labels 和 class_names 统计每个类别的样本数量。"""
    counter = Counter(labels)
    return {
        class_name: int(counter.get(i, 0)) for i, class_name in enumerate(class_names)
    }


def parse_ratios(ratios, num_classes):
    """
    解析用户输入的生成比例，返回字典 {class_idx: ratio}。

    例子：
    --ratios 0:1.0 2:2.0
    会得到：
    {0: 1.0, 1: 0.0, 2: 2.0, ...}
    """
    gen_ratios = {c: 0.0 for c in range(num_classes)}
    if ratios is None:
        return gen_ratios

    for item in ratios:
        class_idx, ratio = item.split(":")
        gen_ratios[int(class_idx)] = float(ratio)
    return gen_ratios


def get_class_counts_from_dataset(dataset):
    """获取数据集中每个类别的样本数量。"""
    if not hasattr(dataset, "labels"):
        raise AttributeError("dataset 必须有 labels 属性")
    return dict(Counter(dataset.labels))


class Summary(Enum):
    """
    AverageMeter 的汇总方式。
    当前代码里没有复杂使用，但保留不影响训练。
    """

    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """
    用于统计某个指标的当前值、累计和、平均值。
    常见于训练循环里记录 loss / accuracy。
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """把统计量清零。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计量。
        val: 当前 batch 的指标值
        n: 当前 batch 的样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    计算 top-k 准确率。

    output: 模型输出 logits，shape [B, C]
    target: 真实标签，shape [B]
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
