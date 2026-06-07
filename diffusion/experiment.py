import csv
import json
import os
from datetime import datetime


def make_experiment_name(args):
    """
    根据关键参数生成实验名。

    不要把所有参数都塞进目录名，否则路径会非常长。
    """
    parts = [
        args.mode,
        f"res{args.resolution}",
        f"bs{args.train_batch_size}",
        f"lr{args.learning_rate}",
    ]

    if getattr(args, "data_mode", "all") == "single_label":
        parts.append(f"label_{args.target_label}")

    if getattr(args, "exclude_train_nv", False):
        parts.append("exclude_nv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)

    return "_".join(parts)


def create_experiment_folders(args):
    if getattr(args, "exp_dir", None) is not None:
        exp_dir = args.exp_dir
        exp_name = os.path.basename(os.path.normpath(exp_dir))
    else:
        exp_name = make_experiment_name(args)
        exp_dir = os.path.join(args.output_root, exp_name)

    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    samples_dir = os.path.join(exp_dir, "samples")
    eval_dir = os.path.join(exp_dir, "evaluation")
    metrics_dir = os.path.join(exp_dir, "metrics")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    return {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "samples_dir": samples_dir,
        "eval_dir": eval_dir,
        "metrics_dir": metrics_dir,
        "metadata_json_path": os.path.join(exp_dir, "metadata.json"),
        "metrics_csv_path": os.path.join(exp_dir, "metrics.csv"),
    }


def save_json(obj, path):
    """
    保存 JSON 文件。

    experiment.py 里也保留一个 save_json，
    是为了让实验相关模块不再依赖 utils.py。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_dict_to_csv(path, row):
    """
    追加一行 dict 到 CSV。

    如果文件不存在，会自动写入表头。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_args_snapshot(args, path):
    """
    保存命令行参数快照。
    """
    args_dict = {}

    for key, value in vars(args).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            args_dict[key] = value
        else:
            args_dict[key] = str(value)

    save_json(args_dict, path)
