import json
import os
import shutil
from collections import Counter
from datetime import datetime

import pandas as pd
import torch


def make_experiment_name(args):
    # 用当前时间生成实验时间戳，便于区分不同运行
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # data_mode 用来区分是全类别训练还是单类别训练
    mode_tag = args.data_mode

    # 如果是 single_label，就把目标类别写进实验名里，方便后续查看
    # 如果是 all，则统一记为 all_labels
    label_tag = (
        f"label_{args.target_label}"
        if args.data_mode == "single_label"
        else "all_labels"
    )

    # 根据是否开启类别条件（class conditioning）写入 cond / uncond 标记
    cond_tag = "cond" if args.use_class_conditioning else "uncond"

    # 最终实验名包含：时间、任务类型、条件方式、数据模式、分辨率、batch size、随机种子
    return (
        f"{timestamp}_ddpm_{cond_tag}_{mode_tag}_{label_tag}"
        f"_res{args.resolution}_bs{args.train_batch_size}_seed{args.seed}"
    )


def setup_experiment_folders(base_dir, exp_name):
    # 实验总目录
    exp_dir = os.path.join(base_dir, exp_name)

    # 统一管理训练过程中会用到的各类子目录
    folders = {
        "exp_dir": exp_dir,
        "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
        "metrics_dir": os.path.join(exp_dir, "metrics"),
        "metadata_dir": os.path.join(exp_dir, "metadata"),
        "samples_dir": os.path.join(exp_dir, "samples"),
        "fid_dir": os.path.join(exp_dir, "fid"),
        "fid_generated_dir": os.path.join(exp_dir, "fid_generated_images"),
    }

    # 如果目录不存在就创建；exist_ok=True 表示目录已存在时不报错
    for path in folders.values():
        os.makedirs(path, exist_ok=True)

    return folders


def make_runtime_run_name(args):
    # 运行时（val_only / infer_only）的名字也带时间戳，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 默认认为没有从 checkpoint 恢复
    ckpt_tag = "no_ckpt"

    # 如果指定了 checkpoint，就把 checkpoint 文件名写进运行名中
    if args.resume_from_checkpoint is not None:
        ckpt_tag = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]

    # 记录当前采样器类型：DDPM 或 DDIM
    sampler_tag = "ddim" if args.use_ddim_sampling else "ddpm"

    # val_only 模式下，把关键评估信息写入名字中
    if args.run_mode == "val_only":
        return (
            f"{timestamp}_val_{ckpt_tag}_{sampler_tag}"
            f"_steps{args.ddpm_num_inference_steps}_seed{args.seed}"
        )

    # infer_only 模式下，再额外记录生成类别和生成数量
    if args.run_mode == "infer_only":
        label_tag = str(args.infer_label) if args.infer_label is not None else "none"
        return (
            f"{timestamp}_infer_{ckpt_tag}_{label_tag}_{sampler_tag}"
            f"_n{args.infer_num_images}_steps{args.ddpm_num_inference_steps}_seed{args.seed}"
        )

    # 其他模式暂时只返回时间戳
    return timestamp


def setup_runtime_run_folders(exp_dir, run_mode, run_name):
    # 根据运行模式决定根目录
    if run_mode == "val_only":
        root_dir = os.path.join(exp_dir, "run_vals")
    elif run_mode == "infer_only":
        root_dir = os.path.join(exp_dir, "run_infers")
    else:
        raise ValueError(f"Unsupported run_mode for runtime folder: {run_mode}")

    # 单次运行的总目录
    run_dir = os.path.join(root_dir, run_name)

    # 这个目录结构主要服务于“只验证”和“只推理”两类运行
    folders = {
        "root_dir": root_dir,
        "run_dir": run_dir,
        "metrics_dir": os.path.join(run_dir, "metrics"),
        "generated_dir": os.path.join(run_dir, "generated_images"),
        "metadata_dir": os.path.join(run_dir, "metadata"),
        "run_config_json": os.path.join(run_dir, "run_config.json"),
        "run_summary_json": os.path.join(run_dir, "run_summary.json"),
    }

    # 只有目录需要 mkdir，json 文件路径本身不需要提前创建
    for key, path in folders.items():
        if key.endswith("_json"):
            continue
        os.makedirs(path, exist_ok=True)

    return folders


def save_json(data, json_path):
    # 保存 JSON，ensure_ascii=False 可以正常保存中文
    # indent=4 让文件更易读
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def update_epoch_metrics_csv(metrics_csv_path, row_dict):
    # 先把本轮指标转成单行 DataFrame
    row_df = pd.DataFrame([row_dict])

    # 如果历史 CSV 已存在，就读出来并在末尾追加新行
    if os.path.exists(metrics_csv_path):
        old_df = pd.read_csv(metrics_csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        # 第一次写入时直接使用当前这一行
        new_df = row_df

    # utf-8-sig 方便在一些表格软件里正确显示中文
    new_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")


def update_epoch_metrics_json(metrics_json_path, row_dict):
    # 如果历史 JSON 已存在，先读出原列表
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # 第一次写入时初始化为空列表
        data = []

    # 把当前 epoch 的结果追加到列表中
    data.append(row_dict)

    # 再整体写回 JSON 文件
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def count_labels_from_indices(labels, indices, class_names):
    # 根据给定样本下标统计每个类别出现的次数
    # labels: 所有样本的整数标签列表
    # indices: 想统计的样本下标
    # class_names: 类别名列表，用于把类别索引映射成类别名
    counter = Counter([labels[i] for i in indices])

    # 输出格式为：
    # {
    #   "MEL": 123,
    #   "NV": 456,
    #   ...
    # }
    return {
        class_name: int(counter.get(class_idx, 0))
        for class_idx, class_name in enumerate(class_names)
    }


def format_count_ratio_dict(count_dict):
    # 先求总样本数
    total = sum(count_dict.values())

    # 把每个类别的数量格式化成：
    # "123 (45.67%)"
    return {
        class_name: f"{count} ({((count / total) * 100.0 if total > 0 else 0.0):.2f}%)"
        for class_name, count in count_dict.items()
    }


def print_class_distribution(title, count_dict):
    # 以更清晰的方式把类别分布打印到控制台
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    total_count = sum(count_dict.values())

    for class_name, count in count_dict.items():
        ratio = (count / total_count * 100.0) if total_count > 0 else 0.0
        print(f"{class_name}: {count} ({ratio:.2f}%)")

    print(f"Total: {total_count} (100.00%)")
    print(f"{'=' * 60}\n")


def save_checkpoint(state, is_best, save_dir, filename="last.pth.tar"):
    # 确保 checkpoint 目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 当前 checkpoint 路径
    filepath = os.path.join(save_dir, filename)

    # 约定“最佳模型”固定保存为这个名字
    best_filepath = os.path.join(save_dir, "model_best.pth.tar")

    # 保存当前状态
    torch.save(state, filepath)

    # 如果这是当前最佳模型，再额外复制一份到 model_best.pth.tar
    if is_best:
        shutil.copyfile(filepath, best_filepath)


def disable_pipeline_progress_bar(pipeline):
    # 有些 diffusers pipeline 支持 set_progress_bar_config
    # 这里做一个安全判断，避免没有这个方法时报错
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)


def save_diffusers_model_index_copy(exp_dir, metadata_dir):
    # diffusers 保存模型后，通常会在实验目录下生成 model_index.json
    src = os.path.join(exp_dir, "model_index.json")

    # 这里额外复制一份到 metadata 目录，方便统一归档
    dst = os.path.join(metadata_dir, "diffusers_pipeline_model_index.json")

    if os.path.exists(src):
        shutil.copyfile(src, dst)

    return dst


def recover_exp_dir_from_checkpoint(checkpoint_path, checkpoint_data):
    # 优先使用 checkpoint 内部记录的 exp_dir
    # 这是最稳妥的恢复方式
    if "exp_dir" in checkpoint_data:
        return checkpoint_data["exp_dir"]

    # 如果老 checkpoint 里没有 exp_dir，就通过路径反推：
    # checkpoint_path -> checkpoints_dir -> exp_dir
    checkpoints_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    return os.path.dirname(checkpoints_dir)


def sync_experiment_metadata_for_resume(experiment_metadata, args, start_epoch, global_step):
    # 记录本次恢复训练的更新时间
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_metadata["updated_time"] = now_str

    # 记录当前使用的 checkpoint 路径
    experiment_metadata["resume_from_checkpoint"] = args.resume_from_checkpoint

    # 保存本次运行时的完整参数，便于之后排查配置差异
    experiment_metadata["last_runtime_args"] = vars(args)

    # 同步 data 相关配置
    experiment_metadata.setdefault("data", {}).update(
        {
            "train_gt_csv_path": args.train_gt_csv_path,
            "val_gt_csv_path": args.val_gt_csv_path,
            "train_img_dir": args.train_img_dir,
            "val_img_dir": args.val_img_dir,
            "data_mode": args.data_mode,
            "target_label": args.target_label,
            "use_class_conditioning": args.use_class_conditioning,
        }
    )

    # 同步 model 相关配置
    experiment_metadata.setdefault("model", {}).update(
        {
            "resolution": args.resolution,
            "ddpm_num_steps": args.ddpm_num_steps,
            "ddpm_num_inference_steps": args.ddpm_num_inference_steps,
            "ddpm_beta_schedule": args.ddpm_beta_schedule,
            "use_ddim_sampling": args.use_ddim_sampling,
            "ddim_eta": args.ddim_eta,
            "use_class_conditioning": args.use_class_conditioning,
        }
    )

    # 同步 training 相关配置
    experiment_metadata.setdefault("training", {}).update(
        {
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "start_epoch": start_epoch,
            "initial_global_step": global_step,
        }
    )

    return experiment_metadata
