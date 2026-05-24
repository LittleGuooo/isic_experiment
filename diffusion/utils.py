import json
import os
import shutil
from collections import Counter
from datetime import datetime
import gc

import pandas as pd
import torch
import math
from PIL import Image
import random
import numpy as np
from pathlib import Path
from argparse import Namespace


def cleanup_after_generation(accelerator):
    """
    在生成可视化样本 / 生成评估之后清理临时显存和共享 CUDA 引用。

    说明：
    1. gc.collect()：触发 Python 垃圾回收，释放已经没有引用的 Python 对象。
    2. torch.cuda.empty_cache()：释放 PyTorch CUDA caching allocator 中未被占用的缓存显存。
    3. torch.cuda.ipc_collect()：清理 CUDA IPC 相关的共享内存引用，适合多进程 / DataLoader / Accelerator 场景。
    4. accelerator.wait_for_everyone()：多卡训练时保证所有进程都完成生成后再清理。
    """
    accelerator.wait_for_everyone()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    accelerator.wait_for_everyone()


def set_seed(seed):
    # 固定 Python / NumPy / PyTorch 随机种子，便于复现实验
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_experiment_name(args):
    """
    生成简洁、可读的实验文件夹名称。

    命名格式示例：
        260429-1421_res128_ddpm_cond_all_seed42
        260429-1421_res128_cfg_cond_all_scale3_seed42
        260429-1421_res128_cg_cond_all_gs1_seed42
        260429-1421_res128_ldm_ae_ae4_seed42
        260429-1421_res128_latent_ddpm_z4_cond_all_seed42

    含义：
        260429-1421 : 年月日-时分
        res128      : 图像分辨率
        ddpm/cfg... : 当前扩散模型类型
        cond/uncond : 是否使用类别条件
        all/label_x : 数据模式
        seed42      : 随机种子
    """
    timestamp = datetime.now().strftime("%y%m%d-%H%M")

    # 当前训练模式：ddpm / cfg / cg / ldm_ae / latent_ddpm
    mode_tag = str(args.mode)

    # 分辨率
    res_tag = f"res{args.resolution}"

    # 类别条件信息
    if args.use_class_conditioning:
        cond_tag = "cond"
    elif args.use_cross_attention_conditioning:
        cond_tag = "cross_att"
    elif args.mode == "sd_full":
        cond_tag = ""
    else:
        cond_tag = "uncond"

    # 数据模式信息
    if args.data_mode == "single_label":
        label_tag = f"label{args.target_label}"
    else:
        label_tag = "all"

    # seed
    seed_tag = f"seed{args.seed}"

    # 不同模式补充最关键的信息，避免文件夹名过长
    extra_tags = []

    if args.mode == "cfg":
        # CFG 的核心超参是 guidance scale 和 label dropout
        extra_tags.append(f"scale{args.cfg_scale:g}")
        extra_tags.append(f"drop{args.cond_drop_prob:g}")

    elif args.mode == "cg":
        # CG 的核心超参是 classifier guidance scale
        extra_tags.append(f"gs{args.classifier_guidance_scale:g}")

    elif args.mode == "ldm_ae":
        # Autoencoder 阶段最关键的是 latent_channels
        extra_tags.append(f"ae{args.ae_latent_channels}")

    elif args.mode == "latent_ddpm":
        # latent diffusion 阶段最关键的是 latent_channels 和下采样倍率
        extra_tags.append(f"z{args.ae_latent_channels}")
        extra_tags.append(f"down{args.ae_downsample_factor}")

    name_parts = [
        timestamp,
        res_tag,
        mode_tag,
        cond_tag,
        label_tag,
        *extra_tags,
        seed_tag,
    ]

    return "_".join(name_parts)


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
    # 单次推理运行名，带时间戳，避免覆盖历史生成结果。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_tag = "no_ckpt"
    if args.resume_from_checkpoint is not None:
        ckpt_tag = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]

    sampler_tag = "ddim" if args.use_ddim_sampling else "ddpm"

    if args.run_mode == "infer_only":
        label_tag = str(args.infer_label) if args.infer_label is not None else "none"
        return (
            f"{timestamp}_infer_{ckpt_tag}_{label_tag}_{sampler_tag}"
            f"_n{args.infer_num_images}_steps{args.ddpm_num_inference_steps}_seed{args.seed}"
        )

    return timestamp


def setup_runtime_run_folders(exp_dir, run_mode, run_name):
    # 当前只保留 infer_only。
    # infer_only 的输出会放在原实验目录的 run_infers 子目录下。
    if run_mode == "infer_only":
        root_dir = os.path.join(exp_dir, "run_infers")
    else:
        raise ValueError(f"Unsupported runtime folder mode: {run_mode}")

    run_dir = os.path.join(root_dir, run_name)

    folders = {
        "root_dir": root_dir,
        "run_dir": run_dir,
        "metrics_dir": os.path.join(run_dir, "metrics"),
        "generated_dir": os.path.join(run_dir, "generated_images"),
        "metadata_dir": os.path.join(run_dir, "metadata"),
        "run_config_json": os.path.join(run_dir, "run_config.json"),
        "run_summary_json": os.path.join(run_dir, "run_summary.json"),
    }

    for key, path in folders.items():
        if key.endswith("_json"):
            continue
        os.makedirs(path, exist_ok=True)

    return folders


def create_experiment_folders(args):
    """
    Runtime 入口使用的统一实验目录创建函数。

    train:
        - 如果没有 --resume_from_checkpoint：创建新的实验目录。
        - 如果有 --resume_from_checkpoint：复用 checkpoint 所属的原实验目录。

    train_classifier:
        - 如果提供 --classifier_ckpt_path 且该文件存在：复用 classifier checkpoint 所属实验目录。
        - 否则创建新的 classifier 实验目录。

    val_only / infer_only:
        - 必须提供 --resume_from_checkpoint。
        - 复用 checkpoint 所属实验目录，并在其中创建 run_vals / run_infers 子目录。
    """

    run_mode = getattr(args, "run_mode", "train")

    # =========================================================
    # 1) train: 普通 diffusion / ldm_ae / latent_ddpm 恢复训练
    # =========================================================
    if run_mode == "train":
        resume_path = getattr(args, "resume_from_checkpoint", None)

        if resume_path is not None:
            # 从 checkpoint 读取原实验目录
            checkpoint_data = torch.load(resume_path, map_location="cpu")
            exp_dir = recover_exp_dir_from_checkpoint(
                checkpoint_path=resume_path,
                checkpoint_data=checkpoint_data,
            )

            folders = {
                "exp_dir": exp_dir,
                "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
                "metrics_dir": os.path.join(exp_dir, "metrics"),
                "metadata_dir": os.path.join(exp_dir, "metadata"),
                "samples_dir": os.path.join(exp_dir, "samples"),
                "fid_dir": os.path.join(exp_dir, "fid"),
                "fid_generated_dir": os.path.join(exp_dir, "fid_generated_images"),
            }

            for path in folders.values():
                os.makedirs(path, exist_ok=True)

            folders["exp_name"] = os.path.basename(exp_dir)
            folders["metadata_json_path"] = os.path.join(
                folders["metadata_dir"],
                "experiment_metadata.json",
            )
            folders["config_json_path"] = os.path.join(
                folders["metadata_dir"],
                "config.json",
            )

            # 写到 args 里，后面保存 checkpoint 时可以直接记录 exp_dir
            args.exp_dir = exp_dir

            return folders

        # 没有 resume 时，创建全新实验目录
        exp_name = make_experiment_name(args)
        folders = setup_experiment_folders(args.output_dir, exp_name)

        folders["exp_name"] = exp_name
        folders["metadata_json_path"] = os.path.join(
            folders["metadata_dir"],
            "experiment_metadata.json",
        )
        folders["config_json_path"] = os.path.join(
            folders["metadata_dir"],
            "config.json",
        )

        args.exp_dir = folders["exp_dir"]

        return folders

    # =========================================================
    # 2) train_classifier: CG guidance classifier 训练
    # =========================================================
    if run_mode == "train_classifier":
        classifier_resume_path = getattr(args, "classifier_ckpt_path", None)

        if classifier_resume_path is not None and os.path.isfile(
            classifier_resume_path
        ):
            checkpoint_data = torch.load(classifier_resume_path, map_location="cpu")
            exp_dir = recover_exp_dir_from_checkpoint(
                checkpoint_path=classifier_resume_path,
                checkpoint_data=checkpoint_data,
            )

            folders = {
                "exp_dir": exp_dir,
                "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
                "metrics_dir": os.path.join(exp_dir, "metrics"),
                "metadata_dir": os.path.join(exp_dir, "metadata"),
                "samples_dir": os.path.join(exp_dir, "samples"),
                "fid_dir": os.path.join(exp_dir, "fid"),
                "fid_generated_dir": os.path.join(exp_dir, "fid_generated_images"),
            }

            for path in folders.values():
                os.makedirs(path, exist_ok=True)

            folders["exp_name"] = os.path.basename(exp_dir)
            folders["metadata_json_path"] = os.path.join(
                folders["metadata_dir"],
                "experiment_metadata.json",
            )
            folders["config_json_path"] = os.path.join(
                folders["metadata_dir"],
                "config.json",
            )

            args.exp_dir = exp_dir

            return folders

        # 没有 classifier resume 时，创建新实验目录
        exp_name = make_experiment_name(args)
        folders = setup_experiment_folders(args.output_dir, exp_name)

        folders["exp_name"] = exp_name
        folders["metadata_json_path"] = os.path.join(
            folders["metadata_dir"],
            "experiment_metadata.json",
        )
        folders["config_json_path"] = os.path.join(
            folders["metadata_dir"],
            "config.json",
        )

        args.exp_dir = folders["exp_dir"]

        return folders

    # =========================================================
    # 3) val_only / infer_only: 单独验证或推理
    # =========================================================
    if run_mode == "infer_only":
        if getattr(args, "resume_from_checkpoint", None) is None:
            raise ValueError(
                "infer_only requires --resume_from_checkpoint so the experiment directory can be recovered."
            )

        checkpoint_path = args.resume_from_checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        exp_dir = recover_exp_dir_from_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_data=checkpoint_data,
        )

        run_name = make_runtime_run_name(args)
        run_folders = setup_runtime_run_folders(
            exp_dir=exp_dir,
            run_mode=run_mode,
            run_name=run_name,
        )

        folders = {
            "exp_dir": exp_dir,
            "run_name": run_name,
            "run_dir": run_folders["run_dir"],
            "metrics_dir": run_folders["metrics_dir"],
            "samples_dir": run_folders["generated_dir"],
            "fid_dir": os.path.join(exp_dir, "fid"),
            "fid_generated_dir": run_folders["generated_dir"],
            "metadata_dir": run_folders["metadata_dir"],
            "metadata_json_path": run_folders["run_summary_json"],
            "config_json_path": run_folders["run_config_json"],
            "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
        }

        args.exp_dir = exp_dir

        return folders

    raise ValueError(f"Unsupported run_mode: {run_mode}")


def make_json_serializable(obj):
    """
    把 Python / PyTorch / NumPy 中不能直接写入 JSON 的对象，
    转换成 JSON 可以保存的基础类型。
    """

    # 1. torch.Tensor
    if isinstance(obj, torch.Tensor):
        # 标量 tensor，例如 tensor(0.123)
        if obj.numel() == 1:
            return obj.detach().cpu().item()

        # 多元素 tensor，例如 tensor([1, 2, 3])
        return obj.detach().cpu().tolist()

    # 2. NumPy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 3. NumPy 标量，例如 np.float32 / np.int64
    if isinstance(obj, np.generic):
        return obj.item()

    # 4. argparse.Namespace
    if isinstance(obj, Namespace):
        return vars(obj)

    # 5. pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # 6. 字典：递归处理 value
    if isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}

    # 7. 列表 / 元组：递归处理每个元素
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]

    # 8. 其他 JSON 原生支持的类型，直接返回
    return obj


def save_json(data, path):
    """
    保存 JSON 文件。
    data: 要保存的数据
    path: 保存路径
    """

    # 关键修改：
    # 先把 Tensor / ndarray / Namespace 等对象转换成 JSON 可保存格式
    data = make_json_serializable(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
        )


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


def sync_experiment_metadata_for_resume(
    experiment_metadata, args, start_epoch, global_step
):
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


def _tensor_to_pil_image(image_tensor):
    """
    Convert a tensor image to PIL.Image.

    Supports:
        [C, H, W]
        value range [-1, 1] or [0, 1]
    """

    if image_tensor.ndim != 3:
        raise ValueError(
            f"Expected image tensor with shape [C, H, W], got {tuple(image_tensor.shape)}"
        )

    image_tensor = image_tensor.detach().cpu().float()

    # If image looks normalized to [-1, 1], convert to [0, 1].
    if image_tensor.min().item() < 0:
        image_tensor = (image_tensor + 1.0) / 2.0

    image_tensor = image_tensor.clamp(0.0, 1.0)

    # [C, H, W] -> [H, W, C]
    image_tensor = image_tensor.permute(1, 2, 0)

    image_uint8 = (image_tensor * 255.0).round().to(torch.uint8).numpy()

    if image_uint8.shape[-1] == 1:
        image_uint8 = image_uint8[:, :, 0]
        return Image.fromarray(image_uint8, mode="L")

    if image_uint8.shape[-1] == 3:
        return Image.fromarray(image_uint8, mode="RGB")

    if image_uint8.shape[-1] == 4:
        return Image.fromarray(image_uint8, mode="RGBA")

    raise ValueError(
        f"Unsupported channel count: {image_uint8.shape[-1]}. "
        "Expected 1, 3, or 4 channels."
    )


def save_image_grid(
    images,
    save_path,
    nrow=None,
    padding=2,
    background_color=(255, 255, 255),
):
    """
    Save a batch/list of images as a single grid image.

    Args:
        images:
            - torch.Tensor [B, C, H, W]
            - torch.Tensor [C, H, W]
            - list/tuple of PIL.Image
            - list/tuple of torch.Tensor [C, H, W]
        save_path:
            Output image path.
        nrow:
            Number of images per row. If None, uses ceil(sqrt(num_images)).
        padding:
            Pixel padding between images.
        background_color:
            RGB background color for the canvas.
    """

    if torch.is_tensor(images):
        if images.ndim == 3:
            pil_images = [_tensor_to_pil_image(images)]
        elif images.ndim == 4:
            pil_images = [_tensor_to_pil_image(img) for img in images]
        else:
            raise ValueError(
                f"Expected tensor shape [C, H, W] or [B, C, H, W], got {tuple(images.shape)}"
            )

    elif isinstance(images, (list, tuple)):
        pil_images = []

        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif torch.is_tensor(img):
                pil_images.append(_tensor_to_pil_image(img).convert("RGB"))
            else:
                raise TypeError(
                    f"Unsupported image type inside list: {type(img)}. "
                    "Expected PIL.Image or torch.Tensor."
                )

    else:
        raise TypeError(
            f"Unsupported images type: {type(images)}. "
            "Expected torch.Tensor, list, or tuple."
        )

    if len(pil_images) == 0:
        raise ValueError("Cannot save an empty image grid.")

    # Ensure all images have the same size.
    widths, heights = zip(*(img.size for img in pil_images))
    target_width = max(widths)
    target_height = max(heights)

    resized_images = []
    for img in pil_images:
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.BICUBIC)
        resized_images.append(img.convert("RGB"))

    num_images = len(resized_images)

    if nrow is None:
        nrow = int(math.ceil(math.sqrt(num_images)))

    nrow = max(1, int(nrow))
    ncol = int(math.ceil(num_images / nrow))

    grid_width = nrow * target_width + padding * (nrow - 1)
    grid_height = ncol * target_height + padding * (ncol - 1)

    grid = Image.new("RGB", (grid_width, grid_height), background_color)

    for idx, img in enumerate(resized_images):
        row = idx // nrow
        col = idx % nrow

        x = col * (target_width + padding)
        y = row * (target_height + padding)

        grid.paste(img, (x, y))

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    grid.save(save_path)
