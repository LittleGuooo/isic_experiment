import json
import os
import shutil
from datetime import datetime

import torch

from .utils import save_json


def make_experiment_name(args):
    """构造简洁实验名：日期时间 + 分类器骨架 + 增强状态 + 学习率。"""
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    mode_name_map = {
        "ddpm": "ddpm",
        "cfg": "cfg",
        "cg": "cg",
        "latent_ddpm": "ldm",
        "sd_full": "sdfull",
    }

    if args.use_diffusion_augmentation:
        mode_tag = mode_name_map.get(args.mode, args.mode)
        aug_tag = f"diff-{mode_tag}"
    else:
        aug_tag = "noaug"

    return f"{timestamp}_{args.arch}_{aug_tag}_lr_{args.lr}"


def setup_experiment_folders(base_dir, exp_name):
    """创建实验目录及固定子目录。"""
    exp_dir = os.path.join(base_dir, exp_name)
    folders = {
        "exp_dir": exp_dir,
        "checkpoints_dir": os.path.join(exp_dir, "checkpoints"),
        "metrics_dir": os.path.join(exp_dir, "metrics"),
        "metadata_dir": os.path.join(exp_dir, "metadata"),
        "roc_dir": os.path.join(exp_dir, "roc_curves"),
        "cm_dir": os.path.join(exp_dir, "confusion_matrices"),
        "predictions_dir": os.path.join(exp_dir, "predictions"),
    }

    for path in folders.values():
        os.makedirs(path, exist_ok=True)
    return folders


def reuse_experiment_folders(exp_dir):
    """断点恢复训练时复用旧实验目录。"""
    return setup_experiment_folders(os.path.dirname(exp_dir), os.path.basename(exp_dir))


def load_experiment_metadata(exp_dir):
    """读取旧实验 metadata。文件不存在时返回 None。"""
    metadata_path = os.path.join(exp_dir, "metadata", "experiment_metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    state, is_best, save_dir="checkpoints", filename="checkpoint.pth.tar"
):
    """保存 checkpoint；如果 is_best=True，则额外复制为 model_best.pth.tar。"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(save_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_filepath)


def resolve_aug_output_dir(args, checkpoint, exp_folders):
    """
    决定本次训练使用哪个合成图目录。

    优先级：
    1. 用户显式传入 --aug-output-dir
    2. checkpoint 中保存的 augmentation.aug_output_dir
    3. 旧实验 metadata 中保存的 diffusion_augmentation.aug_output_dir
    4. 当前实验目录下 train_augmented_data
    """
    if args.aug_output_dir is not None:
        return args.aug_output_dir

    if checkpoint is not None:
        ckpt_aug = checkpoint.get("augmentation", {})
        ckpt_aug_dir = ckpt_aug.get("aug_output_dir")
        if ckpt_aug_dir:
            args.use_diffusion_augmentation = bool(
                ckpt_aug.get("enabled", args.use_diffusion_augmentation)
            )
            return ckpt_aug_dir

        metadata = load_experiment_metadata(checkpoint["exp_dir"])
        if metadata is not None:
            meta_aug = metadata.get("diffusion_augmentation", {})
            meta_aug_dir = meta_aug.get("aug_output_dir")
            if meta_aug_dir:
                args.use_diffusion_augmentation = bool(
                    meta_aug.get("enabled", args.use_diffusion_augmentation)
                )
                return meta_aug_dir

    return os.path.join(exp_folders["exp_dir"], "train_augmented_data")


def build_mode_specific_params(args):
    """整理只属于当前扩散增强模式的关键参数，避免 metadata 过杂。"""
    if args.mode == "ddpm":
        return {}
    if args.mode == "cfg":
        return {
            "cfg_scale": args.cfg_scale,
            "cond_drop_prob": args.cond_drop_prob,
        }
    if args.mode == "cg":
        return {
            "classifier_ckpt_path": args.classifier_ckpt_path,
            "classifier_guidance_scale": args.classifier_guidance_scale,
            "classifier_num_heads": args.classifier_num_heads,
            "classifier_use_rotary": bool(args.classifier_use_rotary),
            "classifier_feat_size": args.classifier_feat_size,
        }
    if args.mode == "latent_ddpm":
        return {
            "autoencoder_ckpt_path": args.autoencoder_ckpt_path,
            "use_cross_attention_conditioning": bool(
                args.use_cross_attention_conditioning
            ),
            "cross_attention_dim": args.cross_attention_dim,
        }
    if args.mode == "sd_full":
        return {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "sd_enable_gradient_checkpointing": bool(
                args.sd_enable_gradient_checkpointing
            ),
            "sd_enable_xformers": bool(args.sd_enable_xformers),
        }
    return {}


def save_metadata(metadata, metadata_dir):
    """保存实验 metadata，并返回保存路径。"""
    metadata_path = os.path.join(metadata_dir, "experiment_metadata.json")
    save_json(metadata, metadata_path)
    return metadata_path
