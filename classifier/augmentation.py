import os
from collections import Counter

import torch
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm


from .dataset import SavedSyntheticISICDataset

from diffusion.modeling import (
    build_model,
    build_noise_scheduler,
    build_sampling_scheduler,
)
from diffusion.modes.ddpm import build_ddpm
from diffusion.modes.cfg import build_cfg
from diffusion.modes.cg import build_cg
from diffusion.modes.ldm import build_latent_ddpm

# 支持的模式工厂
MODE_FACTORY = {
    "ddpm": build_ddpm,
    "cfg": build_cfg,
    "cg": build_cg,
    "latent_ddpm": build_latent_ddpm,
}


def parse_ratios(ratios, num_classes):
    """解析用户输入的生成比例，返回字典 {class_idx: ratio}"""
    gen_ratios = {c: 0.0 for c in range(num_classes)}
    if ratios is None:
        return gen_ratios

    for item in ratios:
        class_idx, ratio = item.split(":")
        gen_ratios[int(class_idx)] = float(ratio)
    return gen_ratios


def get_class_counts_from_dataset(dataset):
    """获取数据集中每个类别的样本数量"""
    if not hasattr(dataset, "labels"):
        raise AttributeError("dataset 必须有 labels 属性")
    return dict(Counter(dataset.labels))


def _has_existing_generated_images(output_dir, class_names):
    """检查磁盘上是否已有生成图"""
    if not os.path.isdir(output_dir):
        return False

    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if any(
            fname.lower().endswith((".jpg", ".jpeg", ".png"))
            for fname in os.listdir(class_dir)
        ):
            return True
    return False


def _print_generation_plan(original_class_counts, active_ratios, class_names):
    """
    在真正开始生成前，先打印每一类计划生成多少张。
    这里只做展示，不改任何原逻辑。
    """
    print("\n[build_train_dataset] 生成计划汇总", flush=True)

    total_to_generate = 0
    for class_idx, ratio in active_ratios.items():
        original_count = original_class_counts.get(class_idx, 0)
        target_count = int(original_count * ratio)
        total_to_generate += target_count

        print(
            f"  类别 {class_idx} ({class_names[class_idx]}): "
            f"原始 {original_count} 张 -> 生成 {target_count} 张 (ratio={ratio})",
            flush=True,
        )

    print(f"[build_train_dataset] 计划总生成数: {total_to_generate}", flush=True)


def _print_augmented_distribution(original_class_counts, synth_dataset, class_names):
    """
    打印增强后数据集分布：
    - 原始数量
    - 合成数量
    - 增强后总数量
    - 百分比（占增强后总数据集的比例）
    """
    synth_class_counts = Counter(synth_dataset.labels)
    final_class_counts = {}

    total_final = 0
    for class_idx in range(len(class_names)):
        final_count = original_class_counts.get(class_idx, 0) + synth_class_counts.get(
            class_idx, 0
        )
        final_class_counts[class_idx] = final_count
        total_final += final_count

    print("\n[build_train_dataset] 增强后数据集分布", flush=True)
    print(
        "  class_idx | class_name | original | synthetic | final | percentage",
        flush=True,
    )

    for class_idx, class_name in enumerate(class_names):
        original_count = original_class_counts.get(class_idx, 0)
        synthetic_count = synth_class_counts.get(class_idx, 0)
        final_count = final_class_counts[class_idx]
        percentage = 100.0 * final_count / total_final if total_final > 0 else 0.0

        print(
            f"  {class_idx:>9} | "
            f"{class_name:<10} | "
            f"{original_count:>8} | "
            f"{synthetic_count:>9} | "
            f"{final_count:>5} | "
            f"{percentage:>8.2f}%",
            flush=True,
        )

    print(f"[build_train_dataset] 增强后总样本数: {total_final}", flush=True)


@torch.no_grad()
def build_diffusion_model(args, class_names, num_classes, device, output_dir):
    """加载扩散模型并生成图像所需组件"""
    if args.diffusion_checkpoint is None:
        raise ValueError("启用扩散增强时，必须提供 --diffusion_checkpoint。")

    print("\n[build_train_dataset] 开始加载扩散模型", flush=True)
    print(f"  mode: {args.mode}", flush=True)
    print(f"  checkpoint: {args.diffusion_checkpoint}", flush=True)
    print(f"  device: {device}", flush=True)

    checkpoint = torch.load(args.diffusion_checkpoint, map_location="cpu")

    mode_ops = MODE_FACTORY[args.mode](args)
    model = build_model(args, num_classes)
    noise_scheduler = build_noise_scheduler(args)
    sampling_scheduler = build_sampling_scheduler(
        noise_scheduler, args.use_ddim_sampling
    )
    extra_components = mode_ops["build_extra_components"](
        num_classes=num_classes, device=device
    )

    # 加载模型权重
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("[build_train_dataset] checkpoint 中使用键: state_dict", flush=True)
    else:
        state_dict = checkpoint
        print("[build_train_dataset] checkpoint 直接作为 state_dict 使用", flush=True)

    model.load_state_dict(state_dict, strict=True)
    mode_ops["load_checkpoint_extra_state"](
        checkpoint=checkpoint,
        extra_components=extra_components,
        device=device,
    )
    model = model.to(device)
    model.eval()

    print("[build_train_dataset] 扩散模型加载完成，进入采样准备阶段", flush=True)

    return model, sampling_scheduler, mode_ops, extra_components


@torch.no_grad()
def build_train_dataset(
    args, train_dataset, class_names, num_classes, device, output_dir
):
    """构建训练集，若开启扩散增强则生成合成图并合并"""

    print("\n[build_augmented_train_dataset] 开始构建增强训练数据集", flush=True)

    if not args.use_diffusion_augmentation:
        print(
            "[build_augmented_train_dataset] 未启用扩散增强，直接返回原始训练集",
            flush=True,
        )
        return train_dataset, None, output_dir

    gen_ratios = parse_ratios(args.ratios, num_classes)
    active_ratios = {k: v for k, v in gen_ratios.items() if v > 0}

    print(
        f"[build_train_dataset] 生效类别比例: {active_ratios if active_ratios else '无'}",
        flush=True,
    )

    if not active_ratios:
        print(
            "[build_train_dataset] 没有任何类别需要生成，直接返回原始训练集", flush=True
        )
        return train_dataset, None, output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 先统计原始数据集分布，并在生成前汇总每类要生成多少张
    original_class_counts = get_class_counts_from_dataset(train_dataset)
    _print_generation_plan(original_class_counts, active_ratios, class_names)

    # 如果已有生成图，直接加载并复用
    if _has_existing_generated_images(output_dir, class_names):
        print(
            "[build_train_dataset] 检测到 aug-output-dir 中已有合成图片，直接复用",
            flush=True,
        )

        synth_dataset = SavedSyntheticISICDataset(
            root_dir=output_dir,
            class_names=class_names,
            transform=train_dataset.transform,
        )

        print(
            f"[build_train_dataset] 复用合成数据集大小: {len(synth_dataset)}",
            flush=True,
        )
        _print_augmented_distribution(original_class_counts, synth_dataset, class_names)

        print(
            f"[build_train_dataset] 拼接后总大小: {len(train_dataset) + len(synth_dataset)}",
            flush=True,
        )

        return ConcatDataset([train_dataset, synth_dataset]), synth_dataset, output_dir

    print("[build_train_dataset] 未复用已有图片，开始生成新的合成图片", flush=True)

    model, sampling_scheduler, mode_ops, extra_components = build_diffusion_model(
        args=args,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
    )

    synth_samples = []

    for class_idx, ratio in active_ratios.items():
        target_count = int(original_class_counts.get(class_idx, 0) * ratio)
        if target_count <= 0:
            continue

        class_name = class_names[class_idx]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(
            f"\n[build_train_dataset] 类别 {class_idx} ({class_name}) 开始生成",
            flush=True,
        )
        print(
            f"  原始数量: {original_class_counts.get(class_idx, 0)} | "
            f"ratio: {ratio} | 目标生成数: {target_count}",
            flush=True,
        )

        generated_this_class = 0

        pbar = tqdm(
            total=target_count,
            desc=f"Generating {class_name}",
            dynamic_ncols=True,
            leave=True,
        )

        while generated_this_class < target_count:
            cur_bs = min(args.gen_batch_size, target_count - generated_this_class)

            generator = torch.Generator(device=device).manual_seed(
                args.seed + class_idx * 100000 + generated_this_class
            )

            class_labels = None
            if args.use_class_conditioning:
                class_labels = torch.full(
                    (cur_bs,),
                    fill_value=class_idx,
                    device=device,
                    dtype=torch.long,
                )

            samples_uint8 = mode_ops["sample_images"](
                model=model,
                sampling_scheduler=sampling_scheduler,
                device=device,
                resolution=args.resolution,
                batch_size=cur_bs,
                num_inference_steps=args.ddpm_num_inference_steps,
                generator=generator,
                class_labels=class_labels,
                extra_components=extra_components,
                return_pil_safe_uint8=True,
            )

            # 保存生成图
            for i in range(samples_uint8.size(0)):
                sample_id = f"synth_{class_name}_{generated_this_class + i:06d}"
                img_path = os.path.join(class_dir, f"{sample_id}.jpg")

                image = samples_uint8[i].permute(1, 2, 0).cpu().numpy()
                from PIL import Image

                Image.fromarray(image).save(img_path)
                synth_samples.append((img_path, class_idx, sample_id))

            generated_this_class += cur_bs
            pbar.update(cur_bs)

        pbar.close()

        print(
            f"[build_train_dataset] 类别 {class_idx} ({class_name}) 生成完成，共 {generated_this_class} 张",
            flush=True,
        )

    synth_dataset = SavedSyntheticISICDataset(
        root_dir=output_dir,
        class_names=class_names,
        transform=train_dataset.transform,
    )

    print(f"\n[build_train_dataset] 合成数据集大小: {len(synth_dataset)}", flush=True)
    print(f"[build_train_dataset] 保存目录: {output_dir}", flush=True)
    # _print_augmented_distribution(original_class_counts, synth_dataset, class_names)

    print(flush=True)

    return ConcatDataset([train_dataset, synth_dataset]), synth_dataset, output_dir
