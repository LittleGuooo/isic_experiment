import os
from collections import Counter

import torch
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from .dataset import SavedSyntheticISICDataset
from .utils import parse_ratios, get_class_counts_from_dataset

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
    """打印本次扩散生成计划，只显示真正需要生成的类别。"""
    rows = []
    total_to_generate = 0

    for class_idx, ratio in active_ratios.items():
        original_count = original_class_counts.get(class_idx, 0)
        target_count = int(original_count * ratio)
        total_to_generate += target_count

        rows.append(
            (
                class_names[class_idx],
                original_count,
                ratio,
                target_count,
            )
        )

    print("\nDiffusion generation plan", flush=True)
    print("-" * 56, flush=True)
    print(f"{'class':<16}{'original':>10}{'ratio':>10}{'generate':>12}", flush=True)
    print("-" * 56, flush=True)

    for class_name, original_count, ratio, target_count in rows:
        print(
            f"{class_name:<16}{original_count:>10}{ratio:>10.2f}{target_count:>12}",
            flush=True,
        )

    print("-" * 56, flush=True)
    print(f"{'Total':<36}{total_to_generate:>20}", flush=True)


@torch.no_grad()
def build_diffusion_model(args, class_names, num_classes, device, output_dir):
    """加载扩散模型并生成图像所需组件"""
    if args.diffusion_checkpoint is None:
        raise ValueError("启用扩散增强时，必须提供 --diffusion_checkpoint。")

    print("\n 开始加载扩散模型", flush=True)
    print(f"  mode: {args.mode}", flush=True)
    print(f"  checkpoint: {args.diffusion_checkpoint}", flush=True)

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
        print("checkpoint 中使用键: state_dict", flush=True)
    else:
        state_dict = checkpoint
        print("checkpoint 直接作为 state_dict 使用", flush=True)

    model.load_state_dict(state_dict, strict=True)
    mode_ops["load_checkpoint_extra_state"](
        checkpoint=checkpoint,
        extra_components=extra_components,
        device=device,
    )
    model = model.to(device)
    model.eval()

    print("扩散模型加载完成，进入采样准备阶段", flush=True)

    return model, sampling_scheduler, mode_ops, extra_components


@torch.no_grad()
def build_train_dataset(
    args, train_dataset, class_names, num_classes, device, output_dir
):
    """构建训练集：可选择复用已有合成图，或现场生成新的合成图。"""

    print("\n开始构建训练数据集", flush=True)

    # 1) 只要 output_dir 里已经有合成图，就直接复用
    #    这种情况不需要 ratios，也不需要 diffusion_checkpoint。
    has_existing_images = _has_existing_generated_images(output_dir, class_names)

    if has_existing_images:
        print(f"检测到已有合成图片，直接复用: {output_dir}", flush=True)

        synth_dataset = SavedSyntheticISICDataset(
            root_dir=output_dir,
            class_names=class_names,
            transform=train_dataset.transform,
        )

        print(f"复用合成数据集大小: {len(synth_dataset)}", flush=True)

        return ConcatDataset([train_dataset, synth_dataset]), synth_dataset, output_dir

    # 2) 没有已有合成图，并且没有启用扩散增强，则只用原始训练集。
    if not args.use_diffusion_augmentation:
        print("未启用扩散增强，直接返回原始训练集", flush=True)
        return train_dataset, None, output_dir

    # 3) 启用了扩散增强，但没有可复用图片，此时必须给 ratios 生成新图。
    gen_ratios = parse_ratios(args.ratios, num_classes)
    active_ratios = {k: v for k, v in gen_ratios.items() if v > 0}

    if not active_ratios:
        raise ValueError(
            "启用了 --use-diffusion-augmentation，但没有检测到已有合成图，"
            "因此必须提供 --ratios 用于生成新合成图。"
        )

    print(f"生效类别比例: {active_ratios}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    original_class_counts = get_class_counts_from_dataset(train_dataset)
    _print_generation_plan(original_class_counts, active_ratios, class_names)

    print("未复用已有图片，开始生成新的合成图片", flush=True)

    model, sampling_scheduler, mode_ops, extra_components = build_diffusion_model(
        args=args,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
    )

    # 生成增强图像数据
    synth_samples = []
    for class_idx, ratio in active_ratios.items():
        target_count = int(original_class_counts.get(class_idx, 0) * ratio)
        if target_count <= 0:
            continue

        class_name = class_names[class_idx]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(
            f"\n类别 {class_idx} ({class_name}) 开始生成",
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

            # 生成图像
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
            f"类别 {class_idx} ({class_name}) 生成完成，共 {generated_this_class} 张",
            flush=True,
        )

    synth_dataset = SavedSyntheticISICDataset(
        root_dir=output_dir,
        class_names=class_names,
        transform=train_dataset.transform,
    )

    print(f"\n合成数据集大小: {len(synth_dataset)}", flush=True)
    print(f"保存目录: {output_dir}", flush=True)
    # _print_augmented_distribution(original_class_counts, synth_dataset, class_names)

    print(flush=True)

    return ConcatDataset([train_dataset, synth_dataset]), synth_dataset, output_dir
