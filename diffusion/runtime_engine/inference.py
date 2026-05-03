import os
import torch

from ..modeling import build_sampling_scheduler
from ..utils import save_image_grid


@torch.no_grad()
def run_inference_only(
    args,
    accelerator,
    model,
    noise_scheduler,
    class_names,
    modes,
    extra_components,
    output_dir,
):
    """
    只推理、不训练。

    无条件模型：
        直接生成一组图像并保存成 grid。

    条件模型：
        对每个类别分别构造 class_labels，
        每个类别生成 infer_samples_per_class 张图，
        分别保存成 infer_{class_name}.png。
    """
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    sampling_scheduler = build_sampling_scheduler(
        noise_scheduler=noise_scheduler,
        use_ddim_sampling=args.use_ddim_sampling,
    )

    device = accelerator.device

    if args.use_class_conditioning:
        # 如果是条件生成，默认每个类别生成 8 张；
        # 这个参数可以在 config.py 中显式定义。
        samples_per_class = getattr(args, "infer_samples_per_class", 8)

        all_images = []
        for class_idx, class_name in enumerate(class_names):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + class_idx * 1000
            )

            class_labels = torch.full(
                (samples_per_class,),
                fill_value=class_idx,
                device=device,
                dtype=torch.long,
            )

            images = modes["sample_images"](
                model=model,
                sampling_scheduler=sampling_scheduler,
                device=device,
                resolution=args.resolution,
                batch_size=samples_per_class,
                num_inference_steps=args.ddpm_num_inference_steps,
                generator=generator,
                class_labels=class_labels,
                extra_components=extra_components,
                return_pil_safe_uint8=True,
            )

            all_images.append(images)

            save_image_grid(
                images,
                os.path.join(output_dir, f"infer_{class_name}.png"),
            )

        return

    generator = torch.Generator(device=device).manual_seed(args.seed)

    images = modes["sample_images"](
        model=model,
        sampling_scheduler=sampling_scheduler,
        device=device,
        resolution=args.resolution,
        batch_size=args.eval_batch_size,
        num_inference_steps=args.ddpm_num_inference_steps,
        generator=generator,
        class_labels=None,
        extra_components=extra_components,
        return_pil_safe_uint8=True,
    )

    save_image_grid(
        images,
        os.path.join(output_dir, "infer_samples.png"),
    )
