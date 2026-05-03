import os

from ..metrics import evaluate_split_with_overall_and_per_class_metrics
from ..utils import save_json


def run_generation_evaluation(
    args,
    split_name,
    real_loader,
    accelerator,
    model,
    noise_scheduler,
    class_names,
    dataset_count_dict,
    num_total_samples,
    exp_folders,
    epoch,
    modes,
    extra_components,
    compute_per_class_metrics=False,
):
    """
    运行一次生成质量评估。

    这里主要负责把 runtime 层参数转交给 metrics.py。
    评估指标包括：
        FID: Fréchet Inception Distance，衡量真实图像分布与生成图像分布的距离。
        KID: Kernel Inception Distance，也用于衡量真实/生成分布差异。
        IPR: Improved Precision/Recall，也就是基于特征流形的 precision/recall。
    """

    model.eval()

    result = evaluate_split_with_overall_and_per_class_metrics(
        split_name=split_name,
        real_loader=real_loader,
        accelerator=accelerator,
        model=model,
        noise_scheduler=noise_scheduler,
        class_names=class_names,
        dataset_count_dict=dataset_count_dict,
        num_total_samples=num_total_samples,
        fid_dir=exp_folders["fid_dir"],
        fid_generated_dir=exp_folders["fid_generated_dir"],
        epoch=epoch,
        resolution=args.resolution,
        eval_batch_size=args.eval_batch_size,
        # 原来是 args.num_inference_steps；
        # config.py 中已有参数名是 args.ddpm_num_inference_steps。
        num_inference_steps=args.ddpm_num_inference_steps,
        use_ddim_sampling=args.use_ddim_sampling,
        ddim_eta=args.ddim_eta,
        use_class_conditioning=args.use_class_conditioning,
        ipr_k=args.ipr_k,
        # KID 相关参数：如果 config.py 暂时没加，就用默认值。
        kid_subsets=getattr(args, "kid_subsets", 50),
        kid_subset_size=getattr(args, "kid_subset_size", 50),
        compute_per_class_metrics=compute_per_class_metrics,
        per_class_max_real_samples=getattr(args, "per_class_max_real_samples", 300),
        # 新增：控制是否计算各类生成质量指标。
        # 默认 True，保持你原来“全部计算”的行为。
        compute_fid=getattr(args, "compute_fid", True),
        compute_kid=getattr(args, "compute_kid", True),
        compute_ipr=getattr(args, "compute_ipr", True),
        modes=modes,
        extra_components=extra_components,
    )

    return result


def save_evaluation_summary(exp_folders, epoch, split_results):
    path = os.path.join(
        exp_folders["metrics_dir"],
        f"epoch_{epoch:03d}_evaluation_summary.json",
    )

    save_json(
        {
            "epoch": int(epoch),
            "results": split_results,
        },
        path,
    )

    return path
