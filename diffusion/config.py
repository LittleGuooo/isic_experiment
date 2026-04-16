import argparse


def parse_args():
    # argparse.ArgumentParser 用来统一管理命令行参数
    parser = argparse.ArgumentParser(
        description="DDPM / CFG / CG baseline for ISIC2018 dermoscopy images"
    )

    # =========================
    # ldm 模式相关参数
    # =========================

    parser.add_argument(
        "--autoencoder_ckpt_path",
        type=str,
        default=None,
        help="AutoencoderKL 的 checkpoint 路径",
    )
    parser.add_argument(
        "--ae_downsample_factor",
        type=int,
        default=8,
        help="AutoencoderKL 的 downsample factor",
    )
    parser.add_argument(
        "--latent_train_sample_posterior",
        action="store_true",
        help="在训练 latent diffusion 时是否从后验分布采样 z",
    )

    # ===== LDM Autoencoder (KL) =====
    parser.add_argument(
        "--ae_latent_channels",
        type=int,
        default=4,
        help="AutoencoderKL 的 latent channels，LDM 常用 4",
    )
    parser.add_argument(
        "--ae_block_out_channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="AutoencoderKL 每层通道数，例如 64 128 256 512",
    )
    parser.add_argument(
        "--ae_layers_per_block",
        type=int,
        default=2,
        help="AutoencoderKL 每个 block 的 ResNet 层数",
    )
    parser.add_argument(
        "--ae_norm_num_groups", type=int, default=32, help="GroupNorm 的 group 数"
    )
    parser.add_argument(
        "--ae_mid_block_add_attention",
        action="store_true",
        help="是否在 VAE 的 mid block 中加入 attention",
    )
    parser.add_argument(
        "--ae_scaling_factor",
        type=float,
        default=0.18215,
        help="latent scaling factor；diffusers 文档中 AutoencoderKL 默认值为 0.18215",
    )

    # ===== Loss weights =====
    parser.add_argument(
        "--ae_recon_loss_type",
        type=str,
        default="l1",
        choices=["l1", "mse"],
        help="重建损失类型",
    )
    parser.add_argument(
        "--ae_recon_loss_weight", type=float, default=1.0, help="重建损失权重"
    )
    parser.add_argument(
        "--ae_kl_loss_weight",
        type=float,
        default=1e-6,
        help="KL 损失权重；先给一个适合最小实现的较小默认值",
    )
    parser.add_argument(
        "--ae_patch_loss_weight",
        type=float,
        default=0.0,
        help="patch-based 损失项权重；当前默认关闭，仅预留接口",
    )
    parser.add_argument(
        "--ae_perceptual_loss_weight",
        type=float,
        default=0.0,
        help="感知损失权重；默认 0 表示关闭",
    )
    parser.add_argument(
        "--ae_perceptual_resize",
        type=int,
        default=224,
        help="送入感知网络前的 resize 尺寸",
    )

    # ===== Sampling / reconstruction behavior =====
    parser.add_argument(
        "--ae_sample_posterior",
        action="store_true",
        help="训练和可视化时是否从后验分布采样 z；默认 False 时使用 posterior mean/mode，更稳定",
    )
    parser.add_argument(
        "--ae_use_slicing",
        action="store_true",
        help="是否启用 AutoencoderKL slicing 以减少显存",
    )
    parser.add_argument(
        "--ae_use_tiling",
        action="store_true",
        help="是否启用 AutoencoderKL tiling 以减少高分辨率显存",
    )

    # ----------------------------
    # cg模式相关参数
    # ----------------------------
    parser.add_argument(
        "--cg_diffusion_ckpt_path",
        type=str,
        default=None,
        help="CG 模式下，用一个已训练好的 diffusion checkpoint 来构建并训练 guidance classifier；"
        "若提供该参数，则 runtime 会跳过 diffusion 训练，只训练 classifier。",
    )
    parser.add_argument(
        "--classifier_train_epochs",
        type=int,
        default=30,
        help="CG 模式下 classifier 的训练轮数",
    )
    parser.add_argument(
        "--classifier_train_lr",
        type=float,
        default=1e-4,
        help="CG 模式下 classifier 的学习率",
    )
    parser.add_argument(
        "--classifier_train_batch_size",
        type=int,
        default=None,
        help="CG 模式下 classifier 训练 batch size；默认复用 train_batch_size",
    )
    parser.add_argument(
        "--classifier_ckpt_path",
        type=str,
        default=None,
        help="CG 模式下分类器 checkpoint 路径",
    )
    parser.add_argument(
        "--classifier_guidance_scale",
        type=float,
        default=1.0,
        help="CG 模式下分类器 guidance scale",
    )

    # ----------------------------
    # cfg模式相关参数
    # ----------------------------
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="CFG 采样时的 guidance scale，仅 mode=cfg 时生效",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.1,
        help="CFG 训练时 label dropout 概率，仅 mode=cfg 时生效",
    )

    # ----------------------------
    # 模式相关参数
    # ----------------------------
    parser.add_argument(
        "--mode",
        type=str,
        default="ddpm",
        choices=["ddpm", "cfg", "cg", "ldm_ae", "latent_ddpm"],
        help="运行模式",
    )

    # UNet / ResNet 中时间嵌入的融合方式
    parser.add_argument(
        "--resnet_time_scale_shift",
        type=str,
        default="default",
        choices=["default", "scale_shift"],
        help="ResNet 时间尺度移位方式",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="指定 .pth.tar checkpoint 路径以从上次训练中断处继续，会自动复用原实验文件夹",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="train",
        choices=["train", "val_only", "infer_only"],
        help="train: 正常训练；val_only: 仅评估；infer_only: 仅推理生成图片",
    )
    parser.add_argument(
        "--infer_label",
        type=str,
        default=None,
        choices=[
            None,
            "MEL",
            "NV",
            "BCC",
            "AKIEC",
            "BKL",
            "DF",
            "VASC",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ],
        help="infer_only 模式下指定生成类别；若模型是 conditional，则建议必须指定",
    )
    parser.add_argument(
        "--infer_num_images",
        type=int,
        default=0,
        help="infer_only 模式下要生成的图片数量",
    )
    parser.add_argument(
        "--use_ddim_sampling",
        action="store_true",
        help="推理/评估时使用 DDIM 采样器替代 DDPM，可大幅加快生成速度",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM 随机性系数：0.0 为完全确定性采样，1.0 退化为 DDPM",
    )

    # ----------------------------
    # 数据集路径相关参数
    # ----------------------------
    parser.add_argument(
        "--train_gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        help="训练集 GroundTruth CSV 路径",
    )
    parser.add_argument(
        "--val_gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        help="验证集 GroundTruth CSV 路径",
    )
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
        help="训练集图片目录",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Validation_Input",
        help="验证集图片目录",
    )

    # data_mode="all" 表示使用全部类别
    # data_mode="single_label" 表示只训练 / 评估某一个类别
    parser.add_argument(
        "--data_mode",
        type=str,
        default="all",
        choices=["all", "single_label"],
        help="all: 使用全部类别; single_label: 只使用一个类别（需配合 --target_label）",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        choices=[
            "MEL",
            "NV",
            "BCC",
            "AKIEC",
            "BKL",
            "DF",
            "VASC",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ],
        help="当 data_mode=single_label 时指定目标类别；可用类别名或对应索引",
    )
    parser.add_argument(
        "--use_class_conditioning",
        action="store_true",
        help="开启类别条件 DDPM/CFG/CG",
    )
    parser.add_argument(
        "--exclude_train_nv",
        action="store_true",
        help="若开启，则仅在训练集构建时剔除 NV 类样本；验证集不受影响",
    )

    # ----------------------------
    # 实验输出与训练配置
    # ----------------------------
    parser.add_argument(
        "--output_root",
        type=str,
        default="experiments",
        help="所有实验结果的根目录",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="训练和生成图像的分辨率，同时决定 UNet 输入尺寸",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="训练时每个 GPU 的 batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="评估/生成时的 batch size",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader 的并行读取进程数",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="总训练轮数",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="AdamW 基础学习率"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="AdamW 权重衰减系数"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="AdamW epsilon",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="学习率调度策略，传入 diffusers get_scheduler 的 name 参数",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率 warmup 步数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="训练时启用 EMA（Exponential Moving Average）权重；评估/推理/保存时优先使用 EMA 权重",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA 衰减系数",
    )

    # ----------------------------
    # 扩散过程相关参数
    # ----------------------------
    parser.add_argument(
        "--ddpm_num_steps", type=int, default=1000, help="DDPM 训练时的总扩散步数 T"
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=1000,
        help="推理/采样时的去噪步数",
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        help="噪声调度方案",
    )

    # ----------------------------
    # 评估相关参数
    # ----------------------------
    parser.add_argument(
        "--enable_per_class_metrics",
        action="store_true",
        help="Whether to compute per-class FID/KID metrics (only effective for train split).",
    )
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=20,
        help="每隔多少 epoch 保存一批可视化生成样本",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=1,
        help="每隔多少 epoch 保存一次模型 checkpoint",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=20,
        help="每隔多少 epoch 计算一次 FID 和 Precision/Recall",
    )
    parser.add_argument(
        "--num_fid_samples_train",
        type=int,
        default=1024,
        help="用于计算训练集 FID 的生成图片数量；0 表示跳过训练集 FID",
    )
    parser.add_argument(
        "--num_fid_samples_val",
        "--num_fid_samples_valid",
        dest="num_fid_samples_val",
        type=int,
        default=0,
        help="用于计算验证集 FID 的生成图片数量；0 表示跳过验证集 FID",
    )
    parser.add_argument(
        "--ipr_k",
        type=int,
        default=3,
        help="流形估计的 k 近邻数",
    )

    # 随机种子用于复现实验
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")

    return parser.parse_args()


def validate_args(args):

    # =========================
    # ldm_ae 模式相关参数检查
    # =========================
    if args.mode == "ldm_ae":
        # 当前只是 AE 预训练，不需要 diffusion 采样相关约束
        if args.use_class_conditioning:
            raise ValueError(
                "ldm_ae 模式下当前不使用类别条件，请关闭 --use_class_conditioning"
            )

        if args.ae_kl_loss_weight < 0:
            raise ValueError("--ae_kl_loss_weight 必须 >= 0")

        if args.ae_patch_loss_weight < 0:
            raise ValueError("--ae_patch_loss_weight 必须 >= 0")

        if args.ae_recon_loss_weight < 0:
            raise ValueError("--ae_recon_loss_weight 必须 >= 0")

    # cg相关的参数检查
    if args.mode == "cg":
        # 只有 val_only / infer_only 才强制要求 classifier checkpoint
        if (
            args.run_mode in ["val_only", "infer_only"]
            and args.classifier_ckpt_path is None
        ):
            raise ValueError(
                "CG mode with val_only / infer_only requires --classifier_ckpt_path."
            )

    # 仅验证 / 仅推理时，必须提供 diffusion checkpoint
    if (
        args.run_mode in ["val_only", "infer_only"]
        and args.resume_from_checkpoint is None
    ):
        raise ValueError(
            f"When run_mode='{args.run_mode}', --resume_from_checkpoint must be provided."
        )

    # single_label 模式必须指定目标类别
    if args.data_mode == "single_label" and args.target_label is None:
        raise ValueError(
            "When data_mode='single_label', --target_label must be provided."
        )

    # 如果只训练单一类别 NV，同时又要求剔除训练集 NV，会导致训练集为空
    if (
        args.exclude_train_nv
        and args.data_mode == "single_label"
        and str(args.target_label).upper() in ["NV", "1"]
    ):
        raise ValueError(
            "When --exclude_train_nv is enabled, target_label cannot be NV in single_label mode."
        )

    # 仅推理时，生成张数必须 > 0
    if args.run_mode == "infer_only" and args.infer_num_images <= 0:
        raise ValueError("When run_mode='infer_only', --infer_num_images must be > 0.")

    # CFG / CG 都依赖类别条件
    if args.mode in ["cfg", "cg"] and not args.use_class_conditioning:
        raise ValueError("mode='cfg' or 'cg' requires --use_class_conditioning.")

    # CFG 的标签丢弃概率需要在 [0, 1) 内
    if args.mode == "cfg":
        if not (0.0 <= args.cond_drop_prob < 1.0):
            raise ValueError("--cond_drop_prob must be in [0, 1).")

    # 条件推理时必须给 infer_label
    if args.run_mode == "infer_only":
        if args.use_class_conditioning and args.infer_label is None:
            raise ValueError(
                "When use_class_conditioning=True and run_mode='infer_only', --infer_label must be specified."
            )
        if (not args.use_class_conditioning) and args.infer_label is not None:
            raise ValueError(
                "Current run is unconditional (use_class_conditioning=False), so --infer_label should not be specified."
            )

    return args
