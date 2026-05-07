import argparse


def parse_args():
    # argparse.ArgumentParser 用来统一管理命令行参数
    parser = argparse.ArgumentParser(
        description="DDPM / CFG / CG baseline for ISIC2018 dermoscopy images"
    )

    # =========================
    # Stable Diffusion Textual Inversion
    # =========================
    parser.add_argument(
        "--ti_placeholder_tokens",
        type=str,
        nargs="+",
        default=[
            "<isic-mel>",
            "<isic-nv>",
            "<isic-bcc>",
            "<isic-akiec>",
            "<isic-bkl>",
            "<isic-df>",
            "<isic-vasc>",
        ],
    )

    parser.add_argument(
        "--ti_initializer_tokens",
        type=str,
        nargs="+",
        default=[
            "melanoma",
            "nevus",
            "carcinoma",
            "lesion",
            "keratosis",
            "fibroma",
            "vascular",
        ],
    )

    # =========================
    # Stable Diffusion Full Fine-tuning
    # =========================
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="C:/Users/Admin/.cache/huggingface/hub/models--nota-ai--bk-sdm-small/snapshots/572238db7ed3a10858900803f3fc8cca53e893e0",
        # default="C:/Users/Admin/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        help="Stable Diffusion 预训练模型名称或本地路径，仅 mode=sd_full 时使用。",
    )
    parser.add_argument(
        "--sd_enable_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sd_full 模式下是否开启 UNet gradient checkpointing。默认开启以降低显存。",
    )
    parser.add_argument(
        "--sd_enable_xformers",
        action="store_true",
        help="sd_full 模式下是否开启 xFormers memory efficient attention。需要你本地已正确安装 xformers。",
    )

    # =========================
    # LDM Cross-Attention 条件注入
    # =========================
    parser.add_argument(
        "--use_cross_attention_conditioning",
        action="store_true",
        help=(
            "开启 latent_ddpm 的 cross-attention 类别条件注入方式。"
            "当前最小实现为 class label -> nn.Embedding -> encoder_hidden_states。"
            "不能与 --use_class_conditioning 同时开启。"
        ),
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=256,
        help="cross-attention 条件 token 的特征维度，即 encoder_hidden_states 最后一维。",
    )

    parser.add_argument(
        "--attention_head_dim",
        type=int,
        default=8,
        help="UNet2DConditionModel 中 attention head 的维度。",
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
        default=0.05,
        help="感知损失权重；默认 0 表示关闭",
    )
    parser.add_argument(
        "--ae_perceptual_resize",
        type=int,
        default=224,
        help="送入感知网络前的 resize 尺寸",
    )

    # ===== 对抗损失项 =====
    parser.add_argument(
        "--ae_adv_loss_weight",
        type=float,
        default=0.01,
        help="生成器对抗损失权重；默认 0 表示关闭 PatchGAN 对抗训练",
    )
    parser.add_argument(
        "--ae_adv_start_step",
        type=int,
        default=1000,
        help="从第多少个 step 开始启用对抗损失；建议先让 AE 学会基本重建",
    )
    parser.add_argument(
        "--ae_discriminator_base_channels",
        type=int,
        default=64,
        help="PatchGAN 判别器基础通道数",
    )
    parser.add_argument(
        "--ae_discriminator_lr",
        type=float,
        default=1e-4,
        help="PatchGAN 判别器学习率",
    )
    parser.add_argument(
        "--ae_discriminator_beta1",
        type=float,
        default=0.5,
        help="PatchGAN 判别器 AdamW beta1",
    )
    parser.add_argument(
        "--ae_discriminator_beta2",
        type=float,
        default=0.999,
        help="PatchGAN 判别器 AdamW beta2",
    )
    parser.add_argument(
        "--ae_discriminator_weight_decay",
        type=float,
        default=0.0,
        help="PatchGAN 判别器 AdamW weight decay",
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

    parser.add_argument(
        "--classifier_base_channels",
        type=int,
        default=128,
        help="Base channels for standalone noisy timestep classifier.",
    )

    parser.add_argument(
        "--classifier_time_dim",
        type=int,
        default=512,
        help="Timestep embedding dimension for standalone noisy timestep classifier.",
    )

    parser.add_argument(
        "--classifier_dropout",
        type=float,
        default=0.1,
        help="Dropout for standalone noisy timestep classifier.",
    )

    parser.add_argument(
        "--classifier_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for standalone noisy timestep classifier.",
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
        choices=[
            "ddpm",
            "cfg",
            "cg",
            "ldm_ae",
            "latent_ddpm",
            "sd_full",
            "sd_textual_inversion",
        ],
        help="运行模式",
    )

    # UNet / ResNet 中时间嵌入的融合方式
    parser.add_argument(
        "--resnet_time_scale_shift",
        type=str,
        default="scale_shift",
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
        choices=["train", "train_classifier"],
        help=(
            "train: 训练 diffusion / autoencoder / latent diffusion；"
            "train_classifier: 只训练 CG guidance classifier；"
        ),
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
        "--use_tensorboard",
        action="store_true",
        help="是否启用 TensorBoard 日志记录；不传该参数时默认关闭",
    )
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
        default=100,
        help="总训练轮数",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 EMA（Exponential Moving Average）权重。默认开启；可用 --no-use_ema 关闭。",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA 衰减系数",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="训练扩散模型时是否启用 WeightedRandomSampler，以提高少数类被采样的概率。",
    )

    # ----------------------------
    # 扩散过程相关参数
    # ----------------------------
    parser.add_argument(
        "--ddpm_num_steps", type=int, default=1000, help="DDPM 训练时的总扩散步数"
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=100,
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
        "--num_visual_samples",
        type=int,
        default=32,
        help=(
            "训练过程中每次保存多少张可视化样本；"
            "对普通 diffusion 表示生成图数量；"
            "对 ldm_ae 表示重建对比中的原图数量"
        ),
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=10,
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
        default=0,
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
    parser.add_argument(
        "--compute_fid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算 FID。关闭方式：--no-compute_fid",
    )

    parser.add_argument(
        "--compute_kid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算 KID。关闭方式：--no-compute_kid",
    )

    parser.add_argument(
        "--compute_ipr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算 manifold precision/recall，也就是 IPR。关闭方式：--no-compute_ipr",
    )

    parser.add_argument(
        "--kid_subsets",
        type=int,
        default=50,
        help="KID 计算时使用的 subset 数量",
    )

    parser.add_argument(
        "--kid_subset_size",
        type=int,
        default=50,
        help="KID 每个 subset 的样本数",
    )

    parser.add_argument(
        "--per_class_max_real_samples",
        type=int,
        default=300,
        help="per-class 指标计算时，每类最多使用多少真实图像",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值",
    )

    # 随机种子用于复现实验
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")

    return parser.parse_args()


def validate_args(args):
    # sd_full 模式相关参数检查
    if args.mode == "sd_full":
        if args.pretrained_model_name_or_path is None:
            raise ValueError("mode='sd_full' requires --pretrained_model_name_or_path.")

    # use_class_conditioning 和 use_cross_attention_conditioning 是两种不同的条件注入路径。
    # 当前最小实现中不允许同时开启，避免类别条件被重复注入。
    if args.use_class_conditioning and args.use_cross_attention_conditioning:
        raise ValueError(
            "--use_class_conditioning and --use_cross_attention_conditioning cannot be enabled at the same time. "
            "--use_class_conditioning uses num_class_embeds + class_labels; "
            "--use_cross_attention_conditioning uses class label -> nn.Embedding -> encoder_hidden_states."
        )

    # 当前 cross-attention 条件注入只给 latent_ddpm 使用。
    if args.use_cross_attention_conditioning and args.mode != "latent_ddpm":
        raise ValueError(
            "--use_cross_attention_conditioning is currently only supported when mode='latent_ddpm'."
        )

    # ldm_ae 模式相关参数检查
    if args.mode == "ldm_ae":
        # 当前只是 AE 预训练，不需要 diffusion 采样相关约束
        if args.use_class_conditioning:
            raise ValueError(
                "ldm_ae 模式下当前不使用类别条件，请关闭 --use_class_conditioning"
            )

    if args.mode == "cg" and args.run_mode == "train":
        if args.classifier_ckpt_path is None:
            # 允许先训练 diffusion；没有 classifier 时，CG 采样不会启用 classifier guidance。
            pass

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

    # CFG / CG 都依赖类别条件
    if args.mode in ["cfg", "cg"] and not args.use_class_conditioning:
        raise ValueError("mode='cfg' or 'cg' requires --use_class_conditioning.")

    # 兼容别名
    args.output_dir = args.output_root

    return args
