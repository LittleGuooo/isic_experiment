import argparse


def add_stable_diffusion_lora_args(parser):
    """
    Stable Diffusion LoRA 参数。

    只负责注册 LoRA 相关命令行参数。
    不在这里写训练逻辑。
    """
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank。越大可训练容量越强，但显存占用和过拟合风险也更高。",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha。通常先设成和 rank 相同。",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout。医学小数据可先用 0.0 或 0.05。",
    )
    parser.add_argument(
        "--lora_init",
        type=str,
        default="gaussian",
        choices=["gaussian", "default"],
        help="LoRA 初始化方式。diffusers 官方示例常用 gaussian。",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="LoRA 注入到 UNet attention 的哪些线性层。",
    )


def add_stable_diffusion_textual_inversion_args(parser):
    """
    Stable Diffusion Textual Inversion 参数。
    """
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
        help="Textual Inversion 为每个类别学习的占位 token。",
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
        help="Textual Inversion 占位 token 的初始化词。",
    )


def add_stable_diffusion_full_args(parser):
    """
    Stable Diffusion full fine-tuning 参数。
    """
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=(
            "C:/Users/Admin/.cache/huggingface/hub/"
            "models--nota-ai--bk-sdm-small/"
            "snapshots/572238db7ed3a10858900803f3fc8cca53e893e0"
        ),
        help="Stable Diffusion 预训练模型名称或本地路径。",
    )
    parser.add_argument(
        "--sd_enable_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sd_full 模式下是否开启 UNet gradient checkpointing。",
    )
    parser.add_argument(
        "--sd_enable_xformers",
        action="store_true",
        help="sd_full 模式下是否开启 xFormers memory efficient attention。",
    )


def add_ldm_conditioning_args(parser):
    """
    latent_ddpm 的 cross-attention 条件注入参数。
    """
    parser.add_argument(
        "--use_cross_attention_conditioning",
        action="store_true",
        help=(
            "开启 latent_ddpm 的 cross-attention 类别条件注入。"
            "不能与 --use_class_conditioning 同时开启。"
        ),
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=256,
        help="cross-attention 条件 token 的特征维度。",
    )
    parser.add_argument(
        "--attention_head_dim",
        type=int,
        default=8,
        help="UNet2DConditionModel 中 attention head 的维度。",
    )


def add_ldm_autoencoder_args(parser):
    """
    LDM 第一阶段 AutoencoderKL 训练参数。
    """
    parser.add_argument(
        "--autoencoder_ckpt_path",
        type=str,
        default=None,
        help="AutoencoderKL 的 Diffusers save_pretrained 目录。",
    )
    parser.add_argument(
        "--ae_downsample_factor",
        type=int,
        default=8,
        help="AutoencoderKL 的下采样倍率。",
    )
    parser.add_argument(
        "--latent_train_sample_posterior",
        action="store_true",
        help="latent diffusion 训练时是否从 VAE posterior 采样 z。",
    )
    parser.add_argument(
        "--ae_latent_channels",
        type=int,
        default=4,
        help="AutoencoderKL latent channels，LDM 常用 4。",
    )
    parser.add_argument(
        "--ae_block_out_channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="AutoencoderKL 每层通道数。",
    )
    parser.add_argument(
        "--ae_layers_per_block",
        type=int,
        default=2,
        help="AutoencoderKL 每个 block 的 ResNet 层数。",
    )
    parser.add_argument(
        "--ae_norm_num_groups",
        type=int,
        default=32,
        help="GroupNorm 的 group 数。",
    )
    parser.add_argument(
        "--ae_mid_block_add_attention",
        action="store_true",
        help="是否在 VAE mid block 中加入 attention。",
    )
    parser.add_argument(
        "--ae_scaling_factor",
        type=float,
        default=0.18215,
        help="latent scaling factor；Stable Diffusion VAE 常用 0.18215。",
    )


def add_ldm_autoencoder_loss_args(parser):
    """
    LDM AutoencoderKL 的损失项参数。
    """
    parser.add_argument(
        "--ae_recon_loss_type",
        type=str,
        default="l1",
        choices=["l1", "mse"],
        help="重建损失类型。",
    )
    parser.add_argument(
        "--ae_recon_loss_weight",
        type=float,
        default=1.0,
        help="重建损失权重。",
    )
    parser.add_argument(
        "--ae_kl_loss_weight",
        type=float,
        default=1e-6,
        help="KL 损失权重。",
    )
    parser.add_argument(
        "--ae_patch_loss_weight",
        type=float,
        default=0.0,
        help="patch-based 损失项权重；当前默认关闭。",
    )
    parser.add_argument(
        "--ae_perceptual_loss_weight",
        type=float,
        default=0.05,
        help="感知损失权重；0 表示关闭。",
    )
    parser.add_argument(
        "--ae_perceptual_resize",
        type=int,
        default=224,
        help="送入感知网络前的 resize 尺寸。",
    )


def add_ldm_autoencoder_gan_args(parser):
    """
    LDM AutoencoderKL 的 PatchGAN 对抗训练参数。
    """
    parser.add_argument(
        "--ae_adv_loss_weight",
        type=float,
        default=0.01,
        help="生成器对抗损失权重；0 表示关闭 PatchGAN。",
    )
    parser.add_argument(
        "--ae_adv_start_step",
        type=int,
        default=1000,
        help="从第多少个 step 开始启用对抗损失。",
    )
    parser.add_argument(
        "--ae_discriminator_base_channels",
        type=int,
        default=64,
        help="PatchGAN 判别器基础通道数。",
    )
    parser.add_argument(
        "--ae_discriminator_lr",
        type=float,
        default=1e-4,
        help="PatchGAN 判别器学习率。",
    )
    parser.add_argument(
        "--ae_discriminator_beta1",
        type=float,
        default=0.5,
        help="PatchGAN 判别器 AdamW beta1。",
    )
    parser.add_argument(
        "--ae_discriminator_beta2",
        type=float,
        default=0.999,
        help="PatchGAN 判别器 AdamW beta2。",
    )
    parser.add_argument(
        "--ae_discriminator_weight_decay",
        type=float,
        default=0.0,
        help="PatchGAN 判别器 AdamW weight decay。",
    )


def add_ldm_autoencoder_runtime_args(parser):
    """
    LDM AutoencoderKL 的运行行为参数。
    """
    parser.add_argument(
        "--ae_sample_posterior",
        action="store_true",
        help="训练和可视化时是否从后验分布采样 z。",
    )
    parser.add_argument(
        "--ae_use_slicing",
        action="store_true",
        help="是否启用 AutoencoderKL slicing 以减少显存。",
    )
    parser.add_argument(
        "--ae_use_tiling",
        action="store_true",
        help="是否启用 AutoencoderKL tiling 以减少高分辨率显存。",
    )


def add_classifier_guidance_args(parser):
    """
    Classifier Guidance 相关参数。

    注意：
    classifier 的独立训练已经移动到单独脚本。
    这些参数仍保留，因为 cg 采样/训练可能仍需要读取分类器 checkpoint。
    """
    parser.add_argument(
        "--classifier_train_epochs",
        type=int,
        default=30,
        help="CG classifier 独立训练轮数。",
    )
    parser.add_argument(
        "--classifier_train_lr",
        type=float,
        default=1e-4,
        help="CG classifier 学习率。",
    )
    parser.add_argument(
        "--classifier_ckpt_path",
        type=str,
        default=None,
        help="CG 模式下分类器 checkpoint 路径。",
    )
    parser.add_argument(
        "--classifier_guidance_scale",
        type=float,
        default=1.0,
        help="CG 采样时的 classifier guidance scale。",
    )
    parser.add_argument(
        "--classifier_base_channels",
        type=int,
        default=128,
        help="Standalone noisy timestep classifier 的基础通道数。",
    )
    parser.add_argument(
        "--classifier_time_dim",
        type=int,
        default=512,
        help="Standalone noisy timestep classifier 的 timestep embedding 维度。",
    )
    parser.add_argument(
        "--classifier_dropout",
        type=float,
        default=0.1,
        help="Standalone noisy timestep classifier 的 dropout。",
    )
    parser.add_argument(
        "--classifier_weight_decay",
        type=float,
        default=0.01,
        help="Standalone noisy timestep classifier 的 weight decay。",
    )


def add_cfg_args(parser):
    """
    Classifier-Free Guidance 参数。
    """
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="CFG 采样时的 guidance scale。",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.1,
        help="CFG 训练时 label dropout 概率。",
    )


def add_mode_args(parser):
    """
    模式选择参数。
    """
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
            "sd_lora",
            "sd_textual_inversion",
        ],
        help="运行模式。",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="train",
        choices=["train"],
        help=(
            "主训练入口只支持 train。"
            "CG classifier 独立训练请使用 scripts/train_classifier.py。"
            "infer_only 已删除。"
        ),
    )
    parser.add_argument(
        "--resnet_time_scale_shift",
        type=str,
        default="scale_shift",
        choices=["default", "scale_shift"],
        help="ResNet 时间嵌入的融合方式。",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="指定 .pth.tar checkpoint 路径以继续训练。",
    )


def add_data_args(parser):
    """
    数据集路径和类别选择参数。
    """
    parser.add_argument(
        "--train_gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        help="训练集 GroundTruth CSV 路径。",
    )
    parser.add_argument(
        "--val_gt_csv_path",
        type=str,
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        help="验证集 GroundTruth CSV 路径。",
    )
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
        help="训练集图片目录。",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Validation_Input",
        help="验证集图片目录。",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="all",
        choices=["all", "single_label"],
        help="all 使用全部类别；single_label 只使用一个类别。",
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
        help="data_mode=single_label 时指定目标类别。",
    )
    parser.add_argument(
        "--use_class_conditioning",
        action="store_true",
        help="开启类别条件 DDPM/CFG/CG。",
    )
    parser.add_argument(
        "--exclude_train_nv",
        action="store_true",
        help="仅训练集剔除 NV 类样本。",
    )


def add_training_args(parser):
    """
    通用训练参数。
    """
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="是否启用 TensorBoard。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="experiments",
        help="实验输出根目录。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="训练和生成图像分辨率。",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="训练 batch size。",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="评估/生成 batch size。",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader 并行读取进程数。",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="总训练轮数。",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="AdamW 基础学习率。",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="AdamW beta1。")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2。")
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-6,
        help="AdamW weight decay。",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="AdamW epsilon。",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="学习率调度策略。",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率 warmup 步数。",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练。",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值。",
    )
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 EMA。可用 --no-use_ema 关闭。",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA 衰减系数。",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="是否启用 WeightedRandomSampler。",
    )


def add_diffusion_args(parser):
    """
    扩散过程参数。
    """
    parser.add_argument(
        "--ddpm_num_steps",
        type=int,
        default=1000,
        help="DDPM 训练扩散步数。",
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=100,
        help="推理/评估采样步数。",
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        help="噪声调度方案。",
    )
    parser.add_argument(
        "--use_ddim_sampling",
        action="store_true",
        help="评估/采样时使用 DDIM。",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM eta；0 为确定性采样。",
    )


def add_evaluation_args(parser):
    """
    评估和可视化参数。
    """
    parser.add_argument(
        "--enable_per_class_metrics",
        action="store_true",
        help="是否计算 per-class FID/KID/IPR。",
    )
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=10,
        help="每隔多少 epoch 保存可视化样本。",
    )
    parser.add_argument(
        "--num_visual_samples",
        type=int,
        default=32,
        help="每次保存多少张可视化样本。",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=10,
        help="每隔多少 epoch 保存 checkpoint。",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=20,
        help="每隔多少 epoch 计算生成质量指标。",
    )
    parser.add_argument(
        "--num_fid_samples_train",
        type=int,
        default=0,
        help="训练集 FID 生成样本数；0 表示跳过。",
    )
    parser.add_argument(
        "--num_fid_samples_val",
        "--num_fid_samples_valid",
        dest="num_fid_samples_val",
        type=int,
        default=0,
        help="验证集 FID 生成样本数；0 表示跳过。",
    )
    parser.add_argument(
        "--ipr_k",
        type=int,
        default=3,
        help="IPR 流形估计的 k 近邻数量。",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM / CFG / CG / LDM / Stable Diffusion baselines for ISIC2018"
    )

    add_stable_diffusion_lora_args(parser)
    add_stable_diffusion_textual_inversion_args(parser)
    add_stable_diffusion_full_args(parser)

    add_ldm_conditioning_args(parser)
    add_ldm_autoencoder_args(parser)
    add_ldm_autoencoder_loss_args(parser)
    add_ldm_autoencoder_gan_args(parser)
    add_ldm_autoencoder_runtime_args(parser)

    add_classifier_guidance_args(parser)
    add_cfg_args(parser)
    add_mode_args(parser)

    add_data_args(parser)
    add_training_args(parser)
    add_diffusion_args(parser)
    add_evaluation_args(parser)

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    """
    参数一致性检查。

    这里只做会直接导致训练逻辑矛盾的检查。
    不做复杂工程配置系统。
    """
    if args.data_mode == "single_label" and args.target_label is None:
        raise ValueError("data_mode='single_label' 时必须指定 --target_label。")

    if args.use_cross_attention_conditioning and args.use_class_conditioning:
        raise ValueError(
            "--use_cross_attention_conditioning 不能与 "
            "--use_class_conditioning 同时开启。"
        )

    if args.mode == "cg" and args.classifier_ckpt_path is None:
        print(
            "[Warning] mode='cg' 但未指定 --classifier_ckpt_path。"
            "如果你要做 classifier guidance 采样，通常需要先训练 classifier。"
        )

    if args.mode != "latent_ddpm" and args.use_cross_attention_conditioning:
        raise ValueError(
            "--use_cross_attention_conditioning 目前只建议用于 mode='latent_ddpm'。"
        )

    if args.mode == "sd_textual_inversion":
        if len(args.ti_placeholder_tokens) != len(args.ti_initializer_tokens):
            raise ValueError(
                "ti_placeholder_tokens 和 ti_initializer_tokens 数量必须一致。"
            )

    if args.train_batch_size <= 0:
        raise ValueError("--train_batch_size 必须大于 0。")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps 必须大于 0。")
