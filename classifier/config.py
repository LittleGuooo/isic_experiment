import argparse

# 当前 trainer.py 只支持带 model.fc 的 ResNet 类模型
# 因此这里先限制 choices，避免误选 densenet / efficientnet / convnext 后运行报错
model_names = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def parse_args():
    """
    解析命令行参数。

    当前参数主要有 3 类：
    1. 分类器训练 / 验证 / 断点恢复
    2. 分类器测试（test-only，有 ground truth）
    3. 用扩散模型为分类训练集生成合成样本（diffusion augmentation）
    """
    parser = argparse.ArgumentParser(
        description="Unified ISIC2018 classifier training with optional diffusion augmentation"
    )

    # ============================================================
    # 运行模式相关参数
    # ============================================================

    # Stable Diffusion Full Fine-tuning
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

    # test-only
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="只在测试集上评估，不训练；测试集必须有 ground truth。",
    )
    parser.add_argument(
        "--test-gt-csv",
        default="dataset/ISIC2018_Task3_Test_GroundTruth.csv",
        type=str,
        help="测试集 ground truth CSV 路径。",
    )
    parser.add_argument(
        "--test-img-dir",
        default="dataset/ISIC2018_Task3_Test_Input",
        type=str,
        help="测试集图像目录路径。",
    )
    parser.add_argument(
        "--test-checkpoint",
        default=None,
        type=str,
        help="test-only 模式下加载的分类器 checkpoint 路径。",
    )

    # ============================================================
    # 分类器基础训练参数
    # ============================================================
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="恢复训练或评估时加载的分类器 checkpoint 路径。",
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=model_names,
        help="分类器骨干网络名称，例如 resnet18 / resnet50。",
    )
    parser.add_argument(
        "--weights",
        default="DEFAULT",
        type=str,
        help="torchvision 预训练权重名称；为 None 表示从头训练。",
    )
    parser.add_argument(
        "--epochs",
        default=80,
        type=int,
        help="分类器训练总 epoch 数。",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        dest="batch_size",
        help="分类器训练 / 验证的 batch size。",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="DataLoader 的 num_workers；debug 时建议设为 0。",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        dest="lr",
        help="分类器初始学习率。",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="SGD 的 momentum。",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        dest="weight_decay",
        help="权重衰减（weight decay）。",
    )
    parser.add_argument(
        "--min-lr",
        default=1e-6,
        type=float,
        dest="min_lr",
        help="CosineAnnealingLR 的最小学习率。",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.05,
        type=float,
        dest="label_smoothing",
        help="CrossEntropyLoss 的 label smoothing 系数。",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="是否启用自动混合精度（AMP）。",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="是否按类别频次自动构造 class weights。",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="随机种子。",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="使用哪块 GPU，例如 0 表示 cuda:0。",
    )

    # ============================================================
    # 数据路径参数
    # ============================================================
    parser.add_argument(
        "--train-gt-csv",
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        type=str,
        help="训练集 ground truth CSV 路径。",
    )
    parser.add_argument(
        "--val-gt-csv",
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        type=str,
        help="验证集 ground truth CSV 路径。",
    )
    parser.add_argument(
        "--train-img-dir",
        default="dataset/ISIC2018_Task3_Training_Input",
        type=str,
        help="训练集图像目录路径。",
    )
    parser.add_argument(
        "--val-img-dir",
        default="dataset/ISIC2018_Task3_Validation_Input",
        type=str,
        help="验证集图像目录路径。",
    )

    # ============================================================
    # 验证、保存、早停参数
    # ============================================================
    parser.add_argument(
        "--eval-freq",
        default=5,
        type=int,
        help="每隔多少个 epoch 做一次验证。",
    )
    parser.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="每隔多少个 epoch 保存一次checkpoint。",
    )
    parser.add_argument(
        "--early-stop-patience",
        default=0,
        type=int,
        dest="early_stop_patience",
        help="早停 patience；0 表示关闭早停。",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        default=0.0,
        type=float,
        dest="early_stop_min_delta",
        help="验证指标至少提升多少才视为真正 improvement。",
    )

    # ============================================================
    # 扩散增强总开关
    # ============================================================
    parser.add_argument(
        "--use-diffusion-augmentation",
        action="store_true",
        help="是否启用扩散增强：为部分类别生成合成图，并拼接到训练集。",
    )
    parser.add_argument(
        "--aug-output-dir",
        default=None,
        type=str,
        help="合成图保存目录；为空时自动保存在实验目录下。",
    )

    # ============================================================
    # 扩散模型通用参数
    # ============================================================
    parser.add_argument(
        "--mode",
        default="ddpm",
        choices=["ddpm", "cfg", "cg", "latent_ddpm", "sd_full"],
        help="扩散增强使用的模式：ddpm / cfg / cg / latent_ddpm。",
    )
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        default=None,
        help="扩散模型 checkpoint 路径；启用扩散增强时必须提供。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="扩散模型生成图像的分辨率。",
    )
    parser.add_argument(
        "--ddpm_num_steps",
        type=int,
        default=1000,
        help="扩散训练时的总噪声步数（noise steps）。",
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=100,
        help="采样时实际使用的推理步数（inference steps）。",
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="squaredcos_cap_v2",
        help="beta schedule 类型。",
    )
    parser.add_argument(
        "--use_ddim_sampling",
        action="store_true",
        help="是否用 DDIM 采样；否则通常走 DDPM 采样。",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM 采样中的 eta 参数。",
    )
    parser.add_argument(
        "--use_class_conditioning",
        action="store_true",
        help="是否启用类别条件（class conditioning）。",
    )
    parser.add_argument(
        "--resnet_time_scale_shift",
        choices=["default", "scale_shift"],
        default="scale_shift",
        help="扩散 UNet 中 time embedding 的 scale shift 方式。",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="训练扩散模型时是否启用 WeightedRandomSampler，以提高少数类被采样的概率。",
    )

    # ============================================================
    # 合成样本生成控制参数
    # ============================================================
    parser.add_argument(
        "--ratios",
        type=str,
        nargs="+",
        default=None,
        help=(
            "每个类别要额外生成原始数量多少倍的合成样本。"
            "例如：--ratios 0:1.0 2:2.0 3:0.5"
        ),
    )
    parser.add_argument(
        "--gen-batch-size",
        default=32,
        type=int,
        dest="gen_batch_size",
        help="扩散模型生成合成图时的 batch size。",
    )

    # ============================================================
    # CFG（Classifier-Free Guidance）模式专属参数
    # ============================================================
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="CFG guidance scale。",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.1,
        help="CFG 训练中 condition dropout 的概率。",
    )

    # ============================================================
    # CG（Classifier Guidance）模式专属参数
    # ============================================================
    parser.add_argument(
        "--classifier_ckpt_path",
        type=str,
        default=None,
        help="CG 模式下 guidance classifier 的 checkpoint 路径。",
    )
    parser.add_argument(
        "--classifier_guidance_scale",
        type=float,
        default=1.0,
        help="CG guidance scale。",
    )
    parser.add_argument(
        "--classifier_num_heads",
        type=int,
        default=8,
        help="guidance classifier 中注意力池化层的 attention heads 数。",
    )
    parser.add_argument(
        "--classifier_use_rotary",
        action="store_true",
        help="guidance classifier 是否使用 rotary embedding。",
    )
    parser.add_argument(
        "--classifier_feat_size",
        type=int,
        default=4,
        help="guidance classifier 的特征图大小参数。",
    )

    # ============================================================
    # latent_ddpm / AutoencoderKL 相关参数
    # ============================================================

    # LDM Cross-Attention 条件注入
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
    parser.add_argument(
        "--autoencoder_ckpt_path",
        type=str,
        default=None,
        help="AutoencoderKL checkpoint 路径；latent_ddpm 模式通常需要。",
    )
    parser.add_argument(
        "--latent_train_sample_posterior",
        action="store_true",
        help="latent diffusion 训练时是否从后验分布采样 latent z。",
    )

    # Autoencoder 结构参数
    parser.add_argument(
        "--ae_downsample_factor",
        type=int,
        default=8,
        help="AutoencoderKL 的下采样倍数。",
    )
    parser.add_argument(
        "--ae_latent_channels",
        type=int,
        default=4,
        help="AutoencoderKL latent channels。",
    )
    parser.add_argument(
        "--ae_block_out_channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="AutoencoderKL 每层 block 的输出通道数。",
    )
    parser.add_argument(
        "--ae_layers_per_block",
        type=int,
        default=2,
        help="AutoencoderKL 每个 block 的层数。",
    )
    parser.add_argument(
        "--ae_norm_num_groups",
        type=int,
        default=32,
        help="AutoencoderKL 中 GroupNorm 的组数。",
    )
    parser.add_argument(
        "--ae_mid_block_add_attention",
        action="store_true",
        help="AutoencoderKL 的 mid block 是否加入 attention。",
    )
    parser.add_argument(
        "--ae_scaling_factor",
        type=float,
        default=0.18215,
        help="latent scaling factor；diffusers 中常见默认值为 0.18215。",
    )

    # Autoencoder loss 参数
    parser.add_argument(
        "--ae_recon_loss_type",
        type=str,
        default="l1",
        choices=["l1", "mse"],
        help="Autoencoder 重建损失类型。",
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
        help="patch-based 损失权重；当前默认关闭。",
    )
    parser.add_argument(
        "--ae_perceptual_loss_weight",
        type=float,
        default=0.0,
        help="感知损失权重；当前默认关闭。",
    )
    parser.add_argument(
        "--ae_perceptual_resize",
        type=int,
        default=224,
        help="感知损失网络前的 resize 尺寸。",
    )

    #  Autoencoder 推理行为参数
    parser.add_argument(
        "--ae_sample_posterior",
        action="store_true",
        help="重建 / 可视化时是否从后验分布采样 latent。",
    )
    parser.add_argument(
        "--ae_use_slicing",
        action="store_true",
        help="是否启用 AutoencoderKL slicing 以节省显存。",
    )
    parser.add_argument(
        "--ae_use_tiling",
        action="store_true",
        help="是否启用 AutoencoderKL tiling 以节省高分辨率显存。",
    )

    args = parser.parse_args()

    # 命令行里只能传字符串，将--weights none 替换成 python 的 None
    if isinstance(args.weights, str) and args.weights.lower() == "none":
        args.weights = None

    # 只要显式给了 --aug-output-dir，就说明本次训练要复用已有增强数据。
    # 即使没有写 --use-diffusion-augmentation，也应视为启用了扩散增强。
    if args.aug_output_dir is not None:
        args.use_diffusion_augmentation = True

    return args
