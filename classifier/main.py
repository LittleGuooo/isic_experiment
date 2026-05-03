import argparse

import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

from .dataset import ISICResNetDataset
from .trainer import run_test, run_training
from .utils import setup_seed_and_device

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
    # 1) 运行模式相关参数
    # ============================================================

    # =========================
    # val-only
    # =========================
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="只在验证集上做评估，不进入正常训练循环；需要配合 --resume 使用。",
    )

    # =========================
    # test-only
    # =========================
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
    # 2) 分类器基础训练参数
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
    # 3) 数据路径参数
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
    # 4) 验证、保存、早停参数
    # ============================================================
    parser.add_argument(
        "--eval-freq",
        default=5,
        type=int,
        help="每隔多少个 epoch 做一次验证。",
    )
    parser.add_argument(
        "--save-every-eval",
        action="store_true",
        help="是否在每次验证时都额外保存一个 checkpoint。",
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
    # 5) 扩散增强总开关
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
    # 6) 扩散模型通用参数
    # ============================================================
    parser.add_argument(
        "--mode",
        default="ddpm",
        choices=["ddpm", "cfg", "cg", "latent_ddpm"],
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
        default="default",
        help="扩散 UNet 中 time embedding 的 scale shift 方式。",
    )

    # ============================================================
    # 7) 合成样本生成控制参数
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
    # 8) CFG（Classifier-Free Guidance）模式专属参数
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
    # 9) CG（Classifier Guidance）模式专属参数
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
    # 10) latent_ddpm / AutoencoderKL 相关参数
    # ============================================================
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

    # ===== Autoencoder 结构参数 =====
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

    # ===== Autoencoder loss 参数 =====
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

    # ===== Autoencoder 推理行为参数 =====
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


def main():
    """
    主函数：
    1. 读取参数
    2. 设置随机种子与设备
    3. 构建图像预处理
    4. 根据模式进入：
       - test-only: 只做测试集评估
       - 否则：正常训练 / 验证 / evaluate
    """
    args = parse_args()

    # 设置随机种子和运行设备（CPU / GPU）
    device = setup_seed_and_device(args)
    print(f"Using device: {device}")

    # 普通数据增加
    classifier_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 把图片统一缩放到 224x224
            transforms.ToTensor(),  # 转成 [C, H, W] 且像素范围变成 [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 更多数据增强
    classifier_train_transforms = transforms.Compose(
        [
            # 随机裁剪到 224x224，提高模型对尺度和局部区域变化的鲁棒性
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            # 皮肤镜图像方向通常不固定，水平/垂直翻转是常用增强
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 小角度旋转，避免模型过度依赖固定方向
            transforms.RandomRotation(degrees=20),
            # 轻微颜色扰动，模拟成像亮度和颜色差异
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    classifier_eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 从训练集 CSV 里读取类别名
    # 这样可以保证 train / val / test 的类别顺序一致
    gt_df = pd.read_csv(args.train_gt_csv)
    class_names = [c for c in gt_df.columns if c != "image"]
    num_classes = len(class_names)

    # =========================
    # 测试模式 test-only
    # =========================
    if args.test_only:
        if args.test_gt_csv is None:
            raise ValueError("启用 --test-only 时，必须提供 --test-gt-csv。")
        if args.test_img_dir is None:
            raise ValueError("启用 --test-only 时，必须提供 --test-img-dir。")
        if args.test_checkpoint is None:
            raise ValueError("启用 --test-only 时，必须提供 --test-checkpoint。")

        test_dataset = ISICResNetDataset(
            args.test_gt_csv,
            args.test_img_dir,
            classifier_eval_transforms,
        )

        run_test(
            args=args,
            test_dataset=test_dataset,
            class_names=class_names,
            num_classes=num_classes,
            device=device,
        )
        return

    # 构建训练集和验证集
    train_dataset = ISICResNetDataset(
        args.train_gt_csv, args.train_img_dir, classifier_train_transforms
    )
    val_dataset = ISICResNetDataset(
        args.val_gt_csv, args.val_img_dir, classifier_eval_transforms
    )

    # =========================
    # 进入训练主流程
    # =========================
    run_training(
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
    )


if __name__ == "__main__":
    main()
