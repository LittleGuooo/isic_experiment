import argparse

import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

from .dataset import ISICResNetDataset
from .trainer import run_training, setup_seed_and_device


# 从 torchvision.models 里自动收集可用模型名称
# 这里会得到类似 resnet18 / resnet50 / densenet121 等名字
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def parse_args():
    """
    解析命令行参数。
    这里集中定义了训练分类器时会用到的所有可配置项。
    """
    parser = argparse.ArgumentParser(
        description="Unified ISIC2018 classifier training with optional diffusion augmentation"
    )

    # =========================
    # 基础训练参数
    # =========================
    parser.add_argument("--arch", default="resnet50", choices=model_names)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=128, type=int, dest="batch_size")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", "--learning-rate", default=0.001, type=float, dest="lr")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument(
        "--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay"
    )
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--print-freq", default=10, type=int)

    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, dest="label_smoothing"
    )
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--min-lr", default=1e-6, type=float, dest="min_lr")
    parser.add_argument("--use-class-weights", action="store_true")

    # =========================
    # 数据路径参数
    # =========================
    parser.add_argument(
        "--train-gt-csv",
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        type=str,
    )
    parser.add_argument(
        "--val-gt-csv",
        default="dataset/ISIC2018_Task3_Validation_GroundTruth.csv",
        type=str,
    )
    parser.add_argument(
        "--train-img-dir", default="dataset/ISIC2018_Task3_Training_Input", type=str
    )
    parser.add_argument(
        "--val-img-dir", default="dataset/ISIC2018_Task3_Validation_Input", type=str
    )

    # =========================
    # 验证与保存相关参数
    # =========================
    parser.add_argument("--eval-freq", default=1, type=int)
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument(
        "--early-stop-patience", default=0, type=int, dest="early_stop_patience"
    )
    parser.add_argument(
        "--early-stop-min-delta", default=0.0, type=float, dest="early_stop_min_delta"
    )

    # =========================
    # 是否启用扩散增强（Diffusion Augmentation）
    # =========================
    parser.add_argument("--use-diffusion-augmentation", action="store_true")
    parser.add_argument("--aug-output-dir", default=None, type=str)

    # =========================
    # 扩散模型相关参数
    # 这些参数会传递给 augmentation.py 和对应的扩散采样逻辑
    # =========================
    parser.add_argument("--mode", default="ddpm", choices=["ddpm", "cfg", "cg"])
    parser.add_argument("--use_class_conditioning", action="store_true")
    parser.add_argument("--use_ddim_sampling", action="store_true")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument(
        "--resnet_time_scale_shift",
        choices=["default", "scale_shift"],
        default="default",
    )
    parser.add_argument("--diffusion_checkpoint", type=str, default=None)

    # =========================
    # CFG（Classifier-Free Guidance）相关参数
    # =========================
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)

    # =========================
    # CG（Classifier Guidance）相关参数
    # =========================
    parser.add_argument("--classifier_ckpt_path", type=str, default=None)
    parser.add_argument("--classifier_guidance_scale", type=float, default=1.0)
    parser.add_argument("--classifier_num_heads", type=int, default=8)
    parser.add_argument("--classifier_use_rotary", action="store_true")
    parser.add_argument("--classifier_feat_size", type=int, default=4)

    # =========================
    # 每个类别生成多少比例的合成样本
    # 例如:
    # --ratios 0:1.0 2:2.0
    # 表示:
    # 类别0额外生成原始数量的1倍
    # 类别2额外生成原始数量的2倍
    # =========================
    parser.add_argument(
        "--ratios",
        type=str,
        nargs="+",
        default=None,
        help="例: 0:1.0 2:2.0 3:0.5，表示额外生成原类别数量的多少倍。",
    )
    parser.add_argument("--gen-batch-size", default=32, type=int, dest="gen_batch_size")

    return parser.parse_args()


def main():
    """
    主函数：
    1. 读取参数
    2. 设置随机种子与设备
    3. 构建图像预处理
    4. 构建训练/验证数据集
    5. 读取类别名
    6. 调用 trainer.run_training 开始训练或评估
    """
    args = parse_args()

    # 设置随机种子和运行设备（CPU / GPU）
    device = setup_seed_and_device(args)
    print(f"Using device: {device}")

    # 分类器输入预处理
    # 这里使用的是 ImageNet 常见的均值和标准差，
    # 因为 torchvision 的很多预训练模型默认按这个分布训练
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

    # 构建训练集和验证集
    train_dataset = ISICResNetDataset(
        args.train_gt_csv, args.train_img_dir, classifier_transforms
    )
    val_dataset = ISICResNetDataset(
        args.val_gt_csv, args.val_img_dir, classifier_transforms
    )

    # 从训练集 CSV 里读取类别名
    gt_df = pd.read_csv(args.train_gt_csv)
    class_names = [c for c in gt_df.columns if c != "image"]
    num_classes = len(class_names)

    # 进入训练主流程
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
