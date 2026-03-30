# 项目目录结构说明

- `pretrained_models/`：用于存放下载的预训练模型权重（如 resnet18、resnet50 的 .pth 或 .safetensors 文件）。
- `checkpoints/`：用于存放你训练过程中保存的模型权重（如 epoch_10.pth、best_model.pth 等）。
- `logs/`：用于存放训练日志、tensorboard 日志、实验记录等。
- `datasets/`：用于存放原始数据集或处理后的数据集副本。
- `test.Ipynb`、`day1.py` 等：你的代码和实验脚本。

## 推荐文件命名规范
- 预训练模型：`resnet18.a1_in1k.safetensors`、`resnet50.a1_in1k.safetensors`
- 训练权重：`model_epoch10.pth`、`best_model.pth`
- 日志：`train_2026-03-28.log`、`tensorboard/`

## 说明
- 请将下载的 timm 或 torchvision 预训练权重放入 `pretrained_models/` 目录。
- 训练过程中自动保存的模型权重建议放入 `checkpoints/`。
- 日志和可视化文件建议放入 `logs/`。
- 数据集建议统一放在 `datasets/`，便于管理和备份。

如需自动加载本地权重或保存模型，请在代码中指定对应目录路径。

## 执行代码
python main.py --pretrained

python diffusion.py ---