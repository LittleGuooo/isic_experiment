# 项目目录结构说明

- `pretrained_models/`：用于存放下载的预训练模型权重（如 resnet18、resnet50 的 .pth 或 .safetensors 文件）。
- `checkpoints/`：用于存放你训练过程中保存的模型权重（如 epoch_10.pth、best_model.pth 等）。
- `logs/`：用于存放训练日志、tensorboard 日志、实验记录等。
- `datasets/`：用于存放原始数据集或处理后的数据集副本。
- `test.Ipynb`、`day1.py` 等：你的代码和实验脚本。

# 执行代码
python classifier.py --epochs 30

python diffusion.py --use_ddim_sampling --ddpm_num_inference_steps 100 --resolution 128 --num_fid_samples_train 1024 --num_fid_samples_val 0 --mixed_precision fp16 --num_epochs 80


# git代码
git checkout --orphan clean_branch
git status
git rm -r --cached .
git add .
git commit -m "clean project (no large files)"
git push -u origin clean_branch

git count-objects -vH
git rev-list --objects --all | git cat-file --batch-check="%(objecttype) %(objectname) %(objectsize) %(rest)" | sort -k3 -n | tail -20

# 更新说明
## version 1.0
最初始的版本

## version 1.1
diffusion.py加入了数据管线，加入了评价指标逻辑

## version 1.2
diffusion.py使用了DDIM加速采样

## version 1.3
加入实验结果和数据集

## version 2.0
数据集修改成ISIC2018

## version 2.1
优化了classifier.py和diffusion.py的代码逻辑。
使用了更完善的评价指标