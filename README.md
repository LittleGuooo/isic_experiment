# 项目目录结构说明

- `pretrained_models/`：用于存放下载的预训练模型权重（如 resnet18、resnet50 的 .pth 或 .safetensors 文件）。
- `checkpoints/`：用于存放你训练过程中保存的模型权重（如 epoch_10.pth、best_model.pth 等）。
- `logs/`：用于存放训练日志、tensorboard 日志、实验记录等。
- `datasets/`：用于存放原始数据集或处理后的数据集副本。
- `test.Ipynb`、`day1.py` 等：你的代码和实验脚本。

# 执行代码
python classifier.py --epochs 50

python diffusion.py --run_mode train --use_class_conditioning --num_epochs 150 --use_ddim_sampling --resolution 128 --num_fid_samples_train 1024 --ddpm_num_inference_steps 100 --train_batch_size 20 --eval_batch_size 20 --resume_from_checkpoint experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar

python diffusion.py --run_mode val_only --use_class_conditioning --use_ddim_sampling --resolution 128 --num_fid_samples_train 1024 --ddpm_num_inference_steps 100 --train_batch_size 20 --eval_batch_size 20 --resume_from_checkpoint experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar

python CFG_diffusion.py --run_mode train --use_class_conditioning --num_epochs 120 --use_ddim_sampling --resolution 128  --num_fid_samples_train 1024 --ddpm_num_inference_steps 100 --train_batch_size 20 --eval_batch_size 20 --resume_from_checkpoint experiments\20260409_182550_ddpm_cond_all_all_labels_res128_bs24_seed42\checkpoints\last.pth.tar

python classifier_augment.py --epochs 100 --diffusion_checkpoint experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar --ratios 5:1.0 6:1.0 --diffusion_module diffusion --num_classes 7 --time_scale_shift default --scheduler_config experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\scheduler\scheduler_config.json --gen_batch_size 64 --train_batch_size 96 --resume experiments\20260412_200106_resnet50_scratch_lr0.001_bs32_seed42\checkpoints\checkpoint_epoch_015.pth.tar --eval-freq 5 --save-every-eval

python classifier_augment.py --epochs 100 --diffusion_checkpoint experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar --ratios 3:1.0 5:2.0 6:2.0 --diffusion_module diffusion --num_classes 7 --time_scale_shift default --scheduler_config experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\scheduler\scheduler_config.json --gen_batch_size 64 --train_batch_size 96 --eval-freq 5 --save-every-eval

python classifier_augment.py --epochs 80 --diffusion_checkpoint experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar --ratios 3:1.0 5:1.0 6:1.0 --diffusion_module diffusion --num_classes 7 --time_scale_shift default --scheduler_config experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\scheduler\scheduler_config.json --gen_batch_size 64 --train_batch_size 96 --eval-freq 5 --save-every-eval


python classifier_augment.py --epochs 100 --diffusion_checkpoint experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar --ratios 5:2.0 6:2.0 --diffusion_module diffusion --num_classes 7 --time_scale_shift default --scheduler_config experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\scheduler\scheduler_config.json --gen_batch_size 64 --train_batch_size 96 --eval-freq 5 --save-every-eval --resume experiments\20260413_012012_resnet50_scratch_lr0.001_bs96_seed42\checkpoints\last.pth.tar

    #   0=MEL(Melanoma), 1=NV(Melanocytic nevus), 2=BCC(Basal cell carcinoma)
    #   3=AKIEC(Actinic keratosis/Bowen's disease), 4=BKL(Benign keratosis)
    #   5=DF(Dermatofibroma), 6=VASC(Vascular lesion)

python CG_diffusion.py --diffusion_checkpoint "experiments\20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\last.pth.tar" --resolution 128 --num_classes 7 --ddpm_num_steps 1000 --ddpm_beta_schedule squaredcos_cap_v2 --num_inference_steps 100 --guidance_scale 3 --ddim_eta 0.0 --classifier_epochs 120 --classifier_lr 1e-4 --batch_size 96 --workers 4 --classifier_feat_size 4 --classifier_num_heads 8 --num_generate_total 2048 --guided_gen_batch_size 32 --use_class_conditioning --resume experiments\20260413_025550_ResNet50_scratch_lr0.0001_bs96_seed42\checkpoints\classifier_last.pth.tar

python diffusion.py --run_mode train --use_class_conditioning --num_epochs 150 --use_ddim_sampling --resolution 128 --num_fid_samples_train 1024 --ddpm_num_inference_steps 100 --train_batch_size 20 --eval_batch_size 20 --resnet_time_scale_shift scale_shift

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
优化了classifier.py和diffusion.py的代码逻辑;使用了更完善的评价指标

## version 2.2
diffusion.py变成条件扩散模型

## version 2.4
加入了仅评估/仅推理模式

## version 2.5
上传GitHub

## version 2.6
创建CFG_diffusion.py

## version 2.9
代码重构前的版本