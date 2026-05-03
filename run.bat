@echo off
chcp 65001 >nul
REM 激活conda环境

REM 执行Python命令，失败也继续


python -m classifier.main --epochs 120 --ratios 2:1 3:1 5:5.0 6:4.0 --resnet_time_scale_shift scale_shift --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --use_ddim_sampling --use_class_conditioning --resolution 128 --mode cfg --ddpm_num_inference_steps 100 --ddpm_num_steps 1000 --use-amp --aug-output-dir experiments\分类器(增强_2_1_3_2_5_5_6_4)_CFG_gscale_0.3_resnet50_scratch_diffaug_lr0.001_bs96_seed42\train_augmented_data --resume experiments\分类器(增强_2_1_3_2_5_5_6_4)_CFG_gscale_0.3_resnet50_scratch_diffaug_lr0.001_bs96_seed42\checkpoints\last.pth.tar --resume experiments\分类器(增强_2_1_3_2_5_5_6_4)_CFG_gscale_0.3_余弦退火_条件交叉熵_resnet50_scratch_diffaug_lr0.001_bs128_seed42\checkpoints\last.pth.tar
|| echo script1.py failed, continue

python -m classifier.main --batch-size 128 --eval-freq 5 --save-every-eval --test-only --test-checkpoint experiments\分类器(增强_2_1_3_2_5_5_6_4)_CFG_gscale_0.3_余弦退火_条件交叉熵_resnet50_scratch_diffaug_lr0.001_bs128_seed42\checkpoints\last.pth.tar
|| echo script1.py failed, continue


REM 结束
echo All commands attempted.
pause