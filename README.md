# 项目目录结构说明

- `pretrained_models/`：用于存放下载的预训练模型权重（如 resnet18、resnet50 的 .pth 或 .safetensors 文件）。
- `checkpoints/`：用于存放你训练过程中保存的模型权重（如 epoch_10.pth、best_model.pth 等）。
- `logs/`：用于存放训练日志、tensorboard 日志、实验记录等。
- `datasets/`：用于存放原始数据集或处理后的数据集副本。
- `test.Ipynb`、`day1.py` 等：你的代码和实验脚本。


# 训练扩散模型
python -m diffusion.main --run_mode train --mode cfg --resolution 224 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 1024 --use_class_conditioning --resnet_time_scale_shift scale_shift --exclude_train_nv --cfg_scale 0.3 --cond_drop_prob 0.15 --use_ema --num_epochs 120 --train_batch_size 20 --eval_batch_size 20 --resume_from_checkpoint experiments\20260415_021917_ddpm_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --eval_epochs 40

## 训练ldm
python -m diffusion.main --run_mode train --mode latent_ddpm --resolution 256 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 1024 --num_fid_samples_val 193 --use_class_conditioning --resnet_time_scale_shift scale_shift --exclude_train_nv --use_ema --num_epochs 120 --train_batch_size 64 --eval_batch_size 32 --autoencoder_ckpt_path experiments\ldmae_exluNV_labels_res256_bs8_seed42\autoencoder --resume_from_checkpoint experiments\20260416_165303_ddpm_cond_all_all_labels_res256_bs64_seed42\checkpoints\last.pth.tar

# 训练cg
python -m diffusion.main --run_mode train --mode cg --cg_diffusion_ckpt_path experiments\有条件(scale_shift)_ddpm_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --classifier_train_epochs 80 --classifier_train_batch_size 128 --resolution 128 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 1024 --use_class_conditioning --resnet_time_scale_shift scale_shift --use_ema --classifier_ckpt_path experiments\20260416_120535_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\classifier_last.pth.tar


# 训练分类器
python -m classifier.main --epochs 100 --batch-size 128 --eval-freq 5 --save-every-eval --use-amp --arch resnet101

## cg分类器
python -m classifier.main --epochs 100 --diffusion_checkpoint experiments\有条件(scale_shift)_ddpm_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 --resnet_time_scale_shift scale_shift --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --use_ddim_sampling --use_class_conditioning --resolution 128 --mode cg --ddpm_num_inference_steps 100 --ddpm_num_steps 1000 --use-amp --classifier_guidance_scale 10 --classifier_ckpt_path experiments\20260416_120910_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\classifier_last.pth.tar --classifier_guidance_scale 10

## ldm分类器
python -m classifier.main --mode latent_ddpm --resolution 256 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --use_class_conditioning --resnet_time_scale_shift scale_shift --epochs 100 --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --autoencoder_ckpt_path experiments\ldmae_exluNV_labels_res256_bs8_seed42\autoencoder --diffusion_checkpoint experiments\20260416_165303_ddpm_cond_all_all_labels_res256_bs64_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0

## experiments\20260415_225814_resnet50_scratch_diffaug_lr0.001_bs128_seed42
python -m classifier.main --epochs 100 --diffusion_checkpoint experiments\有条件(scale)_CFG_eculNV_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 --resnet_time_scale_shift scale_shift --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --use_ddim_sampling --use_class_conditioning --resolution 128 --mode cfg --cfg_scale 0.3 --cond_drop_prob 0.15 --ddpm_num_inference_steps 100 --ddpm_num_steps 1000 --use-amp --resume experiments\20260415_225814_resnet50_scratch_diffaug_lr0.001_bs128_seed42\checkpoints\last.pth.tar

'''
    #   0=MEL(Melanoma), 1=NV(Melanocytic nevus), 2=BCC(Basal cell carcinoma)
    #   3=AKIEC(Actinic keratosis/Bowen's disease), 4=BKL(Benign keratosis)
    #   5=DF(Dermatofibroma), 6=VASC(Vascular lesion)
'''

# 训练autoencoder
python -m diffusion.main --run_mode train --mode ldm_ae --ae_mid_block_add_attention --ae_perceptual_loss_weight 0.2 --resolution 256 --ddpm_num_steps 1000 --exclude_train_nv --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 0 --use_ema --train_batch_size 8 --eval_epochs 10 --save_images_epochs 10 --num_epochs 50 --resume experiments\20260416_013429_ddpm_uncond_all_all_labels_res256_bs8_seed42\checkpoints\last.pth.tar




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

## version 2.10
重构了代码

## version 2.11
加入了LDM的实现