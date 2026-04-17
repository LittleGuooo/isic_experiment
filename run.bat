@echo off
chcp 65001 >nul
REM 激活conda环境

REM 执行Python命令，失败也继续

python -m classifier.main --epochs 100 --diffusion_checkpoint experiments\有条件(scale_shift)_ddpm_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 --resnet_time_scale_shift scale_shift --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --use_ddim_sampling --use_class_conditioning --resolution 128 --mode cg --ddpm_num_inference_steps 100 --ddpm_num_steps 1000 --use-amp --classifier_guidance_scale 10 --classifier_ckpt_path experiments\有条件(scale)_cg_cond_all_all_labels_res128_bs32_seed42\checkpoints\classifier_last.pth.tar --classifier_guidance_scale 10 || echo script1.py failed, continue

python -m classifier.main --mode latent_ddpm --resolution 256 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --use_class_conditioning --resnet_time_scale_shift scale_shift --epochs 100 --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --autoencoder_ckpt_path experiments\ldmae_exluNV_labels_res256_bs8_seed42\autoencoder --diffusion_checkpoint experiments\20260416_165303_ddpm_cond_all_all_labels_res256_bs64_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 --resume experiments\20260416_190424_resnet50_scratch_diffaug_lr0.001_bs128_seed42\checkpoints\last.pth.tar|| echo script1.py failed, continue

REM 结束
echo All commands attempted.
pause