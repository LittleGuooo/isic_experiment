python -m classifier.main --epochs 100 --diffusion_checkpoint experiments\有条件(scale)_CFG_eculNV_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 --resnet_time_scale_shift scale_shift --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --use_ddim_sampling --use_class_conditioning --resolution 128 --mode cfg --cfg_scale 0.3 --cond_drop_prob 0.15 --ddpm_num_inference_steps 100 --ddpm_num_steps 1000 --use-amp --resume experiments\20260415_225814_resnet50_scratch_diffaug_lr0.001_bs128_seed42\checkpoints\last.pth.tar

python -m diffusion.main --run_mode train --mode ldm_ae --ae_mid_block_add_attention --ae_perceptual_loss_weight 0.2 --resolution 256 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 0 --use_ema --train_batch_size 8 --eval_epochs 10 --save_images_epochs 10 --num_epochs 50 --resume 

python -m diffusion.main --run_mode train --mode latent_ddpm --resolution 256 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 1024 --num_fid_samples_val 193 --use_class_conditioning --resnet_time_scale_shift scale_shift --exclude_train_nv --use_ema --num_epochs 120 --train_batch_size 20 --eval_batch_size 20 --eval_epochs 120 --autoencoder_ckpt_path experiments\20260416_003712_ddpm_uncond_all_all_labels_res224_bs12_seed42\autoencoder --eval_epochs 1 --save_images_epochs 1

python -m diffusion.main --run_mode train --mode cg --cg_diffusion_ckpt_path experiments\20260413_031512_ddpm_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --classifier_train_epochs 80 --classifier_train_batch_size 128 --resolution 128 --ddpm_num_steps 1000 --ddpm_num_inference_steps 100 --use_ddim_sampling --num_fid_samples_train 1024 --use_class_conditioning --resnet_time_scale_shift scale_shift --use_ema --classifier_ckpt_path experiments\20260415_015104_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\classifier_last.pth.tar

python -m classifier.main --epochs 100 --diffusion_checkpoint experiments\20260413_031512_ddpm_cond_all_all_labels_res128_bs20_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 --resnet_time_scale_shift scale_shift --gen-batch-size 64 --batch-size 128 --eval-freq 5 --save-every-eval --use-diffusion-augmentation --use_ddim_sampling --use_class_conditioning --resolution 128 --mode cg --ddpm_num_inference_steps 100 --ddpm_num_steps 1000 --use-amp --classifier_ckpt_path experiments\20260415_015104_ddpm_cond_all_all_labels_res128_bs32_seed42\checkpoints\classifier_last.pth.tar --classifier_guidance_scale 10 

@echo off
REM 激活conda环境

REM 执行Python命令，失败也继续
python script1.py || echo script1.py failed, continue
python script2.py || echo script2.py failed, continue
python script3.py || echo script3.py failed, continue

REM 结束
echo All commands attempted.
pause