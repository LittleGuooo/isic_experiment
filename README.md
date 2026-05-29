python -m classifier.export_hard_samples --checkpoint "experiments/260521-1710_resnet50_noaug_lr_0.001/checkpoints/last.pth.tar" --arch resnet50 --gt-csv "dataset/ISIC2018_Task3_Training_GroundTruth.csv" --img-dir "dataset/ISIC2018_Task3_Training_Input" --output-csv "experiments/hard_sample/hard_samples_train_20pct.csv" --resolution 256 --batch-size 128 --workers 4 --hard-ratio 0.2 --gpu 0

python -m diffusion.sd_lora_img2img_sampling --seed_strategy random --pretrained_model_name_or_path "C:/Users/Admin/.cache/huggingface/hub/models--nota-ai--bk-sdm-small/snapshots/572238db7ed3a10858900803f3fc8cca53e893e0" --sd_lora_ckpt_path "experiments\260526-2040_res256_sd_lora_uncond_all_seed42\checkpoints\last.pth.tar" --exclude_seed_csv "experiments\260526-2040_res256_sd_lora_uncond_all_seed42\sampling_img2img\hard\res256_seed20_aug5_s0p45_gs5p0_steps100_runseed42_hr0p2_expand\selected_seeds_hard.csv" --num_seed_per_class 40 --num_aug_per_seed 5 --overwrite_run_dir

python -m diffusion.sd_lora_img2img_sampling --seed_strategy hard --classifier_checkpoint "experiments\260528-0056_resnet50_noaug_lr_0.001\checkpoints\last.pth.tar" --pretrained_model_name_or_path "C:/Users/Admin/.cache/huggingface/hub/models--nota-ai--bk-sdm-small/snapshots/572238db7ed3a10858900803f3fc8cca53e893e0" --sd_lora_ckpt_path "experiments\260526-2040_res256_sd_lora_uncond_all_seed42\checkpoints\last.pth.tar" --expand_hard_pool_if_needed --overwrite_run_dir

python -m classifier.main --arch resnet50 --batch-size 64 --workers 4 --epochs 100 --eval-freq 5 --save-every-eval --lr 0.001 --use-class-weights --use-amp --use-diffusion-augmentation --use_weighted_sampler --mode sd_full --aug-output-dir experiments\260521-2002_res256_sd_lora_uncond_all_seed42\img2img_random\random --resolution 256 --ddpm_num_steps 1000 --ddpm_num_inference_steps 250 --use_ddim_sampling --gen-batch-size 24 --diffusion_checkpoint experiments\260505-0115_res512_sd_full_uncond_all_seed42\checkpoints\last.pth.tar --ratios 2:1 3:1 5:5.0 6:4.0 0:0.5 4:0.5

python -m classifier.main --test-only --test-checkpoint experiments\260528-2007_resnet50_diff-ddpm_lr_0.001\checkpoints\model_best.pth.tar

classifier_ddpm_cond_res128_ratio_0_0_1_1_0_5_4_resnet50_scratch_lr0.001
classifier_CFG0.3_exluNV_res128_ratio_0_0_1_1_0_5_4_resnet50_scratch_lr0.001

python -m classifier.build_metadata_merged --root "experiments\260526-2040_res256_sd_lora_uncond_all_seed42\sampling_img2img\hard\res256_seed20_aug5_s0p45_gs5p0_steps100_runseed42_hr0p2_mix400"

实验项目_简单的日期信息能够区分就行_增强方法_分辨率_增强比例（例如ratio_0_0_1_1_0_5_4）_分类器网络_学习率

# git代码
git checkout --orphan clean_branch
git status
git rm -r --cached .
git add .
git commit -m "clean project (no large files)"
git push -u origin clean_branch

git count-objects -vH
git rev-list --objects --all | git cat-file --batch-check="%(objecttype) %(objectname) %(objectsize) %(rest)" | sort -k3 -n | tail -20

# 图片尺寸
128 × 128
192 × 192
256 × 256
320 × 320
384 × 384
448 × 448
512 × 512
640 × 640
768 × 768

# 数据分布
"train_dataset": {
    "MEL": "1113 (11.11%)",
    "NV": "6705 (66.95%)",
    "BCC": "514 (5.13%)",
    "AKIEC": "327 (3.27%)",
    "BKL": "1099 (10.97%)",
    "DF": "115 (1.15%)",
    "VASC": "142 (1.42%)",
    "Total": "10015 (100.00%)"
}


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

## version 2.14
实现了LDM的cross-attention机制

## version 2.15
重构了classifier代码