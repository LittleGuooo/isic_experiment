@echo off
chcp 65001 >nul

echo ========================================
echo Run 1:
echo ========================================
python -m classifier.main --test-only --test-checkpoint experiments\260526-1728_resnet50_noaug_lr_0.003\checkpoints\model_best.pth.tar
if errorlevel 1 (
    echo Run 1 failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Run 2
echo ========================================
python -m classifier.main --batch-size 128 --epochs 100 --lr 0.001 --use-amp --aug-output-dir "experiments\260526-2040_res256_sd_lora_uncond_all_seed42\sampling_img2img\hard\res256_seed20_aug5_s0p45_gs5p0_steps100_runseed42_hr0p2_expand"
if errorlevel 1 (
    echo Run 2 failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo All runs finished successfully.
echo ========================================
pause