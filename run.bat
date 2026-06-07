@echo off
chcp 65001 >nul


echo ========================================
echo Run 3:
echo ========================================
python -m classifier.main --batch-size 64 --epochs 100 --lr 0.0005 --seed 42 --use-amp --aug-output-dir "experiments\260526-2040_res256_sd_lora_uncond_all_seed42\sampling_img2img\hard\res256_seed60_aug10_s0p45_gs5p0_steps100_runseed42_hr0p3_expand"
if errorlevel 1 (
       echo Run 3 failed, continue...
)



echo.
echo ========================================
echo All runs finished (some may have failed).
echo ========================================
pause