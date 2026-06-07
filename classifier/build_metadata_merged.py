import os
import pandas as pd

root_dir = "experiments\\260526-2040_res256_sd_lora_uncond_all_seed42\\sampling_img2img\\hard\\res256_seed20_aug5_s0p45_gs5p0_steps100_runseed42_hr0p2_expand_mix400"  # 替换为你的增强数据集目录
out_csv = os.path.join(root_dir, "metadata_hard.csv")

class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
rows = []

for label_idx, label in enumerate(class_names):
    class_dir = os.path.join(root_dir, label)
    if not os.path.isdir(class_dir):
        continue

    for fname in sorted(os.listdir(class_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_dir, fname)
        stem = os.path.splitext(fname)[0]

        rows.append(
            {
                "source_image": stem.split("_label-")[0],
                "label": label,
                "label_idx": label_idx,
                "seed_strategy": "hard",
                "strength": 0.45,
                "guidance_scale": 5.0,
                "num_inference_steps": 100,
                "aug_idx": 0,
                "generator_seed": -1,
                "output_path": img_path,
                "source_confidence": -1,
                "pred": label,
                "pred_confidence": -1,
                "correct": 1,
            }
        )

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(f"saved: {out_csv}, rows={len(df)}")
