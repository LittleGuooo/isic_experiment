import os
from collections import Counter

import matplotlib.pyplot as plt

from classifier.dataset import ISICResNetDataset

# =========================
# 1. 修改成你自己的路径
# =========================
gt_csv_path = "dataset\ISIC2018_Task3_Training_GroundTruth.csv"
img_dir = "dataset\ISIC2018_Task3_Training_Input"

save_path = "train_class_distribution.png"


# =========================
# 2. 读取训练集
# 不需要 transform，因为这里只统计标签，不读图也行
# =========================
dataset = ISICResNetDataset(
    gt_csv_path=gt_csv_path,
    img_dir=img_dir,
    transform=None,
)

class_names = dataset.class_columns
labels = dataset.labels


# =========================
# 3. 统计每个类别数量
# =========================
counter = Counter(labels)

counts = [counter[i] for i in range(len(class_names))]
total = sum(counts)
percentages = [c / total * 100 for c in counts]


# =========================
# 4. 打印统计结果
# =========================
print("训练集类别分布：")
for name, count, pct in zip(class_names, counts, percentages):
    print(f"{name:10s}: {count:5d} 张，占比 {pct:6.2f}%")

print(f"\n总样本数: {total}")


# =========================
# 5. 画更美观的柱状图
# =========================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 按数量从大到小排序，更容易看出类别不平衡
order = np.argsort(counts)[::-1]
class_names_sorted = [class_names[i] for i in order]
counts_sorted = [counts[i] for i in order]
percentages_sorted = [percentages[i] for i in order]

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=180)

# 使用横向柱状图，比竖向柱状图更适合类别名
bars = ax.barh(
    class_names_sorted,
    counts_sorted,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.85,
)

# 最大类别放在最上面
ax.invert_yaxis()

# 标题和坐标轴
ax.set_title(
    "ISIC 2018 训练集类别分布",
    fontsize=18,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("图像数量", fontsize=13)
ax.set_ylabel("类别", fontsize=13)

# 去掉上边框和右边框，让图更干净
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# x 方向网格线
ax.grid(axis="x", linestyle="--", alpha=0.35)

# 给右侧留空间，防止文字挤出去
max_count = max(counts_sorted)
ax.set_xlim(0, max_count * 1.18)

# 在每个柱子后面标注：数量 + 百分比
for bar, count, pct in zip(bars, counts_sorted, percentages_sorted):
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2

    ax.text(
        width + max_count * 0.015,
        y,
        f"{count} 张  ({pct:.1f}%)",
        va="center",
        ha="left",
        fontsize=11,
    )

# 在图上加总样本数
ax.text(
    0.99,
    0.02,
    f"Total: {total}",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    alpha=0.75,
)

plt.tight_layout()
plt.savefig("train_class_distribution_beautiful.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\n柱状图已保存到: {os.path.abspath(save_path)}")
