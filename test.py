import pandas as pd

# 读取 CSV
df = pd.read_csv(
    "experiments\\20260403_223203_resnet50_scratch_lr0.001_bs128_seedNone\\metrics\\epoch_metrics.csv"
)

# 保留第一列（epoch）是 5 的倍数的行
df_filtered = df[df.iloc[:, 0] % 5 == 0]

# 保存结果
df_filtered.to_csv(
    "experiments\\20260403_223203_resnet50_scratch_lr0.001_bs128_seedNone\\metrics\\epoch_metrics.csv",
    index=False,
)

print("处理完成 🎉，结果已保存到 output.csv")
