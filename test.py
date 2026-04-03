import pandas as pd

def analyze_classes(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 总样本数（行数）
    total_samples = df.shape[0]
    
    # 去掉 image 列
    label_df = df.drop(columns=['image'])
    
    # 类别总数
    num_classes = label_df.shape[1]
    
    # 每个类别的样本数
    class_counts = label_df.sum(axis=0)
    
    # 实际出现的类别数（非0）
    non_zero_classes = (class_counts > 0).sum()
    
    print(f"总样本数: {total_samples}")
    print(f"类别总数: {num_classes}")
    print(f"实际出现的类别数: {non_zero_classes}")
    print("各类别样本数:")
    print(class_counts)
    
    return total_samples, num_classes, non_zero_classes, class_counts

total_samples, num_classes, non_zero_classes, class_counts = analyze_classes("dataset/ISIC2018_Task3_Training_GroundTruth.csv")