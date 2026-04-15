import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def save_json(data, json_path):
    """
    把 Python 对象保存成 JSON 文件。
    ensure_ascii=False 表示保存中文时不转义。
    indent=4 表示格式化缩进，便于阅读。
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def update_epoch_metrics_csv(metrics_csv_path, row_dict):
    """
    把当前 epoch 的指标追加到 CSV 文件中。
    如果文件已存在，就先读出来再拼接；
    如果不存在，就直接新建。
    """
    row_df = pd.DataFrame([row_dict])
    if os.path.exists(metrics_csv_path):
        old_df = pd.read_csv(metrics_csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df
    new_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")


def update_epoch_metrics_json(metrics_json_path, row_dict):
    """
    把当前 epoch 的指标追加到 JSON 文件中。
    JSON 文件内部保存为 list，每个 epoch 对应一个字典。
    """
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(row_dict)
    save_json(data, metrics_json_path)


def save_detailed_metrics_json(detailed_metrics, save_dir, epoch):
    """
    保存更详细的分类指标。
    文件名中包含 epoch，便于区分不同轮次结果。
    """
    json_path = os.path.join(save_dir, f"epoch_{epoch:03d}_detailed_metrics.json")
    save_json(detailed_metrics, json_path)
    return json_path


def save_confusion_matrix_artifacts(y_true, y_pred, class_names, save_dir, epoch):
    """
    保存混淆矩阵（Confusion Matrix）的 CSV 和 PNG 图片。

    y_true: 真实标签
    y_pred: 预测标签
    class_names: 类别名列表
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 保存 CSV，方便后续分析
    cm_csv_path = os.path.join(save_dir, f"epoch_{epoch:03d}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    # 保存可视化图片
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()

    cm_png_path = os.path.join(save_dir, f"epoch_{epoch:03d}_confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return cm_csv_path, cm_png_path


def save_multiclass_roc_artifacts(y_true, y_prob, class_names, save_dir, epoch):
    """
    保存多分类 ROC 曲线及其数值点。

    y_true: shape [N]
    y_prob: shape [N, C]，每类的预测概率
    """
    num_classes = len(class_names)

    # 多分类 ROC 一般先把真实标签转成 one-vs-rest 的二值矩阵
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    roc_points = {
        "epoch": epoch,
        "num_classes": num_classes,
        "class_names": class_names,
        "per_class": {},
        "macro_average_auc": None,
    }

    plt.figure(figsize=(10, 8))
    all_fpr = []

    for i in range(num_classes):
        class_name = class_names[i]
        y_true_i = y_true_bin[:, i]
        y_prob_i = y_prob[:, i]

        # 如果这个类别在当前验证集里没有正负两类，ROC 无法定义
        if len(np.unique(y_true_i)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_i, y_prob_i)
        roc_auc = auc(fpr, tpr)

        roc_points["per_class"][class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc),
        }

        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={roc_auc:.3f})")
        all_fpr.append(fpr)

    # 计算 macro-average ROC
    if all_fpr:
        mean_fpr = np.unique(np.concatenate(all_fpr))
        mean_tpr = np.zeros_like(mean_fpr)

        for class_result in roc_points["per_class"].values():
            fpr = np.array(class_result["fpr"])
            tpr = np.array(class_result["tpr"])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

        mean_tpr /= max(len(roc_points["per_class"]), 1)
        macro_auc = auc(mean_fpr, mean_tpr)
        roc_points["macro_average_auc"] = float(macro_auc)

        plt.plot(
            mean_fpr,
            mean_tpr,
            linestyle="--",
            linewidth=3,
            color="navy",
            label=f"Macro-average ROC (AUC={macro_auc:.3f})",
        )

    # 画对角线，表示随机分类器水平
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - Epoch {epoch}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_png_path = os.path.join(save_dir, f"epoch_{epoch:03d}_roc.png")
    plt.savefig(roc_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    roc_json_path = os.path.join(save_dir, f"epoch_{epoch:03d}_roc_points.json")
    save_json(roc_points, roc_json_path)
    return roc_png_path, roc_json_path


def save_val_predictions_csv(
    sample_ids, y_true, y_pred, y_prob, class_names, save_dir, epoch
):
    """
    保存验证集逐样本预测结果。
    每一行对应一个样本，包含：
    - 样本 ID
    - 真实标签
    - 预测标签
    - 每个类别的预测概率
    """
    data = {
        "case_id": list(sample_ids),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    for i in range(y_prob.shape[1]):
        data[f"prob_{class_names[i]}"] = y_prob[:, i].tolist()

    pred_df = pd.DataFrame(data)
    pred_csv_path = os.path.join(save_dir, f"val_predictions_epoch_{epoch:02d}.csv")
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    return pred_csv_path


def safe_div(a, b):
    """
    安全除法：
    分母不为 0 时正常除，
    分母为 0 时返回 nan，避免程序报错。
    """
    return float(a) / float(b) if b != 0 else float("nan")


def compute_auc80(y_true_binary, y_score, sensitivity_low=0.8):
    """
    计算 AUC80。
    这里的含义是：只统计 sensitivity >= 0.8 这段 ROC 曲线下的面积。

    常用于某些医学任务里更关注高敏感性（high sensitivity）区间的表现。
    """
    if len(np.unique(y_true_binary)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    keep_idx = np.where(tpr >= sensitivity_low)[0]

    if len(keep_idx) == 0:
        return 0.0

    start_idx = keep_idx[0]

    # 如果 ROC 曲线在某个区间跨过 sensitivity_low，需要线性插值补一个起点
    if tpr[start_idx] > sensitivity_low and start_idx > 0:
        x0, y0 = fpr[start_idx - 1], tpr[start_idx - 1]
        x1, y1 = fpr[start_idx], tpr[start_idx]
        interp_fpr = (
            x0 + (sensitivity_low - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
        )
        fpr_part = np.concatenate([[interp_fpr], fpr[start_idx:]])
        tpr_part = np.concatenate([[sensitivity_low], tpr[start_idx:]])
    else:
        fpr_part = fpr[start_idx:]
        tpr_part = tpr[start_idx:]

    if len(fpr_part) < 2:
        return 0.0

    return float(auc(fpr_part, tpr_part))


def compute_detailed_classification_metrics(y_true, y_pred, y_prob, class_names):
    """
    计算详细分类指标。

    返回结果包含两部分：
    1. overall：整体多分类指标
    2. per_class：每个类别单独的一组 one-vs-rest 指标
    """
    num_classes = len(class_names)
    labels = np.arange(num_classes)

    # 混淆矩阵 shape [C, C]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 把真实标签转成 one-hot / 二值化形式，便于计算 per-class AUC
    y_true_bin = label_binarize(y_true, classes=labels)

    per_class_metrics = {}

    # 这些列表用于后面做宏平均（macro mean）
    per_class_auc_list = []
    per_class_sensitivity_list = []
    per_class_specificity_list = []
    per_class_accuracy_list = []
    per_class_ap_list = []
    per_class_f1_list = []
    per_class_ppv_list = []
    per_class_npv_list = []

    total_samples = cm.sum()

    for i, class_name in enumerate(class_names):
        # one-vs-rest 视角下的 TP / FN / FP / TN
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total_samples - tp - fn - fp

        sensitivity = safe_div(tp, tp + fn)  # 召回率 / 真阳性率
        specificity = safe_div(tn, tn + fp)  # 特异性
        accuracy_i = safe_div(tp + tn, total_samples)
        ppv = safe_div(tp, tp + fp)  # 阳性预测值（Precision）
        npv = safe_div(tn, tn + fn)  # 阴性预测值

        # F1 需要 precision 和 recall
        if np.isnan(ppv) or np.isnan(sensitivity) or (ppv + sensitivity) == 0:
            f1_i = 0.0
            ppv = 0.0 if np.isnan(ppv) else ppv
        else:
            f1_i = 2 * ppv * sensitivity / (ppv + sensitivity)

        y_true_i = y_true_bin[:, i]
        y_prob_i = y_prob[:, i]

        # 当前类若没有正负两类样本，则 AUC / AP 无法定义
        if len(np.unique(y_true_i)) < 2:
            auc_i = float("nan")
            ap_i = float("nan")
        else:
            auc_i = float(roc_auc_score(y_true_i, y_prob_i))
            ap_i = float(average_precision_score(y_true_i, y_prob_i))

        # 只对 MEL 类单独计算 auc80
        melanoma_auc80 = (
            compute_auc80(y_true_i, y_prob_i, sensitivity_low=0.8)
            if class_name == "MEL"
            else float("nan")
        )

        per_class_metrics[class_name] = {
            "sensitivity": (
                float(sensitivity) if not np.isnan(sensitivity) else float("nan")
            ),
            "specificity": (
                float(specificity) if not np.isnan(specificity) else float("nan")
            ),
            "accuracy": float(accuracy_i) if not np.isnan(accuracy_i) else float("nan"),
            "auc": float(auc_i) if not np.isnan(auc_i) else float("nan"),
            "average_precision": float(ap_i) if not np.isnan(ap_i) else float("nan"),
            "f1_score": float(f1_i),
            "ppv": float(ppv) if not np.isnan(ppv) else 0.0,
            "npv": float(npv) if not np.isnan(npv) else float("nan"),
            "auc80": (
                float(melanoma_auc80) if not np.isnan(melanoma_auc80) else float("nan")
            ),
        }

        per_class_auc_list.append(auc_i)
        per_class_sensitivity_list.append(sensitivity)
        per_class_specificity_list.append(specificity)
        per_class_accuracy_list.append(accuracy_i)
        per_class_ap_list.append(ap_i)
        per_class_f1_list.append(f1_i)
        per_class_ppv_list.append(ppv)
        per_class_npv_list.append(npv)

    # =========================
    # 整体多分类指标
    # =========================
    overall_accuracy = accuracy_score(y_true, y_pred) * 100.0
    balanced_macro_recall = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)

    # 多分类宏平均 AUC（One-vs-Rest）
    try:
        multiclass_macro_auc_ovr = float(
            roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        multiclass_macro_auc_ovr = float("nan")

    # =========================
    # 恶性 vs 良性 二分类 AUC
    # 这里把 MEL / BCC / AKIEC 视为 malignant
    # =========================
    malignant_names = [name for name in ["MEL", "BCC", "AKIEC"] if name in class_names]
    malignant_indices = [class_names.index(name) for name in malignant_names]

    if malignant_indices:
        y_true_malignant = np.isin(y_true, malignant_indices).astype(int)
        y_prob_malignant = y_prob[:, malignant_indices].sum(axis=1)

        if len(np.unique(y_true_malignant)) < 2:
            malignant_vs_benign_auc = float("nan")
        else:
            malignant_vs_benign_auc = float(
                roc_auc_score(y_true_malignant, y_prob_malignant)
            )
    else:
        malignant_vs_benign_auc = float("nan")

    return {
        "overall": {
            "accuracy": float(overall_accuracy),
            "balanced_multiclass_accuracy": float(balanced_macro_recall),
            "macro_recall": float(balanced_macro_recall),
            "macro_precision": float(macro_precision),
            "macro_f1": float(macro_f1),
            "multiclass_macro_auc_ovr": float(multiclass_macro_auc_ovr),
            "mean_auc_all_diagnoses": float(np.nanmean(per_class_auc_list)),
            "mean_average_precision_all_diagnoses": float(
                np.nanmean(per_class_ap_list)
            ),
            "mean_sensitivity": float(np.nanmean(per_class_sensitivity_list)),
            "mean_specificity": float(np.nanmean(per_class_specificity_list)),
            "mean_accuracy": float(np.nanmean(per_class_accuracy_list)),
            "mean_f1": float(np.nanmean(per_class_f1_list)),
            "mean_ppv": float(np.nanmean(per_class_ppv_list)),
            "mean_npv": float(np.nanmean(per_class_npv_list)),
            "melanoma_auc80": (
                float(per_class_metrics["MEL"]["auc80"])
                if "MEL" in per_class_metrics
                else float("nan")
            ),
            "malignant_vs_benign_auc": (
                float(malignant_vs_benign_auc)
                if not np.isnan(malignant_vs_benign_auc)
                else float("nan")
            ),
        },
        "per_class": per_class_metrics,
        "malignant_definition_used": malignant_names,
    }
