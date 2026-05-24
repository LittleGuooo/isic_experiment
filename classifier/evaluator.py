import numpy as np
import torch
from tqdm.auto import tqdm


from .metrics import (
    compute_detailed_classification_metrics,
    save_confusion_matrix_artifacts,
    save_detailed_metrics_json,
    save_multiclass_roc_artifacts,
    save_val_predictions_csv,
)
from .utils import AverageMeter, Summary, accuracy


def evaluate(
    loader,
    model,
    criterion,
    device,
    class_names,
    epoch,
    output_dirs,
    split_name="val",
):
    """
    在 val/test loader 上评估，并保存评估产物。

    split_name 只用于进度条和返回字段说明；保存文件名沿用原 metrics.py 的命名，
    这样尽量少改已有评估产物逻辑。
    """
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)

    # 切换到评估模式，关闭 Dropout 并使用 BatchNorm 的滑动统计量。
    model.eval()
    # 先收集所有 batch 的预测结果，再统一计算 ROC、混淆矩阵和宏平均指标。
    all_targets, all_preds, all_probs, all_sample_ids = [], [], [], []

    with torch.no_grad():
        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=f"Evaluate {split_name} epoch {epoch}",
            leave=False,
        )

        for images, target, sample_ids in progress_bar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_sample_ids.extend(sample_ids)

            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            progress_bar.set_postfix(
                {"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"}
            )

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    cm_csv_path, cm_png_path = save_confusion_matrix_artifacts(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        save_dir=output_dirs["cm_dir"],
        epoch=epoch,
    )
    roc_png_path, roc_json_path = save_multiclass_roc_artifacts(
        y_true=all_targets,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=output_dirs["roc_dir"],
        epoch=epoch,
    )
    predictions_csv_path = save_val_predictions_csv(
        sample_ids=all_sample_ids,
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
        save_dir=output_dirs["predictions_dir"],
        epoch=epoch,
    )

    detailed_metrics = compute_detailed_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names,
    )
    detailed_metrics[f"{split_name}_loss"] = float(losses.avg)

    detailed_metrics_json_path = save_detailed_metrics_json(
        detailed_metrics, output_dirs["metrics_dir"], epoch
    )

    return {
        "loss": float(losses.avg),
        "top1": float(top1.avg),
        "overall": detailed_metrics["overall"],
        "per_class": detailed_metrics["per_class"],
        "roc_curve_path": roc_png_path,
        "roc_points_json_path": roc_json_path,
        "confusion_matrix_csv_path": cm_csv_path,
        "confusion_matrix_png_path": cm_png_path,
        "predictions_csv_path": predictions_csv_path,
        "detailed_metrics_json_path": detailed_metrics_json_path,
    }
