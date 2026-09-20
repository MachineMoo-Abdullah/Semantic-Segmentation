import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import CFG


def update_confusion_matrix(cm, preds, targets, num_classes=None, ignore_index=None):
    num_classes = num_classes or CFG.NUM_CLASSES
    ignore_index = ignore_index if ignore_index is not None else CFG.IGNORE_INDEX

    valid = targets != ignore_index
    if valid.sum() == 0:
        return cm
    p = preds[valid].detach().cpu().numpy().ravel()
    t = targets[valid].detach().cpu().numpy().ravel()
    batch_cm = confusion_matrix(t, p, labels=list(range(num_classes)))
    return cm + batch_cm


def metrics_from_confusion_matrix(cm, class_names=None):
    class_names = class_names or CFG.CLASS_NAMES
    num_classes = cm.shape[0]
    ious, dices, precisions, recalls = [], [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(precision))
        recalls.append(float(recall))

    pixel_acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return {
        "per_class": pd.DataFrame({
            "Class": class_names, "IoU": ious, "Dice": dices,
            "Precision": precisions, "Recall": recalls,
        }),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "pixel_accuracy": pixel_acc,
    }
