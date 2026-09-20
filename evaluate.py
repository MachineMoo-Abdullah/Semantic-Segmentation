import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CFG, device
from model import build_model
from train import validate_one_epoch, evaluate_with_tta


def load_final_model(final_ckpt_name):
    best_model = build_model(encoder_weights=None)

    ckpt_path = os.path.join(CFG.OUTPUT_DIR, final_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    epoch_label = checkpoint["epoch"] + 1 if isinstance(checkpoint["epoch"], int) else checkpoint["epoch"]
    print(f"Loaded {ckpt_path} (epoch {epoch_label}, val_miou={checkpoint['val_miou']:.4f})")

    return best_model


def full_evaluation(best_model, val_loader, criterion):
    val_loss_final, val_metrics_final, val_cm_final = validate_one_epoch(best_model, val_loader, criterion)

    print("Validation results with the best checkpoint (no TTA)")
    print("=" * 70)
    print(f"Mean IoU:       {val_metrics_final['mean_iou']:.4f}")
    print(f"Mean Dice:      {val_metrics_final['mean_dice']:.4f}")
    print(f"Pixel Accuracy: {val_metrics_final['pixel_accuracy']:.4f}")
    print("-" * 70)
    print(val_metrics_final["per_class"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)

    if CFG.USE_TTA:
        val_metrics_tta, val_cm_tta = evaluate_with_tta(best_model, val_loader)
        print("\nValidation results WITH TTA (flip-averaged inference)")
        print("=" * 70)
        print(f"Mean IoU:       {val_metrics_tta['mean_iou']:.4f}  "
              f"(delta vs no-TTA: {val_metrics_tta['mean_iou'] - val_metrics_final['mean_iou']:+.4f})")
        print(f"Mean Dice:      {val_metrics_tta['mean_dice']:.4f}")
        print(f"Pixel Accuracy: {val_metrics_tta['pixel_accuracy']:.4f}")
        print("-" * 70)
        print(val_metrics_tta["per_class"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 70)
        val_cm_final = val_cm_tta

    return val_metrics_final, val_cm_final


def plot_confusion_matrix(cm, class_names, title, normalize=True, save_path=None):
    cm = cm.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_display[i, j]
            text_color = "white" if (normalize and value > 0.5) else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def report_confusion_matrix(val_cm_final):
    plot_confusion_matrix(
        val_cm_final, CFG.CLASS_NAMES,
        title=f"Validation Confusion Matrix{' (with TTA)' if CFG.USE_TTA else ''} (row-normalized)",
        normalize=True,
        save_path=os.path.join(CFG.OUTPUT_DIR, "val_confusion_matrix.png"),
    )

    barren_idx = CFG.CLASS_NAMES.index("Barren")
    row = val_cm_final[barren_idx].astype(np.float64)
    row_norm = row / row.sum() if row.sum() > 0 else row
    confusion_ranked = sorted(
        ((CFG.CLASS_NAMES[j], row_norm[j]) for j in range(len(CFG.CLASS_NAMES)) if j != barren_idx),
        key=lambda x: -x[1],
    )
    print("Barren ground-truth pixels are most often predicted as:")
    for name, frac in confusion_ranked:
        print(f"  {name:12s}: {frac * 100:5.2f}%")
