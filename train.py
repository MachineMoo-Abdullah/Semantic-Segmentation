import os
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from config import CFG, device
from metrics import update_confusion_matrix, metrics_from_confusion_matrix

scaler = torch.amp.GradScaler("cuda")


def train_one_epoch(model, loader, optimizer, scheduler, criterion, accum_steps):
    model.train()
    running_loss, running_components = 0.0, defaultdict(float)
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, (images, masks, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss, components = criterion(logits, masks)
            loss_to_backprop = loss / accum_steps

        scaler.scale(loss_to_backprop).backward()

        is_last_batch = (step + 1) == len(loader)
        if (step + 1) % accum_steps == 0 or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item()
        for k, v in components.items():
            running_components[k] += v
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n_batches = len(loader)
    avg_loss = running_loss / n_batches
    avg_components = {k: v / n_batches for k, v in running_components.items()}
    return avg_loss, avg_components


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, num_classes=None, ignore_index=None):
    num_classes = num_classes or CFG.NUM_CLASSES
    ignore_index = ignore_index if ignore_index is not None else CFG.IGNORE_INDEX

    model.eval()
    running_loss = 0.0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, masks, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss, _ = criterion(logits, masks)

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        cm = update_confusion_matrix(cm, preds, masks, num_classes, ignore_index)

    avg_loss = running_loss / len(loader)
    metrics = metrics_from_confusion_matrix(cm)
    return avg_loss, metrics, cm


@torch.no_grad()
def predict_with_tta(model, images):
    model.eval()
    with torch.amp.autocast("cuda"):
        probs = torch.softmax(model(images), dim=1)
        probs = probs + torch.softmax(model(torch.flip(images, dims=[3])), dim=1).flip(dims=[3])
        probs = probs + torch.softmax(model(torch.flip(images, dims=[2])), dim=1).flip(dims=[2])
    return probs / 3.0


@torch.no_grad()
def evaluate_with_tta(model, loader, num_classes=None, ignore_index=None):
    num_classes = num_classes or CFG.NUM_CLASSES
    ignore_index = ignore_index if ignore_index is not None else CFG.IGNORE_INDEX

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks, _ in tqdm(loader, desc="Eval (TTA)", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        probs = predict_with_tta(model, images)
        preds = torch.argmax(probs, dim=1)
        cm = update_confusion_matrix(cm, preds, masks, num_classes, ignore_index)
    metrics = metrics_from_confusion_matrix(cm)
    return metrics, cm


def run_training(model, train_loader, val_loader, optimizer, scheduler, criterion):
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pixel_acc": [], "lr": []}

    best_val_miou = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_miou_path = os.path.join(CFG.OUTPUT_DIR, CFG.BEST_MIOU_CKPT)
    best_loss_path = os.path.join(CFG.OUTPUT_DIR, CFG.BEST_LOSS_CKPT)
    last_ckpt_path = os.path.join(CFG.OUTPUT_DIR, CFG.LAST_CKPT)

    for epoch in range(CFG.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{CFG.NUM_EPOCHS}")

        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, CFG.ACCUM_STEPS
        )
        val_loss, val_metrics, val_cm = validate_one_epoch(model, val_loader, criterion)

        current_lr = optimizer.param_groups[1]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_metrics["mean_iou"])
        history["val_pixel_acc"].append(val_metrics["pixel_accuracy"])
        history["lr"].append(current_lr)

        print(
            f"Train Loss: {train_loss:.4f} "
            f"(ce={train_components.get('ce', 0):.4f}, "
            f"dice={train_components.get('dice', 0):.4f}, "
            f"focal={train_components.get('focal', 0):.4f}, "
            f"lovasz={train_components.get('lovasz', 0):.4f})"
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_metrics['mean_iou']:.4f} | Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
        print(f"Val Barren IoU: {val_metrics['per_class'].loc[4, 'IoU']:.4f}")

        improved_miou = val_metrics["mean_iou"] > best_val_miou

        if improved_miou:
            best_val_miou = val_metrics["mean_iou"]
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "val_miou": best_val_miou,
                "val_loss": val_loss,
                "history": history,
                "best_val_miou": best_val_miou,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "config": {k: v for k, v in vars(CFG).items() if not k.startswith("_")}
            }, best_miou_path)

            print(f"New best val mIoU: {best_val_miou:.4f} -> saved to {CFG.BEST_MIOU_CKPT}")
        else:
            epochs_without_improvement += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "val_loss": best_val_loss,
                "val_miou": val_metrics["mean_iou"],
                "history": history,
                "best_val_miou": best_val_miou,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "config": {k: v for k, v in vars(CFG).items() if not k.startswith("_")}
            }, best_loss_path)

            print(f"New best val loss: {best_val_loss:.4f} -> saved to {CFG.BEST_LOSS_CKPT}")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_miou": val_metrics["mean_iou"],
            "val_loss": val_loss,
            "history": history,
            "best_val_miou": best_val_miou,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": {k: v for k, v in vars(CFG).items() if not k.startswith("_")}
        }, last_ckpt_path)

        print("Checkpoint saved: last_checkpoint.pth")

        if epochs_without_improvement >= CFG.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no val mIoU improvement for {CFG.EARLY_STOPPING_PATIENCE} epochs.")
            break

    print(f"\nBest val mIoU achieved: {best_val_miou:.4f}")
    print(f"Best val loss achieved: {best_val_loss:.4f}")
    print(f"Best mIoU checkpoint: {best_miou_path}")
    print(f"Best loss checkpoint: {best_loss_path}")
    print(f"Last checkpoint: {last_ckpt_path}")

    return history, best_val_miou, best_val_loss
