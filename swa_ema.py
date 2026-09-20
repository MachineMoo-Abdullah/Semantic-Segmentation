import copy
import os

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm.auto import tqdm

from config import CFG, device
from train import scaler, train_one_epoch, validate_one_epoch


def train_one_epoch_swa(model, loader, optimizer, criterion, accum_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, (images, masks, _) in enumerate(tqdm(loader, desc="SWA/EMA train", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss, _ = criterion(logits, masks)
            loss = loss / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * accum_steps
    return running_loss / len(loader)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def load_best_checkpoint(model):
    best_miou_path = os.path.join(CFG.OUTPUT_DIR, CFG.BEST_MIOU_CKPT)
    print(f"Loading best model from: {best_miou_path}")
    ckpt = torch.load(best_miou_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    best_val_miou = ckpt["val_miou"]
    print(f"Loaded best model -> val mIoU = {best_val_miou:.4f}")
    return model, best_val_miou


def fine_tune_phase(model, train_loader, val_loader, criterion, best_val_miou):
    if CFG.FINE_TUNE_EPOCHS <= 0:
        return model, best_val_miou

    print(f"\nStarting extra fine-tuning for {CFG.FINE_TUNE_EPOCHS} epochs...")

    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "encoder" in name.lower() or "backbone" in name.lower():
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": CFG.SWA_LR * CFG.ENCODER_LR_MULT},
        {"params": decoder_params, "lr": CFG.SWA_LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    for epoch in range(CFG.FINE_TUNE_EPOCHS):
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, CFG.ACCUM_STEPS
        )
        val_loss, val_metrics, _ = validate_one_epoch(model, val_loader, criterion)

        print(f"Fine-tune {epoch+1:02d}/{CFG.FINE_TUNE_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_mIoU={val_metrics['mean_iou']:.4f}")

        if val_metrics["mean_iou"] > best_val_miou:
            best_val_miou = val_metrics["mean_iou"]
            torch.save({
                "epoch": f"ft_{epoch}",
                "model_state_dict": model.state_dict(),
                "val_miou": best_val_miou,
            }, os.path.join(CFG.OUTPUT_DIR, CFG.BEST_MIOU_CKPT))
            print(f"  -> New best mIoU: {best_val_miou:.4f}")

    return model, best_val_miou


def swa_ema_phase(model, train_loader, val_loader, criterion, best_val_miou):
    if not (CFG.USE_SWA or CFG.USE_EMA):
        return model, best_val_miou, CFG.BEST_MIOU_CKPT

    final_ckpt_name = CFG.BEST_MIOU_CKPT

    swa_model = AveragedModel(model) if CFG.USE_SWA else None

    ema_model = None
    if CFG.USE_EMA:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    encoder_params = list(model.encoder.parameters()) if hasattr(model, "encoder") else []
    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n]

    swa_optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": CFG.SWA_LR * CFG.ENCODER_LR_MULT},
        {"params": decoder_params, "lr": CFG.SWA_LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    swa_scheduler = SWALR(swa_optimizer, swa_lr=CFG.SWA_LR)

    print(f"\nStarting SWA/EMA fine-tuning: {CFG.SWA_EPOCHS} epochs @ LR={CFG.SWA_LR}")
    for swa_epoch in range(CFG.SWA_EPOCHS):
        train_loss = train_one_epoch_swa(model, train_loader, swa_optimizer, criterion, CFG.ACCUM_STEPS)

        if CFG.USE_SWA:
            swa_model.update_parameters(model)
        if CFG.USE_EMA:
            update_ema(ema_model, model, CFG.EMA_DECAY)

        swa_scheduler.step()
        print(f"SWA/EMA epoch {swa_epoch+1}/{CFG.SWA_EPOCHS}  train_loss={train_loss:.4f}")

    print("Recomputing BatchNorm statistics...")
    if CFG.USE_SWA:
        update_bn(train_loader, swa_model, device=device)
    if CFG.USE_EMA:
        update_bn(train_loader, ema_model, device=device)

    candidates = []
    if CFG.USE_SWA:
        val_loss, val_metrics, _ = validate_one_epoch(swa_model, val_loader, criterion)
        print(f"SWA  val mIoU: {val_metrics['mean_iou']:.4f}")
        candidates.append(("swa", swa_model, val_metrics["mean_iou"], val_loss))

    if CFG.USE_EMA:
        val_loss, val_metrics, _ = validate_one_epoch(ema_model, val_loader, criterion)
        print(f"EMA  val mIoU: {val_metrics['mean_iou']:.4f}")
        candidates.append(("ema", ema_model, val_metrics["mean_iou"], val_loss))

    best_name, best_model, best_miou, best_loss = max(candidates, key=lambda x: x[2])
    print(f"Best averaging method: {best_name.upper()} -> mIoU {best_miou:.4f}")

    if best_miou > best_val_miou:
        print(f"{best_name.upper()} improved val mIoU -- saving as new best.")
        ckpt_name = CFG.SWA_CKPT if best_name == "swa" else CFG.EMA_CKPT
        state = best_model.module.state_dict() if hasattr(best_model, "module") else best_model.state_dict()
        torch.save({
            "epoch": best_name,
            "model_state_dict": state,
            "val_miou": best_miou,
            "val_loss": best_loss,
        }, os.path.join(CFG.OUTPUT_DIR, ckpt_name))
        final_ckpt_name = ckpt_name
        best_val_miou = best_miou
    else:
        print("Averaging did not beat the previous best checkpoint.")

    print(f"\nFinal checkpoint selected: {final_ckpt_name}")

    return model, best_val_miou, final_ckpt_name


def run_swa_ema_pipeline(model, train_loader, val_loader, criterion):
    model, best_val_miou = load_best_checkpoint(model)
    model, best_val_miou = fine_tune_phase(model, train_loader, val_loader, criterion, best_val_miou)
    model, best_val_miou, final_ckpt_name = swa_ema_phase(model, train_loader, val_loader, criterion, best_val_miou)
    return model, best_val_miou, final_ckpt_name
