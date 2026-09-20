import math

from torch.utils.data import DataLoader

from config import CFG, device
from utils import set_seed, gpu_memory_report, gpu_cleanup
from dataset import get_datasets
from transforms import get_train_transform, get_val_transform
from class_frequency import compute_class_weights
from model import build_model
from losses import build_criterion
from optim import build_optimizer, build_scheduler
from train import run_training
from swa_ema import run_swa_ema_pipeline
from evaluate import load_final_model, full_evaluation, report_confusion_matrix


def run_pipeline():
    print(f"Using device: {device}")

    set_seed(CFG.SEED)
    gpu_cleanup()
    gpu_memory_report("startup")

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset, val_dataset, test_dataset = get_datasets(train_transform, val_transform)

    class_weights = compute_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    model = build_model()
    gpu_memory_report("after model to device")

    criterion = build_criterion(class_weights)

    optimizer = build_optimizer(model)
    steps_per_epoch = math.ceil(len(train_loader) / CFG.ACCUM_STEPS)
    scheduler, warmup_steps, total_steps = build_scheduler(optimizer, steps_per_epoch)
    print(f"Steps/epoch (post-accumulation): {steps_per_epoch} | "
          f"Warmup steps: {warmup_steps} | Total steps: {total_steps}")

    gpu_memory_report("before training")
    history, best_val_miou, best_val_loss = run_training(
        model, train_loader, val_loader, optimizer, scheduler, criterion
    )
    gpu_memory_report("after training")

    model, best_val_miou, final_ckpt_name = run_swa_ema_pipeline(
        model, train_loader, val_loader, criterion
    )

    best_model = load_final_model(final_ckpt_name)

    val_metrics_final, val_cm_final = full_evaluation(best_model, val_loader, criterion)

    report_confusion_matrix(val_cm_final)

    return best_model, val_metrics_final


if __name__ == "__main__":
    run_pipeline()
