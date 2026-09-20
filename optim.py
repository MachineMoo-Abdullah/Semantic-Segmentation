import math

import torch

from config import CFG


def build_optimizer(model):
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": CFG.ENCODER_LR, "name": "encoder"},
        {"params": decoder_params, "lr": CFG.DECODER_LR, "name": "decoder"},
    ], weight_decay=CFG.WEIGHT_DECAY)

    return optimizer


def build_scheduler(optimizer, steps_per_epoch):
    warmup_steps = CFG.WARMUP_EPOCHS * steps_per_epoch
    total_steps = CFG.NUM_EPOCHS * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, warmup_steps, total_steps
