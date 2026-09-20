import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, LovaszLoss

from config import CFG


class CombinedLoss(nn.Module):

    def __init__(self, class_weights, ce_weight, dice_weight, focal_weight, lovasz_weight, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(mode="multiclass", ignore_index=ignore_index, from_logits=True)
        self.focal = FocalLoss(mode="multiclass", ignore_index=ignore_index)
        self.lovasz = LovaszLoss(mode="multiclass", ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)
        total = (self.ce_weight * ce_loss + self.dice_weight * dice_loss
                 + self.focal_weight * focal_loss + self.lovasz_weight * lovasz_loss)
        components = {"ce": ce_loss.detach().item(),
                      "dice": dice_loss.detach().item(),
                      "focal": focal_loss.detach().item(),
                      "lovasz": lovasz_loss.detach().item()}
        return total, components


def build_criterion(class_weights):
    return CombinedLoss(
        class_weights=class_weights,
        ce_weight=CFG.CE_WEIGHT,
        dice_weight=CFG.DICE_WEIGHT,
        focal_weight=CFG.FOCAL_WEIGHT,
        lovasz_weight=CFG.LOVASZ_WEIGHT,
        ignore_index=CFG.IGNORE_INDEX,
    )
