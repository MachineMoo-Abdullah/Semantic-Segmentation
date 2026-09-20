import segmentation_models_pytorch as smp

from config import CFG, device


def build_model(encoder_weights=None):
    weights = CFG.ENCODER_WEIGHTS if encoder_weights is None else encoder_weights

    model = smp.UnetPlusPlus(
        encoder_name=CFG.ENCODER_NAME,
        encoder_weights=weights,
        in_channels=3,
        classes=CFG.NUM_CLASSES,
        decoder_channels=CFG.DECODER_CHANNELS,
        decoder_attention_type=CFG.DECODER_ATTENTION_TYPE,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: UnetPlusPlus + {CFG.ENCODER_NAME} ({weights}) | "
          f"{n_params / 1e6:.1f}M params total, {n_trainable / 1e6:.1f}M trainable")

    return model
