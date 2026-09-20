import os
import torch


class CFG:
    DATASET_ROOT = "/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation"
    NUM_CLASSES = 7
    CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
    IGNORE_INDEX = 255

    ARCHITECTURE = "UnetPlusPlus"
    ENCODER_NAME = "timm-efficientnet-b5"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)
    DECODER_ATTENTION_TYPE = "scse"

    IMAGE_SIZE = 384
    BATCH_SIZE = 8
    ACCUM_STEPS = 2
    NUM_WORKERS = 2

    ENCODER_LR = 1e-5
    DECODER_LR = 1e-4
    WEIGHT_DECAY = 1e-4

    WARMUP_EPOCHS = 3
    NUM_EPOCHS = 40
    EARLY_STOPPING_PATIENCE = 8

    CE_WEIGHT = 0.4
    DICE_WEIGHT = 0.2
    FOCAL_WEIGHT = 0.15
    LOVASZ_WEIGHT = 0.25

    USE_TTA = True
    USE_SWA = True
    USE_EMA = True
    SWA_EPOCHS = 15
    SWA_LR = 5e-5
    EMA_DECAY = 0.999
    FINE_TUNE_EPOCHS = 20
    ENCODER_LR_MULT = 0.1
    TTA_SCALES = [0.75, 1.0, 1.25]

    SEED = 42
    OUTPUT_DIR = "saved_models"
    BEST_MIOU_CKPT = "loveda_unetplusplus_efficientnetb5_best_miou.pth"
    BEST_LOSS_CKPT = "loveda_unetplusplus_efficientnetb5_best_valloss.pth"
    LAST_CKPT = "last_checkpoint.pth"
    SWA_CKPT = "swa_best.pth"
    EMA_CKPT = "ema_best.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
