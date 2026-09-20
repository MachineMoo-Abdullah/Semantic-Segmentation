import albumentations as A
import cv2

from config import CFG


def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(
            size=(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.11),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(
            height=CFG.IMAGE_SIZE, width=CFG.IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
        ),
    ])
