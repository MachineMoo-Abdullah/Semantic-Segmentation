from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CFG, IMAGENET_MEAN, IMAGENET_STD


class LoveDADataset(Dataset):

    def __init__(self, root, split="Train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []

        split_root = self.root / split / split

        for scene in ["Rural", "Urban"]:
            scene_root = split_root / scene
            image_dir = scene_root / "images_png"
            mask_dir = scene_root / "masks_png"

            if not image_dir.exists():
                continue

            image_files = sorted(image_dir.glob("*.png"), key=lambda x: int(x.stem))

            for image_path in image_files:
                if split in ["Train", "Val"]:
                    mask_path = mask_dir / image_path.name
                    if not mask_path.exists():
                        continue
                else:
                    candidate = mask_dir / image_path.name
                    mask_path = candidate if candidate.exists() else None

                scene_label = 0 if scene == "Urban" else 1
                self.samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "scene": scene_label,
                    "scene_name": scene,
                })

        self.has_masks = any(s["mask"] is not None for s in self.samples)
        print(f"{split}: {len(self.samples)} images | masks available: {self.has_masks}")

    def __len__(self):
        return len(self.samples)

    def _load_converted_mask(self, mask_path, shape_hw):
        if mask_path is None:
            return np.full(shape_hw, CFG.IGNORE_INDEX, dtype=np.uint8)
        try:
            raw = np.array(Image.open(mask_path), dtype=np.int64)
        except Exception as e:
            print(f"[WARN] Could not read mask {mask_path}: {e} -- treating as fully ignored.")
            return np.full(shape_hw, CFG.IGNORE_INDEX, dtype=np.uint8)

        converted = np.full_like(raw, CFG.IGNORE_INDEX)
        valid = raw > 0
        converted[valid] = raw[valid] - 1
        bad = valid & ((converted < 0) | (converted >= CFG.NUM_CLASSES))
        if bad.any():
            converted[bad] = CFG.IGNORE_INDEX
        return converted.astype(np.uint8)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            image = Image.open(sample["image"]).convert("RGB")
            image = np.array(image, dtype=np.uint8)
        except Exception as e:
            raise IOError(f"Failed to read image {sample['image']}: {e}")

        mask = self._load_converted_mask(sample["mask"], image.shape[:2])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        mask = torch.from_numpy(mask.astype(np.int64))
        scene_label = torch.tensor(sample["scene"], dtype=torch.long)
        return image, mask, scene_label


def get_datasets(train_transform, val_transform, root=None):
    root = root or CFG.DATASET_ROOT
    train_dataset = LoveDADataset(root, split="Train", transform=train_transform)
    val_dataset = LoveDADataset(root, split="Val", transform=val_transform)
    test_dataset = LoveDADataset(root, split="Test", transform=val_transform)
    return train_dataset, val_dataset, test_dataset
