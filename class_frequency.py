import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from config import CFG, device


def scan_unique_mask_values(dataset, max_samples=300):
    seen = set()
    samples = dataset.samples[:max_samples] if len(dataset.samples) > max_samples else dataset.samples
    for s in tqdm(samples, desc="Scanning mask values"):
        if s["mask"] is None:
            continue
        raw = np.array(Image.open(s["mask"]), dtype=np.int64)
        converted = np.full_like(raw, CFG.IGNORE_INDEX)
        valid = raw > 0
        converted[valid] = raw[valid] - 1
        seen.update(np.unique(converted).tolist())
    expected = set(range(CFG.NUM_CLASSES)) | {CFG.IGNORE_INDEX}
    unexpected = seen - expected
    print(f"Unique converted mask values found (sample of {len(samples)} masks): {sorted(seen)}")
    if unexpected:
        print(f"[WARNING] Unexpected mask values detected: {sorted(unexpected)} -- "
              f"these will be silently treated as ignore by the defensive clip in the Dataset.")
    else:
        print("OK -- only expected values {0..6, 255} found.")


def compute_class_frequencies(dataset, num_classes=None, ignore_index=None):
    num_classes = num_classes or CFG.NUM_CLASSES
    ignore_index = ignore_index if ignore_index is not None else CFG.IGNORE_INDEX

    counts = np.zeros(num_classes, dtype=np.int64)
    for sample in tqdm(dataset.samples, desc="Computing training class frequencies"):
        if sample["mask"] is None:
            continue
        try:
            raw = np.array(Image.open(sample["mask"]), dtype=np.int64)
        except Exception as e:
            print(f"[WARN] Skipping unreadable mask {sample['mask']}: {e}")
            continue
        valid = raw > 0
        converted = raw[valid] - 1
        converted = converted[(converted >= 0) & (converted < num_classes)]
        counts += np.bincount(converted, minlength=num_classes)[:num_classes]
    return counts


def compute_class_weights(train_dataset):
    scan_unique_mask_values(train_dataset)

    class_pixel_counts = compute_class_frequencies(train_dataset)
    total_pixels = class_pixel_counts.sum()
    class_freq = class_pixel_counts / total_pixels

    print("\nTraining pixel class distribution (ignore_index=255 excluded):")
    for name, count, freq in zip(CFG.CLASS_NAMES, class_pixel_counts, class_freq):
        print(f"{name:12s}: {count:>12,d} px  ({freq * 100:5.2f}%)")

    class_weights_np = 1.0 / np.log(1.02 + class_freq)
    class_weights_np = class_weights_np / class_weights_np.mean()
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    print("\nDerived class weights (mean-normalized log-inverse-frequency):")
    for name, w in zip(CFG.CLASS_NAMES, class_weights_np):
        print(f"{name:12s}: {w:.3f}")

    return class_weights
