import gc
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_memory_report(tag=""):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    label = f" [{tag}]" if tag else ""
    print(f"GPU memory{label} -- name: {torch.cuda.get_device_name(0)} | "
          f"total: {total:.2f} GB | allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


def gpu_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
