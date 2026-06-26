import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def center_roi(width: int, height: int, ratio: float,
               aspect: float = 1.0) -> tuple[int, int, int, int]:

    box_h = int(height * ratio)
    box_w = int(box_h * aspect)
    max_w = int(width * 0.95)
    if box_w > max_w:
        box_w = max_w
        box_h = int(box_w / aspect)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    return x1, y1, x1 + box_w, y1 + box_h
