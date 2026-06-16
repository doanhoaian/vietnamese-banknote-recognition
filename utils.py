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


def center_roi(width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    """Ô vuông căn giữa khung, cạnh = ratio * cạnh nhỏ. Trả về (x1, y1, x2, y2)."""
    side = int(min(width, height) * ratio)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return x1, y1, x1 + side, y1 + side
