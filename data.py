import random
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from utils import seed_worker


def _list_samples():
    per_class = {}
    for idx, cls in enumerate(config.CLASS_DIRS):
        folder = config.DATASET_DIR / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục lớp: {folder}")
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise RuntimeError(f"Thư mục {folder} không có ảnh .png")
        per_class[idx] = paths
    return per_class


def _stratified_split(per_class, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for idx, paths in per_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        n = len(paths)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test += [(p, idx) for p in paths[:n_test]]
        val += [(p, idx) for p in paths[n_test:n_test + n_val]]
        train += [(p, idx) for p in paths[n_test + n_val:]]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class CurrencyDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _build_transforms(train: bool):
    norm = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            norm,
        ])
    # Không lật ngang: tiền giấy lật gương không tồn tại trong thực tế.
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        norm,
    ])


infer_transform = _build_transforms(train=False)


def _class_weights(samples, num_classes: int) -> torch.Tensor:
    """Trọng số nghịch tần suất, chuẩn hoá quanh 1.0 để bù mất cân bằng lớp."""
    counts = torch.zeros(num_classes)
    for _, label in samples:
        counts[label] += 1
    counts = counts.clamp(min=1)
    return counts.sum() / (num_classes * counts)


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_weights: torch.Tensor


def get_dataloaders(seed: int = None) -> DataBundle:
    seed = config.SEED if seed is None else seed
    per_class = _list_samples()
    train_s, val_s, test_s = _stratified_split(
        per_class, config.VAL_RATIO, config.TEST_RATIO, seed
    )

    generator = torch.Generator().manual_seed(seed)
    train_ds = CurrencyDataset(train_s, _build_transforms(train=True))
    val_ds = CurrencyDataset(val_s, _build_transforms(train=False))
    test_ds = CurrencyDataset(test_s, _build_transforms(train=False))

    loader_kwargs = dict(batch_size=config.BATCH_SIZE,
                         num_workers=config.NUM_WORKERS,
                         worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"Số ảnh: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return DataBundle(
        train=train_loader, val=val_loader, test=test_loader,
        class_weights=_class_weights(train_s, len(config.CLASS_DIRS)),
    )
