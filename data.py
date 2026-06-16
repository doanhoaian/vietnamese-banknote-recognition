import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


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
        paths = paths[:]            # copy
        rng.shuffle(paths)
        n = len(paths)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test_p = paths[:n_test]
        val_p = paths[n_test:n_test + n_val]
        train_p = paths[n_test + n_val:]
        train += [(p, idx) for p in train_p]
        val += [(p, idx) for p in val_p]
        test += [(p, idx) for p in test_p]
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
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        norm,
    ])


infer_transform = _build_transforms(train=False)


def get_dataloaders():
    per_class = _list_samples()
    train_s, val_s, test_s = _stratified_split(
        per_class, config.VAL_RATIO, config.TEST_RATIO, config.SEED
    )

    train_ds = CurrencyDataset(train_s, _build_transforms(train=True))
    val_ds = CurrencyDataset(val_s, _build_transforms(train=False))
    test_ds = CurrencyDataset(test_s, _build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=2)

    print(f"Số ảnh: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader
