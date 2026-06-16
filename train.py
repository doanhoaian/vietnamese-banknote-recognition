"""Huấn luyện mô hình nhận diện tiền Việt Nam, lưu checkpoint tốt nhất và biểu đồ."""
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data import get_dataloaders
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val  "
    with ctx:
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return loss_sum / total, correct / total


def plot_history(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bỏ qua vẽ biểu đồ (chưa cài matplotlib).")
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    out = config.OUTPUT_DIR / "training_curves.png"
    fig.savefig(out)
    print(f"Đã lưu biểu đồ: {out}")


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Thiết bị: {device}")

    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:2d}/{config.EPOCHS} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_dirs": config.CLASS_DIRS,
                "val_acc": va_acc,
            }, config.MODEL_PATH)
            print(f"  ↳ Lưu model tốt nhất (val acc={va_acc:.3f}) -> {config.MODEL_PATH}")

    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\nKết quả TEST: loss {te_loss:.3f} | accuracy {te_acc:.3f}")
    print(f"Best val accuracy: {best_val_acc:.3f}")

    plot_history(history)


if __name__ == "__main__":
    main()
