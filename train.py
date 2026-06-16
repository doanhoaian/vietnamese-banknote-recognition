import torch
import torch.nn as nn
from tqdm import tqdm

import config
import metrics
from data import get_dataloaders
from model import build_model, param_groups
from utils import get_device, seed_everything


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        """Trả về True nếu chỉ số cải thiện (đáng lưu checkpoint)."""
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, y_true, y_pred = 0.0, [], []
    for imgs, labels in tqdm(loader, desc="eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss_sum += criterion(out, labels).item() * imgs.size(0)
        y_true.append(labels.cpu())
        y_pred.append(out.argmax(1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    acc = float((y_true == y_pred).mean())
    return loss_sum / len(y_true), acc, y_true, y_pred


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
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {out}")


def main():
    seed_everything(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Thiết bị: {device}")

    data = get_dataloaders()
    model = build_model().to(device)

    weights = data.class_weights.to(device) if config.USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        param_groups(model, config.HEAD_LR, config.BACKBONE_LR),
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_FACTOR, patience=config.LR_PATIENCE)
    stopper = EarlyStopping(config.EARLY_STOPPING_PATIENCE, config.EARLY_STOPPING_MIN_DELTA)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, data.train, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, data.val, criterion, device)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:2d}/{config.EPOCHS} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if stopper.step(va_acc):
            torch.save({
                "model_state": model.state_dict(),
                "class_dirs": config.CLASS_DIRS,
                "val_acc": va_acc,
                "epoch": epoch,
            }, config.MODEL_PATH)
            print(f"  ↳ Lưu model tốt nhất (val acc={va_acc:.3f}) -> {config.MODEL_PATH}")
        elif stopper.should_stop:
            print(f"  ↳ Early stopping tại epoch {epoch} "
                  f"(best val acc={stopper.best:.3f}).")
            break

    ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc, y_true, y_pred = evaluate(model, data.test, criterion, device)
    print(f"\nKết quả TEST: loss {te_loss:.3f} | accuracy {te_acc:.3f}")
    print(f"Best val accuracy: {stopper.best:.3f}\n")

    cm = metrics.confusion_matrix(y_true, y_pred, len(config.CLASS_DIRS))
    report = metrics.report_text(cm, config.CLASS_DIRS)
    print(report)
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    metrics.save_confusion_matrix(
        cm, config.CLASS_DIRS, config.OUTPUT_DIR / "confusion_matrix.png", "Test")

    plot_history(history)


if __name__ == "__main__":
    main()
