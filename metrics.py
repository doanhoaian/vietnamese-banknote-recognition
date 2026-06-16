import numpy as np

import config


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_metrics(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def report_text(cm: np.ndarray, class_dirs) -> str:
    precision, recall, f1, support = per_class_metrics(cm)
    total = int(cm.sum())
    accuracy = np.diag(cm).sum() / total if total else 0.0

    header = f"{'Lớp':<16}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Hỗ trợ':>9}"
    lines = [header, "-" * len(header)]
    for i, cls in enumerate(class_dirs):
        name = config.LABELS_VI.get(cls, cls)
        lines.append(f"{name:<16}{precision[i]:>11.3f}{recall[i]:>9.3f}"
                     f"{f1[i]:>8.3f}{int(support[i]):>9}")
    lines.append("-" * len(header))
    lines.append(f"{'Accuracy':<16}{'':>11}{'':>9}{accuracy:>8.3f}{total:>9}")
    lines.append(f"{'Macro avg':<16}{precision.mean():>11.3f}{recall.mean():>9.3f}"
                 f"{f1.mean():>8.3f}{total:>9}")
    return "\n".join(lines)


def save_confusion_matrix(cm: np.ndarray, class_dirs, path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bỏ qua vẽ confusion matrix (chưa cài matplotlib).")
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_dirs)), class_dirs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_dirs)), class_dirs, fontsize=8)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Đã lưu confusion matrix: {path}")
