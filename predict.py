"""Dự đoán mệnh giá tiền từ MỘT ảnh.

Cách dùng:
    python predict.py đường_dẫn_ảnh.png
"""
import sys

import torch
from PIL import Image

import config
from data import infer_transform
from model import build_model


def load_model(device):
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model = build_model(num_classes=len(ckpt["class_dirs"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["class_dirs"]


def predict_image(path, model, class_dirs, device):
    img = Image.open(path).convert("RGB")
    x = infer_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    cls = class_dirs[idx.item()]
    return config.LABELS_VI.get(cls, cls), conf.item(), probs


def main():
    if len(sys.argv) < 2:
        print("Dùng: python predict.py <đường_dẫn_ảnh>")
        sys.exit(1)
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_dirs = load_model(device)

    label, conf, probs = predict_image(sys.argv[1], model, class_dirs, device)
    print(f"\n=> Dự đoán: {label}  (độ tin cậy {conf*100:.1f}%)\n")

    top = torch.topk(probs, k=min(3, len(class_dirs)))
    print("Top dự đoán:")
    for p, i in zip(top.values, top.indices):
        cls = class_dirs[i.item()]
        print(f"  {config.LABELS_VI.get(cls, cls):>15s} : {p.item()*100:5.1f}%")


if __name__ == "__main__":
    main()
