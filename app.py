import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from data import infer_transform
from model import build_model


def load_model(device):
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model = build_model(num_classes=len(ckpt["class_dirs"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["class_dirs"]


def load_font():
    for path in config.FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, config.FONT_SIZE)
    print("Cảnh báo: Không tìm thấy font Unicode.")
    return ImageFont.load_default()


def draw_label(frame_bgr, text, color_rgb, font):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    x, y = 12, 12
    tb = draw.textbbox((x, y), text, font=font)
    draw.rectangle([tb[0] - 6, tb[1] - 4, tb[2] + 6, tb[3] + 4], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_dirs = load_model(device)
    font = load_font()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được Webcam.")
        return
    print("Đang chạy App... Nhấn 'Q' để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # lật ngang cho đúng hướng gương

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = infer_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = probs.max(0)
        conf = conf.item()
        cls = class_dirs[idx.item()]
        label = config.LABELS_VI.get(cls, cls)

        text = f"{label} ({conf*100:.0f}%)"
        color_rgb = (0, 220, 0) if conf >= config.CONF_THRESHOLD else (255, 165, 0)
        frame = draw_label(frame, text, color_rgb, font)

        cv2.imshow("VietNam Currency Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
