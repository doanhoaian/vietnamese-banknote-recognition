"""Nhận diện tiền Việt Nam theo thời gian thực qua webcam.

Cách dùng:
    python app.py
Nhấn 'q' để thoát.
"""
import cv2
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


def main():
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_dirs = load_model(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    print("Đang chạy webcam... Nhấn 'q' để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # BGR (OpenCV) -> RGB (PIL) rồi qua transform suy luận.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = infer_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = probs.max(0)
        cls = class_dirs[idx.item()]
        label = config.LABELS_VI.get(cls, cls)

        text = f"{label} ({conf.item()*100:.0f}%)"
        color = (0, 200, 0) if conf.item() > 0.6 else (0, 165, 255)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2, cv2.LINE_AA)
        cv2.imshow("Nhan dien tien VND - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
