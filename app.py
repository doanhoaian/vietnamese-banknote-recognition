"""Nhận diện tiền Việt Nam theo thời gian thực qua webcam.

Cách dùng:
    python app.py
Nhấn 'q' để thoát.
"""
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
    """Tìm font TrueType hỗ trợ tiếng Việt; nếu không có, dùng font mặc định."""
    for path in config.FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, config.FONT_SIZE)
    print("Cảnh báo: không tìm thấy font Unicode, chữ có dấu có thể hiển thị sai.")
    return ImageFont.load_default()


class GradCAM:
    """Trích bản đồ chú ý (Grad-CAM) ở lớp conv cuối của MobileNetV2.

    Vùng activation cao tương ứng với chỗ model nhìn để quyết định mệnh giá,
    tức là tờ tiền -> dùng làm cơ sở vẽ khung.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x):
        """x: tensor [1,3,H,W] (requires_grad=True). Trả (cam[h,w], idx, conf)."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        logits[0, idx].backward()

        acts = self.activations[0]                  # [C,h,w]
        weights = self.gradients[0].mean(dim=(1, 2))  # [C] - gộp gradient theo không gian
        cam = torch.relu((weights[:, None, None] * acts).sum(0))  # [h,w]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), idx.item(), conf.item()


def cam_to_bbox(cam, frame_shape, thresh):
    """Ngưỡng hoá bản đồ Grad-CAM -> bounding box của vùng nóng lớn nhất."""
    h, w = frame_shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    mask = (cam_resized >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(biggest)  # (x, y, w, h)


def smooth_bbox(prev, cur, alpha):
    """Làm mượt bounding box theo thời gian (EMA) để khung đỡ giật."""
    if prev is None or cur is None:
        return cur
    return tuple(int(alpha * p + (1 - alpha) * c) for p, c in zip(prev, cur))


def draw_box_with_label(frame_bgr, bbox, text, color_rgb, font):
    """Vẽ khung quanh tờ tiền và nhãn mệnh giá phía trên cạnh khung (PIL Unicode)."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox

    # Khung bao quanh tờ tiền.
    draw.rectangle([x, y, x + w, y + h], outline=color_rgb, width=3)

    # Nhãn nằm trên cạnh trên của khung, có nền cho dễ đọc.
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    label_y = max(0, y - th - 8)              # đẩy lên trên cạnh khung
    draw.rectangle([x, label_y, x + tw + 12, label_y + th + 8], fill=color_rgb)
    draw.text((x + 6, label_y + 2), text, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_dirs = load_model(device)
    font = load_font()
    # Lớp conv cuối của MobileNetV2 dùng để trích Grad-CAM.
    gradcam = GradCAM(model, model.features[-1])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    print("Đang chạy webcam... Nhấn 'q' để thoát.")

    smoothed_bbox = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Phân loại + Grad-CAM trên TOÀN khung hình. Cần gradient nên không no_grad.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = infer_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        x.requires_grad_(True)
        cam, idx, conf = gradcam.generate(x)
        cls = class_dirs[idx]

        # Chỉ vẽ khung khi: có tiền (khác lớp 000000) và đủ tin cậy.
        is_money = cls != config.NO_MONEY_CLASS and conf >= config.CONF_THRESHOLD
        if is_money:
            bbox = cam_to_bbox(cam, frame.shape, config.GRADCAM_THRESHOLD)
            smoothed_bbox = smooth_bbox(smoothed_bbox, bbox, config.BOX_SMOOTHING)
            if smoothed_bbox is not None:
                label = config.LABELS_VI.get(cls, cls)
                text = f"{label} ({conf*100:.0f}%)"
                frame = draw_box_with_label(frame, smoothed_bbox, text,
                                            (0, 220, 0), font)
        else:
            smoothed_bbox = None  # không có tiền -> bỏ khung

        cv2.imshow("Nhan dien tien VND - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
