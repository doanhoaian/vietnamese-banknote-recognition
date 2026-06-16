import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from inference import CurrencyClassifier
from utils import center_roi

ESC_KEY = 27

PANEL = (0, 0, 0)
COL_MUTED = (154, 163, 175)
COL_TEXT = (232, 234, 237)
COL_GREEN = (52, 208, 88)       # tự tin
COL_AMBER = (239, 159, 39)      # chưa chắc
COL_NEUTRAL = (180, 184, 191)   # không có tiền


def build_fonts(scale):
    path = next((p for p in config.FONT_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        print("Cảnh báo: không tìm thấy font Unicode, chữ có dấu có thể sai.")

    def f(size):
        sz = max(11, int(size * scale))
        return ImageFont.truetype(path, sz) if path else ImageFont.load_default()

    return {"title": f(22), "label": f(16), "big": f(58), "row": f(20), "hint": f(15)}


def accent_for(pred):
    if not pred.is_money:
        return COL_NEUTRAL
    return COL_GREEN if pred.is_confident else COL_AMBER


def classify_roi(classifier, frame_bgr):
    """Phân loại trong ROI giữa khung; trả về (Prediction, hộp ROI hoặc None)."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if config.ROI_RATIO is None:
        return classifier.predict(Image.fromarray(rgb)), None
    x1, y1, x2, y2 = center_roi(w, h, config.ROI_RATIO)
    return classifier.predict(Image.fromarray(rgb[y1:y2, x1:x2])), (x1, y1, x2, y2)


def draw_roi(frame_bgr, box, accent):
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), tuple(reversed(accent)), 2)


def draw_hud(frame_bgr, label, conf, accent, top3, fps, fonts, scale):
    """Vẽ HUD (tiêu đề, mệnh giá, độ tin cậy, top-3), mọi kích thước theo `scale`."""
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    W, H = base.size

    def s(v):
        return int(v * scale)

    pad = s(18)
    radius = s(16)
    title_h = s(52)
    panel_h = s(190)
    panel_top = H - panel_h - pad
    main_w = int(W * 0.46)
    top_x = int(W * 0.66)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, 0, W, title_h], fill=PANEL + (175,))
    od.rounded_rectangle([pad, panel_top, main_w, H - pad],
                         radius=radius, fill=PANEL + (205,))
    od.rounded_rectangle([top_x, panel_top, W - pad, H - pad],
                         radius=radius, fill=PANEL + (205,))
    base = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(base)

    def vcenter(font):
        asc, desc = font.getmetrics()
        return (title_h - (asc + desc)) // 2

    d.text((pad, vcenter(fonts["title"])), "Nhận diện tiền Việt Nam",
           font=fonts["title"], fill=COL_TEXT)
    hint = "Nhấn Esc để thoát"
    hw = d.textlength(hint, font=fonts["label"])
    d.text(((W - hw) / 2, vcenter(fonts["label"])), hint,
           font=fonts["label"], fill=COL_MUTED)
    fps_text = f"{fps:.0f} FPS"
    fw = d.textlength(fps_text, font=fonts["label"])
    d.text((W - fw - pad, vcenter(fonts["label"])), fps_text,
           font=fonts["label"], fill=COL_MUTED)

    mx, my = pad + s(16), panel_top + s(16)
    d.text((mx, my), "Mệnh giá", font=fonts["label"], fill=COL_MUTED)
    d.text((mx, my + s(24)), label, font=fonts["big"], fill=accent)

    bar_h = s(10)
    bar_y = H - pad - s(34)
    bar_w = main_w - mx - s(72)
    d.rounded_rectangle([mx, bar_y, mx + bar_w, bar_y + bar_h],
                        radius=bar_h // 2, fill=(255, 255, 255, 50))
    d.rounded_rectangle([mx, bar_y, mx + int(bar_w * conf), bar_y + bar_h],
                        radius=bar_h // 2, fill=accent)
    d.text((mx + bar_w + s(10), bar_y - s(6)), f"{conf*100:.0f}%",
           font=fonts["row"], fill=COL_TEXT)

    tx, ty = top_x + s(16), panel_top + s(16)
    d.text((tx, ty), "Top 3", font=fonts["label"], fill=COL_MUTED)
    for i, (name, p) in enumerate(top3):
        ry = ty + s(30) + i * s(30)
        color = COL_TEXT if i == 0 else (196, 201, 209)
        d.text((tx, ry), name, font=fonts["row"], fill=color)
        pct = f"{p*100:.0f}%"
        pw = d.textlength(pct, font=fonts["row"])
        d.text((W - pad - s(16) - pw, ry), pct, font=fonts["row"], fill=COL_MUTED)

    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        return

    classifier = CurrencyClassifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    print("Đang chạy app... Nhấn Esc để thoát.")

    fps = 0.0
    prev_t = time.time()
    fonts, scale = None, 1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        if fonts is None:
            scale = frame.shape[0] / 720.0
            fonts = build_fonts(scale)

        pred, roi = classify_roi(classifier, frame)
        accent = accent_for(pred)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now

        draw_roi(frame, roi, accent)
        frame = draw_hud(frame, pred.label, pred.confidence, accent,
                         pred.topk, fps, fonts, scale)
        cv2.imshow("VietNam Currency Detector", frame)

        if cv2.waitKey(1) & 0xFF == ESC_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
