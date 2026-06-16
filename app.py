import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from data import infer_transform
from model import build_model

PANEL = (15, 18, 24)            # Nền Panel
COL_MUTED = (154, 163, 175)     # Chữ phụ
COL_TEXT = (232, 234, 237)      # Chữ chính
COL_GREEN = (52, 208, 88)       # Tự tin
COL_AMBER = (239, 159, 39)      # Chưa chắc
COL_NEUTRAL = (180, 184, 191)   # Không có tiền


def load_model(device):
    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model = build_model(num_classes=len(ckpt["class_dirs"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["class_dirs"]


def load_fonts():
    """Tải font Unicode (tiếng Việt) ở nhiều cỡ cho các thành phần HUD."""
    path = next((p for p in config.FONT_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        print("Cảnh báo: không tìm thấy font Unicode, chữ có dấu có thể sai.")
        d = ImageFont.load_default()
        return {"title": d, "label": d, "big": d, "row": d, "hint": d}
    return {
        "title": ImageFont.truetype(path, 18),
        "label": ImageFont.truetype(path, 13),
        "big": ImageFont.truetype(path, 40),
        "row": ImageFont.truetype(path, 14),
        "hint": ImageFont.truetype(path, 12),
    }


def draw_hud(frame_bgr, label, conf, accent, top3, fps, fonts):
    """Vẽ giao diện HUD lên khung hình: thanh tiêu đề, panel mệnh giá + thanh
    tin cậy, panel top-3, dòng hướng dẫn. Trả về khung BGR đã vẽ.
    """
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    W, H = base.size
    pad = 12
    title_h = max(34, H // 14)
    panel_h = max(118, H // 4)
    panel_top = H - panel_h - pad
    main_w = int(W * 0.58)
    top_x = int(W * 0.62)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, 0, W, title_h], fill=PANEL + (140,))
    od.rounded_rectangle([pad, panel_top, main_w, H - pad],
                         radius=14, fill=PANEL + (165,))
    od.rounded_rectangle([top_x, panel_top, W - pad, H - pad],
                         radius=14, fill=PANEL + (165,))
    base = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(base)

    d.text((pad, title_h // 2 - 10), "Nhận diện tiền Việt Nam",
           font=fonts["title"], fill=COL_TEXT)
    fps_text = f"{fps:.0f} FPS"
    fw = d.textlength(fps_text, font=fonts["label"])
    d.text((W - fw - pad, title_h // 2 - 8), fps_text,
           font=fonts["label"], fill=COL_MUTED)

    mx, my = pad + 14, panel_top + 12
    d.text((mx, my), "Mệnh giá", font=fonts["label"], fill=COL_MUTED)
    d.text((mx, my + 18), label, font=fonts["big"], fill=accent)

    bar_y = H - pad - 26
    bar_w = main_w - mx - 56
    d.rounded_rectangle([mx, bar_y, mx + bar_w, bar_y + 8],
                        radius=4, fill=(255, 255, 255, 46))
    d.rounded_rectangle([mx, bar_y, mx + int(bar_w * conf), bar_y + 8],
                        radius=4, fill=accent)
    d.text((mx + bar_w + 8, bar_y - 4), f"{conf*100:.0f}%",
           font=fonts["row"], fill=COL_TEXT)

    tx, ty = top_x + 12, panel_top + 12
    d.text((tx, ty), "Top 3", font=fonts["label"], fill=COL_MUTED)
    for i, (name, p) in enumerate(top3):
        ry = ty + 22 + i * 22
        color = COL_TEXT if i == 0 else (196, 201, 209)
        d.text((tx, ry), name, font=fonts["row"], fill=color)
        pct = f"{p*100:.0f}%"
        pw = d.textlength(pct, font=fonts["row"])
        d.text((W - pad - 12 - pw, ry), pct, font=fonts["row"], fill=COL_MUTED)

    hint = "Nhấn Q để thoát"
    hw = d.textlength(hint, font=fonts["hint"])
    d.text(((W - hw) / 2, H - pad - 2), hint, font=fonts["hint"], fill=(107, 114, 128))

    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_dirs = load_model(device)
    fonts = load_fonts()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return
    print("Đang chạy app... Nhấn 'q' để thoát.")

    fps = 0.0
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # lật ngang cho đúng hướng gương

        # Phân loại trên toàn khung hình.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = infer_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = probs.max(0)
        conf = conf.item()
        cls = class_dirs[idx.item()]
        label = config.LABELS_VI.get(cls, cls)

        # Top-3 dự đoán.
        top = torch.topk(probs, k=min(3, len(class_dirs)))
        top3 = [(config.LABELS_VI.get(class_dirs[i.item()], class_dirs[i.item()]),
                 p.item()) for p, i in zip(top.values, top.indices)]

        # Màu nhấn theo trạng thái.
        if cls == config.NO_MONEY_CLASS:
            accent = COL_NEUTRAL
        else:
            accent = COL_GREEN if conf >= config.CONF_THRESHOLD else COL_AMBER

        # Ước lượng FPS (làm mượt).
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now

        frame = draw_hud(frame, label, conf, accent, top3, fps, fonts)
        cv2.imshow("VietNam Currency Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
