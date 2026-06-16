import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from data import infer_transform
from model import build_model

PANEL = (0, 0, 0)            # Nền Panel
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


def build_fonts(scale):
    """Tải font Unicode (tiếng Việt) ở nhiều cỡ, co giãn theo kích thước khung."""
    path = next((p for p in config.FONT_CANDIDATES if os.path.exists(p)), None)

    def f(size):
        sz = max(11, int(size * scale))
        return ImageFont.truetype(path, sz) if path else ImageFont.load_default()

    if path is None:
        print("Cảnh báo: không tìm thấy font Unicode, chữ có dấu có thể sai.")
    return {"title": f(22), "label": f(16), "big": f(58), "row": f(20), "hint": f(15)}


def draw_hud(frame_bgr, label, conf, accent, top3, fps, fonts, scale):
    """Vẽ giao diện HUD lên khung hình: thanh tiêu đề, panel mệnh giá + thanh
    tin cậy, panel top-3, dòng hướng dẫn. Mọi kích thước co giãn theo `scale`.
    """
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    W, H = base.size

    def s(v):  # quy đổi giá trị px theo scale
        return int(v * scale)

    pad = s(18)
    radius = s(16)
    title_h = s(52)
    panel_h = s(190)
    panel_top = H - panel_h - pad
    main_w = int(W * 0.46)
    top_x = int(W * 0.66)

    # Lớp panel bán trong suốt (đặc hơn để chữ dễ đọc).
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, 0, W, title_h], fill=PANEL + (175,))
    od.rounded_rectangle([pad, panel_top, main_w, H - pad],
                         radius=radius, fill=PANEL + (205,))
    od.rounded_rectangle([top_x, panel_top, W - pad, H - pad],
                         radius=radius, fill=PANEL + (205,))
    base = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(base)

    # Thanh tiêu đề: tên (trái) - hướng dẫn (giữa) - FPS (phải), canh giữa dọc.
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

    # Panel mệnh giá.
    mx, my = pad + s(16), panel_top + s(16)
    d.text((mx, my), "Mệnh giá", font=fonts["label"], fill=COL_MUTED)
    d.text((mx, my + s(24)), label, font=fonts["big"], fill=accent)

    # Thanh độ tin cậy.
    bar_h = s(10)
    bar_y = H - pad - s(34)
    bar_w = main_w - mx - s(72)
    d.rounded_rectangle([mx, bar_y, mx + bar_w, bar_y + bar_h],
                        radius=bar_h // 2, fill=(255, 255, 255, 50))
    d.rounded_rectangle([mx, bar_y, mx + int(bar_w * conf), bar_y + bar_h],
                        radius=bar_h // 2, fill=accent)
    d.text((mx + bar_w + s(10), bar_y - s(6)), f"{conf*100:.0f}%",
           font=fonts["row"], fill=COL_TEXT)

    # Panel top-3.
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_dirs = load_model(device)

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

        frame = cv2.flip(frame, 1)  # lật ngang cho đúng hướng gương

        # Tính scale + dựng font một lần theo kích thước khung (chuẩn theo 720p).
        if fonts is None:
            scale = frame.shape[0] / 720.0
            fonts = build_fonts(scale)

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

        frame = draw_hud(frame, label, conf, accent, top3, fps, fonts, scale)
        cv2.imshow("VietNam Currency Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 27 = phím Esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
