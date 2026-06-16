import sys
import time

import cv2
import torch
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

import config
from data import infer_transform
from model import build_model

# Màu nhấn (hex) theo trạng thái.
GREEN = "#34D058"
AMBER = "#EF9F27"
NEUTRAL = "#B4B8BF"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CurrencyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện tiền Việt Nam")
        self.resize(1100, 680)

        self.device = get_device()
        self.model, self.class_dirs = self._load_model()

        self.cap = None
        self.fps = 0.0
        self.prev_t = time.time()

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

    # ---------- mô hình ----------
    def _load_model(self):
        ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
        model = build_model(num_classes=len(ckpt["class_dirs"]), pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        return model, ckpt["class_dirs"]

    # ---------- giao diện ----------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Vùng video (trái).
        self.video = QLabel("Nhấn Bắt đầu để mở camera")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 360)
        self.video.setStyleSheet("background:#0e1117; color:#6b7280; font-size:15px;")
        root.addWidget(self.video, stretch=1)

        # Panel thông tin (phải).
        panel = QFrame()
        panel.setFixedWidth(320)
        panel.setStyleSheet("background:#161a22;")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(20, 22, 20, 22)
        pl.setSpacing(14)

        title = QLabel("Tiền Việt Nam")
        title.setStyleSheet("color:#e8eaed; font-size:18px; font-weight:600;")
        pl.addWidget(title)

        cap_denom = self._caption("MỆNH GIÁ")
        pl.addWidget(cap_denom)
        self.lbl_denom = QLabel("—")
        self.lbl_denom.setStyleSheet(f"color:{NEUTRAL}; font-size:34px; font-weight:700;")
        pl.addWidget(self.lbl_denom)

        # Thanh độ tin cậy + %.
        row_conf = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self._set_bar_color(NEUTRAL)
        row_conf.addWidget(self.bar, stretch=1)
        self.lbl_pct = QLabel("0%")
        self.lbl_pct.setStyleSheet("color:#e8eaed; font-size:14px;")
        self.lbl_pct.setFixedWidth(44)
        self.lbl_pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_conf.addWidget(self.lbl_pct)
        pl.addLayout(row_conf)

        pl.addSpacing(8)
        pl.addWidget(self._caption("TOP 3"))
        self.top_rows = []
        for _ in range(3):
            r = QHBoxLayout()
            name = QLabel("—")
            name.setStyleSheet("color:#c4c9d1; font-size:14px;")
            pct = QLabel("")
            pct.setStyleSheet("color:#9aa3af; font-size:14px;")
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            r.addWidget(name, stretch=1)
            r.addWidget(pct)
            pl.addLayout(r)
            self.top_rows.append((name, pct))

        pl.addStretch(1)

        # Chọn camera + nút Start/Stop.
        self.combo = QComboBox()
        for i in range(3):
            self.combo.addItem(f"Camera {i}", i)
        self.combo.setStyleSheet(
            "QComboBox{background:#222834; color:#e8eaed; border:1px solid #333a47;"
            "border-radius:6px; padding:6px 10px; font-size:14px;}")
        pl.addWidget(self.combo)

        self.btn = QPushButton("Bắt đầu")
        self.btn.setCursor(Qt.PointingHandCursor)
        self._style_button(start=True)
        self.btn.clicked.connect(self._toggle)
        pl.addWidget(self.btn)

        self.lbl_fps = QLabel("")
        self.lbl_fps.setStyleSheet("color:#6b7280; font-size:12px;")
        self.lbl_fps.setAlignment(Qt.AlignRight)
        pl.addWidget(self.lbl_fps)

        root.addWidget(panel)

    def _caption(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#9aa3af; font-size:11px; letter-spacing:1px;")
        return lbl

    def _set_bar_color(self, hex_color):
        self.bar.setStyleSheet(
            "QProgressBar{background:#2a2f3a; border:none; border-radius:5px;}"
            f"QProgressBar::chunk{{background:{hex_color}; border-radius:5px;}}")

    def _style_button(self, start: bool):
        color, hover = ("#2563eb", "#1d4ed8") if start else ("#a33636", "#8f2d2d")
        self.btn.setText("Bắt đầu" if start else "Dừng")
        self.btn.setStyleSheet(
            f"QPushButton{{background:{color}; color:white; border:none;"
            "border-radius:8px; padding:11px; font-size:15px; font-weight:600;}"
            f"QPushButton:hover{{background:{hover};}}")

    # ---------- điều khiển camera ----------
    def _toggle(self):
        if self.timer.isActive():
            self._stop()
        else:
            self._start()

    def _start(self):
        cam_id = self.combo.currentData()
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            self.video.setText(f"Không mở được Camera {cam_id}")
            self.cap = None
            return
        self.combo.setEnabled(False)
        self._style_button(start=False)
        self.prev_t = time.time()
        self.timer.start(15)  # ~ tối đa 60 fps, thực tế tuỳ model/CPU

    def _stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.combo.setEnabled(True)
        self._style_button(start=True)
        self.video.setText("Đã dừng. Nhấn Bắt đầu để mở lại.")

    # ---------- vòng lặp xử lý ----------
    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)

        # Phân loại.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = infer_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        conf = conf.item()
        cls = self.class_dirs[idx.item()]

        # Cập nhật panel.
        if cls == config.NO_MONEY_CLASS:
            accent = NEUTRAL
        else:
            accent = GREEN if conf >= config.CONF_THRESHOLD else AMBER
        self.lbl_denom.setText(config.LABELS_VI.get(cls, cls))
        self.lbl_denom.setStyleSheet(
            f"color:{accent}; font-size:34px; font-weight:700;")
        self.bar.setValue(int(conf * 100))
        self._set_bar_color(accent)
        self.lbl_pct.setText(f"{conf*100:.0f}%")

        top = torch.topk(probs, k=min(3, len(self.class_dirs)))
        for (name_lbl, pct_lbl), p, i in zip(self.top_rows, top.values, top.indices):
            c = self.class_dirs[i.item()]
            name_lbl.setText(config.LABELS_VI.get(c, c))
            pct_lbl.setText(f"{p.item()*100:.0f}%")

        # FPS.
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.prev_t, 1e-6))
        self.prev_t = now
        self.lbl_fps.setText(f"{self.fps:.0f} FPS")

        # Hiển thị video: letterbox giữ tỉ lệ theo kích thước label hiện tại.
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        return
    app = QApplication(sys.argv)
    win = CurrencyApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
