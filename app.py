import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

import config
from inference import CurrencyClassifier
from utils import center_roi

GREEN = "#34D058"
AMBER = "#EF9F27"
NEUTRAL = "#B4B8BF"
PLACEHOLDER = "—"
FRAME_INTERVAL_MS = 15
SMOOTH_WINDOW = 8                # số frame gần nhất để trung bình xác suất


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _accent_for(pred):
    if not pred.is_money:
        return NEUTRAL
    return GREEN if pred.is_confident else AMBER


def _open_camera(cam_id: int):
    if sys.platform == "win32":
        return cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    return cv2.VideoCapture(cam_id)


class CurrencyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện tiền Việt Nam")
        self.resize(1100, 680)
        self.setStyleSheet("CurrencyApp{background:#0e1117;}")

        self.clf = CurrencyClassifier()
        self.snap_font = self._load_snap_font()

        self.cap = None
        self.fps = 0.0
        self.prev_t = time.time()
        self.last_frame = None       # BGR, dùng cho _snapshot
        self.last_text = ""
        self.prob_history = deque(maxlen=SMOOTH_WINDOW)   # xác suất các frame gần nhất

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

    def _load_snap_font(self):
        path = next((p for p in config.FONT_CANDIDATES if os.path.exists(p)), None)
        return ImageFont.truetype(path, 30) if path else ImageFont.load_default()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 14, 14)
        self.video = QLabel("Nhấn Bắt đầu để mở camera")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 360)
        self.video.setStyleSheet(
            "background:#000; color:#6b7280; font-size:15px;"
            "border:1px solid #222834; border-radius:12px;")
        ll.addWidget(self.video)
        root.addWidget(left, stretch=1)

        panel = QFrame()
        panel.setFixedWidth(320)
        panel.setStyleSheet("background:#161a22;")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(20, 22, 20, 22)
        pl.setSpacing(14)

        title = QLabel("Tiền Việt Nam")
        title.setStyleSheet("color:#e8eaed; font-size:18px; font-weight:600;")
        pl.addWidget(title)

        srow = QHBoxLayout()
        srow.setSpacing(8)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(10, 10)
        self.status_text = QLabel()
        self.status_text.setStyleSheet("color:#9aa3af; font-size:13px;")
        srow.addWidget(self.status_dot)
        srow.addWidget(self.status_text, stretch=1)
        pl.addLayout(srow)

        pl.addWidget(self._caption("MỆNH GIÁ"))
        self.lbl_denom = QLabel(PLACEHOLDER)
        self.lbl_denom.setStyleSheet(f"color:{NEUTRAL}; font-size:34px; font-weight:700;")
        pl.addWidget(self.lbl_denom)

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
        pl.addWidget(self._caption("KHẢ NĂNG KHÁC"))
        self.top_rows = []
        for _ in range(3):
            r = QHBoxLayout()
            name = QLabel(PLACEHOLDER)
            name.setStyleSheet("color:#c4c9d1; font-size:14px;")
            pct = QLabel("")
            pct.setStyleSheet("color:#9aa3af; font-size:14px;")
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            r.addWidget(name, stretch=1)
            r.addWidget(pct)
            pl.addLayout(r)
            self.top_rows.append((name, pct))

        pl.addStretch(1)

        self.lbl_msg = QLabel("")
        self.lbl_msg.setStyleSheet("color:#6b7280; font-size:12px;")
        self.lbl_msg.setWordWrap(True)
        pl.addWidget(self.lbl_msg)

        self.combo = QComboBox()
        for i in range(3):
            self.combo.addItem(f"Camera {i}", i)
        self.combo.setStyleSheet(
            "QComboBox{background:#222834; color:#e8eaed; border:1px solid #333a47;"
            "border-radius:6px; padding:6px 10px; font-size:14px;}")
        pl.addWidget(self.combo)

        self.btn_snap = QPushButton("Chụp ảnh")
        self.btn_snap.setCursor(Qt.PointingHandCursor)
        self.btn_snap.setEnabled(False)
        self.btn_snap.setStyleSheet(
            "QPushButton{background:#222834; color:#e8eaed; border:1px solid #333a47;"
            "border-radius:8px; padding:9px; font-size:14px;}"
            "QPushButton:hover{background:#2a313f;}"
            "QPushButton:disabled{color:#555b66; border-color:#262c38;}")
        self.btn_snap.clicked.connect(self._snapshot)
        pl.addWidget(self.btn_snap)

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
        self._set_status(running=False)

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

    def _set_status(self, running: bool):
        color = GREEN if running else "#5f6672"
        text = "Đang nhận diện" if running else "Đã dừng"
        self.status_dot.setStyleSheet(f"background:{color}; border-radius:5px;")
        self.status_text.setText(text)

    def _reset_panel(self):
        self.lbl_denom.setText(PLACEHOLDER)
        self.lbl_denom.setStyleSheet(f"color:{NEUTRAL}; font-size:34px; font-weight:700;")
        self.bar.setValue(0)
        self._set_bar_color(NEUTRAL)
        self.lbl_pct.setText("")
        for name_lbl, pct_lbl in self.top_rows:
            name_lbl.setText(PLACEHOLDER)
            pct_lbl.setText("")
        self.lbl_fps.setText("")

    def _toggle(self):
        self._stop() if self.timer.isActive() else self._start()

    def _start(self):
        cam_id = self.combo.currentData()
        self.cap = _open_camera(cam_id)
        if not self.cap.isOpened():
            self.video.setText(f"Không mở được Camera {cam_id}")
            self.cap = None
            return
        self.prob_history.clear()
        self.combo.setEnabled(False)
        self.btn_snap.setEnabled(True)
        self._style_button(start=False)
        self._set_status(running=True)
        self.lbl_msg.setText("")
        self.prev_t = time.time()
        self.timer.start(FRAME_INTERVAL_MS)

    def _stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.last_frame = None
        self.combo.setEnabled(True)
        self.btn_snap.setEnabled(False)
        self._style_button(start=True)
        self._set_status(running=False)
        self._reset_panel()
        self.video.setText("Đã dừng. Nhấn Bắt đầu để mở lại.")

    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        self.last_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi = None
        region = rgb
        if config.ROI_RATIO is not None:
            roi = center_roi(rgb.shape[1], rgb.shape[0], config.ROI_RATIO,
                             getattr(config, "ROI_ASPECT", 1.0))
            region = rgb[roi[1]:roi[3], roi[0]:roi[2]]

        probs = self.clf.predict_probs(Image.fromarray(region))
        self.prob_history.append(probs)
        avg_probs = sum(self.prob_history) / len(self.prob_history)
        pred = self.clf.resolve(avg_probs, topk=4)
        accent = _accent_for(pred)
        conf = pred.confidence
        self.last_text = f"{pred.label} ({conf*100:.0f}%)"

        if roi is not None:
            cv2.rectangle(rgb, roi[:2], roi[2:], _hex_to_rgb(accent), 2)

        self.lbl_denom.setText(pred.label)
        self.lbl_denom.setStyleSheet(
            f"color:{accent}; font-size:34px; font-weight:700;")
        self.bar.setValue(int(conf * 100))
        self._set_bar_color(accent)
        self.lbl_pct.setText(f"{conf*100:.0f}%")

        for (name_lbl, pct_lbl), (name, prob) in zip(self.top_rows, pred.topk[1:4]):
            name_lbl.setText(name)
            pct_lbl.setText(f"{prob*100:.0f}%")

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.prev_t, 1e-6))
        self.prev_t = now
        self.lbl_fps.setText(f"{self.fps:.0f} FPS")

        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def _snapshot(self):
        if self.last_frame is None:
            return
        img = Image.fromarray(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        tb = d.textbbox((0, 0), self.last_text, font=self.snap_font)
        d.rectangle([10, 10, 10 + tb[2] - tb[0] + 16, 10 + tb[3] - tb[1] + 12],
                    fill=(0, 0, 0))
        d.text((18, 14), self.last_text, font=self.snap_font, fill=(52, 208, 88))

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        img.save(config.OUTPUT_DIR / name)
        self.lbl_msg.setText(f"Đã lưu: outputs/{name}")

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
