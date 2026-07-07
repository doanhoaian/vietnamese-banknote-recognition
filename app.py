import sys
import time

import cv2
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QGridLayout, QLabel, QVBoxLayout, QWidget,
)

import config
import speech
from inference import CurrencyClassifier
from utils import center_roi

# Màu theo trạng thái
GREEN = "#34D058"
AMBER = "#EF9F27"
NEUTRAL = "#8b93a1"
BG = "#0e1117"

FRAME_INTERVAL_MS = 15
SMOOTH_WINDOW = 8
SCAN_INTERVAL_MS = 16
SCAN_DURATION_MS = 1100

SEARCHING = "SEARCHING"
GUIDING = "GUIDING"
SCANNING = "SCANNING"
RESULT = "RESULT"


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
    def __init__(self, cam_id: int = 0):
        super().__init__()
        self.setWindowTitle("Nhận diện tiền Việt Nam")
        self.resize(1000, 640)
        self.setStyleSheet(f"CurrencyApp{{background:{BG};}}")

        self.cam_id = cam_id
        self.clf = CurrencyClassifier()
        self.speaker = speech.Speaker(config.VOICE_NAME, config.VOICE_RATE)

        self.cap = None
        self.prob_history = []          # xác suất các frame gần nhất (làm mượt)

        # Máy trạng thái + chọn frame tốt nhất
        self.state = SEARCHING
        self.stable_cls = None
        self.stable_streak = 0
        self.best = None                # {score, rgb, roi, pred}
        self.captured_cls = None
        self.absent = 0
        self.last_guide_t = 0.0
        self.last_result_speech = ""    # để đọc lại khi chạm màn hình

        # Animation quét
        self.scan_rgb = None
        self.scan_roi = None
        self.scan_pred = None
        self.scan_progress = 0.0
        self.scan_speed = SCAN_INTERVAL_MS / SCAN_DURATION_MS
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._scan_step)

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        # Tự khởi động camera + chào ngay khi mở app.
        QTimer.singleShot(0, self._start)

    # ---------------- Giao diện ----------------

    def _build_ui(self):
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setAttribute(Qt.WA_TransparentForMouseEvents)
        grid.addWidget(self.video, 0, 0)

        overlay = QWidget()
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay.setStyleSheet("background:transparent;")
        ov = QVBoxLayout(overlay)
        ov.setContentsMargins(40, 30, 40, 34)

        self.status_lbl = QLabel()
        self.status_lbl.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.status_lbl.setStyleSheet(
            "color:#c4c9d1; font-size:16px; letter-spacing:2px;"
            "background:rgba(0,0,0,90); border-radius:12px; padding:6px 14px;")
        ov.addWidget(self.status_lbl, alignment=Qt.AlignHCenter)

        ov.addStretch(1)
        self.big_lbl = QLabel()
        self.big_lbl.setAlignment(Qt.AlignCenter)
        self.big_lbl.setWordWrap(True)
        ov.addWidget(self.big_lbl)

        self.sub_lbl = QLabel()
        self.sub_lbl.setAlignment(Qt.AlignCenter)
        self.sub_lbl.setStyleSheet("color:#e8eaed; font-size:22px;")
        ov.addWidget(self.sub_lbl)
        ov.addStretch(1)

        self.hint_lbl = QLabel("Esc để thoát")
        self.hint_lbl.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.hint_lbl.setStyleSheet("color:#6b7280; font-size:14px;")
        ov.addWidget(self.hint_lbl, alignment=Qt.AlignHCenter)

        grid.addWidget(overlay, 0, 0)

    def _set_overlay(self, status, big, sub, color):
        self.status_lbl.setText(status)
        self.big_lbl.setText(big)
        self.big_lbl.setStyleSheet(
            f"color:{color}; font-size:64px; font-weight:800;")
        self.sub_lbl.setText(sub)

    def _show_rgb(self, rgb):
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    # ---------------- Camera ----------------

    def _start(self):
        self.cap = _open_camera(self.cam_id)
        if not self.cap.isOpened():
            self.cap = None
            self._set_overlay("LỖI", "Không mở\nđược camera", "", AMBER)
            self.speaker.speak(config.SPEECH_NO_CAMERA, interrupt=True)
            return
        self.prob_history.clear()
        self._reset_capture()
        self._render_searching()
        self.speaker.speak(config.SPEECH_GREETING, interrupt=True)
        self.timer.start(FRAME_INTERVAL_MS)

    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi = None
        region = rgb
        if config.ROI_RATIO is not None:
            roi = center_roi(rgb.shape[1], rgb.shape[0], config.ROI_RATIO,
                             getattr(config, "ROI_ASPECT", 1.0))
            region = rgb[roi[1]:roi[3], roi[0]:roi[2]]

        probs = self.clf.predict_probs(Image.fromarray(region))
        self.prob_history.append(probs)
        if len(self.prob_history) > SMOOTH_WINDOW:
            self.prob_history.pop(0)
        avg = sum(self.prob_history) / len(self.prob_history)
        pred = self.clf.resolve(avg, topk=4)
        conf = pred.confidence

        self._process(rgb, region, roi, pred, conf)

        if self.state in (SCANNING, RESULT):
            return

        if roi is not None:
            cv2.rectangle(rgb, roi[:2], roi[2:], _hex_to_rgb(_accent_for(pred)), 3)
        self._show_rgb(rgb)

    # ---------------- Máy trạng thái ----------------

    def _reset_capture(self):
        self.anim_timer.stop()
        self.state = SEARCHING
        self.stable_cls = None
        self.stable_streak = 0
        self.best = None
        self.captured_cls = None
        self.absent = 0
        self.scan_rgb = None
        self.scan_roi = None
        self.scan_pred = None

    def _process(self, rgb, region, roi, pred, conf):
        stable = (pred.is_money and pred.is_confident
                  and conf >= config.AUTO_STABLE_CONF)

        if self.state in (SCANNING, RESULT):
            if self.state == RESULT and self.speaker.is_speaking():
                self.absent = 0
                return
            if not pred.is_money:
                self.absent += 1
                if self.state == RESULT and self.absent >= config.AUTO_REARM_FRAMES:
                    self._go_next()
            else:
                self.absent = 0
                if (self.state == RESULT and stable
                        and pred.cls != self.captured_cls):
                    self._resume_live()
            return

        if stable:
            if pred.cls == self.stable_cls:
                self.stable_streak += 1
            else:
                self.stable_cls = pred.cls
                self.stable_streak = 1
                self.best = None
            self._keep_best(rgb, region, roi, pred, conf)
            self.state = GUIDING
            self._render_guiding(conf)
            if self.stable_streak >= config.AUTO_STABLE_FRAMES:
                self._start_scan()
        elif pred.is_money:
            self.stable_cls = None
            self.stable_streak = 0
            self.best = None
            self.state = GUIDING
            self._render_guiding(conf)
            self._maybe_guide()
        else:
            self.stable_cls = None
            self.stable_streak = 0
            self.best = None
            self.state = SEARCHING
            self._render_searching()

    def _keep_best(self, rgb, region, roi, pred, conf):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = conf * sharpness
        if self.best is None or score > self.best["score"]:
            full_roi = roi if roi is not None else (
                0, 0, rgb.shape[1], rgb.shape[0])
            self.best = {"score": score, "rgb": rgb.copy(),
                         "roi": full_roi, "pred": pred}

    def _maybe_guide(self):
        now = time.time()
        if now - self.last_guide_t >= config.GUIDE_COOLDOWN_S:
            self.last_guide_t = now
            self.speaker.speak(config.SPEECH_GUIDE_HOLD)

    # ---------------- Hiển thị từng trạng thái ----------------

    def _render_searching(self):
        self._set_overlay("ĐANG TÌM", "Đưa tờ tiền\nvào khung", "", NEUTRAL)

    def _render_guiding(self, conf):
        self._set_overlay("ĐANG NHẬN DIỆN", "Giữ yên…",
                          f"{conf*100:.0f}%", AMBER)

    # ---------------- Animation quét ----------------

    def _start_scan(self):
        self.scan_rgb = self.best["rgb"]
        self.scan_roi = self.best["roi"]
        self.scan_pred = self.best["pred"]
        self.scan_progress = 0.0
        self.state = SCANNING
        speech.play_cue("scan")
        self._set_overlay("ĐANG QUÉT", "", "", GREEN)
        self.anim_timer.start(SCAN_INTERVAL_MS)

    def _scan_step(self):
        self.scan_progress = min(self.scan_progress + self.scan_speed, 1.0)
        x1, y1, x2, y2 = self.scan_roi
        y = int(y1 + (y2 - y1) * self.scan_progress)
        green = (52, 208, 88)

        img = self.scan_rgb.copy()
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y), green, -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.line(img, (x1, y), (x2, y), green, 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)
        self._show_rgb(img)

        if self.scan_progress >= 1.0:
            self.anim_timer.stop()
            self._reveal()

    def _reveal(self):
        self.state = RESULT
        pred = self.scan_pred
        accent = _accent_for(pred)
        x1, y1, x2, y2 = self.scan_roi

        img = cv2.convertScaleAbs(self.scan_rgb, alpha=0.4)   # làm tối nền
        img[y1:y2, x1:x2] = self.scan_rgb[y1:y2, x1:x2]       # giữ sáng tờ tiền
        cv2.rectangle(img, (x1, y1), (x2, y2), _hex_to_rgb(accent), 3)
        self._show_rgb(img)

        self._set_overlay("KẾT QUẢ", pred.label, "", accent)

        self.captured_cls = pred.cls
        self.absent = 0
        value = config.LABELS_SPEECH.get(pred.cls, pred.label)
        self.last_result_speech = f"{config.SPEECH_SUCCESS_PREFIX} {value}"
        speech.play_cue("ok")
        self.speaker.speak(self.last_result_speech, interrupt=True)

    def _resume_live(self):
        self.anim_timer.stop()
        self.state = SEARCHING
        self.stable_cls = None
        self.stable_streak = 0
        self.best = None
        self.captured_cls = None
        self.absent = 0
        self.scan_rgb = self.scan_roi = self.scan_pred = None

    def _go_next(self):
        self._resume_live()
        self._render_searching()
        self.speaker.speak(config.SPEECH_NEXT, interrupt=False)

    # ---------------- Tương tác ----------------

    def _repeat(self):
        if self.last_result_speech and self.state == RESULT:
            self.speaker.speak(self.last_result_speech, interrupt=True)
        elif self.state == SEARCHING:
            self.speaker.speak(config.SPEECH_GREETING, interrupt=True)
        elif self.last_result_speech:
            self.speaker.speak(self.last_result_speech, interrupt=True)

    def mousePressEvent(self, event):
        self._repeat()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self._repeat()
        elif key == Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, event):
        self.timer.stop()
        self.anim_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.speaker.stop()
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
