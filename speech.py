import os
import platform
import queue
import shutil
import subprocess
import threading


class Speaker:
    def __init__(self, voice: str = "Linh", rate: int = 180):
        self.voice = voice
        self.rate = rate
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._proc = None
        self._proc_lock = threading.Lock()
        self._engine = None
        self._backend = self._pick_backend()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def backend(self) -> str:
        return self._backend

    def _pick_backend(self) -> str:
        if platform.system() == "Darwin" and shutil.which("say"):
            return "say"
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except Exception:
            return "none"

    def speak(self, text: str, interrupt: bool = False) -> None:
        if not text:
            return
        if interrupt:
            self._flush()
        self._q.put(text)

    def stop(self) -> None:
        self._flush()
        self._q.put(None)

    def _flush(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def _run(self) -> None:
        if self._backend == "pyttsx3":
            self._run_pyttsx3()
        else:
            self._run_say()

    def _run_say(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            if self._backend == "none":
                print(f"[TTS] {text}")
                continue
            cmd = ["say", "-v", self.voice, "-r", str(self.rate), text]
            try:
                with self._proc_lock:
                    self._proc = subprocess.Popen(cmd)
                self._proc.wait()
            except Exception as exc:
                print(f"[TTS lỗi] {exc}")

    def _run_pyttsx3(self) -> None:
        import pyttsx3
        self._engine = pyttsx3.init()
        try:
            self._engine.setProperty("rate", self.rate)
            for v in self._engine.getProperty("voices"):
                if "vi" in (getattr(v, "languages", []) or []) or \
                        "vietnam" in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"[TTS lỗi] {exc}")


# --- Âm báo ngắn (không lời) cho các mốc trạng thái ---
_MAC_SOUNDS = {"scan": "Tink", "ok": "Glass", "fail": "Basso"}


def play_cue(kind: str) -> None:
    """Phát âm báo ngắn: 'scan' khi bắt đầu quét, 'ok'/'fail' cho kết quả."""
    if platform.system() == "Darwin" and shutil.which("afplay"):
        path = f"/System/Library/Sounds/{_MAC_SOUNDS.get(kind, 'Tink')}.aiff"
        if os.path.exists(path):
            try:
                subprocess.Popen(["afplay", path])
            except Exception:
                pass
