import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading

import config

_WIN_PLAY_PS1 = r"""
$ErrorActionPreference = 'Stop'
$path = $env:TTS_FILE
$sig = @'
[DllImport("winmm.dll", CharSet=CharSet.Auto)]
public static extern int mciSendString(string command, System.Text.StringBuilder ret, int len, System.IntPtr cb);
'@
$mci = Add-Type -MemberDefinition $sig -Name 'Mci' -Namespace 'Win32Tts' -PassThru
[void]$mci::mciSendString("open `"$path`" type mpegvideo alias ttsclip", $null, 0, [System.IntPtr]::Zero)
[void]$mci::mciSendString("play ttsclip wait", $null, 0, [System.IntPtr]::Zero)
[void]$mci::mciSendString("close ttsclip", $null, 0, [System.IntPtr]::Zero)
"""

_LINUX_PLAYERS = [
    ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("mpg123", ["mpg123", "-q"]),
    ("mpv", ["mpv", "--no-video", "--really-quiet"]),
    ("cvlc", ["cvlc", "--play-and-exit", "--quiet"]),
]


class Speaker:
    def __init__(self, voice: str = "Linh", rate: int = 180):
        self.voice = voice
        self.rate = rate
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._proc = None
        self._proc_lock = threading.Lock()
        self._engine = None            
        self._win_ps1 = None           
        self._backend = self._pick_backend()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def backend(self) -> str:
        return self._backend

    def _pick_backend(self) -> str:
        # macOS: giữ nguyên cách cũ — lệnh `say` với giọng Linh.
        if platform.system() == "Darwin" and shutil.which("say"):
            return "say"
        # Windows/Linux: ưu tiên edge-tts (giọng nữ tiếng Việt, cần internet).
        try:
            import edge_tts  # noqa: F401
            return "edgetts"
        except Exception:
            pass
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

    # ---------------- Vòng lặp phát theo backend ----------------

    def _run(self) -> None:
        if self._backend == "say":
            self._run_say()
        elif self._backend == "edgetts":
            self._run_edgetts()
        elif self._backend == "pyttsx3":
            self._run_pyttsx3()
        else:
            self._run_none()

    def _run_none(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            print(f"[TTS] {text}")

    def _run_say(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            cmd = ["say", "-v", self.voice, "-r", str(self.rate), text]
            try:
                with self._proc_lock:
                    self._proc = subprocess.Popen(cmd)
                self._proc.wait()
            except Exception as exc:
                print(f"[TTS lỗi] {exc}")

    def _run_pyttsx3(self) -> None:
        engine = self._init_pyttsx3()
        if engine is None:
            self._run_none()
            return
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS lỗi] {exc}")

    def _run_edgetts(self) -> None:
        import asyncio
        import edge_tts

        voice = getattr(config, "VOICE_EDGE", "vi-VN-HoaiMyNeural")
        rate = getattr(config, "VOICE_EDGE_RATE", "+0%")
        tmp_dir = tempfile.mkdtemp(prefix="vncur_tts_")
        warned = False

        while True:
            text = self._q.get()
            if text is None:
                self._cleanup_dir(tmp_dir)
                return
            mp3 = os.path.join(tmp_dir, "utt.mp3")
            try:
                async def _synth():
                    await edge_tts.Communicate(text, voice, rate=rate).save(mp3)
                asyncio.run(_synth())
                self._play_file(mp3)
            except Exception as exc:
                # Mất mạng / edge-tts lỗi → tụt xuống pyttsx3 cho câu này.
                if not warned:
                    print(f"[TTS] edge-tts không dùng được ({exc}), "
                          f"chuyển sang giọng offline.")
                    warned = True
                self._say_with_pyttsx3(text)
            finally:
                try:
                    if os.path.exists(mp3):
                        os.remove(mp3)
                except OSError:
                    pass

    # ---------------- Tiện ích ----------------

    def _init_pyttsx3(self):
        if self._engine is not None:
            return self._engine
        try:
            import pyttsx3
            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", self.rate)
                for v in engine.getProperty("voices"):
                    if "vietnam" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            except Exception:
                pass
            self._engine = engine
            return engine
        except Exception:
            return None

    def _say_with_pyttsx3(self, text: str) -> None:
        engine = self._init_pyttsx3()
        if engine is None:
            print(f"[TTS] {text}")
            return
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"[TTS lỗi] {exc}")

    def _play_file(self, mp3_path: str) -> None:
        """Phát file mp3, chặn tới khi xong. Lưu proc để interrupt cắt được."""
        system = platform.system()
        if system == "Windows":
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-File", self._ensure_win_ps1()]
            env = {**os.environ, "TTS_FILE": mp3_path}
            with self._proc_lock:
                self._proc = subprocess.Popen(cmd, env=env)
            self._proc.wait()
            return
        if system == "Darwin" and shutil.which("afplay"):
            with self._proc_lock:
                self._proc = subprocess.Popen(["afplay", mp3_path])
            self._proc.wait()
            return
        # Linux: thử các trình phát dòng lệnh phổ biến.
        for name, base in _LINUX_PLAYERS:
            if shutil.which(name):
                with self._proc_lock:
                    self._proc = subprocess.Popen(base + [mp3_path])
                self._proc.wait()
                return
        raise RuntimeError("Không tìm thấy trình phát mp3 trên hệ thống")

    def _ensure_win_ps1(self) -> str:
        if self._win_ps1 and os.path.exists(self._win_ps1):
            return self._win_ps1
        fd, path = tempfile.mkstemp(suffix=".ps1", prefix="vncur_play_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_WIN_PLAY_PS1)
        self._win_ps1 = path
        return path

    @staticmethod
    def _cleanup_dir(path: str) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


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
