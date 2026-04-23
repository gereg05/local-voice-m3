from __future__ import annotations

import enum
import gc
import os
import queue
import threading
import warnings

import numpy as np
import sounddevice as sd

# Use cached model only — no network calls to Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

# Suppress harmless multiprocessing semaphore warning on shutdown
warnings.filterwarnings("ignore", message=".*resource_tracker.*leaked.*")

SAMPLE_RATE = 16000
MIN_AUDIO_FRAMES = 8000  # 0.5 seconds at 16kHz
MODEL_REPO = "mlx-community/whisper-small-mlx"
TRANSCRIBE_TIMEOUT = 45  # seconds before late results are discarded

_CACHE_LIMIT = 1024 * 1024 * 1024  # 1 GB
_mlx = None
_mlx_configured = False
_mlx_lock = threading.Lock()


def _get_mlx():
    global _mlx, _mlx_configured

    with _mlx_lock:
        if _mlx is None:
            import mlx.core as mx

            _mlx = mx

        if not _mlx_configured:
            # Cap MLX metal cache to prevent runaway memory on 8 GB machines.
            if hasattr(_mlx, "set_cache_limit"):
                _mlx.set_cache_limit(_CACHE_LIMIT)
            else:
                _mlx.metal.set_cache_limit(_CACHE_LIMIT)
            _mlx_configured = True

        return _mlx


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


class DictationWorker:
    def __init__(self, result_queue: queue.Queue, recording_event: threading.Event):
        self._result_queue = result_queue
        self._recording_event = recording_event

        self._state = State.IDLE
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._record_lock = threading.Lock()
        self._transcription_id = 0

        self._language: str | None = None  # None = auto-detect
        self._language_lock = threading.Lock()

    # --- language setting (called from Main Thread) ---

    def set_language(self, lang: str | None) -> None:
        with self._language_lock:
            self._language = lang

    def _get_language(self) -> str | None:
        with self._language_lock:
            return self._language

    # --- audio callback ---

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[audio] {status}")
        self._audio_chunks.append(indata.copy())

    # --- cleanup (called on app shutdown) ---

    def cleanup(self) -> None:
        with self._record_lock:
            self._close_stream()
            self._audio_chunks.clear()
            self._state = State.IDLE
            self._recording_event.clear()

    def _close_stream(self) -> None:
        """Close the audio stream if open. Caller must hold _record_lock."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # --- recording control ---

    def start_recording(self) -> None:
        with self._record_lock:
            if self._state is not State.IDLE:
                return

            self._state = State.RECORDING
            self._audio_chunks.clear()
            self._recording_event.set()

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            print("[worker] recording ...")

    def stop_recording_and_transcribe(self) -> None:
        """Stop recording and kick off transcription on a background thread."""
        with self._record_lock:
            if self._state is not State.RECORDING:
                return

            self._close_stream()
            self._state = State.TRANSCRIBING
            self._recording_event.clear()

            if not self._audio_chunks:
                self._state = State.IDLE
                return

            audio = np.concatenate(self._audio_chunks).flatten()
            self._audio_chunks.clear()

        if len(audio) < MIN_AUDIO_FRAMES:
            self._finish_transcription()
            return

        with self._record_lock:
            self._transcription_id += 1
            transcription_id = self._transcription_id

        # offload transcription — pynput thread stays free
        t = threading.Thread(
            target=self._transcribe_safe,
            args=(audio, transcription_id),
            daemon=True,
        )
        t.start()

    def _transcribe_safe(self, audio: np.ndarray, transcription_id: int) -> None:
        """Run transcription with a timeout watchdog."""
        duration = len(audio) / SAMPLE_RATE
        print(f"[worker] transcribing {duration:.1f}s audio ...")

        finished = threading.Event()
        result_holder = {"text": ""}

        def _run() -> None:
            try:
                result_holder["text"] = self._transcribe(audio)
            except Exception as e:
                print(f"[worker] error: {e}")
            finally:
                self._finish_transcription(transcription_id)
                finished.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        if finished.wait(timeout=TRANSCRIBE_TIMEOUT):
            text = result_holder["text"]
            if text:
                self._result_queue.put(text)
            return

        if result_holder["text"]:
            print("[worker] late transcription discarded")
        else:
            print(f"[worker] TIMEOUT after {TRANSCRIBE_TIMEOUT}s — skipping")
        # Can't kill the thread. The worker stays in TRANSCRIBING until the
        # daemon finishes, so no second MLX job starts in parallel.

    def _finish_transcription(self, transcription_id: int | None = None) -> None:
        with self._record_lock:
            if transcription_id is None or transcription_id == self._transcription_id:
                self._state = State.IDLE

        gc.collect()

    def _transcribe(self, audio: np.ndarray) -> str:
        mx = _get_mlx()
        import mlx_whisper

        lang = self._get_language()
        kwargs: dict = {
            "path_or_hf_repo": MODEL_REPO,
            "fp16": True,
            "condition_on_previous_text": False,
        }
        if lang is not None:
            kwargs["language"] = lang

        result = mlx_whisper.transcribe(audio, **kwargs)
        text = result.get("text", "").strip()

        # free transcription tensors + MLX GPU cache
        del result, audio
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
        gc.collect()

        if text:
            print(f"[worker] done: {text[:60]}...")

        return text

    # --- key listener (runs in its own thread via pynput) ---

    def run_key_listener(self) -> None:
        from pynput import keyboard
        from pynput.keyboard import Key, KeyCode

        try:
            right_cmd = Key.cmd_r
        except AttributeError:
            right_cmd = KeyCode.from_vk(0x36)  # Right Command virtual keycode

        def on_press(key):
            if key == right_cmd:
                self.start_recording()

        def on_release(key):
            if key == right_cmd:
                self.stop_recording_and_transcribe()

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
