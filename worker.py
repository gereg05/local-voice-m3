from __future__ import annotations

import enum
import gc
import os
import queue
import threading

import numpy as np
import sounddevice as sd

# Use cached model only — no network calls to Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

SAMPLE_RATE = 16000
MIN_AUDIO_FRAMES = 8000  # 0.5 seconds at 16kHz
MODEL_REPO = "mlx-community/whisper-small-mlx"


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
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._audio_chunks.clear()
        self._recording_event.clear()

    # --- recording control ---

    def start_recording(self) -> None:
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

    def stop_recording_and_transcribe(self) -> None:
        if self._state is not State.RECORDING:
            return

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            self._recording_event.clear()
            self._state = State.TRANSCRIBING

            if not self._audio_chunks:
                return

            audio = np.concatenate(self._audio_chunks).flatten()
            self._audio_chunks.clear()  # free chunk list immediately

            if len(audio) < MIN_AUDIO_FRAMES:
                return

            self._transcribe(audio)

        except Exception as e:
            print(f"[worker] error: {e}")
        finally:
            self._audio_chunks.clear()
            self._state = State.IDLE
            self._recording_event.clear()
            gc.collect()

    def _transcribe(self, audio: np.ndarray) -> None:
        import mlx_whisper

        lang = self._get_language()
        kwargs: dict = {
            "path_or_hf_repo": MODEL_REPO,
            "fp16": True,          # half-precision – faster on M3, less RAM
            "condition_on_previous_text": False,  # skip context window – lower latency
        }
        if lang is not None:
            kwargs["language"] = lang

        result = mlx_whisper.transcribe(audio, **kwargs)
        text = result.get("text", "").strip()

        # free transcription tensors before pasting
        del result, audio
        gc.collect()

        if text:
            self._result_queue.put(text)

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
