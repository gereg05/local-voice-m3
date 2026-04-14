# Push-to-Talk Dictation Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a global push-to-talk dictation tool for macOS that records audio while holding Right Cmd, transcribes locally via mlx-whisper, and pastes the result into the active window.

**Architecture:** Two-thread design — Main Thread runs a Rumps menubar app (UI, clipboard, paste), Worker Thread (daemon) handles pynput key listening, sounddevice recording, and mlx-whisper transcription. Communication via `threading.Event`, `queue.Queue`, and `threading.Lock`.

**Tech Stack:** Python 3.11+, rumps, pynput, sounddevice, numpy, pyperclip, mlx-whisper

**Spec:** `docs/superpowers/specs/2026-04-14-push-to-talk-dictation-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Python dependencies |
| `worker.py` | Worker thread: key listener, audio recording, transcription, state machine |
| `dictation_app.py` | Entry point: Rumps menubar app, timer polling, clipboard+paste, language menu |
| `assets/mic_outlineTemplate.png` | Menubar icon: idle state (microphone outline, 18x18px monochrome) |
| `assets/mic_filledTemplate.png` | Menubar icon: recording state (filled microphone, 18x18px monochrome) |
| `README.md` | Setup instructions with brew + pip + permissions |

---

### Task 1: requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
rumps
pynput
sounddevice
numpy
pyperclip
mlx-whisper
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements.txt with all dependencies"
```

---

### Task 2: Template Icon Assets

**Files:**
- Create: `assets/mic_outlineTemplate.png`
- Create: `assets/mic_filledTemplate.png`

- [ ] **Step 1: Generate mic_outlineTemplate.png**

Use Python + Pillow to generate an 18x18px monochrome microphone outline icon. The `Template` suffix in the filename tells macOS to treat it as a template image (auto-adapts to Light/Dark Mode).

```python
from PIL import Image, ImageDraw

img = Image.new("RGBA", (18, 18), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
# Microphone body outline (rounded rect approximation)
draw.ellipse([5, 1, 12, 8], outline=(0, 0, 0, 255), width=1)
# Microphone stand
draw.line([9, 8, 9, 13], fill=(0, 0, 0, 255), width=1)
# Base arc
draw.arc([4, 6, 14, 14], start=0, end=180, fill=(0, 0, 0, 255), width=1)
# Base line
draw.line([6, 14, 12, 14], fill=(0, 0, 0, 255), width=1)
img.save("assets/mic_outlineTemplate.png")
```

- [ ] **Step 2: Generate mic_filledTemplate.png**

```python
from PIL import Image, ImageDraw

img = Image.new("RGBA", (18, 18), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
# Filled microphone body
draw.ellipse([5, 1, 12, 8], fill=(0, 0, 0, 255))
# Microphone stand
draw.line([9, 8, 9, 13], fill=(0, 0, 0, 255), width=1)
# Base arc
draw.arc([4, 6, 14, 14], start=0, end=180, fill=(0, 0, 0, 255), width=1)
# Base line
draw.line([6, 14, 12, 14], fill=(0, 0, 0, 255), width=1)
img.save("assets/mic_filledTemplate.png")
```

- [ ] **Step 3: Verify icons exist and are 18x18**

```bash
python3 -c "from PIL import Image; img=Image.open('assets/mic_outlineTemplate.png'); print(img.size)"
python3 -c "from PIL import Image; img=Image.open('assets/mic_filledTemplate.png'); print(img.size)"
```

Expected: `(18, 18)` for both.

- [ ] **Step 4: Commit**

```bash
git add assets/
git commit -m "feat: add menubar template icons (outline + filled)"
```

---

### Task 3: worker.py — State Machine, Key Listener, Audio Recording, Transcription

**Files:**
- Create: `worker.py`

- [ ] **Step 1: Create worker.py with all worker logic**

```python
import enum
import queue
import threading

import numpy as np
import sounddevice as sd

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

            if len(audio) < MIN_AUDIO_FRAMES:
                return

            self._transcribe(audio)

        except Exception as e:
            print(f"[worker] error: {e}")
        finally:
            self._state = State.IDLE
            self._recording_event.clear()

    def _transcribe(self, audio: np.ndarray) -> None:
        import mlx_whisper

        lang = self._get_language()
        kwargs: dict = {
            "path_or_hf_repo": MODEL_REPO,
        }
        if lang is not None:
            kwargs["language"] = lang

        result = mlx_whisper.transcribe(audio, **kwargs)
        text = result.get("text", "").strip()

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
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('worker.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add worker.py
git commit -m "feat: add worker with state machine, key listener, audio recording, transcription"
```

---

### Task 4: dictation_app.py — Rumps Menubar App, Timer, Paste Logic

**Files:**
- Create: `dictation_app.py`

- [ ] **Step 1: Create dictation_app.py**

```python
import os
import queue
import threading
import time

import pyperclip
import rumps
from pynput.keyboard import Controller, Key

from worker import DictationWorker

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ICON_IDLE = os.path.join(ASSETS_DIR, "mic_outlineTemplate.png")
ICON_RECORDING = os.path.join(ASSETS_DIR, "mic_filledTemplate.png")

LANGUAGE_MAP = {
    "Auto": None,
    "Deutsch": "de",
    "Englisch": "en",
}


class DictationApp(rumps.App):
    def __init__(self):
        super().__init__("Dictation", icon=ICON_IDLE, template=True)

        self._result_queue: queue.Queue[str] = queue.Queue()
        self._recording_event = threading.Event()
        self._worker = DictationWorker(self._result_queue, self._recording_event)
        self._keyboard = Controller()
        self._was_recording = False

        # --- build menu ---
        self._language_items: dict[str, rumps.MenuItem] = {}
        language_menu = rumps.MenuItem("Sprache")

        for label in LANGUAGE_MAP:
            item = rumps.MenuItem(label, callback=self._on_language_select)
            self._language_items[label] = item
            language_menu.add(item)

        self._language_items["Auto"].state = True  # default

        self.menu = [language_menu, None]  # None = separator before Quit

        # --- start worker thread ---
        worker_thread = threading.Thread(
            target=self._worker.run_key_listener,
            daemon=True,
        )
        worker_thread.start()

    def _on_language_select(self, sender: rumps.MenuItem) -> None:
        for item in self._language_items.values():
            item.state = False
        sender.state = True

        lang_code = LANGUAGE_MAP[sender.title]
        self._worker.set_language(lang_code)

    @rumps.timer(0.1)
    def _poll(self, _timer) -> None:
        # update icon based on recording state
        is_recording = self._recording_event.is_set()
        if is_recording and not self._was_recording:
            self.icon = ICON_RECORDING
        elif not is_recording and self._was_recording:
            self.icon = ICON_IDLE
        self._was_recording = is_recording

        # check for transcription results
        try:
            text = self._result_queue.get_nowait()
        except queue.Empty:
            return

        self._paste_text(text)

    def _paste_text(self, text: str) -> None:
        pyperclip.copy(text)
        time.sleep(0.05)
        self._keyboard.press(Key.cmd)
        self._keyboard.tap("v")
        self._keyboard.release(Key.cmd)


if __name__ == "__main__":
    DictationApp().run()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('dictation_app.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dictation_app.py
git commit -m "feat: add rumps menubar app with timer polling, language menu, paste logic"
```

---

### Task 5: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# Push-to-Talk Dictation Tool

Globales Diktier-Tool für macOS (Apple Silicon). Halte die **rechte Command-Taste** gedrückt, sprich, und der transkribierte Text wird automatisch in das aktive Fenster eingefügt.

- Lokale Transkription via **mlx-whisper** (whisper-small) — keine Cloud, keine API-Keys
- Menübar-App mit Sprachauswahl (Auto/Deutsch/Englisch)
- Modell bleibt im RAM — minimale Latenz nach dem ersten Aufruf

## Voraussetzungen

- macOS auf Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Homebrew

## Installation

### 1. System-Dependency installieren

```bash
brew install portaudio
```

### 2. Python-Umgebung einrichten

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. App starten

```bash
python dictation_app.py
```

## macOS-Berechtigungen

Beim ersten Start werden zwei Berechtigungen benötigt:

1. **Bedienungshilfen (Accessibility):** Systemeinstellungen → Datenschutz & Sicherheit → Bedienungshilfen → Terminal (oder IDE) hinzufügen
2. **Mikrofon:** macOS zeigt automatisch einen Dialog

## Verwendung

1. **Rechte Command-Taste gedrückt halten** → Aufnahme startet (Icon wechselt zu ausgefülltem Mikrofon)
2. **Sprechen**
3. **Taste loslassen** → Audio wird transkribiert und in das aktive Fenster eingefügt

### Sprachauswahl

Über das Menübar-Icon → Sprache:
- **Auto** (Standard): Whisper erkennt die Sprache automatisch
- **Deutsch**: Erzwingt deutsche Transkription
- **Englisch**: Erzwingt englische Transkription

## Hinweise

- Der erste Transkriptionsaufruf dauert ~1-2 Sekunden (Modell wird geladen). Danach bleibt das Modell im RAM.
- Audio kürzer als 0.5 Sekunden wird ignoriert (versehentliche Taps).
- Das Tool nutzt die Zwischenablage (Cmd+V) zum Einfügen. Der vorherige Inhalt der Zwischenablage wird dabei überschrieben.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

### Task 6: Integration Test (Manual)

- [ ] **Step 1: Install dependencies**

```bash
cd "/Users/g.schaunig/Library/Mobile Documents/com~apple~CloudDocs/Voice to text Model"
brew install portaudio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install Pillow  # for icon generation only
```

- [ ] **Step 2: Generate icons**

```bash
python3 -c "
from PIL import Image, ImageDraw
import os
os.makedirs('assets', exist_ok=True)

# Outline icon
img = Image.new('RGBA', (18, 18), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse([5, 1, 12, 8], outline=(0, 0, 0, 255), width=1)
d.line([9, 8, 9, 13], fill=(0, 0, 0, 255), width=1)
d.arc([4, 6, 14, 14], start=0, end=180, fill=(0, 0, 0, 255), width=1)
d.line([6, 14, 12, 14], fill=(0, 0, 0, 255), width=1)
img.save('assets/mic_outlineTemplate.png')

# Filled icon
img = Image.new('RGBA', (18, 18), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse([5, 1, 12, 8], fill=(0, 0, 0, 255))
d.line([9, 8, 9, 13], fill=(0, 0, 0, 255), width=1)
d.arc([4, 6, 14, 14], start=0, end=180, fill=(0, 0, 0, 255), width=1)
d.line([6, 14, 12, 14], fill=(0, 0, 0, 255), width=1)
img.save('assets/mic_filledTemplate.png')
print('Icons created')
"
```

- [ ] **Step 3: Launch and test**

```bash
python dictation_app.py
```

Verify:
1. Menubar icon appears (microphone outline)
2. Click icon → "Sprache" submenu with Auto/Deutsch/Englisch
3. Hold Right Cmd → icon changes to filled microphone
4. Speak and release → text appears in active window
5. Click "Beenden" → app quits cleanly

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: push-to-talk dictation tool — complete implementation"
```
