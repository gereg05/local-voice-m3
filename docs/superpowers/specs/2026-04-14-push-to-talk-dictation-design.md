# Push-to-Talk Diktier-Tool für macOS — Design Spec

**Datum:** 2026-04-14
**Status:** Approved
**Plattform:** macOS (Apple Silicon)

---

## 1. Überblick

Ein globales Push-to-Talk Diktier-Tool, das über die rechte Command-Taste aktiviert wird. Audio wird lokal mit mlx-whisper transkribiert und automatisch in das aktive Fenster eingefügt. Das Whisper-Modell bleibt permanent im RAM, um die Latenz zu minimieren.

## 2. Architektur

Zwei-Thread-Architektur mit Queue-Kommunikation.

### Main Thread
- **Rumps Menübar-App** mit Template-Icons und Sprache-Dropdown
- **Timer-Callback** (100ms Intervall) pollt `recording_event` (Icon-Wechsel) und `result_queue` (Paste-Aktion)
- **Paste-Logik:** `pyperclip.copy(text)` → `time.sleep(0.05)` → Cmd+V via pynput

### Worker Thread (daemon=True)
- **pynput KeyListener:** Right Cmd als Hold-to-Talk Trigger
- **sounddevice InputStream:** 16kHz, mono, float32, callback-basiert
- **mlx-whisper Transkription:** In-Memory, Modell bleibt nach erstem Aufruf im RAM
- Ergebnisse werden via `result_queue.put(text)` an den Main Thread übergeben

### Kommunikation
- `threading.Event` — Recording-Status (Worker → Main, für Icon-Wechsel)
- `queue.Queue` — Transkriptions-Ergebnisse (Worker → Main)
- `threading.Lock` — Thread-safe Zugriff auf Language-Einstellung (Main → Worker)

### Architektur-Diagramm

```
┌──────────────────────────────────────┐
│           Main Thread                │
│  ┌────────────────────────────────┐  │
│  │  Rumps Menübar-App             │  │
│  │  - Template Icons (outline/    │  │
│  │    filled)                     │  │
│  │  - Dropdown: Auto/DE/EN        │  │
│  │  - "Beenden"-Button            │  │
│  └────────────────────────────────┘  │
│         ▲ rumps.timer (0.1s)         │
│         │ pollt result_queue         │
│  ┌──────┴─────────────────────────┐  │
│  │ pyperclip.copy(text)           │  │
│  │ time.sleep(0.05)               │  │
│  │ pynput Cmd+V Simulation        │  │
│  └────────────────────────────────┘  │
└─────────┬────────────────────────────┘
          │ queue.Queue (text)
┌─────────┼────────────────────────────┐
│         │    Worker Thread (daemon)   │
│  ┌──────┴─────────────────────────┐  │
│  │ pynput KeyListener             │  │
│  │ - Right Cmd press → start rec  │  │
│  │ - Right Cmd release → stop rec │  │
│  └─────────┬──────────────────────┘  │
│  ┌─────────▼──────────────────────┐  │
│  │ sounddevice InputStream        │  │
│  │ - 16kHz mono float32           │  │
│  │ - callback → audio_chunks list │  │
│  └─────────┬──────────────────────┘  │
│  ┌─────────▼──────────────────────┐  │
│  │ mlx-whisper transcribe         │  │
│  │ - mlx-community/whisper-      │  │
│  │   small-mlx                   │  │
│  │ - Permanent im RAM            │  │
│  └─────────┬──────────────────────┘  │
│            │ result_queue.put(text)   │
└────────────┴─────────────────────────┘
```

## 3. Dateistruktur

```
Voice to text Model/
├── dictation_app.py        # Einstiegspunkt, Rumps-App, Main-Thread-Logik
├── worker.py               # Worker-Thread: KeyListener, Audio, Transkription
├── requirements.txt        # Python Dependencies
├── README.md               # Setup-Anleitung
├── assets/
│   ├── mic_outlineTemplate.png   # Menübar: bereit (18x18px, monochrom)
│   └── mic_filledTemplate.png    # Menübar: nimmt auf (18x18px, monochrom)
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-14-push-to-talk-dictation-design.md
```

## 4. Datenfluss

```
Right Cmd gedrückt
    │
    ▼
State: IDLE → RECORDING
recording_event.set()
sounddevice.InputStream.start()  (Context Manager)
    │
    │  callback(indata, frames, time, status):
    │      audio_chunks.append(indata.copy())
    │
Right Cmd losgelassen
    │
    ▼
stream.stop()
State: RECORDING → TRANSCRIBING
audio = np.concatenate(audio_chunks).flatten()   # 1D float32 Array
    │
    ├── len(audio) < 8000? → verwerfen, State → IDLE (Audio-Gate: < 0.5s)
    │
    ▼
mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-small-mlx")
    │                    + language="de"|"en" (oder kein Parameter bei Auto)
    │
    ├── result["text"] leer? → ignorieren, State → IDLE
    │
    ▼
result_queue.put(text)
State: TRANSCRIBING → IDLE
    │
    ▼  (Main Thread Timer pollt)
pyperclip.copy(text)
time.sleep(0.05)
pynput: Cmd+V Simulation
```

## 5. Menübar-UI (Rumps)

```
┌──────────────────────┐
│ 🎤 (Template Icon)   │
├──────────────────────┤
│ Sprache           ▶  │
│  ├ ✓ Auto            │
│  ├   Deutsch         │
│  └   Englisch        │
├──────────────────────┤
│ Beenden              │
└──────────────────────┘
```

- **Icons:** `mic_outlineTemplate.png` (bereit) / `mic_filledTemplate.png` (Aufnahme). Template-Suffix im Dateinamen → macOS behandelt sie automatisch als Template-Images (korrekt in Light/Dark Mode)
- **Sprache-Submenu:** Radio-Button-Logik (genau eine Option aktiv). Default: Auto. Zugriff auf die Variable über `threading.Lock`
- **Timer:** `rumps.Timer(callback, 0.1)` — pollt `recording_event` und `result_queue`
- **Beenden:** Stoppt Worker, `rumps.quit_application()`

## 6. State-Machine

```
IDLE ──(Right Cmd press)──→ RECORDING ──(Right Cmd release)──→ TRANSCRIBING ──(done)──→ IDLE
```

- Tastendrücke werden **nur im IDLE-State** akzeptiert
- Jeder State-Übergang ist atomar
- `finally`-Block garantiert Reset auf IDLE bei jeder Exception

## 7. Error-Handling

| Szenario | Verhalten |
|----------|-----------|
| Audio < 0.5s (versehentlicher Tap) | Verwerfen, keine Transkription, State → IDLE |
| Kein Mikrofon verfügbar | Rumps-Notification beim Start |
| Accessibility-Permission fehlt | pynput Exception → Rumps-Notification |
| Transkription liefert leeren String | Ignorieren, kein Paste |
| Tastendruck während TRANSCRIBING | Wird ignoriert (State-Machine) |
| Erster Aufruf (Modell laden) | ~1-2s Latenz, danach Modell im RAM |
| `Key.cmd_r` nicht verfügbar | `AttributeError`-Catch, Fallback auf `KeyCode` |

## 8. Thread-Safety

- **`recording_event`** (`threading.Event`): Thread-safe by design
- **`result_queue`** (`queue.Queue`): Thread-safe by design
- **`language_setting`** (str): Geschützt durch `threading.Lock`. Main-Thread schreibt (Menü-Callback), Worker-Thread liest (vor Transkription)

## 9. Dependencies

### System (Homebrew)
```
brew install portaudio
```

### Python (requirements.txt)
```
rumps
pynput
sounddevice
numpy
pyperclip
mlx-whisper
```

### Python-Version
3.11+ (Voraussetzung für mlx)

## 10. macOS Permissions

Beim ersten Start werden diese Berechtigungen benötigt:
- **Accessibility:** Für pynput (globaler KeyListener + Tastensimulation). Muss in Systemeinstellungen → Datenschutz & Sicherheit → Bedienungshilfen gewährt werden
- **Mikrofon:** Für sounddevice. macOS zeigt automatisch einen Dialog

## 11. Nicht im Scope

- Streaming-Transkription (nicht nötig für kurze Diktate)
- Konfigurierbare Hotkeys (Right Cmd ist fix)
- Auto-Start bei Login (kann später ergänzt werden)
- Mehrere Whisper-Modellgrößen (fix: whisper-small)
- Interpunktions-/Formatierungs-Postprocessing
