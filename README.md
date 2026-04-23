# Push-to-Talk Dictation Tool

Globales Diktier-Tool für macOS (Apple Silicon). Halte die **rechte Command-Taste** gedrückt, sprich, und der transkribierte Text wird automatisch in das aktive Fenster eingefügt.

- Lokale Transkription via **mlx-whisper** (whisper-small) — keine Cloud, keine API-Keys
- Menübar-App mit Sprachauswahl (Auto/Deutsch/Englisch)
- Optionales Audio-Feedback beim Starten und Stoppen der Aufnahme
- Zwischenablage wird nach dem Einfügen wiederhergestellt
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

1. **Rechte Command-Taste gedrückt halten** — Aufnahme startet (Icon wechselt zu ausgefülltem Mikrofon)
2. **Sprechen**
3. **Taste loslassen** — Audio wird transkribiert und in das aktive Fenster eingefügt

### Sprachauswahl

Über das Menübar-Icon → Sprache:
- **Auto** (Standard): Whisper erkennt die Sprache automatisch
- **Deutsch**: Erzwingt deutsche Transkription
- **Englisch**: Erzwingt englische Transkription

### Audio-Feedback

Über das Menübar-Icon → **Audio-Feedback** kann der kurze Ton beim Starten
und Stoppen der Aufnahme ein- oder ausgeschaltet werden.

## Hinweise

- Der erste Transkriptionsaufruf dauert ~1-2 Sekunden (Modell wird geladen). Danach bleibt das Modell im RAM.
- Audio kürzer als 0.5 Sekunden wird ignoriert (versehentliche Taps).
- Das Tool nutzt die Zwischenablage (Cmd+V) zum Einfügen und stellt den vorherigen Inhalt danach wieder her.
