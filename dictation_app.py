from __future__ import annotations

import os
import queue
import threading
import time

import pyperclip
import rumps
from pynput.keyboard import Controller, Key

from worker import DictationWorker

try:
    from AppKit import NSSound, NSPasteboard, NSPasteboardItem
except Exception:  # pragma: no cover - macOS runtime dependency
    NSSound = None
    NSPasteboard = None
    NSPasteboardItem = None

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ICON_IDLE = os.path.join(ASSETS_DIR, "mic_outlineTemplate.png")
ICON_RECORDING = os.path.join(ASSETS_DIR, "mic_filledTemplate.png")
PASTE_SETTLE_SECONDS = 0.05
CLIPBOARD_RESTORE_DELAY_SECONDS = 0.75

LANGUAGE_MAP = {
    "Auto": None,
    "Deutsch": "de",
    "Englisch": "en",
}


class DictationApp(rumps.App):
    def __init__(self):
        super().__init__("Dictation", icon=ICON_IDLE, template=True)

        self._result_queue: queue.Queue[str] = queue.Queue()
        self._paste_queue: queue.Queue[str] = queue.Queue()
        self._recording_event = threading.Event()
        self._worker = DictationWorker(self._result_queue, self._recording_event)
        self._keyboard = Controller()
        self._was_recording = False
        self._audio_feedback_enabled = True
        self._sounds = {}

        # --- build menu ---
        self._language_items: dict[str, rumps.MenuItem] = {}
        language_menu = rumps.MenuItem("Sprache")
        self._audio_feedback_item = rumps.MenuItem(
            "Audio-Feedback",
            callback=self._on_audio_feedback_toggle,
        )
        self._audio_feedback_item.state = True

        for label in LANGUAGE_MAP:
            item = rumps.MenuItem(label, callback=self._on_language_select)
            self._language_items[label] = item
            language_menu.add(item)

        self._language_items["Auto"].state = True  # default

        self.menu = [language_menu, self._audio_feedback_item, None]

        # --- start worker thread ---
        worker_thread = threading.Thread(
            target=self._worker.run_key_listener,
            daemon=True,
        )
        worker_thread.start()

        paste_thread = threading.Thread(target=self._run_paste_worker, daemon=True)
        paste_thread.start()

    def _on_language_select(self, sender: rumps.MenuItem) -> None:
        for item in self._language_items.values():
            item.state = False
        sender.state = True

        lang_code = LANGUAGE_MAP[sender.title]
        self._worker.set_language(lang_code)

    def _on_audio_feedback_toggle(self, sender: rumps.MenuItem) -> None:
        self._audio_feedback_enabled = not self._audio_feedback_enabled
        sender.state = self._audio_feedback_enabled

    @rumps.timer(0.1)
    def _poll(self, _timer) -> None:
        # update icon based on recording state
        is_recording = self._recording_event.is_set()
        if is_recording and not self._was_recording:
            self.icon = ICON_RECORDING
            self._play_sound("Tink")
        elif not is_recording and self._was_recording:
            self.icon = ICON_IDLE
            self._play_sound("Pop")
        self._was_recording = is_recording

        # check for transcription results
        while True:
            try:
                text = self._result_queue.get_nowait()
            except queue.Empty:
                return

            self._paste_queue.put(text)

    def _run_paste_worker(self) -> None:
        while True:
            text = self._paste_queue.get()
            try:
                self._paste_text(text)
            except Exception as e:
                print(f"[paste] failed: {e}")
            finally:
                self._paste_queue.task_done()

    def _paste_text(self, text: str) -> None:
        clipboard_snapshot = self._capture_clipboard()
        fallback_text = None

        if clipboard_snapshot is None:
            try:
                fallback_text = pyperclip.paste()
            except pyperclip.PyperclipException:
                fallback_text = None

        try:
            pyperclip.copy(text)
            time.sleep(PASTE_SETTLE_SECONDS)
            self._keyboard.press(Key.cmd)
            try:
                self._keyboard.tap("v")
            finally:
                self._keyboard.release(Key.cmd)
            time.sleep(CLIPBOARD_RESTORE_DELAY_SECONDS)
        finally:
            if clipboard_snapshot is not None:
                self._restore_clipboard(clipboard_snapshot)
            elif fallback_text is not None:
                pyperclip.copy(fallback_text)

    def _play_sound(self, sound_name: str) -> None:
        if not self._audio_feedback_enabled or NSSound is None:
            return

        sound = NSSound.soundNamed_(sound_name)
        if sound is not None:
            self._sounds[sound_name] = sound
            sound.play()

    def _capture_clipboard(self):
        if NSPasteboard is None:
            return None

        try:
            pasteboard = NSPasteboard.generalPasteboard()
            items = pasteboard.pasteboardItems() or []
            snapshot = []

            for item in items:
                item_data = []
                for pasteboard_type in item.types():
                    data = item.dataForType_(pasteboard_type)
                    if data is not None:
                        item_data.append((pasteboard_type, data.copy()))
                if item_data:
                    snapshot.append(item_data)

            return snapshot
        except Exception as e:
            print(f"[clipboard] capture failed: {e}")
            return None

    def _restore_clipboard(self, snapshot) -> None:
        if NSPasteboard is None or NSPasteboardItem is None:
            return

        try:
            pasteboard = NSPasteboard.generalPasteboard()
            pasteboard.clearContents()

            if not snapshot:
                return

            restored_items = []
            for item_data in snapshot:
                item = NSPasteboardItem.alloc().init()
                for pasteboard_type, data in item_data:
                    item.setData_forType_(data, pasteboard_type)
                restored_items.append(item)

            pasteboard.writeObjects_(restored_items)
        except Exception as e:
            print(f"[clipboard] restore failed: {e}")


if __name__ == "__main__":
    import atexit
    import signal
    import sys

    app = DictationApp()

    def _shutdown(*_args) -> None:
        app._worker.cleanup()

    def _signal_exit(*_args) -> None:
        _shutdown()
        sys.exit(0)

    atexit.register(_shutdown)
    signal.signal(signal.SIGINT, _signal_exit)
    signal.signal(signal.SIGTERM, _signal_exit)

    app.run()
