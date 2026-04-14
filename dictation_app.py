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
