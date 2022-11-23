import torch

from util import grab_window_image
from pynput import mouse, keyboard
from torchvision.transforms import ToTensor
import time


class Fisherman:
    STATE_UNKNOWN = 0
    STATE_NO_FISHING_OK = 1
    STATE_FISHING_IDLE = 2
    STATE_FISHING_CATCH = 3
    STATE_FISHING_INVALID = 4

    def __init__(self, hwnd, model, model_states, transform, toggle_run_hotkey=None):
        self.hwnd = hwnd
        self.model = model
        self.transform = transform
        self.state_map = model_states
        self.mouse = mouse.Controller()
        self.current_state = self.STATE_UNKNOWN
        self.last_grab_time = 0.0
        self.last_state_change = 0.0
        self.is_catching = False
        self.toggle_run_hotkey = toggle_run_hotkey
        self.is_running = False
        self.pressed_keys = set()

        self.state_label_map = {
            self.STATE_UNKNOWN: "Unknown",
            self.STATE_NO_FISHING_OK: "No fishing",
            self.STATE_FISHING_IDLE: "Fishing, Idle",
            self.STATE_FISHING_CATCH: "Catch!",
        }

    def start(self, max_grab_rate=10):
        grab_interval = 1.0 / max_grab_rate

        listener = keyboard.Listener(on_press=self.hotkey_on_press_listener, on_release=self.hotkey_on_release_listener)
        listener.start()

        while True:
            if not self.is_running:
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time - self.last_grab_time < grab_interval:
                time.sleep(.01)
                continue

            self.last_grab_time = current_time
            image = self.transform(grab_window_image(self.hwnd))
            x = torch.unsqueeze(ToTensor()(image), 0)
            state = self.state_map[self.model(x).argmax(1)]
            if state != self.current_state:
                state_changed = False

                if state == self.STATE_FISHING_CATCH:
                    if self.current_state == self.STATE_FISHING_IDLE and current_time - self.last_state_change > 2.0:
                        state_changed = True
                        self.is_catching = True
                        self.mouse.click(mouse.Button.right)
                elif state == self.STATE_NO_FISHING_OK:
                    if self.current_state == self.STATE_FISHING_CATCH and current_time - self.last_state_change > 1.0:
                        state_changed = True
                    elif self.current_state == self.STATE_UNKNOWN:
                        state_changed = True

                    if state_changed:
                        self.is_catching = False
                        self.mouse.click(mouse.Button.right)
                elif state == self.STATE_FISHING_IDLE:
                    if self.current_state == self.STATE_NO_FISHING_OK:
                        state_changed = True
                    elif self.current_state == self.STATE_FISHING_CATCH and not self.is_catching:
                        state_changed = True

                if state_changed:
                    print(self.state_label_map[state])
                    self.current_state = state
                    self.last_state_change = current_time

    def hotkey_on_press_listener(self, key):
        if self.toggle_run_hotkey is not None and key in self.toggle_run_hotkey:
            self.pressed_keys.add(key)
            if all(k in self.pressed_keys for k in self.toggle_run_hotkey):
                self.is_running = not self.is_running
                print("Running: %s" % self.is_running)

                if not self.is_running:
                    self.current_state = self.STATE_UNKNOWN

    def hotkey_on_release_listener(self, key):
        if self.toggle_run_hotkey is not None and key in self.toggle_run_hotkey:
            self.pressed_keys.remove(key)
