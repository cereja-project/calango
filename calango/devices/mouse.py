"""
MIT License

Copyright (c) 2021 Cereja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Tuple, Union
import cereja as cj

try:
    import pyautogui

    pyautogui_available = True
except:
    pyautogui_available = False

__all__ = ['Mouse']


class Mouse:
    def __init__(self, jump=5, reduce_noise=True, noise_threshold=3):
        assert pyautogui_available, "invalid environment"
        self._jump = jump
        self._last_position = None
        self._reduce_noise = reduce_noise
        self._noise_threshold = noise_threshold

    @property
    def window_size(self):
        """
        Get window Width and Height.
        :return: (w, h)
        """
        w, h = pyautogui.size()
        return w, h

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value):
        self._move(value, self.y)

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, value):
        self._move(self.x, value)

    @property
    def position(self):
        return pyautogui.mouseinfo.position()

    @position.setter
    def position(self, value: Tuple[Union[int, float], Union[int, float]]):
        assert cj.is_numeric_sequence(value) and len(value) == 2, f"Value {value} isn't valid"
        self._move(*value)

    def _move(self, x, y):
        if self._reduce_noise:
            distance = cj.mathtools.distance_between_points((x, y), self._last_position or self.position)
            if distance <= self._noise_threshold:
                return
        self._last_position = self.position
        pyautogui.moveTo(x, y)

    def _get_jump(self, jump):
        if not isinstance(jump, (int, float)):
            return self._jump
        return int(jump)

    def up(self, jump: int = None):
        self.y = self.y - self._get_jump(jump)

    def down(self, jump: int = None):
        self.y = self.y + self._get_jump(jump)

    def right(self, jump: int = None):
        self.x = self.x + self._get_jump(jump)

    def left(self, jump: int = None):
        self.x = self.x - self._get_jump(jump)

    def bottom_left(self, jump: int = None):
        y = self.y + self._get_jump(jump)
        x = self.x - self._get_jump(jump)
        self.position = (x, y)

    def bottom_right(self, jump: int = None):
        y = self.y + self._get_jump(jump)
        x = self.x + self._get_jump(jump)
        self.position = (x, y)

    def top_left(self, jump: int = None):
        y = self.y - self._get_jump(jump)
        x = self.x - self._get_jump(jump)
        self.position = (x, y)

    def top_right(self, jump: int = None):
        y = self.y - self._get_jump(jump)
        x = self.x + self._get_jump(jump)
        self.position = (x, y)

    def center(self):
        w, h = self.window_size
        self.position = (w // 2, h // 2)

    def corner_bottom_left(self):
        w, h = self.window_size
        self.position = 0, h - 1

    def corner_bottom_right(self):
        w, h = self.window_size
        self.position = w - 1, h - 1

    def corner_top_right(self):
        w, h = self.window_size
        self.position = w - 1, 0

    def corner_top_left(self):
        self.position = 1, 1

    def _click(self, button: str, n_clicks: int, **kwargs):
        pyautogui.click(*self.position, button=button, clicks=n_clicks, **kwargs)

    def click_left(self, n_clicks=1, **kwargs):
        self._click('left', n_clicks=n_clicks, **kwargs)

    def click_right(self, n_clicks=1, **kwargs):
        self._click('right', n_clicks=n_clicks, **kwargs)
