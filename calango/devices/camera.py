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
import sys

import cv2
import cereja as cj
from ..media import Image

__all__ = ['Capture']


class Capture:
    def __init__(self, *args, take_rgb=False, flip=False, **kwargs):
        self._args = args or (0,)
        self._is_file = bool(self._args and isinstance(self._args[0], str))
        self._kwargs = kwargs
        self._cap = self._cv2_cap()
        self._take_rgb = take_rgb
        self._flip = flip
        # FIX frame count for webcam
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._is_file else 999999
        self._width, self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self._current_frame = 0
        self._frames = self._generate_frames()

    def _cv2_cap(self):
        return cv2.VideoCapture(*self._args)

    @property
    def current_frame(self):
        return self._current_frame

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def is_file(self):
        return self._is_file

    @property
    def cap(self):
        if self.stopped and not self.is_file:
            self._cap.release()
            self._cap = self._cv2_cap()
        return self._cap

    def _generate_frames(self):
        self._current_frame = 0
        cap = self.cap
        while (cap.isOpened() or not self.stopped) and (self._current_frame < self.frame_count or not self.is_file):
            success, image = cap.read()
            if not success:
                continue
            image = Image(image)
            if self._flip:
                image.flip()
            if self._take_rgb:
                image.bgr_to_rgb()
            self._current_frame += 1
            yield image.data
        self.stop()

    @property
    def frames(self):
        if self.stopped and not self.is_file:
            self._frames = self._generate_frames()
        return self._frames

    @property
    def frame(self):
        return next(self.frames)

    @property
    def stopped(self):
        return not self._cap.isOpened()

    def stop(self):
        self._cap.release()

    def save_frames(self, p: str, start=1, end=None, step=1, img_format='png'):
        filter_map = set(range(start, end or self.frame_count, step))
        p = cj.Path(p)
        size_number = len(str(self.frame_count))
        max_frame = max(filter_map)
        for frame in self:
            prefix = cj.get_zero_mask(self.current_frame, size_number)
            if self.current_frame in filter_map:
                cv2.imwrite(p.join(f'{prefix}.{img_format}').path, frame)
            if self.current_frame >= max_frame:
                break

    def __next__(self):
        return next(self.__iter__())

    def __iter__(self):
        for frame in self.frames:
            yield frame
