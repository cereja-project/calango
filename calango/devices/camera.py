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
import cv2

__all__ = ['Capture']


class Capture:
    def __init__(self, *args, take_rgb=False, flip=False, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._cap = self._cv2_cap()
        self._take_rgb = take_rgb
        self._flip = flip

    def _cv2_cap(self):
        return cv2.VideoCapture(*self._args)

    @property
    def cap(self):
        if self.stopped:
            self._cap.release()
            self._cap = self._cv2_cap()
        return self._cap

    @property
    def frame(self):
        while True:
            success, image = self.cap.read()
            if not success:
                continue
            if self._flip:
                image = cv2.flip(image, 1)
            if self._take_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

    @property
    def stopped(self):
        return not self._cap.isOpened()

    def stop(self):
        self._cap.release()

    def __next__(self):
        return self.frame

    def __iter__(self):
        for i in self.frame:
            yield i
