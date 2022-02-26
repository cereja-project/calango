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
import time

import cv2
import cereja as cj
from ..media import Image
import scipy.fftpack as fftpack
import numpy as np
import platform

__all__ = ['Capture', 'VideoMagnify', 'VideoWriter']


class VideoWriter:
    def __init__(self, p, width=None, height=None, fps=30):
        self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._height, self._width = width, height
        self._path = p
        self.__writer = None
        self._fps = fps
        self._with_context = False

    def add_frame(self, frame):
        assert self._with_context, """Use with context eg.
    with VideoWriter('path/to/video.avi', fps=30) as video:
        video.write(frame)"""
        if self._width is None or self._height is None:
            self._height, self._width, _ = frame.shape
        self._writer.write(frame)

    @property
    def _writer(self):
        if self.__writer is None or not self.__writer.isOpened():
            self.__writer = cv2.VideoWriter(self._path, self._fourcc, self._fps, (self._width, self._height), self._fps)
        return self.__writer

    @classmethod
    def write_frames(cls, p, frames, fps=30, width=None, height=None):
        with cls(p, width=width, height=height, fps=fps) as video:
            for frame in frames:
                if isinstance(frame, (str, cj.Path)):
                    frame = cv2.imread(cj.Path(frame).path)
                video.add_frame(frame)

    def __enter__(self, *args, **kwargs):
        self._with_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._with_context = False
        self._writer.release()


class _VideoView:
    def __init__(self, cap: 'Capture'):
        self._cap = cap

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cap.stop()


class Capture:
    def __init__(self, *args, frame_preprocess_func=None, take_rgb=False, flip=False, **kwargs):
        self._args = args or (0,)
        self._frame_preprocess_func = frame_preprocess_func
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
        with _VideoView(self):
            while not self.stopped:
                success, image = cap.read()
                if not success:
                    continue
                if self._frame_preprocess_func:
                    image = self._frame_preprocess_func(image)
                image = Image(image)
                if self._flip:
                    image.flip()
                if self._take_rgb:
                    image.bgr_to_rgb()
                self._current_frame += 1
                yield image.data

    @property
    def frames(self):
        if self.stopped and not self.is_file:
            self._frames = self._generate_frames()
        return self._frames

    @property
    def frame(self):
        try:
            return next(self.frames)
        except StopIteration:
            return

    @property
    def stopped(self):
        return not (self._cap.isOpened() and (self._current_frame < self.frame_count or not self.is_file))

    def stop(self):
        self._cap.release()
        cv2.destroyAllWindows()

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
        return self.frame

    def __iter__(self):
        for frame in self.frames:
            yield frame

    def show(self):
        with _VideoView(self):
            while not self.stopped:
                frame = self.frame
                if frame is None:
                    break
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) == ord('q'):
                    break


class VideoMagnify(Capture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = 40

    def build_gaussian_pyramid(self, src, level=3):
        s = src.copy()
        pyramid = [s]
        for i in range(level):
            s = cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    def build_gaussian_pyramid_last(self, src, level=3):
        s = src.copy()
        while level >= 1:
            s = cv2.pyrDown(s)
            level -= 1
        return s

    def gaussian_video(self, video_tensor, levels=3):
        vid_data = []
        for frame in video_tensor:
            vid_data.append(self.build_gaussian_pyramid_last(frame, level=levels))
        return np.array(vid_data, dtype=video_tensor.dtype)

    @classmethod
    def temporal_ideal_filter(cls, tensor, low, high, fps, axis=0):
        fft = fftpack.fft(tensor, axis=axis)
        frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff = fftpack.ifft(fft, axis=axis)
        return np.abs(iff)

    @classmethod
    def amplify_video(cls, gaussian_vid, amplification=30):
        return gaussian_vid * amplification

    @classmethod
    def reconstract_frame(cls, amp_video, origin_video, levels=3):
        img = amp_video[-1].copy()
        while levels >= 1:
            img = cv2.pyrUp(img)
            levels -= 1
        img = img + origin_video[-1]
        return img

    def magnify_color(self, data_buffer, fps, low=0.4, high=2, levels=3, amplification=30):
        gau_video = self.gaussian_video(data_buffer, levels=levels)
        filtered_tensor = self.temporal_ideal_filter(gau_video, low, high, fps)
        amplified_video = self.amplify_video(filtered_tensor, amplification=amplification)
        return self.reconstract_frame(amplified_video, data_buffer, levels=levels)

    def _generate_frames(self):
        self._current_frame = 0
        cap = self.cap
        data_buffer = []
        times = []
        t0 = time.time()
        with _VideoView(self):
            while not self.stopped:
                success, image = cap.read()
                if not success:
                    continue
                if self._frame_preprocess_func:
                    image = self._frame_preprocess_func(image)
                image = Image(image)
                image = image.data
                data_buffer.append(image.copy())
                times.append(time.time() - t0)
                L = len(data_buffer)
                self._current_frame += 1
                if L < self.buffer_size:
                    yield image
                if L > self.buffer_size:
                    data_buffer = data_buffer[-self.buffer_size:]
                    times = times[-self.buffer_size:]

                if len(data_buffer) > self.buffer_size - 1:
                    self._fps = float(len(data_buffer)) / (times[-1] - times[0])
                    yield cv2.convertScaleAbs(
                            self.magnify_color(data_buffer=np.array(data_buffer).astype('float'), fps=self._fps)).copy()
