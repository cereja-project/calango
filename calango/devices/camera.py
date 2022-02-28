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
import time

import cv2
import cereja as cj
from ..media import Image
import scipy.fftpack as fftpack
import numpy as np

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


class _VideoCap(cv2.VideoCapture):
    def __init__(self, *args, use_buffer=False, buffer_size=40, buffer_resize_to=(256, 256), **kwargs):
        super().__init__(*args, **kwargs)
        self._use_buffer = use_buffer
        self._buffer_size = buffer_size
        self._buffer_resize_to = buffer_resize_to
        self._buffer = []

    def read(self, image=None):
        success, image = super().read(image)
        if self._use_buffer:
            self._buffer.append(Image(image).crop_by_center(self._buffer_resize_to).data)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size:]

        return success, image

    @property
    def buffer(self):
        return self._buffer

    @property
    def has_buffer(self):
        return bool(self._buffer)

    @property
    def buffer_len(self):
        return len(self.buffer)

    def clear_buffer(self):
        self._buffer = []


class Capture:
    def __init__(self, *args, show=False, draw=False, frame_preprocess_func=None, use_buffer=False, buffer_size=40,
                 **kwargs):
        self._args = args or (0,)
        self._show = show
        self._draw = draw
        self._frame_preprocess_func = frame_preprocess_func
        self._is_file = bool(self._args and isinstance(self._args[0], str))
        self._kwargs = kwargs
        self._use_buffer = use_buffer
        self._buffer_size = buffer_size
        self._cap = self._cv2_cap()
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._is_file else 1
        self._width, self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self._current_frame = 0
        self._frames = self._generate_frames()

    def _cv2_cap(self):
        if self._is_file:
            return _VideoCap(*self._args, use_buffer=self._use_buffer, buffer_size=self._buffer_size, **self._kwargs)
        return _VideoCap(*self._args, cv2.CAP_DSHOW, use_buffer=self._use_buffer, buffer_size=self._buffer_size,
                         **self._kwargs)

    @property
    def draw_info(self):
        return f'FRAME: {self.current_frame} - FPS: {self.fps} ' \
               f'- TIME: {round(time.time() - self._start_time)} seconds'

    @property
    def is_break_view(self) -> bool:
        if self.is_file:
            k = cv2.waitKey(int(self.frame_count // self.fps))
        else:
            k = cv2.waitKey(1)
        return k == ord('q') or k == ord('\x1b')

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
        if self._is_file:
            return self._fps
        return int(round(self.current_frame / (time.time() - self._start_time)))

    @property
    def frame_count(self):
        return self._frame_count if self._is_file else self.current_frame + 1

    @property
    def is_file(self):
        return self._is_file

    @property
    def cap(self):
        if self.stopped:
            self._cap.release()
            self._cap = self._cv2_cap()
        return self._cap

    def _generate_frames(self):
        self._current_frame = 0
        self._start_time = time.time()
        cap = self.cap
        with _VideoView(self):
            while not self.stopped:
                success, image = cap.read()
                self._current_frame += 1
                if not success and self._current_frame > self._frame_count:
                    break
                if image is None:
                    continue
                if self._show:
                    cv2.imshow(f'Video',
                               Image(image.copy()).draw_text(
                                       self.draw_info).data if self._draw else image.copy())

                    if self.is_break_view:
                        break
                yield image

    @property
    def frames(self):
        if self.stopped:
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
        return not self._cap.isOpened()

    def stop(self):
        self._cap.release()
        cv2.destroyAllWindows()

    def save_frames(self, p: str, start=1, end=None, step=1, img_format='png'):
        _frame_count = (self.frame_count if self.is_file else 99999)
        filter_map = set(range(start, end or _frame_count, step))
        p = cj.Path(p)
        size_number = len(str(_frame_count))
        max_frame = max(filter_map)
        for frame in self:
            prefix = cj.get_zero_mask(self.current_frame, size_number)
            if self.current_frame in filter_map:
                cv2.imwrite(p.join(f'{prefix}.{img_format}').path, frame)
            if self.current_frame >= max_frame:
                break

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

    def __next__(self):
        return self.frame

    def __iter__(self):
        for frame in self.frames:
            yield frame

    def show(self):
        _show = self._show
        _draw = self._draw
        self._show = True
        self._draw = True
        for _ in self.frames:
            pass

        self._show = _show
        self._draw = _draw


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

    @property
    def frames(self):
        if self.stopped:
            self._frames = self._generate_frames()
        return self._frames

    def _generate_frames(self):
        self._current_frame = 0
        self._start_time = time.time()
        cap = self.cap
        data_buffer = []
        # times = []
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
                # times.append(time.time() - self._start_time)
                L = len(data_buffer)
                self._current_frame += 1
                if L < self.buffer_size:
                    yield image
                    continue
                if L > self.buffer_size:
                    data_buffer = data_buffer[-self.buffer_size:]

                if len(data_buffer) > self.buffer_size - 1:
                    yield cv2.convertScaleAbs(
                            self.magnify_color(data_buffer=np.array(data_buffer).astype('float'), fps=self.fps)).copy()
