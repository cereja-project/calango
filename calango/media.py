from __future__ import annotations

import io
import math
import subprocess
import threading
import time
from abc import abstractmethod
from typing import Union, Tuple, Sequence, Iterator
from urllib.parse import urlparse

import cereja as cj
import cv2
import numpy as np
import logging
from matplotlib import pyplot as plt

from .devices import Mouse
from .settings import ON_COLAB_JUPYTER

__all__ = ['Image', 'Video', 'VideoWriter']

from .utils import show_local_mp4


def is_url(val):
    try:
        result = urlparse(val)
        return all([result.scheme, result.netloc])
    except:
        return False


class Image(np.ndarray):
    _GRAY_SCALE = 'GRAY_SCALE'
    _color_mode = 'BGR'

    def __new__(cls, im: Union[str, np.ndarray], color_mode: str = 'BGR', **kwargs) -> 'Image':
        if im is None:
            data = np.zeros(kwargs.get('shape', (480, 640)), dtype=np.uint8)
        else:
            assert isinstance(color_mode, str), f'channels {color_mode} is not valid.'
            if isinstance(im, str):
                if is_url(im):
                    with cj.TempDir() as dir_path:
                        req = cj.request.get(im)
                        file_path = dir_path.path.join(req.content_type.replace('/', '.'))
                        cj.FileIO.create(file_path, req.data).save()
                        data = cv2.imread(file_path.path)
                else:
                    p = cj.Path(im)
                    assert p.exists, FileNotFoundError(f'Image {p.path} not found.')
                    data = cv2.imread(p.path)
                if data is None:
                    raise ValueError('The image is not valid.')
                color_mode = 'BGR'
            elif isinstance(im, plt.Figure):
                io_buf = io.BytesIO()
                im.savefig(io_buf, format='raw')
                io_buf.seek(0)
                data = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                  newshape=(int(im.bbox.bounds[3]), int(im.bbox.bounds[2]), -1))
            elif isinstance(im, np.ndarray):
                data = im.copy()
            else:
                raise TypeError(f'{type(im)} is not valid.')

        if data.shape[-1] == 1 or len(data.shape) == 2:
            color_mode = cls._GRAY_SCALE
        elif data.shape[-1] == 4:
            color_mode = 'BGRA'

        color_mode = color_mode.upper()
        assert color_mode in {'BGR', 'RGB', 'BGRA', 'GRAY_SCALE'}, f'channels {color_mode} is not valid.'
        obj = super().__new__(cls, data.shape, dtype=data.dtype, buffer=data)
        obj._color_mode = color_mode
        return obj

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Image({self.shape}, {self._color_mode})'

    def _get_channel_data(self, c):
        if self._color_mode == self._GRAY_SCALE:
            raise ValueError('The image is Gray Scale')
        assert isinstance(c, str), 'send str R, G or B for channel.'
        return self[:, :, self._color_mode.index(c.upper())]

    def get_lower_scale(self, scale: Union[int, float]):
        return self.height // scale, self.width // scale

    def get_high_scale(self, scale: Union[int, float]):
        return int(self.height * scale), int(self.width * scale)

    def set_border(self, size: int = 1, color: Tuple[int, int, int] = (0, 255, 0)):
        cv2.rectangle(self, (size, size), (self.width - 1, self.height - 1), color, size)

    @property
    def mask(self):
        return Image(np.zeros(self.shape[:2], dtype=np.uint8))

    @property
    def top(self):
        return self[:self.height // 2, :]

    @top.setter
    def top(self, value):
        value: Image = Image(value)[:, :, :self.shape[-1]]
        value = value.resize(self.top.shape[:2], keep_scale=False)
        self[:self.height // 2, :] = value

    @property
    def left(self):
        return self[:, :self.width // 2]

    @left.setter
    def left(self, value):
        value: Image = Image(value)[:, :, :self.shape[-1]]
        value = value.resize(self.left.shape[:2], keep_scale=False)
        self[:, :self.width // 2] = value

    @property
    def right(self):
        return self[:, self.width // 2:]

    @right.setter
    def right(self, value):
        value: Image = Image(value)[:, :, :self.shape[-1]]
        value = value.resize(self.right.shape[:2], keep_scale=False)
        self[:, self.width // 2:] = value

    @property
    def center(self) -> Image:
        y, x = self.get_lower_scale(3)
        return self[y:y * 2, :]

    @center.setter
    def center(self, value):
        y, x = self.get_lower_scale(3)
        value: Image = Image(value)[:, :, :self.shape[-1]]
        value = value.resize(self.center.shape[:2], keep_scale=False)
        self[y:y * 2, :] = value

    @property
    def center_crop(self):
        y, x = self.height // 2, self.width // 2
        k = min(y, x)
        return self[y - k // 2:y + k // 2, x - k // 2:x + k // 2]

    @center_crop.setter
    def center_crop(self, value):
        y, x = self.height // 2, self.width // 2
        k = min(y, x)
        value: Image = Image(value)[:, :, :self.shape[-1]]
        value = value.resize(self.center_crop.shape[:2], keep_scale=False)
        self[y - k // 2:y + k // 2, x - k // 2:x + k // 2] = value

    @property
    def bottom(self):
        return self[self.height // 2:, :]

    @bottom.setter
    def bottom(self, value):
        value: Image = Image(value)
        value = value.resize(self.bottom.shape[:2], keep_scale=False)
        self[self.height // 2:, :] = value

    @property
    def zoom_in(self):
        y, x = self.height // 2, self.width // 2
        dif_h = abs(y - self.height) // 2
        dif_w = abs(x - self.width) // 2
        return self[dif_h:self.height - dif_h, dif_w:self.width - dif_w]

    @property
    def zoom_out(self):
        y, x = self.height // 2, self.width // 2
        dif_h = abs(y - self.height) // 2
        dif_w = abs(x - self.width) // 2
        new_img = self.resize((self.height - dif_h, self.width - dif_w), keep_scale=False)
        self[:, :, :] *= 0
        self[dif_h // 2:self.height - dif_h // 2, dif_w // 2:self.width - dif_w // 2] = new_img
        return self

    def flip(self, axis=1) -> Union[np.ndarray, 'Image']:
        return np.flip(self, axis=axis)

    @property
    def r(self) -> np.ndarray:
        return self._get_channel_data('R')

    @property
    def g(self) -> np.ndarray:
        return self._get_channel_data('G')

    @property
    def b(self) -> np.ndarray:
        return self._get_channel_data('B')

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def pyramid_level(self) -> int:
        if self.width < self.height:
            return int(np.log2(self.width))
        return int(np.log2(self.height))

    @property
    def center_position(self) -> Tuple[int, int]:
        return int(self.width // 2), int(self.height // 2)

    @property
    def min_len(self) -> int:
        return min(self._get_h_w(self.shape))

    @property
    def max_len(self) -> int:
        return max(self._get_h_w(self.shape))

    @classmethod
    def _get_h_w(cls, shape):
        assert len(shape) >= 2, f'Image shape invalid. {shape}'
        return shape[:2]

    @classmethod
    def size_proportional(cls, original_shape, new_shape) -> Tuple[int, int]:
        o_h, o_w = cls._get_h_w(original_shape)
        n_h, n_w = cls._get_h_w(new_shape)
        if o_h < o_w:
            return math.floor(o_h / (o_w / n_w)), n_w
        return n_h, math.floor(o_w / (o_h / n_h))

    @classmethod
    def get_empty_image(cls, shape=(480, 640, 3), color=(0, 0, 0)) -> 'Image':
        return cls((np.ones(shape, dtype=np.uint8) * color).astype(np.uint8))

    def bgr_to_rgb(self) -> Image:
        if self._color_mode == 'BGR':
            self._color_mode = 'RGB'
            self[:, ] = cv2.cvtColor(self, cv2.COLOR_BGR2RGB)
        return self

    @property
    def gray_scale(self) -> Image:
        code_ = {'BGR':  cv2.COLOR_BGR2GRAY,
                 'RGB':  cv2.COLOR_RGB2GRAY,
                 'RGBA': cv2.COLOR_RGBA2GRAY,
                 'BGRA': cv2.COLOR_BGRA2GRAY}.get(self._color_mode)
        if code_ is not None:
            return Image(cv2.cvtColor(self, code_))
        return self

    def rgb_to_bgr(self) -> Image:
        if self._color_mode == 'RGB':
            self._color_mode = 'BGR'
            self[:, ] = cv2.cvtColor(self, cv2.COLOR_RGB2BGR)
        elif self._color_mode == 'RGBA':
            self._color_mode = 'BGR'
            self[:, ] = cv2.cvtColor(self, cv2.COLOR_RGBA2BGR)
        elif self._color_mode == 'BGRA':
            self._color_mode = 'BGR'
            self[:, ] = cv2.cvtColor(self, cv2.COLOR_BGRA2BGR)
        return self

    def resize(self, shape: Union[tuple, list], keep_scale: bool = False) -> Image:
        """
        :param shape: is a tuple with HxW
        :param keep_scale: default is False, if True returns the image with the desired size in one of
               the axes, however the shape may be different as it maintains the proportion
        """
        shape = self.size_proportional(self.shape, shape) if keep_scale else self._get_h_w(shape)
        return Image(cv2.resize(self, shape[::-1]))

    def rotate(self, degrees=90):
        assert degrees in (90, 180, -90), ValueError('send integer 90, -90 or 180')
        return cv2.rotate(self, {90:  cv2.ROTATE_90_CLOCKWISE,
                                 -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                 180: cv2.ROTATE_180
                                 }.get(degrees))

    def crop_by_center(self, size=None, keep_scale=False) -> Image:
        assert size is None or isinstance(size, (list, tuple)) and cj.is_numeric_sequence(size) and len(
                size) == 2, 'Send HxW image cropped output'
        if size is None:
            size = self.min_len, self.min_len
        new_h, new_w = size
        if keep_scale:
            new_h, new_w = self.size_proportional(self.shape, (new_h, new_w))
        assert self.width >= new_w and self.height >= new_h, f'This is impossible because the image {self.shape} ' \
                                                             f'has proportions smaller than the size sent {size}'
        im_crop = self.copy()
        bottom = (self.center_position[0], self.center_position[1] + new_h // 2)
        top = (self.center_position[0], self.center_position[1] - new_h // 2)
        left = (self.center_position[0] - new_w // 2, self.center_position[1])
        right = (self.center_position[0] + new_w // 2, self.center_position[1])

        start = left[0], top[-1]

        end = right[0], bottom[-1]

        return Image(cv2.resize(im_crop[start[1]:end[1], start[0]:end[0]], (new_w, new_h)))

    def prune(self) -> Image:
        min_len = self.min_len
        return Image(self.crop_by_center((min_len, min_len)))

    def circle(self, radius=None, position=None, color=(0, 0, 0), thickness=1):
        position = position or self.center_position
        radius = radius or self.min_len // 2
        cv2.circle(self, position, radius, color, thickness)
        return self

    def get_mask_circle(self, radius=None, position=None):
        return self.mask.circle(radius=radius, position=position, color=(255, 255, 255), thickness=-1)

    def crop_circle(self, radius=None, position=None, inverse=False):
        mask = self.get_mask_circle(radius=radius, position=position)
        return Image(cv2.bitwise_and(self, self, mask=mask if not inverse else ~mask))

    def add_circle_from_image(self, image: Image, radius=None, position=None):
        if not isinstance(image, Image):
            image = Image(image)
        assert image.shape == self.shape, f'Image shape must be the same as the image to be added {image.shape}'
        cropped_image = image.crop_circle(radius=radius, position=position)
        cropped_self = self.crop_circle(radius=radius, position=position, inverse=True)
        return Image(cv2.add(cropped_self, cropped_image))

    def plot(self):
        if ON_COLAB_JUPYTER:
            # noinspection PyUnresolvedReferences
            from google.colab.patches import cv2_imshow
            cv2_imshow(self)
        else:
            if self._color_mode == 'BGR':
                plt.imshow(cv2.cvtColor(self, cv2.COLOR_BGR2RGB))
            elif self._color_mode == self._GRAY_SCALE:
                plt.imshow(self, cmap='gray')
            else:
                plt.imshow(self)
            plt.show()

    def save(self, p: str):
        cv2.imwrite(p, self)
        assert cj.Path(p).exists, f'Error saving image {p}'

    def plot_colors_histogram(self):
        # tuple to select colors of each channel line
        colors = ("red", "green", "blue") if self._color_mode == 'RGB' else ('blue', 'green', 'red')
        channel_ids = (0, 1, 2)

        # create the histogram plot, with three lines, one for
        # each color
        plt.figure()
        plt.xlim([0, 256])
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                    self[:, :, channel_id], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=c)

        plt.title("Color Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")

        plt.show()

    def write_text(self, text,
                   pos='left_bottom',
                   font=cv2.FONT_HERSHEY_PLAIN,
                   font_scale=1,
                   font_thickness=1,
                   text_color=(0, 255, 0),
                   text_color_bg=(0, 0, 0)
                   ) -> Image:
        """
        Write text on this Image and returns self.

        :param text: a string
        :param pos: region on write text in the image
        :param font: cv2 font index. cv2.FONT_HERSHEY_PLAIN is default.
        :param font_scale: size of font
        :param font_thickness: thickness of font
        :param text_color: BGR color
        :param text_color_bg: BGR color
        :return: Image
        """
        return self.draw_text(text, pos, font, font_scale, font_thickness, text_color, text_color_bg)

    def draw_text(self, text,
                  pos='left_bottom',
                  font=cv2.FONT_HERSHEY_PLAIN,
                  font_scale=1,
                  font_thickness=1,
                  text_color=(0, 255, 0),
                  text_color_bg=(0, 0, 0)
                  ) -> Image:
        """
        Write text on this Image and returns self.

        :param text: a string
        :param pos: region on write text in the image
        :param font: cv2 font index. cv2.FONT_HERSHEY_PLAIN is default.
        :param font_scale: size of font
        :param font_thickness: thickness of font
        :param text_color: BGR color
        :param text_color_bg: BGR color
        :return: Image
        """

        text = str(text)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        if text_size[0] > self.width:
            font_scale = min(text_size[0] / (self.width - (self.width * 0.2)), font_scale)

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_size = text_size[0], int(text_size[1] * 1.5)
        k = int(self.min_len * 0.03)

        w_top_r_limit = (self.width - text_size[0]) - k
        w_center_limit = int((self.width - text_size[0]) // 2) + k
        h_bottom_limit = (self.height - text_size[1]) - k
        center = w_center_limit, int((self.height - text_size[-1]) // 2)

        pos = {'left_top':      (k, k),
               'left_bottom':   (k, h_bottom_limit),
               'right_top':     (w_top_r_limit, k),
               'right_bottom':  (w_top_r_limit, h_bottom_limit),
               'center_top':    (w_center_limit, k),
               'center_bottom': (w_center_limit, h_bottom_limit),
               'center':        center
               }.get(pos, center)

        x, y = pos

        text_w, text_h = text_size
        cv2.rectangle(self, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(self, text, (x, y + int(text_h * 0.9)), font, font_scale, text_color, font_thickness)

        return self

    def gaussian_pyramid(self, levels=3, *args, **kwargs):

        img = self.copy()
        pyramid = [img]
        while levels >= 1:
            img = cv2.pyrDown(img, *args, **kwargs)
            pyramid.append(img)
            levels -= 1
        return pyramid

    def laplacian_pyramid(self, levels=3) -> Sequence[Image]:
        gaussian_pyramid = self.gaussian_pyramid(levels)
        pyramid = []
        for i in range(levels, 0, -1):
            GE = cv2.pyrUp(gaussian_pyramid[i])
            GE = cv2.resize(GE, gaussian_pyramid[i - 1].shape[::-1][1:])
            L = Image(cv2.subtract(gaussian_pyramid[i - 1], GE))
            pyramid.append(L)
        return pyramid

    @classmethod
    def images_from_dir(cls, dir_path, ext='.jpg'):
        dir_path = cj.Path(dir_path)
        return [cls(im_p.path) for im_p in dir_path.list_files(ext=ext)]


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
            self.__writer = cv2.VideoWriter(self._path, self._fourcc, self._fps, (self._width, self._height))
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


class _IVideo:

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def next_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        pass

    @property
    @abstractmethod
    def total_frames(self) -> int:
        pass

    @property
    @abstractmethod
    def fps(self) -> Union[int, float]:
        pass

    @property
    def name(self) -> str:
        return f'Video({str(id(self))})'

    @property
    @abstractmethod
    def is_webcam(self) -> bool:
        pass

    @property
    def is_stream(self) -> bool:
        pass

    @abstractmethod
    def set_fps(self, fps: Union[int, float]) -> None:
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def stop(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        try:
            cv2.destroyWindow(self.name)
        except:
            cv2.destroyAllWindows()


class _VideoCV2(cv2.VideoCapture, _IVideo):

    def __init__(self, *args, fps=None, **kwargs):
        self._is_webcam = not bool(args and isinstance(args[0], str))
        self._is_stream = cj.request.is_url(args[0]) if not self._is_webcam else False
        args = (*args, cv2.CAP_DSHOW) if self._is_webcam else args
        super().__init__(*args, **kwargs)
        if fps is not None:
            self.set(cv2.CAP_PROP_FPS, fps)
        elif self.get(cv2.CAP_PROP_FPS) == 0:
            self.set(cv2.CAP_PROP_FPS, 30)
        self._fps = self.get(cv2.CAP_PROP_FPS)
        self._total_frames = -1 if self._is_webcam else int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width, self._height = int(self.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def fps(self):
        return self._fps

    def set_fps(self, fps):
        assert isinstance(fps, (int, float)), ValueError(f'{fps} value for fps is not valid. Send int or float.')
        self.set(cv2.CAP_PROP_FPS, fps)
        self._fps = fps

    @property
    def next_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        return self.read()

    @property
    def is_webcam(self):
        return self._is_webcam

    @property
    def is_stream(self):
        return self._is_stream

    @property
    def is_opened(self) -> bool:
        return self.isOpened()

    def stop(self):
        self.release()


class _FrameSequence(_IVideo):
    def __init__(self, frames, fps=30):
        self._fps = fps
        if isinstance(frames, (list, np.ndarray)):
            self._total_frames = len(frames)
            frames = iter(frames)
        elif isinstance(frames, Iterator):
            self._total_frames = -1
        else:
            raise TypeError('Send a frame sequence.')
        self._first_frame = next(frames)
        assert isinstance(self._first_frame, np.ndarray) and len(
                self._first_frame.shape) == 3, f'Frame format {self._first_frame.shape} is invalid'

        self._height, self._width = self._first_frame.shape[:2]
        self._frames = frames
        self._is_opened = True

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_webcam(self):
        return False

    @property
    def is_stream(self) -> bool:
        return False

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def next_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        if self._first_frame is not None:
            res = self._first_frame
            self._first_frame = None
            return True, res
        try:
            return True, next(self._frames)
        except StopIteration:
            self._is_opened = False
            return False, None

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def fps(self):
        return self._fps

    def set_fps(self, fps):
        assert isinstance(fps, (int, float)), ValueError(f'{fps} value for fps is not valid. Send int or float.')
        self._fps = fps

    @classmethod
    def load_from_dir(cls, p):
        paths = cj.Path(p).list_dir()
        obj = cls(cls._read_images_paths(paths))
        obj._total_frames = len(paths)
        return obj

    @classmethod
    def load_from_paths(cls, paths):
        obj = cls(cls._read_images_paths(paths))
        obj._total_frames = len(paths)
        return obj

    @classmethod
    def _read_images_paths(cls, paths):
        for fp in paths:
            if fp.is_file:
                img = cv2.imread(fp.path)
                if img is not None:
                    yield img

    def stop(self):
        self._frames = None
        self._is_opened = False


class Screen(_IVideo):
    def __init__(self, *args, fps=30, **kwargs):
        mouse = Mouse(*args, **kwargs)

        self._width, self._height = mouse.window_size
        self._mon = {'left': 0, 'top': 0, 'width': self._width, 'height': self._height}
        self._capture = True
        self._frames = self.__get_frames()
        self._fps = fps

    def set_fps(self, fps: Union[int, float]) -> None:
        assert isinstance(fps, (int, float)), ValueError(f'{fps} value for fps is not valid. Send int or float.')
        self._fps = fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def next_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        return True, next(self._frames)

    def __get_frames(self):
        from mss import mss
        with mss() as sct:
            while self._capture:
                yield Image(np.array(sct.grab(self._mon)), 'RGB').rgb_to_bgr().bgr_to_rgb().data

    @property
    def total_frames(self) -> int:
        return -1

    @property
    def fps(self) -> Union[int, float]:
        return self._fps

    @property
    def is_webcam(self) -> bool:
        return True

    @property
    def is_stream(self) -> bool:
        return False

    @property
    def is_opened(self) -> bool:
        return self._capture

    def stop(self):
        self._capture = False


class Video:

    def __init__(self, *args, fps=None, **kwargs):
        kwargs['fps'] = fps
        self._speed = 1
        self._args = args
        self._kwargs = kwargs
        self._build()
        self._th_show = None
        self._t0 = None
        self._fps_time = None
        self._count_frames = 0

    def _build(self):
        if len(self._args):
            if isinstance(self._args[0], str):
                if self._args[0] == 'monitor':
                    self._cap = Screen()
                elif cj.request.is_url(self._args[0]):
                    self._cap = _VideoCV2(*self._args, **self._kwargs)
                else:
                    path_ = cj.Path(self._args[0])
                    assert path_.exists, FileNotFoundError(f'{path_.path}')
                    if path_.is_dir:
                        self._cap = _FrameSequence.load_from_dir(path_)
                    else:
                        self._cap = _VideoCV2(*self._args, **self._kwargs)
            elif isinstance(self._args[0], int):
                self._cap = _VideoCV2(*self._args, **self._kwargs)
            elif isinstance(self._args[0], (list, np.ndarray, Iterator)):
                if len(self._args[0]) and isinstance(self._args[0][0], str):
                    self._cap = _FrameSequence.load_from_paths(self._args[0])
                if isinstance(self._args[0], (list, np.ndarray, Iterator)):
                    self._cap = _FrameSequence(self._args[0])
            else:
                raise ValueError('Error on build Video. Arguments is invalid.')
        else:
            self._cap = _VideoCV2(*self._args, **self._kwargs)

        assert hasattr(self, '_cap'), NotImplementedError(
                f'Internal error. Please open new issue on https://github.com/cereja-project/calango')
        self._current_number_frame = 0

        self._th_show_running = False
        self._last_frame = None

    @property
    def current_number_frame(self):
        return self._current_number_frame

    @property
    def width(self):
        return self._cap.width

    @property
    def height(self):
        return self._cap.height

    @property
    def total_frames(self):
        return self._cap.total_frames if not self._cap.is_webcam else self._current_number_frame + 1

    def __get_next_frame(self) -> Union[np.ndarray, Image, None]:
        if self._t0 is None:
            self._t0 = time.time()
            self._fps_time = self._t0  # for fps on show
        _, image = self._cap.next_frame
        image = Image(image)
        self._current_number_frame += 1
        self._count_frames += 1  # for calculate fps correctly
        if self.current_number_frame > self.total_frames:
            if isinstance(self._cap, cv2.VideoCapture):
                self._cap.release()
            return None
        self._last_frame = image
        return image

    @property
    def next_frame(self):
        if self._th_show_running:
            return self._last_frame
        return self.__get_next_frame()

    @property
    def is_opened(self):
        return self._cap.is_opened

    def get_batch_frames(self, kernel_size, strides=1, take_number_frame=False):
        batch_frames = []
        while self.is_opened:
            frame = self.next_frame
            if frame is None:
                continue
            batch_frames.append(frame if not take_number_frame else [self.current_number_frame, frame])
            if len(batch_frames) == kernel_size:
                yield batch_frames
                batch_frames = batch_frames[strides:]

    @property
    def fps(self):
        if self._th_show_running and not self._cap.is_webcam:
            time_it = (time.time() - self._fps_time)
            if time_it >= 1:
                return self._count_frames / time_it

        return self._cap.fps * self._speed

    @property
    def is_break_view(self) -> bool:
        if not self._cap.is_webcam:
            # need to take fps in video view
            time_it = (time.time() - self._fps_time)

            wait_msec = (self._cap.fps * self._speed) + int(
                    abs(self._count_frames - time_it * (self._cap.fps * self._speed)))
        else:
            wait_msec = self._cap.fps * self._speed
        time.sleep(1 / wait_msec)
        k = cv2.waitKey(1)
        return k == ord('q') or k == ord('\x1b')

    @property
    def video_info(self):
        return f"Size: {self._size_info} - FPS: {self._fps_info} - Frames: {self._frames_info} T: {self._time_info} - Speed: {self._speed_info}"

    @property
    def _size_info(self):
        return f'{self.width}x{self.height}'

    @property
    def _fps_info(self):
        return f'{cj.get_zero_mask(self.fps, max_len=3)}'

    @property
    def _time_info(self):
        return f'{round(time.time() - self._t0, 1)} s'

    @property
    def _speed_info(self):
        return f'{self._speed}x'

    @property
    def _frames_info(self):
        return f'{cj.get_zero_mask(self.current_number_frame, max_len=len(str(self.total_frames)))}/{self.total_frames}'

    def _show(self):
        if ON_COLAB_JUPYTER:
            raise NotImplementedError("Not implemented show video on colab")
        self._th_show_running = True
        try:
            for image in self.get_frames():
                cv2.imshow(self._cap.name, image.draw_text(self.video_info))
                if self.is_break_view:
                    self._cap.stop()
        except Exception as e:
            logging.error(e)
            self._cap.stop()
        self._th_show_running = False

    def get_frames(self, n_frames=None):
        assert self._th_show_running, "The video is showing, so you can't get frames"
        if not self.is_opened:
            self._build()

        while self.is_opened:
            frame = self.__get_next_frame()
            if frame is None:
                continue
            yield frame
            if n_frames is not None:
                n_frames -= 1
            if n_frames == 0:
                self._cap.stop()
                break

    def save(self, file_path, n_frames=None):
        VideoWriter.write_frames(file_path, self.get_frames(n_frames=n_frames), fps=self._cap.fps)

    def show(self):
        if ON_COLAB_JUPYTER:
            with cj.system.TempDir() as dir_path:
                video_path = dir_path.path.join(f'{self._cap.name}.mp4')
                self.save_frames(dir_path.path)

                subprocess.run(
                        f'ffmpeg -f image2 -i "{dir_path.path}"/%0{max(len(str(self.total_frames)), 3)}d.jpg -y "{video_path.path}" -hide_banner -loglevel panic',
                        shell=True,
                        stdout=subprocess.PIPE,
                ).check_returncode()
                if video_path.exists:
                    return show_local_mp4(video_path.path)
                raise Exception("Error on show video.")
        else:
            if self._th_show_running:
                return
            if self._th_show is not None:
                self._th_show.join()
            self._th_show = threading.Thread(target=self._show, daemon=True)
            self._th_show.start()

    def save_frames(self, p: str, start=1, end=None, step=1, img_format='jpg', limit_web_cam=500):
        _frame_count = (self.total_frames if not self._cap.is_webcam else limit_web_cam)
        filter_map = set(range(start, end or _frame_count, step))
        p = cj.Path(p)
        size_number = len(str(_frame_count))
        max_frame = max(filter_map)
        for frame in self.get_frames():
            if not (self.current_number_frame < max_frame):
                break
            prefix = cj.get_zero_mask(self.current_number_frame, size_number)
            if self.current_number_frame in filter_map:
                cv2.imwrite(p.join(f'{prefix}.{img_format}').path, frame)

    @property
    def duration(self):
        return self.total_frames / self._cap.fps

    def set_speed(self, speed=1):
        self._fps_time = time.time()
        self._speed = speed
        self._count_frames = 0
