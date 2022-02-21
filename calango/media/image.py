from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Union, Tuple

import cv2
import cereja as cj
from matplotlib import pyplot as plt
import numpy as np

__all__ = ['Image']


class _InterfaceImage(ABC):

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @abstractmethod
    def read(self, p, flags):
        pass

    @property
    @abstractmethod
    def data(self):
        pass


class Image(_InterfaceImage):

    def __init__(self, image_or_path: Union[str, np.ndarray], channels='BGR', **kwargs):
        self._path = None
        assert isinstance(channels, str), f'channels {channels} is not valid.'
        if isinstance(image_or_path, str):
            self._path = cj.Path(image_or_path)
            self._data = self.read(self._path.path, **kwargs)
            self._channels = 'BGR'
        else:
            assert isinstance(image_or_path, np.ndarray) and len(image_or_path.shape) == 3, 'Image format is invalid'
            self._data = image_or_path
            self._channels = channels.upper()

    def _get_channel_data(self, c):
        assert isinstance(c, str), 'send str R, G or B for channel.'
        return self.data[:, :, self._channels.index(c.upper())]

    @property
    def data(self):
        return self._data

    def bgr_to_rgb(self):
        self._data = cv2.cvtColor(self._data, cv2.COLOR_BGR2RGB)
        self._channels = 'RGB'
        return self

    def rgb_to_bgr(self):
        self._data = cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR)
        self._channels = 'BGR'
        return self

    def flip(self):
        self._data = cv2.flip(self._data, 1)
        return self

    @property
    def r(self):
        return self._get_channel_data('R')

    @property
    def g(self):
        return self._get_channel_data('G')

    @property
    def b(self):
        return self._get_channel_data('B')

    @property
    def width(self) -> int:
        return len(self.data)

    @property
    def height(self) -> int:
        return len(self.data)

    @classmethod
    def _get_h_w(cls, shape):
        assert len(shape) >= 2, f'Image shape invalid. {shape}'
        return shape[:2]

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @classmethod
    def size_proportional(cls, original_shape, new_shape) -> Tuple[int, int]:
        o_h, o_w = cls._get_h_w(original_shape)
        n_h, n_w = cls._get_h_w(new_shape)
        if o_h < o_w:
            return n_w, math.floor(o_h / (o_w / n_w))
        return math.floor(o_w / (o_h / n_h)), n_h

    def resize(self, shape: Union[tuple, list], keep_scale: bool = False) -> Image:
        """
        :param shape: is a tuple with HxW
        :param keep_scale: default is False, if True returns the image with the desired size in one of
               the axes, however the shape may be different as it maintains the proportion
        """
        shape = self.size_proportional(self.shape, shape) if keep_scale else self._get_h_w(shape)
        self._data = cv2.resize(self.data, shape)
        return self

    @property
    def center(self) -> Tuple[int, int]:
        return self.width // 2, self.height // 2

    def crop_by_center(self, size) -> Image:
        assert isinstance(size, (list, tuple)) and cj.is_numeric_sequence(size) and len(
                size) == 2, 'Send HxW image cropped output'

        new_h, new_w = size
        assert self.width >= new_w and self.height >= new_h, f'This is impossible because the image {self.shape} ' \
                                                             f'has proportions smaller than the size sent {size}'
        im_crop = self.data.copy()
        bottom = (self.center[0], self.center[1] + new_h // 2)
        top = (self.center[0], self.center[1] - new_h // 2)
        left = (self.center[0] - new_w // 2, self.center[1])
        right = (self.center[0] + new_w // 2, self.center[1])

        start = left[0], top[-1]

        end = right[0], bottom[-1]

        self._data = im_crop[start[1]:end[1], start[0]:end[0]]
        return self

    def prune_shape(self, image):
        h, w, _ = image.shape
        min_len = min((h, w))
        return self.crop_by_center((min_len, min_len))

    def show(self):
        cv2.imshow(f'HxW:{self.shape}', self.data)

    def save(self, p: str):
        cv2.imwrite(p, self.data)
        assert cj.Path(p).exists, f'Error saving image {p}'

    @classmethod
    def read(cls, p, flags=None):
        p = cj.Path(p)
        assert p.exists, FileNotFoundError(f'Image {p.path} not found.')
        return cv2.imread(p.path, flags)

    def plot_colors_histogram(self):
        # tuple to select colors of each channel line
        colors = ("red", "green", "blue") if self._channels == 'RGB' else ('blue', 'green', 'red')
        channel_ids = (0, 1, 2)

        # create the histogram plot, with three lines, one for
        # each color
        plt.figure()
        plt.xlim([0, 256])
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                    self.data[:, :, channel_id], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=c)

        plt.title("Color Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")

        plt.show()

    def gaussian_pyramid(self, levels=3, *args, **kwargs):

        img = self.data.copy()
        pyramid = [img]
        while levels >= 1:
            img = cv2.pyrDown(img, *args, **kwargs)
            pyramid.append(img)
            levels -= 1
        return pyramid

    def laplacian_pyramid(self, levels=3):
        gaussian_pyramid = self.gaussian_pyramid(levels)
        pyramid = []
        for i in range(levels, 0, -1):
            GE = cv2.pyrUp(gaussian_pyramid[i])
            L = cv2.subtract(gaussian_pyramid[i - 1], GE)
            pyramid.append(L)
        return pyramid
