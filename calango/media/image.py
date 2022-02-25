from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Union, Tuple

import cv2
import cereja as cj
from matplotlib import pyplot as plt
import numpy as np

__all__ = ['Image']

try:
    # noinspection PyUnresolvedReferences
    IPYTHON = get_ipython()
    ON_COLAB_JUPYTER = True if "google.colab" in IPYTHON.__str__() else False
except NameError:
    IPYTHON = None
    ON_COLAB_JUPYTER = False


class _InterfaceImage(ABC):

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
    def data(self):
        pass


class Image(_InterfaceImage):

    def __init__(self, image_or_path: Union[str, np.ndarray], channels='BGR'):
        self._path = None
        assert isinstance(channels, str), f'channels {channels} is not valid.'
        if isinstance(image_or_path, str):
            self._path = cj.Path(image_or_path)
            assert self._path.exists, FileNotFoundError(f'Image {self._path.path} not found.')
            self._data = cv2.imread(self._path.path)
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
    def shape(self):
        return self.data.shape

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def center(self) -> Tuple[int, int]:
        return self.width // 2, self.height // 2

    @property
    def min_len(self):
        return min(self._get_h_w(self.shape))

    @classmethod
    def _get_h_w(cls, shape):
        assert len(shape) >= 2, f'Image shape invalid. {shape}'
        return shape[:2]

    @classmethod
    def size_proportional(cls, original_shape, new_shape) -> Tuple[int, int]:
        o_h, o_w = cls._get_h_w(original_shape)
        n_h, n_w = cls._get_h_w(new_shape)
        if o_h < o_w:
            return n_w, math.floor(o_h / (o_w / n_w))
        return math.floor(o_w / (o_h / n_h)), n_h

    def bgr_to_rgb(self):
        if self._channels == 'BGR':
            self._data = cv2.cvtColor(self._data, cv2.COLOR_BGR2RGB)
            self._channels = 'RGB'
        return self

    def rgb_to_bgr(self):
        if self._channels == 'RGB':
            self._data = cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR)
            self._channels = 'BGR'
        return self

    def resize(self, shape: Union[tuple, list], keep_scale: bool = False) -> Image:
        """
        :param shape: is a tuple with HxW
        :param keep_scale: default is False, if True returns the image with the desired size in one of
               the axes, however the shape may be different as it maintains the proportion
        """
        shape = self.size_proportional(self.shape, shape) if keep_scale else self._get_h_w(shape)
        self._data = cv2.resize(self.data, shape)
        return self

    def rotate(self, degrees=90):
        assert degrees in (90, 180, -90), ValueError('send integer 90, -90 or 180')
        self._data = cv2.rotate(self.data, {90:  cv2.ROTATE_90_CLOCKWISE,
                                            -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                            180: cv2.ROTATE_180
                                            }.get(degrees))
        return self

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

    def prune(self):
        min_len = self.min_len
        return self.crop_by_center((min_len, min_len))

    def show(self):
        if ON_COLAB_JUPYTER:
            from google.colab.patches import cv2_imshow
            cv2_imshow(self.data)
        else:
            if self._channels == 'BGR':
                plt.imshow(cv2.cvtColor(self._data, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(self.data)
            plt.show()

    def save(self, p: str):
        cv2.imwrite(p, self.data)
        assert cj.Path(p).exists, f'Error saving image {p}'

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
