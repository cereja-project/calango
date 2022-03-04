from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Union, Tuple, Sequence

import cv2
import cereja as cj
from matplotlib import pyplot as plt
import numpy as np

__all__ = ['Image', 'Video']

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
    _GRAY_SCALE = 'GRAY_SCALE'
    IMAGE_FORMATS = {'.jpg', '.jpeg', '.png'}

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
            self._data = image_or_path.copy()
            if self.shape[-1] == 1:
                self._channels = self._GRAY_SCALE
            else:
                self._channels = channels.upper()

    def _get_channel_data(self, c):
        if self._channels == self._GRAY_SCALE:
            raise ValueError('The image is Gray Scale')
        assert isinstance(c, str), 'send str R, G or B for channel.'
        return self.data[:, :, self._channels.index(c.upper())]

    @property
    def data(self) -> np.ndarray:
        return self._data

    def flip(self) -> Image:
        self._data = cv2.flip(self._data, 1)
        return self

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
    def center(self) -> Tuple[int, int]:
        return int(self.width // 2), int(self.height // 2)

    @property
    def min_len(self) -> int:
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

    def bgr_to_rgb(self) -> Image:
        if self._channels == 'BGR':
            self._data = cv2.cvtColor(self._data, cv2.COLOR_BGR2RGB)
            self._channels = 'RGB'
        else:
            raise Exception("Image isn't BGR")
        return self

    def to_gray_scale(self) -> Image:
        if self._channels == 'BGR':
            self._data = cv2.cvtColor(self._data, cv2.COLOR_BGR2GRAY)
        elif self._channels == 'RGB':
            self._data = cv2.cvtColor(self._data, cv2.COLOR_RGB2GRAY)
        else:
            raise Exception("Image isn't RGB or BGR")
        self._channels = self._GRAY_SCALE
        return self

    def rgb_to_bgr(self) -> Image:
        if self._channels == 'RGB':
            self._data = cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR)
            self._channels = 'BGR'
        else:
            raise Exception("Image isn't RGB")
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

    def prune(self) -> Image:
        min_len = self.min_len
        return self.crop_by_center((min_len, min_len))

    def plot(self):
        if ON_COLAB_JUPYTER:
            from google.colab.patches import cv2_imshow
            cv2_imshow(self.data)
        else:
            if self._channels == 'BGR':
                plt.imshow(cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB))
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
        k = int(self.min_len * 0.03)

        w_top_r_limit = (self.width - text_size[0]) - k
        w_center_limit = int((self.width - text_size[0]) // 2) + k
        h_bottom_limit = (self.height - text_size[1]) - k

        pos = {'left_top':      (k, k),
               'left_bottom':   (k, h_bottom_limit),
               'right_top':     (w_top_r_limit, k),
               'right_bottom':  (w_top_r_limit, h_bottom_limit),
               'center_top':    (w_center_limit, k),
               'center_bottom': (w_center_limit, h_bottom_limit)
               }.get(pos, (k, k))

        x, y = pos

        text_w, text_h = text_size
        cv2.rectangle(self.data, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(self.data, text, (x, y + text_h), font, font_scale, text_color, font_thickness)

        return self

    def gaussian_pyramid(self, levels=3, *args, **kwargs):

        img = self.data.copy()
        pyramid = [Image(img)]
        while levels >= 1:
            img = cv2.pyrDown(img, *args, **kwargs)
            pyramid.append(Image(img))
            levels -= 1
        return pyramid

    def laplacian_pyramid(self, levels=3) -> Sequence[Image]:
        gaussian_pyramid = self.gaussian_pyramid(levels)
        pyramid = []
        for i in range(levels, 0, -1):
            GE = cv2.pyrUp(gaussian_pyramid[i].data)
            GE = cv2.resize(GE, gaussian_pyramid[i - 1].data.shape[::-1][1:])
            L = Image(cv2.subtract(gaussian_pyramid[i - 1].data, GE))
            pyramid.append(L)
        return pyramid

    @classmethod
    def images_from_dir(cls, dir_path, ext='.jpg'):
        dir_path = cj.Path(dir_path)
        return [cls(im_p.path) for im_p in dir_path.list_files(ext=ext)]


class Video:
    def __init__(self, data):
        try:
            self._n_frames = len(data)
        except TypeError:
            self._n_frames = None

        self._data = data

    def show(self, fps=30, window_name='Video', write_info=False):
        start_time = time.time()
        for current_frame, image in enumerate(self._data, start=1):
            start = time.time()

            if not isinstance(image, Image):
                image = Image(image)
            if write_info:
                current_time = time.time() - start_time
                image.draw_text(
                    f'Frame: {current_frame} - Time: {int(current_time)} - FPS: {int(current_frame // current_time) if current_time > 0 else 0.01}')

            cv2.imshow(window_name, image.data)

            if self._n_frames is not None:
                k = cv2.waitKey(int(round(max(1./fps - (time.time() - start), 0), 3) * 1000))
            else:
                k = cv2.waitKey(1)
            if k == ord('q') or k == ord('\x1b'):
                break

        cv2.destroyAllWindows()

    @classmethod
    def load_from_dir(cls, p):
        return cls(Image.images_from_dir(p))
