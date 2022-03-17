from __future__ import annotations

import math
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Union, Tuple, Sequence, Iterator

import cereja as cj
import cv2
import numpy as np
from matplotlib import pyplot as plt

from .devices import Mouse
from .settings import ON_COLAB_JUPYTER

__all__ = ['Image', 'Video', 'VideoWriter']

from .utils import show_local_mp4


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
            # noinspection PyUnresolvedReferences
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
        font_scale = min((self.width - (self.width * 0.2)) / text_size[0], font_scale)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_size = text_size[0], int(text_size[1] * 1.5)
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
        cv2.putText(self.data, text, (x, y + int(text_h * 0.9)), font, font_scale, text_color, font_thickness)

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
        args = (*args, cv2.CAP_DSHOW) if self._is_webcam else args
        super().__init__(*args, **kwargs)
        if fps is not None:
            assert isinstance(fps, (int, float)), ValueError(f'{fps} value for fps is not valid. Send int or float.')
        else:
            fps = self.get(cv2.CAP_PROP_FPS)
        self._fps = fps
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

    @property
    def next_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        return self.read()

    @property
    def is_webcam(self):
        return self._is_webcam

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
            if fp.is_file and fp.suffix in Image.IMAGE_FORMATS:
                yield cv2.imread(fp.path)

    def stop(self):
        self._frames = None
        self._is_opened = False


class Screen(_IVideo):
    def __init__(self, *args):
        mouse = Mouse()

        self._width, self._height = mouse.window_size
        self._mon = {'left': 0, 'top': 0, 'width': self._width, 'height': self._height}
        self._capture = True
        self._frames = self.__get_frames()

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
        return 30

    @property
    def is_webcam(self) -> bool:
        return True

    @property
    def is_opened(self) -> bool:
        return self._capture

    def stop(self):
        self._capture = False


class Video:

    def __init__(self, *args, fps=None, **kwargs):
        kwargs['fps'] = fps
        self._args = args
        self._kwargs = kwargs
        self._build()
        self._th_show = None

    def _build(self):
        if len(self._args):
            if isinstance(self._args[0], str):
                if self._args[0] == 'monitor':
                    self._cap = Screen()
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
        self._start_time = None

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
    def fps(self):
        if not self._cap.is_webcam:
            time_it = time.time() - self._start_time
            if time_it >= 1:
                return int(round(self.current_number_frame / time_it))
            return self._cap.fps
        return int(round(self.current_number_frame / (time.time() - self._start_time)))

    @property
    def total_frames(self):
        return self._cap.total_frames if not self._cap.is_webcam else self._current_number_frame + 1

    def __get_next_frame(self) -> Union[np.ndarray, None]:
        if self._start_time is None:
            self._start_time = time.time()
        _, image = self._cap.next_frame
        self._current_number_frame += 1
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
            if len(batch_frames) < kernel_size:
                batch_frames.append(frame if not take_number_frame else [self.current_number_frame, frame])
            else:
                yield batch_frames
                batch_frames = batch_frames[strides:]

    @property
    def is_break_view(self) -> bool:
        if not self._cap.is_webcam:
            time_it = time.time() - self._start_time
            # need to take fps in video view
            wait_msec = self._cap.fps + int(
                    abs(self.current_number_frame - time_it * self._cap.fps)) * self._cap.fps
        else:
            wait_msec = 1000
        k = cv2.waitKey(int(1000 // wait_msec))
        return k == ord('q') or k == ord('\x1b')

    @property
    def video_info(self):
        return f"Size: {self._size_info} - FPS: {self._fps_info} - Frames: {self._frames_info} T: {self._time_info}"

    @property
    def _size_info(self):
        return f'{self.width}x{self.height}'

    @property
    def _fps_info(self):
        return f'{cj.get_zero_mask(self.fps, max_len=3)}'

    @property
    def _time_info(self):
        return f'{round(time.time() - self._start_time, 1)} s'

    @property
    def _frames_info(self):
        return f'{cj.get_zero_mask(self.current_number_frame, max_len=len(str(self.total_frames)))}/{self.total_frames}'

    def _show(self):
        if ON_COLAB_JUPYTER:
            raise NotImplementedError("Not implemented show video on colab")
        self._th_show_running = True
        while self.is_opened:
            image = self.__get_next_frame()
            if image is None:
                continue
            cv2.imshow(self._cap.name, Image(image).draw_text(self.video_info).data)
            if self.is_break_view:
                self._cap.stop()
        self._th_show_running = False

    def get_frames(self, n_frames=None):
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
            if not self.is_opened:
                self._build()
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
        while self.current_number_frame < max_frame:
            frame = self.next_frame
            if frame is None:
                continue
            prefix = cj.get_zero_mask(self.current_number_frame, size_number)
            if self.current_number_frame in filter_map:
                cv2.imwrite(p.join(f'{prefix}.{img_format}').path, frame)

    @property
    def duration(self):
        return self.total_frames / self._cap.fps
