"""Scoped decoding of frames from local MP4 files."""
from collections.abc import Iterator
from contextlib import closing, contextmanager
from numbers import Integral
from os import PathLike, fspath
from pathlib import Path
from stat import S_ISDIR, S_ISREG

import cv2
import numpy as np


@contextmanager
def read_video_frames(
    path: str | PathLike[str], *, crop: tuple[int, int] | None = None
) -> Iterator[Iterator[tuple[int, np.ndarray]]]:
    """Yield an iterator over (zero-based frame index, contiguous RGB uint8 array).

    Enter the context with a local .mp4 path. An optional (height, width)
    crop copies the central pixels without resizing. Orientation follows the
    installed decoder. The index counts decoded frames; it is not a timestamp.
    Exiting the context closes the decoder, including on break or exception.
    """
    raw_path = fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError('path must be a string or a text path-like object')
    file_path = Path(raw_path)
    if '://' in raw_path or file_path.suffix.lower() != '.mp4':
        raise ValueError('path must be a local .mp4 file')
    mode = file_path.stat().st_mode
    if S_ISDIR(mode):
        raise IsADirectoryError(str(file_path))
    if not S_ISREG(mode):
        raise ValueError('path must be a regular file')

    if crop is not None and (
        not isinstance(crop, tuple) or len(crop) != 2
        or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in crop)
    ):
        raise ValueError('crop must be a (height, width) tuple of positive integers')

    capture = cv2.VideoCapture()
    try:
        if not capture.open(str(file_path.absolute())):
            raise OSError(f'Cannot open MP4: {file_path}')
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise OSError(f'Cannot decode the first frame of MP4: {file_path}')
        first_pixels = _center_view(first_frame, crop) if crop is not None else first_frame
        with closing(_rgb_frames(capture, first_pixels, crop)) as frames:
            del first_frame, first_pixels
            yield frames
    finally:
        capture.release()


def _rgb_frames(capture, pixels, crop):
    index = 0
    while True:
        yield index, cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        del pixels
        ok, frame = capture.read()
        if not ok or frame is None:
            return
        pixels = _center_view(frame, crop) if crop is not None else frame
        del frame
        index += 1


def _center_view(frame, crop):
    height, width = frame.shape[:2]
    crop_height, crop_width = map(int, crop)
    if crop_height > height or crop_width > width:
        raise ValueError(f'crop {crop} does not fit decoded frame {(height, width)}')
    top = height // 2 - crop_height // 2
    left = width // 2 - crop_width // 2
    return frame[top:top + crop_height, left:left + crop_width]
