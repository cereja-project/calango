"""Asynchronous Windows screen recording with Cereja's native capture API."""

import math
import os
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np

from ._recording import pad_to_even, validate_fps, write_timed_frames

__all__ = ["ScreenRecorder"]


def _load_screen_capture():
    try:
        from cereja import ScreenCapture
    except ImportError as exc:
        raise ImportError(
            "Screen recording requires Cereja's native ScreenCapture API. "
            "Install the Cereja checkout containing the screen-capture extension; "
            "see the Calango recorder installation instructions."
        ) from exc
    return ScreenCapture


def _verify_video(path, *, width, height, fps, frames):
    """Decode the complete closed file before reporting a successful recording."""
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise RuntimeError("The recorded video could not be reopened.")
        actual_fps = video.get(cv2.CAP_PROP_FPS)
        if not math.isclose(actual_fps, fps, rel_tol=1e-3, abs_tol=1e-3):
            raise RuntimeError("The recorded video has an unexpected frame rate.")
        if (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)), round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height):
            raise RuntimeError("The recorded video has unexpected dimensions.")
        if round(video.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
            raise RuntimeError("The recorded video has an unexpected frame count.")
        decoded = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise RuntimeError("A recorded frame has unexpected dimensions.")
            decoded += 1
        if decoded != frames:
            raise RuntimeError("The recorded video could not be decoded completely.")
    finally:
        video.release()


class ScreenRecorder:
    """Record one monitor, region or window's client area to an MP4 file.

    Each instance represents a single session. ``start`` and ``stop`` do not
    block on capture or encoding. ``wait`` returns a pathlib.Path after the
    closed video has been verified, or None if stopped before the first frame.
    Capture, encoding and finalization errors are raised by ``wait``; a failed
    recording's recoverable output is exposed through ``partial_path``.
    """

    def __init__(self, output, *, monitor=None, region=None, window=None, fps=30, include_cursor=True):
        self._output = Path(output).expanduser().absolute()
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("The recording destination must end in .mp4.")
        if not self.output.parent.is_dir():
            raise FileNotFoundError("The recording destination directory does not exist.")
        if os.path.lexists(self.output):
            raise FileExistsError(self.output)
        if sum(value is not None for value in (monitor, region, window)) > 1:
            raise ValueError("Choose only one monitor, region or window.")
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("Window must be a positive integer HWND.")
        if region is not None:
            if (not isinstance(region, (tuple, list)) or len(region) != 4
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in region)
                    or region[2] <= 0 or region[3] <= 0):
                raise ValueError("Region must contain integer left, top, positive width and positive height.")
            region = tuple(region)
        self._monitor = monitor
        self._region = region
        self._window = window
        self._fps = validate_fps(fps)
        # The desktop pointer may belong to an application covering this window.
        # It is never composited into a window-only recording.
        self._include_cursor = bool(include_cursor) and window is None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._done = threading.Event()
        self._state = "idle"
        self._error = None
        self._frames_written = 0
        self._started_at = None
        self._ended_at = None
        self._stopped_at = None
        self._result = None
        self._partial_path = None
        self._thread = None

    @property
    def output(self):
        """The validated destination, fixed for the lifetime of this session."""
        return self._output

    @property
    def monitor(self):
        return self._monitor

    @property
    def region(self):
        return self._region

    @property
    def window(self):
        return self._window

    @property
    def fps(self):
        return self._fps

    @property
    def include_cursor(self):
        return self._include_cursor

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def frames_written(self):
        with self._lock:
            return self._frames_written

    @property
    def partial_path(self):
        with self._lock:
            return self._partial_path

    @property
    def elapsed(self):
        with self._lock:
            if self._started_at is None:
                return 0.0
            end = self._ended_at if self._ended_at is not None else self._stopped_at
            return max(0.0, (time.monotonic() if end is None else end) - self._started_at)

    def start(self):
        with self._lock:
            if self._state != "idle":
                raise RuntimeError("A ScreenRecorder instance can only be started once.")
            self._state = "starting"
            self._thread = threading.Thread(target=self._run, name="calango-recorder", daemon=False)
            try:
                self._thread.start()
            except BaseException as exc:
                self._state = "failed"
                self._error = exc
                self._done.set()
                raise
        return self

    def stop(self):
        with self._lock:
            if self._state in ("idle", "finished", "failed") or self._stopped_at is not None:
                return
            self._stopped_at = time.monotonic()
            self._state = "stopping"
            self._stop_event.set()

    def wait(self, timeout=None):
        with self._lock:
            if self._state == "idle":
                raise RuntimeError("Start the recorder before waiting for it.")
        if not self._done.wait(timeout):
            raise TimeoutError("The recording has not finished.")
        with self._lock:
            error, result = self._error, self._result
        if error is not None:
            raise error
        return result

    def _stop_time(self):
        with self._lock:
            return self._stopped_at

    def _progress(self, count, elapsed):
        with self._lock:
            self._frames_written = count
            if self._started_at is None:
                self._started_at = time.monotonic() - elapsed
            if self._state == "starting":
                self._state = "recording"

    def _run(self):
        partial = None
        try:
            if self._stop_event.is_set():
                return
            capture_class = _load_screen_capture()
            from .media import VideoWriter

            if os.path.lexists(self.output):
                raise FileExistsError(self.output)
            descriptor, filename = tempfile.mkstemp(
                prefix=f".{self.output.stem}.", suffix=".partial.mp4", dir=self.output.parent
            )
            os.close(descriptor)
            partial = Path(filename)
            with self._lock:
                self._partial_path = partial
            dimensions = None
            capture_dimensions = None
            with capture_class(include_cursor=self.include_cursor) as capture:
                def get_frame():
                    nonlocal capture_dimensions
                    if self.window is not None:
                        native = capture.grab(window=self.window, only_window_content=True)
                    else:
                        native = capture.grab(monitor=self.monitor, region=self.region)
                    current_dimensions = native.width, native.height
                    if capture_dimensions is None:
                        capture_dimensions = current_dimensions
                    elif current_dimensions != capture_dimensions:
                        raise ValueError("Capture dimensions changed during the recording. Keep the source size unchanged.")
                    pixels = np.frombuffer(native.bgra, dtype=np.uint8).reshape(native.height, native.width, 4)
                    return pad_to_even(pixels[:, :, :3])

                with VideoWriter(str(partial), fourcc="mp4v", fps=self.fps) as writer:
                    def write_frame(frame):
                        nonlocal dimensions
                        height, width = frame.shape[:2]
                        if dimensions is None:
                            dimensions = width, height
                        elif dimensions != (width, height):
                            raise ValueError("Capture dimensions changed during the recording.")
                        writer.add_frame(frame)

                    stats = write_timed_frames(
                        get_frame, write_frame, fps=self.fps, stop_event=self._stop_event,
                        stop_time=self._stop_time, progress=self._progress,
                    )
                with self._lock:
                    self._started_at = stats.started_at
                    self._ended_at = stats.ended_at
            if not stats.frames_written:
                partial.unlink(missing_ok=True)
                with self._lock:
                    self._partial_path = None
                return
            _verify_video(partial, width=dimensions[0], height=dimensions[1], fps=self.fps, frames=stats.frames_written)
            # Windows rename fails if the destination exists, including a race
            # during recording, and also supports volumes without hard links.
            # On POSIX, link supplies the same no-replacement guarantee.
            if os.name == "nt":
                os.rename(partial, self.output)
            else:
                os.link(partial, self.output)
                partial.unlink()
            with self._lock:
                self._partial_path = None
                self._result = self.output
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._state = "failed"
                self._ended_at = self._stopped_at if self._stopped_at is not None else time.monotonic()
            if partial is not None and partial.exists():
                if partial.stat().st_size:
                    exc.add_note(f"Partial recording retained at: {partial}")
                else:
                    partial.unlink(missing_ok=True)
                    with self._lock:
                        self._partial_path = None
        finally:
            with self._lock:
                if self._state != "failed":
                    self._state = "finished"
            self._done.set()
