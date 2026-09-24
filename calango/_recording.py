"""Bounded, constant-frame-rate writing shared by screen recording APIs."""

from dataclasses import dataclass
import math
import operator
import time

import numpy as np


class RecordingPerformanceError(RuntimeError):
    """Capture or encoding fell too far behind the recording clock."""


@dataclass(frozen=True)
class RecordingStats:
    started_at: float | None
    ended_at: float
    frames_written: int


def validate_fps(fps):
    if isinstance(fps, bool):
        raise ValueError("FPS must be a finite positive number.")
    try:
        value = float(fps)
    except (TypeError, ValueError):
        raise ValueError("FPS must be a finite positive number.") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("FPS must be a finite positive number.")
    return value


def pad_to_even(frame):
    """Validate BGR pixels and duplicate at most one bottom/right edge pixel."""
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise ValueError("Frames must be nonempty uint8 BGR arrays with three channels.")
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        frame = np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
    return np.ascontiguousarray(frame)


def write_timed_frames(get_frame, write_frame, *, fps, stop_event=None,
                       stop_time=None, n_frames=None, progress=None,
                       clock=time.monotonic, wait=None, max_lag=2.0):
    """Write on absolute deadlines, repeating the previous image for missed slots.

    ``get_frame`` returns a frame, or None on EOF. ``stop_time`` optionally
    returns the monotonic time of a stop request; supplying it prevents a slow
    in-flight capture from extending the recording. ``progress(count, elapsed)``
    runs in the calling thread. Only the current and previous frames are held.
    """
    fps = validate_fps(fps)
    if n_frames is not None:
        if isinstance(n_frames, bool):
            raise ValueError("n_frames must be a nonnegative integer or None.")
        try:
            n_frames = operator.index(n_frames)
        except TypeError:
            raise ValueError("n_frames must be a nonnegative integer or None.") from None
        if n_frames < 0:
            raise ValueError("n_frames must be a nonnegative integer or None.")
    if not math.isfinite(max_lag) or max_lag <= 0:
        raise ValueError("max_lag must be finite and positive.")
    wait = wait or (stop_event.wait if stop_event is not None else time.sleep)
    started_at = None
    count = 0
    last_frame = None

    def stopped():
        return stop_event is not None and stop_event.is_set()

    def cutoff():
        requested = stop_time() if stop_time is not None else None
        return clock() if requested is None else requested

    def emit(frame):
        nonlocal count
        write_frame(frame)
        count += 1
        if progress is not None:
            progress(count, max(0.0, clock() - started_at))
        # The final write of a limited recording has no next iteration in
        # which to detect encoder lag, so check after every completed write.
        if clock() - (started_at + count / fps) > max_lag:
            raise RecordingPerformanceError("Recording fell more than two seconds behind real time.")

    def fill_until(end):
        # The final frame covers the fraction of a frame before the stop request.
        target = max(1, math.ceil(max(0.0, end - started_at) * fps - 1e-9))
        if n_frames is not None:
            target = min(target, n_frames)
        while count < target:
            if clock() - (started_at + count / fps) > max_lag:
                raise RecordingPerformanceError("Recording fell more than two seconds behind real time.")
            emit(last_frame)

    while n_frames is None or count < n_frames:
        if stopped():
            end = cutoff()
            if last_frame is not None:
                fill_until(end)
            return RecordingStats(started_at, end, count)
        if started_at is not None:
            deadline = started_at + count / fps
            delay = deadline - clock()
            if delay > 0:
                wait(delay)
                continue
            if clock() - deadline > max_lag:
                raise RecordingPerformanceError("Recording fell more than two seconds behind real time.")
        frame = get_frame()
        now = clock()
        if stopped():
            end = cutoff()
            if last_frame is not None:
                fill_until(end)
            return RecordingStats(started_at, end, count)
        if frame is None:
            return RecordingStats(started_at, now, count)
        if started_at is None:
            started_at = now
        else:
            if now - (started_at + count / fps) > max_lag:
                raise RecordingPerformanceError("Recording fell more than two seconds behind real time.")
            slot = math.floor((now - started_at) * fps + 1e-9)
            if n_frames is not None:
                slot = min(slot, n_frames - 1)
            while count < slot:
                emit(last_frame)
                if clock() - (started_at + count / fps) > max_lag:
                    raise RecordingPerformanceError("Recording fell more than two seconds behind real time.")
        last_frame = frame
        emit(last_frame)
    return RecordingStats(started_at, clock(), count)
