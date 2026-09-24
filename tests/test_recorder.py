"""Recorder behavior without a display, plus real MP4 codec integration."""

from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calango._recording import RecordingPerformanceError, pad_to_even, write_timed_frames
from calango.recorder import ScreenRecorder, _verify_video


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class TestTimedRecording(unittest.TestCase):
    def test_absolute_deadlines_and_frame_limit(self):
        clock = Clock()
        captured_at = []
        frames = []

        def capture():
            captured_at.append(clock())
            clock.now += 0.01
            return len(captured_at)

        stats = write_timed_frames(capture, frames.append, fps=10, n_frames=4, clock=clock, wait=clock.wait)
        self.assertEqual(frames, [1, 2, 3, 4])
        np.testing.assert_allclose(captured_at, [0, 0.11, 0.21, 0.31])
        self.assertEqual(stats.frames_written, 4)

    def test_slow_capture_repeats_previous_frame_without_drifting(self):
        clock = Clock()
        frames = []
        calls = 0

        def capture():
            nonlocal calls
            calls += 1
            if calls == 2:
                clock.now += 0.26
            return calls

        write_timed_frames(capture, frames.append, fps=10, n_frames=5, clock=clock, wait=clock.wait)
        self.assertEqual(frames, [1, 1, 1, 2, 3])
        self.assertAlmostEqual(clock(), 0.4)

    def test_stop_during_capture_uses_requested_time(self):
        clock = Clock()
        stop = threading.Event()
        frames = []
        calls = 0

        def capture():
            nonlocal calls
            calls += 1
            if calls == 2:
                clock.now = 0.8
                stop.set()
            return calls

        stats = write_timed_frames(capture, frames.append, fps=10, stop_event=stop,
                                   stop_time=lambda: 0.25, clock=clock, wait=clock.wait)
        self.assertEqual(frames, [1, 1, 1])
        self.assertEqual(stats.ended_at, 0.25)

    def test_stop_during_first_capture_writes_nothing(self):
        stop = threading.Event()
        frames = []

        def capture():
            stop.set()
            return 1

        stats = write_timed_frames(capture, frames.append, fps=30, stop_event=stop)
        self.assertEqual(frames, [])
        self.assertIsNone(stats.started_at)

    def test_capture_and_encoding_lag_fail(self):
        for delayed_stage in ("capture", "write"):
            with self.subTest(delayed_stage=delayed_stage):
                clock = Clock()
                count = 0

                def capture():
                    nonlocal count
                    count += 1
                    if count == 2 and delayed_stage == "capture":
                        clock.now += 2.1
                    return count

                def write(frame):
                    if delayed_stage == "write":
                        clock.now += 2.2

                with self.assertRaises(RecordingPerformanceError):
                    write_timed_frames(capture, write, fps=30, n_frames=3, clock=clock, wait=clock.wait)

    def test_final_limited_frame_still_checks_encoder_lag(self):
        clock = Clock()

        def encode(frame):
            clock.now += 3.0

        with self.assertRaises(RecordingPerformanceError):
            write_timed_frames(lambda: 1, encode, fps=30, n_frames=1, clock=clock, wait=clock.wait)

    def test_no_work_for_zero_limit_or_already_stopped(self):
        for kwargs in ({"n_frames": 0}, {"stop_event": SimpleNamespace(is_set=lambda: True, wait=lambda _: None)}):
            with self.subTest(kwargs=kwargs):
                stats = write_timed_frames(lambda: self.fail("capture called"), lambda _: self.fail("write called"),
                                           fps=30, **kwargs)
                self.assertEqual(stats.frames_written, 0)

    def test_source_eof_and_failure_propagate(self):
        self.assertEqual(write_timed_frames(lambda: None, lambda _: None, fps=30).frames_written, 0)
        with self.assertRaisesRegex(OSError, "capture failed"):
            write_timed_frames(lambda: (_ for _ in ()).throw(OSError("capture failed")), lambda _: None, fps=30)

    def test_invalid_parameters(self):
        for fps in (0, -1, float("nan"), float("inf"), True, None):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                write_timed_frames(lambda: None, lambda _: None, fps=fps)
        for count in (-1, 1.5, True):
            with self.subTest(n_frames=count), self.assertRaises(ValueError):
                write_timed_frames(lambda: None, lambda _: None, fps=30, n_frames=count)

    def test_odd_pixels_are_padded_with_edge_color(self):
        frame = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        padded = pad_to_even(frame)
        self.assertEqual(padded.shape, (4, 6, 3))
        np.testing.assert_array_equal(padded[:3, :5], frame)
        np.testing.assert_array_equal(padded[-1, -1], frame[-1, -1])
        for invalid in (np.zeros((2, 2)), np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                pad_to_even(invalid)


class SyntheticCapture:
    """Exercise buffer conversion and the actual encoder in the worker thread."""
    instances = []

    def __init__(self, include_cursor=True):
        self.include_cursor = include_cursor
        self.owner = threading.get_ident()
        self.closed = False
        self.calls = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        if threading.get_ident() != self.owner:
            raise AssertionError("Capture closed on a different thread.")

    def grab(self, *, monitor=None, region=None):
        if threading.get_ident() != self.owner:
            raise AssertionError("Capture used on a different thread.")
        self.calls.append((monitor, region))
        frame = np.full((47, 63, 4), (20, 60, 200, 255), dtype=np.uint8)
        return SimpleNamespace(left=-63, top=0, width=63, height=47, bgra=frame.tobytes())


class TestScreenRecorder(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "recording.mp4"
        SyntheticCapture.instances = []
        self.capture_patch = patch("calango.recorder._load_screen_capture", return_value=SyntheticCapture)
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)

    def await_frames(self, recorder, minimum=3):
        deadline = time.monotonic() + 5
        while recorder.frames_written < minimum and time.monotonic() < deadline:
            if recorder.state == "failed":
                recorder.wait()
            time.sleep(0.01)
        self.assertGreaterEqual(recorder.frames_written, minimum)

    def stop_safely(self, recorder):
        recorder.stop()
        try:
            recorder.wait(timeout=5)
        except BaseException:
            pass

    def start_recorder(self, **kwargs):
        recorder = ScreenRecorder(self.output, **kwargs)
        self.addCleanup(self.stop_safely, recorder)
        self.assertIs(recorder.start(), recorder)
        return recorder

    def test_real_mp4_roundtrip_state_timing_colors_and_thread_ownership(self):
        recorder = self.start_recorder(fps=30, monitor="monitor-2", include_cursor=False)
        self.await_frames(recorder)
        self.assertEqual(recorder.state, "recording")
        self.assertGreater(recorder.elapsed, 0)
        with self.assertRaises(TimeoutError):
            recorder.wait(timeout=0)
        recorder.stop()
        recorder.stop()
        self.assertEqual(recorder.wait(timeout=5), self.output)
        self.assertEqual(recorder.wait(), self.output)
        self.assertEqual(recorder.state, "finished")
        self.assertIsNone(recorder.error)
        self.assertIsNone(recorder.partial_path)
        self.assertLessEqual(abs(recorder.frames_written / 30 - recorder.elapsed), 1 / 30 + 0.01)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.output])
        capture = SyntheticCapture.instances[-1]
        self.assertTrue(capture.closed)
        self.assertNotEqual(capture.owner, threading.get_ident())
        self.assertFalse(capture.include_cursor)
        self.assertEqual(capture.calls[0], ("monitor-2", None))
        video = cv2.VideoCapture(str(self.output))
        try:
            self.assertEqual(video.get(cv2.CAP_PROP_FRAME_WIDTH), 64)
            self.assertEqual(video.get(cv2.CAP_PROP_FRAME_HEIGHT), 48)
            self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT), recorder.frames_written)
            ok, frame = video.read()
            self.assertTrue(ok)
            np.testing.assert_allclose(frame.mean(axis=(0, 1)), [20, 60, 200], atol=6)
        finally:
            video.release()
        with self.assertRaises(RuntimeError):
            recorder.start()

    def test_cancel_while_first_frame_is_pending(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingCapture(SyntheticCapture):
            def grab(self, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test release timed out.")
                return super().grab(**kwargs)

        with patch("calango.recorder._load_screen_capture", return_value=BlockingCapture):
            recorder = self.start_recorder()
            self.addCleanup(release.set)
            self.assertTrue(entered.wait(5))
            recorder.stop()
            self.assertEqual(recorder.state, "stopping")
            release.set()
            self.assertIsNone(recorder.wait(5))
        self.assertEqual(recorder.state, "finished")
        self.assertEqual(recorder.frames_written, 0)
        self.assertEqual(recorder.elapsed, 0)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_window_recording_targets_only_client_area_without_desktop_cursor(self):
        class WindowCapture(SyntheticCapture):
            def grab(self, *, window, only_window_content):
                self.window_options = window, only_window_content
                return super().grab()

        with patch("calango.recorder._load_screen_capture", return_value=WindowCapture):
            recorder = self.start_recorder(window=1234, include_cursor=True)
            self.await_frames(recorder)
            recorder.stop()
            self.assertEqual(recorder.wait(5), self.output)
        capture = SyntheticCapture.instances[-1]
        self.assertEqual(capture.window_options, (1234, True))
        self.assertFalse(capture.include_cursor)
        self.assertFalse(recorder.include_cursor)
        self.assertEqual(recorder.window, 1234)

    def test_closed_window_fails_without_switching_to_desktop(self):
        calls = []

        class ClosingWindow(SyntheticCapture):
            def grab(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 2:
                    raise OSError("The selected window was closed.")
                return super().grab()

        with patch("calango.recorder._load_screen_capture", return_value=ClosingWindow):
            recorder = self.start_recorder(window=1234)
            with self.assertRaisesRegex(OSError, "window was closed"):
                recorder.wait(5)
        self.assertEqual(calls, [{"window": 1234, "only_window_content": True}] * 2)
        self.assertEqual(recorder.state, "failed")
        self.assertFalse(self.output.exists())
        self.assertTrue(recorder.partial_path.is_file())
        self.assertTrue(SyntheticCapture.instances[-1].closed)

    def test_resize_is_detected_even_when_padded_dimensions_do_not_change(self):
        class ResizingCapture(SyntheticCapture):
            def grab(self, **kwargs):
                frame = super().grab()
                if len(self.calls) > 1:
                    pixels = np.full((48, 64, 4), (20, 60, 200, 255), dtype=np.uint8)
                    frame.width, frame.height, frame.bgra = 64, 48, pixels.tobytes()
                return frame

        with patch("calango.recorder._load_screen_capture", return_value=ResizingCapture):
            recorder = self.start_recorder()
            with self.assertRaisesRegex(ValueError, "dimensions changed"):
                recorder.wait(5)
        self.assertEqual(recorder.state, "failed")
        self.assertEqual(recorder.frames_written, 1)
        self.assertTrue(recorder.partial_path.is_file())
        self.assertFalse(self.output.exists())

    def test_capture_failure_preserves_partial_and_propagates_original_error(self):
        failure = OSError("Capture failed")

        class FailingCapture(SyntheticCapture):
            def grab(self, **kwargs):
                if len(self.calls) == 2:
                    raise failure
                return super().grab(**kwargs)

        with patch("calango.recorder._load_screen_capture", return_value=FailingCapture):
            recorder = self.start_recorder()
            with self.assertRaises(OSError) as raised:
                recorder.wait(5)
        self.assertIs(raised.exception, failure)
        self.assertIs(recorder.error, failure)
        self.assertEqual(recorder.state, "failed")
        self.assertTrue(SyntheticCapture.instances[-1].closed)
        self.assertFalse(self.output.exists())
        self.assertTrue(recorder.partial_path.is_file())
        self.assertTrue(str(recorder.partial_path).endswith(".partial.mp4"))

    def test_codec_error_and_verification_error_never_publish_success(self):
        for target in ("calango.media.VideoWriter.add_frame", "calango.recorder._verify_video"):
            with self.subTest(target=target), patch(target, side_effect=RuntimeError("Encoder failed")):
                recorder = self.start_recorder()
                if target.endswith("_verify_video"):
                    self.await_frames(recorder)
                    recorder.stop()
                with self.assertRaisesRegex(RuntimeError, "Encoder failed"):
                    recorder.wait(5)
                self.assertFalse(self.output.exists())
                self.assertEqual(recorder.state, "failed")

    def test_destination_race_does_not_overwrite(self):
        recorder = self.start_recorder()
        self.await_frames(recorder)
        self.output.write_bytes(b"Created by another process")
        recorder.stop()
        with self.assertRaises(FileExistsError):
            recorder.wait(5)
        self.assertEqual(self.output.read_bytes(), b"Created by another process")
        self.assertTrue(recorder.partial_path.is_file())

    def test_initial_validation_and_wait_before_start(self):
        recorder = ScreenRecorder(self.output)
        recorder.stop()
        self.assertEqual(recorder.state, "idle")
        with self.assertRaises(RuntimeError):
            recorder.wait(0)
        for kwargs in ({"fps": 0}, {"monitor": "m", "region": (0, 0, 1, 1)}, {"region": (0, 0, -1, 2)},
                       {"window": 12, "monitor": "m"}, {"window": 12, "region": (0, 0, 2, 2)},
                       *({"window": value} for value in (True, False, 0, -1, "12", 1.5))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ScreenRecorder(self.output, **kwargs)
        with self.assertRaises(ValueError):
            ScreenRecorder(self.output.with_suffix(".avi"))
        with self.assertRaises(FileNotFoundError):
            ScreenRecorder(self.output.parent / "missing" / "file.mp4")
        self.output.touch()
        with self.assertRaises(FileExistsError):
            ScreenRecorder(self.output)

    def test_configuration_is_read_only_before_and_during_recording(self):
        recorder = ScreenRecorder(self.output, monitor="monitor-2", fps=15, include_cursor=False)
        attempts = {"output": self.output.with_suffix(".txt"), "monitor": "another-monitor",
                    "region": (0, 0, 20, 20), "window": 1234, "fps": 60, "include_cursor": True}
        for phase in ("before start", "during recording"):
            for name, value in attempts.items():
                with self.subTest(phase=phase, option=name), self.assertRaises(AttributeError):
                    setattr(recorder, name, value)
            if phase == "before start":
                self.addCleanup(self.stop_safely, recorder)
                recorder.start()
                self.await_frames(recorder)
        recorder.stop()
        self.assertEqual(recorder.wait(5), self.output)
        self.assertEqual(recorder.fps, 15)
        self.assertIsNone(recorder.region)
        capture = SyntheticCapture.instances[-1]
        self.assertTrue(all(call == ("monitor-2", None) for call in capture.calls))
        self.assertFalse(capture.include_cursor)

    def test_caller_region_list_cannot_change_session_configuration(self):
        region = [-63, 0, 63, 47]
        recorder = ScreenRecorder(self.output, region=region)
        region[:] = [0, 0, 5, 5]
        self.assertEqual(recorder.region, (-63, 0, 63, 47))
        self.addCleanup(self.stop_safely, recorder)
        recorder.start()
        self.await_frames(recorder)
        recorder.stop()
        self.assertEqual(recorder.wait(5), self.output)
        self.assertTrue(all(call == (None, (-63, 0, 63, 47)) for call in SyntheticCapture.instances[-1].calls))

    def test_missing_cereja_capability_is_reported_from_worker(self):
        with patch("calango.recorder._load_screen_capture", side_effect=ImportError("Install the Cereja checkout")):
            recorder = self.start_recorder()
            with self.assertRaisesRegex(ImportError, "Cereja checkout"):
                recorder.wait(5)
        self.assertEqual(recorder.state, "failed")

    def test_verification_rejects_invalid_video(self):
        self.output.write_bytes(b"Not a video")
        with self.assertRaises(RuntimeError):
            _verify_video(self.output, width=64, height=48, fps=30, frames=1)


if __name__ == "__main__":
    unittest.main()
