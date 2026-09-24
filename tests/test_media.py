import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from calango import Image, Video, VideoWriter
from calango.media import Screen


class TestVideoWriter(unittest.TestCase):
    def test_rectangular_frames_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'video.mp4'
            frame = np.zeros((48, 80, 3), dtype=np.uint8)
            frame[:, :40] = (255, 0, 0)
            frame[:, 40:] = (0, 0, 255)
            with VideoWriter(path, fourcc='mp4v', width=80, height=48, fps=15) as writer:
                for _ in range(3):
                    writer.add_frame(Image(frame))
            capture = cv2.VideoCapture(str(path))
            try:
                ok, decoded = capture.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape, (48, 80, 3))
                self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 3)
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 15)
                self.assertGreater(int(decoded[20, 20, 0]), 230)
                self.assertGreater(int(decoded[20, 60, 2]), 230)
            finally:
                capture.release()

    def test_empty_context_does_not_open_encoder(self):
        with mock.patch('calango.media.cv2.VideoWriter') as encoder:
            with VideoWriter('unused.mp4', fourcc='mp4v'):
                pass
            encoder.assert_not_called()

    def test_invalid_frames_are_rejected_before_encoding(self):
        frames = [np.zeros((4, 6), dtype=np.uint8), np.zeros((4, 6, 4), dtype=np.uint8),
                  np.zeros((4, 6, 3), dtype=float), np.zeros((0, 6, 3), dtype=np.uint8)]
        with mock.patch('calango.media.cv2.VideoWriter') as encoder:
            with VideoWriter('unused.mp4', fourcc='mp4v') as writer:
                for frame in frames:
                    with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                        writer.add_frame(frame)
            encoder.assert_not_called()

    def test_dimensions_remain_fixed(self):
        with mock.patch('calango.media.cv2.VideoWriter') as encoder:
            with VideoWriter('unused.mp4', fourcc='mp4v', width=8, height=4) as writer:
                with self.assertRaises(ValueError):
                    writer.add_frame(np.zeros((8, 4, 3), dtype=np.uint8))
            encoder.assert_not_called()

    def test_invalid_configuration_and_closed_writer(self):
        for fps in [0, -1, math.inf, math.nan, True]:
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                VideoWriter('unused.mp4', fps=fps)
        with self.assertRaises(RuntimeError):
            VideoWriter('unused.mp4').add_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_encoder_failure_releases_resource(self):
        with mock.patch('calango.media.cv2.VideoWriter') as encoder:
            encoder.return_value.isOpened.return_value = False
            with self.assertRaises(RuntimeError):
                with VideoWriter('unused.mp4', fourcc='mp4v') as writer:
                    writer.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
            encoder.return_value.release.assert_called_once()


class FakeCapture:
    instances = []

    def __init__(self, include_cursor=True):
        self.closed = False
        self.owner_thread = threading.get_ident()
        self.closed_thread = None
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True
        self.closed_thread = threading.get_ident()

    def list_monitors(self):
        return (SimpleNamespace(id='display', name='Test', left=-64, top=0, width=64, height=32, is_primary=True),)

    def grab(self, **kwargs):
        pixels = np.zeros((32, 64, 4), dtype=np.uint8)
        pixels[:, :, 0] = 255
        pixels[:, :, 3] = 255
        return SimpleNamespace(width=64, height=32, bgra=pixels.tobytes())


class TestScreenCompatibility(unittest.TestCase):
    def setUp(self):
        FakeCapture.instances = []
        self.platform = mock.patch('calango.media.sys.platform', 'win32')
        self.native = mock.patch('cereja.system.ScreenCapture', FakeCapture, create=True)
        self.platform.start()
        self.native.start()
        self.addCleanup(self.native.stop)
        self.addCleanup(self.platform.stop)

    def test_screen_fps_and_bgr_data(self):
        video = Video('monitor', fps=15)
        try:
            self.assertEqual(video._cap.fps, 15)
            self.assertEqual(video.next_frame[0, 0].tolist(), [255, 0, 0])
        finally:
            video.stop()
        self.assertTrue(all(c.closed for c in FakeCapture.instances))

    def test_save_preserves_preprocess_and_output_frame_limit(self):
        def preprocess(frame):
            result = np.zeros_like(frame)
            result[:, :, 2] = 255
            return result
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.mp4'
            video = Video('monitor', fps=15, frame_preprocess=preprocess)
            video.save(str(path), n_frames=3, fourcc='mp4v')
            self.assertFalse(video.is_opened)
            cap = cv2.VideoCapture(str(path))
            try:
                ok, decoded = cap.read()
                self.assertTrue(ok)
                self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 3)
                self.assertGreater(int(decoded[12, 12, 2]), 230)
            finally:
                cap.release()
        self.assertTrue(all(c.closed for c in FakeCapture.instances))

    def test_reader_is_closed_when_capture_fails(self):
        screen = Screen()
        with mock.patch.object(FakeCapture, 'grab', side_effect=OSError('capture failed')):
            with self.assertRaises(OSError):
                screen.next_frame
        self.assertTrue(all(c.closed for c in FakeCapture.instances))

    def test_preview_stopped_from_another_thread_closes_native_reader(self):
        rendered = threading.Event()
        release_display = threading.Event()

        def display(*args):
            rendered.set()
            if not release_display.wait(5):
                raise TimeoutError('Test display release timed out')

        video = Video('monitor')
        with mock.patch('calango.media.cv2.imshow', side_effect=display), \
                mock.patch('calango.media.cv2.waitKey', return_value=-1):
            try:
                video.show()
                self.assertTrue(rendered.wait(5))
                video.stop()
            finally:
                video.stop()
                release_display.set()
                video._th_show.join(5)
        self.assertFalse(video._th_show.is_alive())
        self.assertFalse(video._th_show_running)
        self.assertIsNone(video._cap._reader)
        self.assertTrue(all(c.closed for c in FakeCapture.instances))
        self.assertTrue(all(c.owner_thread == c.closed_thread for c in FakeCapture.instances))

    def test_closing_frame_generator_releases_reader_on_owner_thread(self):
        video = Video('monitor')
        frames = video.get_frames()
        next(frames)
        self.assertFalse(FakeCapture.instances[-1].closed)
        frames.close()
        self.assertIsNone(video._cap._reader)
        self.assertTrue(all(c.closed for c in FakeCapture.instances))
        self.assertTrue(all(c.owner_thread == c.closed_thread for c in FakeCapture.instances))
        video.stop()

    def test_preprocess_failure_releases_generator_reader(self):
        def preprocess(frame):
            raise ValueError('Invalid preprocessing')

        video = Video('monitor', frame_preprocess=preprocess)
        with self.assertRaisesRegex(ValueError, 'Invalid preprocessing'):
            next(video.get_frames())
        self.assertIsNone(video._cap._reader)
        self.assertTrue(all(c.closed for c in FakeCapture.instances))
        video.stop()


class TestImports(unittest.TestCase):
    def test_import_does_not_load_automation_or_open_capture(self):
        code = "import calango, sys; assert 'pyautogui' not in sys.modules; assert 'mss' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_file_and_frame_sources_do_not_load_windows_automation(self):
        code = '''
from pathlib import Path
import sys
import tempfile
import numpy as np
import cereja
from calango import Video, VideoWriter

original_getattr = cereja.__getattr__
def guard(name):
    if name in ('Window', 'Mouse', 'Keyboard', 'ScreenCapture'):
        raise AssertionError('Unexpected desktop API access: ' + name)
    return original_getattr(name)
cereja.__getattr__ = guard

frames = [np.full((12, 18, 3), value, dtype=np.uint8) for value in (0, 60, 120)]
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'input.mp4'
    with VideoWriter(path, fourcc='mp4v', fps=15) as writer:
        for frame in frames:
            writer.add_frame(frame)
    for source in (str(path), frames):
        video = Video(source)
        try:
            decoded = list(video.get_frames())
            assert len(decoded) == 3
            assert all(frame.shape == (12, 18, 3) for frame in decoded)
        finally:
            video.stop()
assert 'pyautogui' not in sys.modules
assert 'mss' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_versions_remain_consistent(self):
        from calango import VERSION, __version__
        from cereja.utils import get_version_pep440_compliant
        self.assertEqual(get_version_pep440_compliant(VERSION), __version__)
