import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import calango


def _exclusive_windows_status(path):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = create_file(str(path), 0x80000000, 0, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        return ctypes.get_last_error()
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())
    return 0


class TestReadVideoFrames(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.directory) / 'fixture.mp4'
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (16, 12))
        try:
            self.assertTrue(writer.isOpened(), 'OpenCV cannot encode the MP4 fixture')
            rows, columns = np.indices((12, 16), dtype=np.uint8)
            for frame_id in range(3):
                frame = np.stack((5 + rows * 4, 10 + columns * 6, 120 + frame_id * 30 + columns), axis=-1)
                writer.write(frame)
        finally:
            writer.release()

    def test_matches_decoded_rgb_without_losing_first_frame(self):
        expected = self._decoded_frames()
        with calango.read_video_frames(str(self.path)) as frames:
            result = list(frames)
        self.assertEqual([i for i, _ in result], [0, 1, 2])
        for (_, rgb), bgr in zip(result, expected, strict=True):
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(rgb.flags.c_contiguous)
            np.testing.assert_array_equal(rgb, bgr[:, :, [2, 1, 0]])

    def test_odd_center_crop_is_pixel_exact_and_owns_output(self):
        expected = self._decoded_frames()
        with calango.read_video_frames(self.path, crop=(3, 5)) as frames:
            result = list(frames)
        for (_, rgb), bgr in zip(result, expected, strict=True):
            self.assertEqual(rgb.shape, (3, 5, 3))
            self.assertTrue(rgb.flags.c_contiguous)
            np.testing.assert_array_equal(rgb, bgr[5:8, 6:11, [2, 1, 0]])
        result[0][1][:] = 0
        np.testing.assert_array_equal(result[1][1], expected[1][5:8, 6:11, [2, 1, 0]])

    def test_rejects_a_nonlocal_or_unsupported_input(self):
        for path, error in [(0, TypeError), ('https://example.invalid/file.mp4', ValueError),
                            (self.path.with_name('missing.mp4'), FileNotFoundError),
                            (self.path.with_name('folder.mp4'), IsADirectoryError),
                            (self.path.with_suffix('.avi'), ValueError)]:
            if error is IsADirectoryError:
                path.mkdir()
            with self.subTest(path=path), self.assertRaises(error):
                with calango.read_video_frames(path):
                    self.fail('invalid input entered the context')

    def test_invalid_crops_never_yield(self):
        for crop in [(0, 5), (3, 17), (13, 4), (True, 5), (3.0, 5), (3,), [3, 5]]:
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                with calango.read_video_frames(self.path, crop=crop):
                    self.fail('invalid crop entered the context')

    def test_unopenable_mp4_raises_oserror(self):
        bad = self.path.with_name('invalid.mp4')
        bad.write_bytes(b'not an MP4 file')
        with self.assertRaises(OSError):
            with calango.read_video_frames(bad):
                self.fail('unopenable input entered the context')

    def test_open_container_without_first_frame_raises_oserror(self):
        # Exercise an uncommon decoder response independent of encoder versions.
        class EmptyCapture:
            def open(self, path):
                return True

            def read(self):
                return False, None

            def release(self):
                pass

        with patch.object(cv2, 'VideoCapture', EmptyCapture), self.assertRaises(OSError):
            with calango.read_video_frames(self.path):
                self.fail('empty input entered the context')

    def test_exiting_after_break_releases_file_with_iterator_retained(self):
        self.assertFalse(self._file_is_held())
        context = calango.read_video_frames(self.path)
        with context as frames:
            for _, _rgb in frames:
                self.assertTrue(self._file_is_held(), 'no native handle observed while decoding')
                break
            else:
                self.fail('missing first frame')
        self.assertFalse(self._file_is_held())
        self.assertIsNone(next(frames, None))

    def test_consumer_exception_propagates_and_releases_real_file(self):
        self.assertFalse(self._file_is_held())
        marker = RuntimeError('consumer failed')
        context = calango.read_video_frames(self.path)
        with self.assertRaises(RuntimeError) as caught:
            with context as frames:
                next(frames)
                self.assertTrue(self._file_is_held(), 'no native handle observed while decoding')
                raise marker
        self.assertIs(caught.exception, marker)
        self.assertFalse(self._file_is_held())
        self.assertIsNone(next(frames, None))

    def _decoded_frames(self):
        capture = cv2.VideoCapture(str(self.path))
        result = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                result.append(bgr)
        finally:
            capture.release()
        self.assertEqual(len(result), 3)
        return result

    def _file_is_held(self):
        if sys.platform == 'win32':
            status = _exclusive_windows_status(self.path)
            self.assertIn(status, (0, 32), f'Unexpected CreateFileW status {status}')
            return status == 32
        descriptors = Path('/proc/self/fd')
        if not descriptors.is_dir():
            self.skipTest('native file probe is available on Linux and Windows')
        for descriptor in descriptors.iterdir():
            try:
                if os.path.samefile(descriptor, self.path):
                    return True
            except FileNotFoundError:
                continue
        return False
