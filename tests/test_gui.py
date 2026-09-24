"""Controller and UI event tests that do not require a display or camera."""

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from calango.gui import (
    RecorderApp, RecorderController, _WindowChoice, _list_recordable_windows, _place_overlay, _region_from_drag, main,
)


class FakeRecorder:
    def __init__(self, output, **options):
        self.output = output
        self.options = options
        self.state = "idle"
        self.elapsed = 0
        self.error = None
        self.stops = 0
        self.waits = []
        self.exit_pending = False

    def start(self):
        self.state = "recording"
        return self

    def stop(self):
        self.stops += 1
        self.state = "stopping"

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.exit_pending:
            raise TimeoutError()
        if self.error is not None:
            raise self.error
        return self.output


class TestRecorderController(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.created = []

        def factory(output, **options):
            recorder = FakeRecorder(output, **options)
            self.created.append(recorder)
            return recorder

        self.controller = RecorderController(factory, clock=lambda: self.now)

    def begin(self):
        self.controller.start(Path("capture.mp4"), fps=15, region=(-10, 20, 8, 6), include_cursor=True)
        self.now += 3
        self.controller.poll()
        return self.created[-1]

    def test_countdown_starts_only_at_deadline_and_passes_configuration(self):
        self.controller.start(Path("capture.mp4"), fps=15, monitor="screen-1")
        self.assertTrue(self.controller.busy)
        self.assertEqual(self.controller.state, "countdown")
        self.now += 1.1
        self.controller.poll()
        self.assertEqual(self.controller.countdown, 2)
        self.assertEqual(self.created, [])
        self.now += 1.9
        self.controller.poll()
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.created[0].options, {"fps": 15, "monitor": "screen-1"})
        self.assertEqual(self.controller.state, "recording")

    def test_cancel_countdown_never_creates_recorder(self):
        self.controller.start("capture.mp4")
        self.controller.stop()
        self.now += 10
        self.controller.poll()
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.state, "idle")
        self.assertEqual(self.created, [])

    def test_close_during_countdown_is_immediate(self):
        self.controller.start("capture.mp4")
        self.controller.close()
        self.assertTrue(self.controller.poll())
        self.assertEqual(self.created, [])

    def test_close_requests_stop_and_waits_nonblocking_for_worker_exit(self):
        recorder = self.begin()
        self.controller.close()
        self.assertEqual(recorder.stops, 1)
        self.assertFalse(self.controller.poll())
        self.assertEqual(recorder.waits, [])
        recorder.state = "finished"
        recorder.exit_pending = True
        self.assertFalse(self.controller.poll())
        self.assertEqual(recorder.waits, [0])
        recorder.exit_pending = False
        self.assertTrue(self.controller.poll())
        self.assertEqual(self.controller.result, Path("capture.mp4"))

    def test_failed_recorder_propagates_error_for_ui(self):
        recorder = self.begin()
        recorder.error = OSError("Encoder unavailable")
        recorder.state = "failed"
        self.controller.poll()
        self.assertIs(self.controller.error, recorder.error)
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.state, "failed")

    def test_capture_timeout_is_not_confused_with_worker_still_finalizing(self):
        recorder = self.begin()
        recorder.error = TimeoutError("Capture backend timed out")
        recorder.state = "failed"
        self.controller.close()
        # A completed worker stays failed when stop is requested.
        recorder.state = "failed"
        self.assertTrue(self.controller.poll())
        self.assertIs(self.controller.error, recorder.error)
        self.assertFalse(self.controller.busy)

    def test_synchronous_initialization_failure_does_not_leave_ui_busy(self):
        self.controller._factory = Mock(side_effect=OSError("Capture unavailable"))
        self.controller.start("capture.mp4")
        self.now += 3
        self.controller.poll()
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.state, "failed")
        self.assertIsInstance(self.controller.error, OSError)

    def test_start_failure_does_not_leave_ui_busy(self):
        recorder = FakeRecorder("capture.mp4")
        recorder.start = Mock(side_effect=RuntimeError("Thread failed"))
        self.controller._factory = Mock(return_value=recorder)
        self.controller.start("capture.mp4")
        self.now += 3
        self.controller.poll()
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.state, "failed")

    def test_new_recording_after_completion_uses_new_instance(self):
        previous = self.begin()
        previous.state = "finished"
        self.controller.poll()
        self.controller.start("second.mp4")
        self.now += 3
        self.controller.poll()
        self.assertIsNot(previous, self.controller.recorder)
        self.assertEqual(len(self.created), 2)

    def test_start_is_rejected_while_active_and_while_closing(self):
        self.controller.start("capture.mp4")
        with self.assertRaises(RuntimeError):
            self.controller.start("other.mp4")
        self.controller.close()
        with self.assertRaises(RuntimeError):
            self.controller.start("other.mp4")

    def test_no_first_frame_does_not_offer_a_saved_path(self):
        recorder = self.begin()
        recorder.output = None
        recorder.state = "finished"
        self.controller.poll()
        self.assertIsNone(self.controller.result)
        self.assertTrue(self.controller.completed)


class TestRegionCoordinates(unittest.TestCase):
    def setUp(self):
        self.monitor = SimpleNamespace(left=-1920, top=-200, width=1920, height=1080)

    def test_negative_desktop_origin_and_reverse_drag(self):
        self.assertEqual(_region_from_drag(self.monitor, (800, 700), (100, 200)), (-1820, 0, 700, 500))

    def test_drag_clamps_to_selected_monitor(self):
        self.assertEqual(_region_from_drag(self.monitor, (-30, -50), (2500, 1400)), (-1920, -200, 1920, 1080))

    def test_empty_selection_is_ignored(self):
        self.assertIsNone(_region_from_drag(self.monitor, (20, 30), (20, 200)))
        self.assertIsNone(_region_from_drag(self.monitor, (20, 30), (200, 30)))

    def test_overlay_uses_signed_physical_coordinates_for_native_window(self):
        user32 = Mock()
        user32.GetParent.return_value = 202
        user32.SetWindowPos.return_value = True
        window = Mock()
        window.winfo_id.return_value = 101
        with patch("calango.gui.ctypes.WinDLL", create=True, return_value=user32):
            _place_overlay(window, self.monitor)
        arguments = user32.SetWindowPos.call_args.args
        self.assertEqual(arguments[0], 202)
        self.assertEqual(arguments[2:6], (-1920, -200, 1920, 1080))


class TestGuiEvents(unittest.TestCase):
    def make_app(self):
        app = RecorderApp.__new__(RecorderApp)
        app.root = Mock()
        app.controller = Mock()
        app.controller.poll.return_value = False
        app.controller.error = None
        app.controller.busy = False
        app.keyboard = Mock()
        app.keyboard.is_pressed.return_value = False
        app.shortcut = Mock()
        app._hotkey_down = False
        app._reported_error = None
        app._refresh = Mock()
        app.selector = None
        return app

    def test_shortcut_stops_once_per_press_even_while_window_is_minimized(self):
        app = self.make_app()
        app.keyboard.is_pressed.side_effect = [True, True, False, True]
        for _ in range(4):
            app._poll()
        self.assertEqual(app.controller.stop.call_count, 2)
        app.keyboard.is_pressed.assert_called_with("CTRL+SHIFT+F9")
        app.root.after.assert_called_with(25, app._poll)

    def test_shortcut_failure_preserves_stop_button_and_polling(self):
        app = self.make_app()
        app.keyboard.is_pressed.side_effect = OSError("Keyboard unavailable")
        app._poll()
        self.assertIsNone(app.keyboard)
        app._refresh.assert_called_once()
        app.root.after.assert_called_once_with(25, app._poll)

    def test_window_only_destroyed_when_controller_reports_safe_close(self):
        app = self.make_app()
        app.controller.poll.return_value = True
        app._poll()
        app.root.destroy.assert_called_once()
        app.root.after.assert_not_called()

    def test_close_requests_stop_without_waiting_in_tk_handler(self):
        app = self.make_app()
        app._close()
        app.controller.close.assert_called_once()
        app.controller.wait.assert_not_called()
        app.root.destroy.assert_not_called()

    def test_failure_on_close_reports_preserved_partial_before_destroying_window(self):
        app = self.make_app()
        app.controller.error = OSError("Encoder failed")
        app.controller.error.add_note("Partial recording retained at: capture.partial.mp4")
        app.controller.poll.return_value = True
        with patch("tkinter.messagebox.showerror") as message:
            app._poll()
        self.assertIn("capture.partial.mp4", message.call_args.args[1])
        app.root.destroy.assert_called_once()

    def test_existing_output_is_rejected_before_countdown(self):
        app = self.make_app()
        app.output = Mock()
        app.output.get.return_value = "existing.mp4"
        with patch("calango.gui.Path.exists", return_value=True), patch("tkinter.messagebox.showerror") as error:
            app._start()
        app.controller.start.assert_not_called()
        error.assert_called_once()
        self.assertIn("already exists", error.call_args.args[1])

    def test_help_does_not_initialize_dpi_or_tk(self):
        with patch("calango.gui._configure_dpi_awareness") as dpi, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        dpi.assert_not_called()


class TestWindowSelection(unittest.TestCase):
    @staticmethod
    def variable(value):
        storage = [value]
        return SimpleNamespace(get=lambda: storage[0], set=lambda new: storage.__setitem__(0, new))

    def make_app(self):
        app = RecorderApp.__new__(RecorderApp)
        app.root = Mock()
        app.root.winfo_id.return_value = 101
        app.root.wm_frame.return_value = "0x100"
        app.controller = Mock()
        app.controller.busy = False
        app.controller.closing = False
        app.windows = ()
        app._window_numbers = {}
        app.source = self.variable("window")
        app.include_cursor = self.variable(True)
        app.output = self.variable("capture.mp4")
        app.fps = self.variable("30")
        app._last_source = "monitor"
        app._desktop_cursor = True
        app._config_widgets = [(Mock(), "normal")]
        app.capture_hint = Mock()
        app.window_status = Mock()
        app.window_box = Mock()
        app.window_box.current.return_value = -1
        app.monitor_box = Mock()
        app.select_region_button = Mock()
        app.cursor_checkbox = Mock()
        app.refresh_windows_button = Mock()
        app.start_button = Mock()
        app._refresh = Mock()
        return app

    def test_listing_filters_unusable_and_own_windows_without_activating_them(self):
        def window(handle, **overrides):
            data = dict(hwnd=handle, title="Editor", is_visible=True, state="Normal", size_window_content=(640, 480))
            data.update(overrides)
            return SimpleNamespace(**data)

        windows = [window(100), window(200), window(300, title=""), window(400, title="UNKNOW"),
                   window(500, state="Minimized"), window(600, size_window_content=(0, 200)),
                   window(700, is_visible=False), window(800, title="Editor"), window(200)]
        provider = SimpleNamespace(get_all_windows=Mock(return_value=windows))
        with patch.dict("sys.modules", {"cereja": SimpleNamespace(Window=provider)}):
            choices = _list_recordable_windows({100})
        self.assertEqual([item.hwnd for item in choices], [200, 800])
        self.assertEqual([item.title for item in choices], ["Editor", "Editor"])
        provider.get_all_windows.assert_called_once_with()

    def test_refresh_preserves_handle_when_same_titles_change_order(self):
        app = self.make_app()
        first = _WindowChoice(200, "Editor", 640, 480)
        second = _WindowChoice(800, "Editor", 640, 480)
        app.windows = (first, second)
        app.window_box.current.return_value = 1
        app._window_provider = Mock(return_value=(second, first))
        app._refresh_windows()
        app._window_provider.assert_called_once_with({101, 256})
        app.window_box.current.assert_called_with(0)
        app.window_box.set.assert_called_once_with("")
        app.window_box.configure.assert_any_call(values=["Editor (640 x 480) [Window 1]", "Editor (640 x 480) [Window 2]"])
        app._window_provider.return_value = (first, second)
        app._refresh_windows()
        app.window_box.configure.assert_any_call(values=["Editor (640 x 480) [Window 2]", "Editor (640 x 480) [Window 1]"])

    def test_refresh_never_selects_replacement_for_closed_target(self):
        app = self.make_app()
        app.windows = (_WindowChoice(200, "Editor", 640, 480),)
        app.window_box.current.return_value = 0
        app._window_provider = Mock(return_value=(_WindowChoice(800, "Editor", 640, 480),))
        app._refresh_windows()
        self.assertEqual(app.window_box.current.call_args_list, [unittest.mock.call()])
        app.window_box.set.assert_called_once_with("")

    def test_empty_window_list_clears_selection_and_explains_recovery(self):
        app = self.make_app()
        app._window_provider = Mock(return_value=())
        app._refresh_windows()
        app.window_box.configure.assert_any_call(values=[])
        app.window_box.configure.assert_any_call(state="disabled")
        self.assertIn("Open or restore", app.window_status.set.call_args.args[0])

    def test_window_target_passes_only_selected_handle_without_cursor(self):
        app = self.make_app()
        app.windows = (_WindowChoice(200, "Editor", 640, 480), _WindowChoice(800, "Editor", 640, 480))
        app.window_box.current.return_value = 1
        with patch("calango.gui.Path.exists", return_value=False), patch("calango.gui.Path.is_dir", return_value=True):
            app._start()
        app.controller.start.assert_called_once_with(Path("capture.mp4").absolute(), fps=30,
                                                     include_cursor=False, window=800)

    def test_window_target_requires_explicit_selection(self):
        app = self.make_app()
        app.windows = (_WindowChoice(200, "Editor", 640, 480),)
        with patch("calango.gui.Path.exists", return_value=False), patch("calango.gui.Path.is_dir", return_value=True), \
                patch("tkinter.messagebox.showerror") as error:
            app._start()
        app.controller.start.assert_not_called()
        self.assertIn("Choose a window", error.call_args.args[1])

    def test_window_mode_disables_cursor_and_restores_desktop_preference(self):
        app = self.make_app()
        app._source_changed()
        self.assertFalse(app.include_cursor.get())
        app.cursor_checkbox.configure.assert_called_with(state="disabled")
        app.monitor_box.configure.assert_called_with(state="disabled")
        app.source.set("monitor")
        app._source_changed()
        self.assertTrue(app.include_cursor.get())
        app.cursor_checkbox.configure.assert_called_with(state="normal")

    def test_recording_locks_window_selection_and_refresh(self):
        app = self.make_app()
        app.windows = (_WindowChoice(200, "Editor", 640, 480),)
        app.controller.busy = True
        app._apply_configuration_state()
        app.window_box.configure.assert_called_with(state="disabled")
        app.refresh_windows_button.configure.assert_called_with(state="disabled")
        app._window_provider = Mock()
        app._refresh_windows()
        app._window_provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
