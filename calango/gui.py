"""Windows screen recorder UI. Importing this module does not initialize Tk."""

import argparse
import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class RecorderController:
    """Coordinate a cancellable countdown and recorder without blocking the UI."""

    def __init__(self, recorder_factory, clock=time.monotonic):
        self._factory = recorder_factory
        self._clock = clock
        self.recorder = None
        self.deadline = None
        self.countdown = None
        self.result = None
        self.error = None
        self.closing = False
        self.completed = False
        self._options = None

    @property
    def busy(self):
        return self.deadline is not None or (self.recorder is not None and not self.completed)

    @property
    def state(self):
        if self.deadline is not None:
            return "countdown"
        if self.error is not None:
            return "failed"
        return self.recorder.state if self.recorder is not None else "idle"

    def start(self, output, **options):
        if self.busy or self.closing:
            raise RuntimeError("A recording is already active or the application is closing.")
        self.recorder = None
        self.result = None
        self.error = None
        self.completed = False
        self._options = (output, options)
        self.deadline = self._clock() + 3.0
        self.countdown = 3

    def stop(self):
        if self.deadline is not None:
            self.deadline = None
            self.countdown = None
            self._options = None
        elif self.recorder is not None and not self.completed:
            self.recorder.stop()

    def close(self):
        self.closing = True
        self.stop()

    def poll(self):
        """Advance the controller. Returns true when the window may close."""
        if self.deadline is not None:
            remaining = self.deadline - self._clock()
            self.countdown = max(0, math.ceil(remaining))
            if remaining <= 0:
                output, options = self._options
                self.deadline = None
                self.countdown = None
                self._options = None
                try:
                    self.recorder = self._factory(output, **options)
                    self.recorder.start()
                except Exception as exc:
                    self.error = exc
                    self.completed = True
        if self.recorder is not None and not self.completed:
            if self.recorder.state in ("finished", "failed"):
                try:
                    self.result = self.recorder.wait(timeout=0)
                except TimeoutError as exc:
                    # The terminal state can become visible just before worker exit.
                    # A capture failure may itself be a TimeoutError; wait raises
                    # that original exception once finalization has completed.
                    if self.recorder.error is exc:
                        self.error = exc
                        self.completed = True
                except Exception as exc:
                    self.error = exc
                    self.completed = True
                else:
                    self.completed = True
        return self.closing and not self.busy


def _region_from_drag(monitor, start, end):
    """Convert local overlay coordinates to a physical desktop rectangle."""
    x1, x2 = sorted(max(0, min(monitor.width, int(x))) for x in (start[0], end[0]))
    y1, y2 = sorted(max(0, min(monitor.height, int(y))) for y in (start[1], end[1]))
    if x1 == x2 or y1 == y2:
        return None
    return monitor.left + x1, monitor.top + y1, x2 - x1, y2 - y1


def _configure_dpi_awareness():
    """Set physical-pixel coordinates before Tk and legacy device imports."""
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
    user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
    user32.SetThreadDpiAwarenessContext.argtypes = [wintypes.HANDLE]
    user32.SetThreadDpiAwarenessContext.restype = wintypes.HANDLE
    if not user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
        error = ctypes.get_last_error()
        if error != 5:  # An embedding host may already have set process awareness.
            raise ctypes.WinError(error)
    previous = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    return user32, previous


def _place_overlay(window, monitor):
    """Place a Tk toplevel at signed desktop coordinates, including left monitors."""
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    window.update_idletasks()
    window_id = window.winfo_id()
    handle = user32.GetParent(window_id) or window_id
    if not user32.SetWindowPos(
        handle, ctypes.c_void_p(-1), monitor.left, monitor.top, monitor.width, monitor.height, 0x0040
    ):
        raise ctypes.WinError(ctypes.get_last_error())


@dataclass(frozen=True)
class _WindowChoice:
    hwnd: int
    title: str
    width: int
    height: int


def _list_recordable_windows(excluded_handles=()):
    """Read Cereja window metadata without activating or capturing any window."""
    from cereja import Window

    choices = []
    seen = set(excluded_handles)
    for window in Window.get_all_windows():
        try:
            handle = getattr(window.hwnd, "value", window.hwnd)
            if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0 or handle in seen:
                continue
            title = window.title.strip()
            if not title or title == "UNKNOW" or not window.is_visible or window.state == "Minimized":
                continue
            width, height = window.size_window_content
            if width <= 0 or height <= 0:
                continue
            choices.append(_WindowChoice(handle, title, width, height))
            seen.add(handle)
        except (AttributeError, OSError, ValueError):
            # A window can close between enumeration and reading its metadata.
            continue
    return tuple(choices)


class RegionSelector:
    """A temporary transparent overlay on the selected monitor."""

    def __init__(self, root, monitor, on_selected):
        import tkinter as tk

        self.root = root
        self.monitor = monitor
        self.on_selected = on_selected
        self.start = None
        self.rectangle = None
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.35)
        self.window.geometry(f"{monitor.width}x{monitor.height}+0+0")
        self.canvas = tk.Canvas(self.window, background="black", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(24, 24, anchor="nw", fill="white", font=("Segoe UI", 18),
                                text="Drag to select a region. Press Esc to cancel.")
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.window.bind("<Escape>", lambda event: self._finish(None))
        self.window.protocol("WM_DELETE_WINDOW", lambda: self._finish(None))
        root.withdraw()
        try:
            self.window.deiconify()
            _place_overlay(self.window, monitor)
            self.window.grab_set()
            self.window.focus_force()
        except Exception:
            self.window.destroy()
            root.deiconify()
            raise

    def _press(self, event):
        self.start = (event.x, event.y)
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                      outline="#00ffcc", width=3)

    def _move(self, event):
        if self.start is not None:
            self.canvas.coords(self.rectangle, *self.start, event.x, event.y)

    def _release(self, event):
        if self.start is not None:
            region = _region_from_drag(self.monitor, self.start, (event.x, event.y))
            if region is not None:
                self._finish(region)

    def _finish(self, region):
        self.window.grab_release()
        self.window.destroy()
        self.root.deiconify()
        self.root.lift()
        self.on_selected(region)


class RecorderApp:
    """Tk widgets for the recorder. All recorder work stays off the Tk thread."""

    def __init__(self, root, recorder_factory=None, capture_factory=None, keyboard=None, window_provider=None):
        import tkinter as tk
        from tkinter import ttk

        if recorder_factory is None:
            from .recorder import ScreenRecorder

            recorder_factory = ScreenRecorder
        if capture_factory is None:
            try:
                from cereja import ScreenCapture
            except ImportError as exc:
                raise RuntimeError(
                    "Install the Cereja checkout containing ScreenCapture before starting the recorder."
                ) from exc
            capture_factory = ScreenCapture
        with capture_factory(include_cursor=False) as capture:
            self.monitors = capture.list_monitors()
        if not self.monitors:
            raise RuntimeError("No monitors are available for capture.")
        self.root = root
        self.controller = RecorderController(recorder_factory)
        self.keyboard = keyboard
        self.region = None
        self.selector = None
        self._hotkey_down = False
        self._reported_error = None
        self._last_busy = None
        self._config_widgets = []
        self._window_provider = window_provider or _list_recordable_windows
        self.windows = ()
        self._window_numbers = {}
        self._last_source = "monitor"
        self._desktop_cursor = True
        self.root.title("Calango Screen Recorder")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        body = ttk.Frame(root, padding=20)
        body.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        ttk.Label(body, text="Screen Recorder", font=("Segoe UI", 18, "bold")).grid(sticky="w")
        ttk.Label(body, text="Record a monitor, region, or window to MP4. Audio is not recorded.").grid(
            sticky="w", pady=(4, 18)
        )
        ttk.Label(body, text="Monitor").grid(sticky="w")
        labels = [f"{m.name} ({m.width} x {m.height})" + (" - Primary" if m.is_primary else "")
                  for m in self.monitors]
        self.monitor_box = ttk.Combobox(body, state="readonly", values=labels)
        self.monitor_box.grid(sticky="ew", pady=(4, 8))
        self.monitor_box.current(next((i for i, m in enumerate(self.monitors) if m.is_primary), 0))
        self.monitor_box.bind("<<ComboboxSelected>>", self._monitor_changed)
        self._config_widgets.append((self.monitor_box, "readonly"))
        source = ttk.Frame(body)
        source.grid(sticky="ew", pady=(0, 4))
        self.source = tk.StringVar(value="monitor")
        full = ttk.Radiobutton(source, text="Entire monitor", variable=self.source, value="monitor",
                               command=self._source_changed)
        full.pack(side="left")
        region = ttk.Radiobutton(
            source, text="Region", variable=self.source, value="region", command=self._source_changed
        )
        region.pack(side="left", padx=(16, 8))
        window_source = ttk.Radiobutton(
            source, text="Window", variable=self.source, value="window", command=self._source_changed
        )
        window_source.pack(side="left", padx=(8, 16))
        self.select_region_button = ttk.Button(source, text="Select region...", command=self._select_region)
        self.select_region_button.pack(side="left")
        self._config_widgets.extend((widget, "normal") for widget in (full, region, window_source, self.select_region_button))
        self.region_text = tk.StringVar(value="No region selected")
        ttk.Label(body, textvariable=self.region_text).grid(sticky="w", pady=(0, 8))
        ttk.Label(body, text="Window (background capture)").grid(sticky="w")
        window_row = ttk.Frame(body)
        window_row.grid(sticky="ew", pady=(4, 4))
        window_row.columnconfigure(0, weight=1)
        self.window_box = ttk.Combobox(window_row, state="disabled")
        self.window_box.grid(row=0, column=0, sticky="ew")
        self.refresh_windows_button = ttk.Button(window_row, text="Refresh", command=self._refresh_windows)
        self.refresh_windows_button.grid(row=0, column=1, padx=(8, 0))
        self.window_status = tk.StringVar(value="Choose a window explicitly when using Window mode.")
        ttk.Label(body, textvariable=self.window_status, wraplength=520).grid(sticky="w", pady=(0, 12))
        ttk.Label(body, text="Save video to").grid(sticky="w")
        destination = ttk.Frame(body)
        destination.grid(sticky="ew", pady=(4, 12))
        destination.columnconfigure(0, weight=1)
        videos = Path.home() / "Videos"
        folder = videos if videos.is_dir() else Path.home()
        self.output = tk.StringVar(value=str(folder / f"calango-{datetime.now():%Y%m%d-%H%M%S}.mp4"))
        entry = ttk.Entry(destination, textvariable=self.output)
        entry.grid(row=0, column=0, sticky="ew")
        browse = ttk.Button(destination, text="Browse...", command=self._browse)
        browse.grid(row=0, column=1, padx=(8, 0))
        self._config_widgets.extend(((entry, "normal"), (browse, "normal")))
        options = ttk.Frame(body)
        options.grid(sticky="ew", pady=(0, 18))
        ttk.Label(options, text="FPS").pack(side="left")
        self.fps = tk.StringVar(value="30")
        fps_box = ttk.Combobox(options, textvariable=self.fps, values=(15, 30), state="readonly", width=5)
        fps_box.pack(side="left", padx=(8, 20))
        self.include_cursor = tk.BooleanVar(value=True)
        self.cursor_checkbox = ttk.Checkbutton(options, text="Include cursor", variable=self.include_cursor)
        self.cursor_checkbox.pack(side="left")
        self._config_widgets.extend(((fps_box, "readonly"), (self.cursor_checkbox, "normal")))
        actions = ttk.Frame(body)
        actions.grid(sticky="ew")
        self.start_button = ttk.Button(actions, text="Start recording", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop", command=self.controller.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.timer = tk.StringVar(value="00:00")
        ttk.Label(actions, textvariable=self.timer, font=("Segoe UI", 14)).pack(side="right")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(body, textvariable=self.status, wraplength=520).grid(sticky="w", pady=(12, 4))
        self.shortcut = tk.StringVar(value="Stop shortcut: Ctrl+Shift+F9" if keyboard else "Stop shortcut unavailable")
        ttk.Label(body, textvariable=self.shortcut).grid(sticky="w")
        self.capture_hint = tk.StringVar()
        ttk.Label(body, textvariable=self.capture_hint, wraplength=520).grid(sticky="w", pady=(4, 12))
        completed = ttk.Frame(body)
        completed.grid(sticky="ew")
        self.open_video = ttk.Button(completed, text="Open video", command=lambda: self._open_result(False), state="disabled")
        self.open_video.pack(side="left")
        self.open_folder = ttk.Button(completed, text="Open folder", command=lambda: self._open_result(True), state="disabled")
        self.open_folder.pack(side="left", padx=8)
        self._refresh_windows()
        self._source_changed()
        self.root.update_idletasks()
        self.root.minsize(max(540, body.winfo_reqwidth()), max(380, body.winfo_reqheight()))
        self.root.after(25, self._poll)

    def _monitor_changed(self, event=None):
        self.region = None
        self.source.set("monitor")
        self.region_text.set("No region selected")
        self._source_changed()

    def _source_changed(self):
        source = self.source.get()
        if source == "window" and self._last_source != "window":
            self._desktop_cursor = self.include_cursor.get()
            self.include_cursor.set(False)
        elif source != "window" and self._last_source == "window":
            self.include_cursor.set(self._desktop_cursor)
        self._last_source = source
        self.capture_hint.set(
            "Records only the selected window, even when covered. Keep its size unchanged and do not minimize it."
            if source == "window" else
            "You may minimize this recorder. Visible recorder controls appear in monitor and region videos."
        )
        self._apply_configuration_state()

    def _refresh_windows(self):
        if self.controller.busy:
            return
        index = self.window_box.current()
        selected = self.windows[index].hwnd if 0 <= index < len(self.windows) else None
        excluded = {self.root.winfo_id()}
        try:
            excluded.add(int(self.root.wm_frame(), 0))
            self.windows = tuple(self._window_provider(excluded))
        except Exception:
            self.windows = ()
            self.window_status.set("Window list unavailable. Click Refresh to retry.")
        else:
            self.window_status.set(
                "Choose a window. Cursor is unavailable for window capture." if self.windows else
                "No recordable windows. Open or restore a window, then click Refresh."
            )
        labels = []
        descriptions = [(item.title, item.width, item.height) for item in self.windows]
        for item in self.windows:
            label = f"{item.title} ({item.width} x {item.height})"
            if descriptions.count((item.title, item.width, item.height)) > 1:
                number = self._window_numbers.setdefault(item.hwnd, len(self._window_numbers) + 1)
                label += f" [Window {number}]"
            labels.append(label)
        self.window_box.configure(values=labels)
        self.window_box.set("")
        for index, item in enumerate(self.windows):
            if item.hwnd == selected:
                self.window_box.current(index)
                break
        self._apply_configuration_state()

    def _apply_configuration_state(self):
        disabled = self.controller.busy or self.controller.closing
        is_window = self.source.get() == "window"
        for widget, enabled_state in self._config_widgets:
            widget.configure(state="disabled" if disabled else enabled_state)
        self.monitor_box.configure(state="disabled" if disabled or is_window else "readonly")
        self.select_region_button.configure(state="disabled" if disabled or is_window else "normal")
        self.cursor_checkbox.configure(state="disabled" if disabled or is_window else "normal")
        self.window_box.configure(state="readonly" if not disabled and is_window and self.windows else "disabled")
        self.refresh_windows_button.configure(state="normal" if not disabled and is_window else "disabled")
        self.start_button.configure(state="disabled" if disabled else "normal")

    def _select_region(self):
        from tkinter import messagebox

        if self.controller.busy or self.selector is not None:
            return
        try:
            self.selector = RegionSelector(self.root, self.monitors[self.monitor_box.current()], self._region_selected)
        except Exception as exc:
            messagebox.showerror("Region selection failed", str(exc), parent=self.root)

    def _region_selected(self, region):
        self.selector = None
        if region is not None:
            self.region = region
            self.source.set("region")
            left, top, width, height = region
            self.region_text.set(f"{width} x {height} pixels at ({left}, {top})")
            self._source_changed()

    def _browse(self):
        from tkinter import filedialog

        current = Path(self.output.get()).expanduser()
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save recording", defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")], initialdir=str(current.parent),
            initialfile=current.name, confirmoverwrite=False
        )
        if path:
            self.output.set(path)

    def _start(self):
        from tkinter import messagebox

        try:
            output_text = self.output.get().strip()
            if not output_text:
                raise ValueError("Choose an output file.")
            output = Path(output_text).expanduser().absolute()
            if output.suffix.lower() != ".mp4":
                raise ValueError("The output filename must end in .mp4.")
            if output.exists():
                raise ValueError("The output file already exists. Choose a new filename.")
            if not output.parent.is_dir():
                raise ValueError("The output folder does not exist.")
            options = {"fps": int(self.fps.get()), "include_cursor": self.include_cursor.get()}
            if self.source.get() == "window":
                index = self.window_box.current()
                if not 0 <= index < len(self.windows):
                    raise ValueError("Choose a window before starting. Use Refresh if the window is not listed.")
                options["window"] = self.windows[index].hwnd
                options["include_cursor"] = False
            elif self.source.get() == "region":
                if self.region is None:
                    raise ValueError("Select a region before starting.")
                options["region"] = self.region
            else:
                options["monitor"] = self.monitors[self.monitor_box.current()].id
            self.controller.start(output, **options)
            self._reported_error = None
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Cannot start recording", str(exc), parent=self.root)

    def _close(self):
        if self.selector is not None:
            self.selector._finish(None)
        self.controller.close()
        self._refresh()

    def _open_result(self, folder):
        from tkinter import messagebox

        if self.controller.result is not None:
            path = Path(self.controller.result)
            try:
                os.startfile(str(path.parent if folder else path))
            except OSError as exc:
                messagebox.showerror("Cannot open recording", str(exc), parent=self.root)

    def _refresh(self):
        busy = self.controller.busy
        if busy != self._last_busy:
            self._apply_configuration_state()
            self._last_busy = busy
        self.stop_button.configure(state="normal" if busy else "disabled",
                                   text="Cancel countdown" if self.controller.deadline is not None else "Stop")
        state = self.controller.state
        if state == "countdown":
            status = f"Recording starts in {self.controller.countdown}..."
        elif state == "finished":
            status = "Saved successfully" if self.controller.result is not None else "Stopped before the first frame"
        elif state == "failed":
            status = f"Recording failed: {self.controller.error}"
        else:
            status = {"idle": "Ready", "starting": "Starting...", "recording": "Recording",
                      "stopping": "Finalizing video..."}.get(state, state)
        if self.controller.closing and busy:
            status = "Finalizing video before closing..."
        self.status.set(status)
        elapsed = self.controller.recorder.elapsed if self.controller.recorder is not None else 0
        seconds = max(0, int(elapsed))
        self.timer.set(f"{seconds // 60:02d}:{seconds % 60:02d}")
        saved = self.controller.result is not None and not busy
        self.open_video.configure(state="normal" if saved else "disabled")
        self.open_folder.configure(state="normal" if saved else "disabled")

    def _poll(self):
        from tkinter import messagebox

        if self.keyboard is not None:
            try:
                pressed = self.keyboard.is_pressed("CTRL+SHIFT+F9")
                if pressed and not self._hotkey_down:
                    self.controller.stop()
                self._hotkey_down = pressed
            except Exception:
                self.keyboard = None
                self.shortcut.set("Stop shortcut unavailable. Use the Stop button.")
        may_close = self.controller.poll()
        self._refresh()
        if self.controller.error is not None and self.controller.error is not self._reported_error:
            self._reported_error = self.controller.error
            message = "\n".join([str(self.controller.error), *getattr(self.controller.error, "__notes__", ())])
            messagebox.showerror("Recording failed", message, parent=self.root)
        if may_close:
            self.root.destroy()
            return
        self.root.after(25, self._poll)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calango screen recorder for Windows 10/11 (MP4 video, no audio).")
    parser.parse_args(argv)
    if sys.platform != "win32":
        parser.exit(1, "The Calango recorder interface requires Windows 10 or Windows 11.\n")
    root = None
    dpi = None
    try:
        dpi = _configure_dpi_awareness()
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            from cereja import Keyboard

            keyboard = Keyboard(hwnd=None)
        except (ImportError, AttributeError, OSError):
            keyboard = None
        RecorderApp(root, keyboard=keyboard)
        root.deiconify()
        root.mainloop()
    except Exception as exc:
        if root is not None:
            from tkinter import messagebox

            messagebox.showerror("Calango recorder", str(exc), parent=root)
            root.destroy()
        else:
            print(f"Cannot open Calango recorder: {exc}", file=sys.stderr)
        return 1
    finally:
        if dpi is not None:
            dpi[0].SetThreadDpiAwarenessContext(dpi[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
