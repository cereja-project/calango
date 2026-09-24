# Screen recorder

The recorder supports Windows 10 version 1703 or newer, Windows 11 and Python
3.11+. It records one monitor, a region inside one monitor, or the client area
of a selected window to a silent MP4 file. Desktop capture supports an optional
cursor. Tkinter supplies the desktop interface; Cereja
supplies native capture without third-party dependencies; OpenCV encodes video.

## Installation

The recorder is introduced in Calango 2.3.0 and requires Cereja 2.3.0 or newer.
Once these versions are published, pip installs the required capture dependency:

```powershell
python -m pip install --upgrade "calango>=2.3.0"
calango-recorder
```

Alternatively, run `python -m calango.gui` in the same environment. Both commands
accept `--help` without opening a window. The package declares `cereja>=2.3.0`
so a normal installation cannot silently resolve to an older capture backend.
An incompatible environment assembled with dependency checks disabled produces
an explicit missing-capability error when recording is requested.

## Development installation

If the required releases are not yet available on PyPI, install both source
checkouts into one virtual environment. Use a Cereja checkout containing
`ScreenCapture(window=...)` and declaring version 2.3.0 or newer. From the
directory containing the Cereja and Calango checkouts:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ./cereja -e ./calango
.venv\Scripts\calango-recorder
```

Use a Python installation with Tcl/Tk support. A Tcl initialization failure is
an environment error; check the interpreter's Tk installation and access to its
Tcl library. No changes to a global Python installation are required.

## Desktop workflow

1. Choose a monitor, region, or window. For a region, use **Select region...** and
   drag within the selected monitor; Escape cancels selection. For a window,
   select its title from the available windows and refresh the list if needed.
2. Choose a new `.mp4` output path and 15 or 30 FPS. Monitor and region recordings
   can include the cursor; window recordings exclude the desktop cursor.
3. Start recording. The three-second countdown can be cancelled.
4. Stop with the button or Ctrl+Shift+F9. The shortcut is polled while the app is
   running; it does not reserve the key combination and very brief presses may
   be missed. The button remains available.
5. Wait for finalization and verification, then open the video or its folder.

Configuration is locked during recording. The recorder window may be minimized. Visible
recorder controls are part of monitor/region recordings and are not automatically
excluded. A window recording captures only the selected application's client area.
Closing the application stops and finalizes the recording before the
window disappears. Failures remain visible and identify any preserved partial
file.

## Python API

```python
import time
from calango import ScreenRecorder

recorder = ScreenRecorder("demo.mp4", fps=30, include_cursor=True).start()
try:
    time.sleep(10)
finally:
    recorder.stop()
output = recorder.wait()
```

Pass `monitor` as a Cereja monitor id or `ScreenMonitor`,
`region=(left, top, width, height)` in physical desktop pixels, or `window` as a
positive integer HWND (for example, `window=selected_window.hwnd` from a Cereja
`Window`). The three selectors are mutually exclusive. Omitting all selects the
primary monitor. Coordinates
can be negative; each region must fit inside a single currently attached monitor.

A recorder represents one session with immutable configuration. `start()` is
asynchronous and returns the recorder; it cannot be repeated. `stop()` is
idempotent and does not block. `wait(timeout=None)` returns the output `Path`
only after closing and decoding the complete video for verification. It raises
`TimeoutError` if still running, or the original capture/encoding error after
failure. A stop before the first frame returns `None` and creates no recording.

Properties are `state`, `error`, `elapsed`, `frames_written` and `partial_path`.
States are `idle`, `starting`, `recording`, `stopping`, `finished` and `failed`.
Use the same API from any frontend, keeping `wait()` off the GUI thread unless
using a zero timeout.

## Window recording and privacy

Window mode captures the selected window's client area through Cereja's native
`PrintWindow` path. The window can remain behind another application without
receiving focus. The recorder never substitutes a screenshot of its desktop
rectangle, even when window capture fails. A desktop cursor can belong to an
overlapping application, so `include_cursor` is always false in window mode.

Keep the target window open, unminimized and at its initial size. Moving it does
not change the selected target. Closing, minimizing or resizing it interrupts the
recording with an error and preserves any nonempty partial video. Refreshing the
GUI's list does not automatically select a replacement for a closed window.

Capture support depends on the target application's rendering. Some applications
may return blank content; check a short sample when recording an application for
the first time. `PrintWindow` is synchronous, so an unresponsive target may delay
completion of an in-flight capture and the stop request.

The same Cereja capture core supplies `ScreenCapture` and
`Window.capture_image_bmp()`, sharing native bindings, DPI handling, pixel buffers
and cleanup. Calango's recorder only controls timing, encoding and output
verification. Existing `Window.mouse` and `Window.keyboard` interactions remain
directed at the target window.

## Output and timing

The default is 30 FPS using OpenCV's `mp4v` encoder. Frames retain their capture
size, with at most one duplicated edge pixel on the right/bottom to make odd
dimensions even. FPS is the output cadence, not a guarantee of equally frequent
new screenshots. Absolute monotonic deadlines prevent accumulated sleep drift;
missed slots repeat the previous frame. More than two seconds of accumulated
lag produces an explicit performance error. Memory use is bounded by frame size,
not recording length.

Existing destinations are never overwritten, including files created while a
recording is in progress. Encoding uses a sibling `.partial.mp4` file. Successful
verification publishes it without replacement; errors preserve nonempty partial
output. A partial file may require repair after interrupted encoding. Forced
process termination and power loss cannot guarantee a valid MP4.

The legacy `Video('monitor', fps=...)` uses Cereja capture on Windows and shares
the timed frame writer for `save()`. `frame_preprocess` still applies, and
`n_frames` limits encoded frames, including repeated frames. The legacy API
retains its existing destination and asynchronous-return behavior; use
`ScreenRecorder` for verified publication, progress and failure reporting.
Preview and saving are not supported simultaneously in the legacy `Video` API.

## Boundaries

This version has no audio, webcam overlay, editing, pause, continuous preview,
executable installer or GUI support on Linux/macOS. Existing non-Windows monitor
capture retains its optional MSS path. Cereja capture follows the visible desktop;
protected media, secure desktops and HDR fidelity are outside its contract.
Animated cursors are drawn using their first image.

## Verification

```text
python -m unittest tests
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
python -m build
calango-recorder --help
```

Automated tests use controlled clocks and frame sources for lifecycle/failure
scenarios and the real MP4 codec for round trips. Native Cereja tests separately
exercise GDI cursor composition with synthetic images. These tests do not replace
manual desktop acceptance on each display/DPI configuration.

## Recorded local acceptance

On Windows with Python 3.14.5, a controlled 1920 x 1080 window was recorded at
30 FPS for 30.005 seconds. The verified MP4 contained 901 frames (30.033 seconds),
a difference of 0.029 seconds. Colors, motion and the visible cursor were checked.
The real Tk interface also passed region selection, countdown cancellation and
stopping while minimized; a 641 x 359 region produced a padded 642 x 360 video.

Window mode was separately exercised through the real GUI using a 320 x 180 Tk
target completely covered by another test window. The resulting 17-frame video
contained the target's colors and changing content, with no pixels from the
covering window. The target never received focus, and foreground remained stable
during recording. The window picker, configuration locking and cursor exclusion
were also checked. Legacy BGRA, BMP and PPM captures used the same target.

Only one physical monitor was available. Negative origins and cursor edge cases
have controlled tests; multiple physical monitors and mixed DPI remain untested.
These observations establish this local run, not performance on other machines.
