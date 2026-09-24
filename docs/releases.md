# Releases

Calango releases are prepared on `main`. Choose the version explicitly in
`calango/_version.py`, keeping both representations consistent:

```python
VERSION = "2.3.0.final.0"
__version__ = "2.3.0"
```

The workflows do not increment versions or rewrite package source. A version
that is already published on PyPI is a no-op, not a request for another patch
release. Submit version changes through the repository's normal review process.

## CI and publication

The `Python package` workflow validates the package and builds distribution
artifacts. `Upload Python Package` follows a successful CI run triggered by a
push to `main`. It uses artifacts from that specific run and attempt instead of
rebuilding a potentially newer branch checkout.

Artifacts keep the attempt number of their producing job. Re-running only
failed CI jobs can reuse an earlier successful build from the same run; the
publisher selects that immutable artifact and preserves its name through deploy.
Artifacts are retained for 30 days. Retry the publishing workflow against the
original CI artifacts if an upload fails. Already uploaded files must match
their SHA-256 digests; only missing files are uploaded, and tags are never moved.
A GitHub release alone does not establish a successful PyPI upload.

The Git tag, GitHub release and PyPI distributions must identify the same
validated revision and declared version. A failed CI run or pull-request run
does not authorize publication. Re-running an existing version does not create
a new version or overwrite a published distribution.

The publisher retains the existing `PYPI_USERNAME` and `PYPI_PASSWORD` GitHub
Actions configuration. These secrets can be defined for the repository or
inherited from its organization. If authentication fails, restore the intended
publish credentials in GitHub Actions and retry the release; changing the
package version does not repair authentication. Never place credential values
in source files or release notes.

## Cereja dependency

Calango 2.3.0 introduces the recorder and requires Python 3.11+ and
`cereja>=2.3.0`. The minimum Cereja version includes native monitor, region and
background-window capture. An unconstrained dependency could install successfully
while leaving the recorder without its required backend.

Publish Cereja 2.3.0 before publishing Calango 2.3.0. While preparing the two
releases, CI may validate against an explicitly pinned Cereja source revision.
That development arrangement does not change the dependency declared in the
Calango wheel or make a checkout-only dependency available to pip users.

Before publication, verify a clean installation resolves the declared Cereja
requirement from PyPI. On Windows, check both `calango-recorder --help` and
`python -m calango.gui --help`, then confirm `ScreenCapture` and `ScreenRecorder`
are importable and complete a short recording. A source-checkout test alone
does not establish that the published dependency chain is installable.

## Recorder release notes

Calango 2.3.0 adds a Tkinter screen recorder and the asynchronous `ScreenRecorder`
API. It records a monitor, region or selected window to silent MP4, preserves
output timing, reports capture and encoding failures, and protects existing
destinations. Windows capture is supplied by Cereja; the recorded file is
verified before success is reported.

Python 3.11 is now the minimum supported version. The recorder requires Windows
10 version 1703 or newer, or Windows 11. Audio and Linux/macOS recorder interfaces
are outside this release. See [recorder usage and validation](screen_recorder.md)
for capabilities, limitations and recorded local acceptance.
