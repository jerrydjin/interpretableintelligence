"""Start the viewer for a run folder: the Rust binary, or the Python server when the binary cannot serve it.

``ii view``, ``ii try``, ``ii run``'s serve stage and ``ii run --remote`` all come through
:func:`start_viewer`. It finds ``interpretableintelligence-server`` with
:func:`server_bin.find_server` and runs ``view RUN --clips … --static … --port …``; the browser is
still opened from Python.

* ``--steer``: the binary forwards ``GET``/``POST /api/steer`` to ``--steer-url``.
  :class:`SteerService` is that service: a small HTTP server in this process that answers those two
  routes from a loaded :class:`~interpretableintelligence.steer.Steerer`. It starts before the
  binary, which asks it for its status once at start-up.
* A run that is still filling in (``ii run --remote``): the status shown on the page is written to
  a JSON file passed as ``--live``. :meth:`Viewer.reload` bumps its ``generation``, and the binary
  loads the run folder again the next time the page asks for it.

The Python server in ``view.py`` is still used when:

* no binary can be found, built or downloaded (``ServerNotFound``; for example Windows or an Intel Mac),
* the run's clips are video files: the binary does not yet decode video frames (thumbnails) or
  transcode avi/mov/mkv, so the Python viewer serves those runs until it does, or
* ``II_VIEWER=python`` is set.

Engineering side of the line (AGENTS.md): this module starts a program and relays steering requests
to the model; the read API itself is Rust.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from interpretableintelligence import server_bin

STEER_LIMIT = 64 * 1024
_START_TIMEOUT_S = 60.0


class ViewerError(OSError):
    """The viewer could not start (a bad run folder, a port in use, a binary that exited)."""


# --- the steering service -----------------------------------------------------------------------


def answer_steer(steerer, body: bytes) -> tuple[int, dict]:
    """HTTP status and JSON reply for one ``POST /api/steer`` body. Shared with the Python viewer."""
    from interpretableintelligence.steer import SteerError  # noqa: PLC0415 (loads torch)

    if not body or len(body) > STEER_LIMIT:
        return 400, {"error": "request body must be a JSON object under 64 KiB"}
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return 400, {"error": "request body is not valid JSON"}
    try:
        return 200, steerer.steer(payload)
    except SteerError as exc:
        return 400, exc.as_dict()


def steer_status(steerer) -> dict:
    """What ``/api/run`` shows under ``steer`` when the playground is on."""
    return {"enabled": True, "sae_usable": bool(steerer.sae_usable)}


class SteerService:
    """``GET /api/steer`` (status) and ``POST /api/steer`` (one steering request) for ``--steer-url``.

    Binds ``127.0.0.1`` only: the binary is the public face, this answers it alone.
    """

    def __init__(self, steerer, host: str = "127.0.0.1", port: int = 0) -> None:
        self.steerer = steerer
        self.httpd = _Server((host, port), _steer_handler(steerer))
        bound_host, bound_port = self.httpd.server_address[:2]
        self.url = f"http://{bound_host}:{bound_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="ii-steer", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _steer_handler(steerer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            del fmt, args

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/steer":
                self._json(404, {"error": "not found"})
                return
            self._json(200, steer_status(steerer))

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/steer":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if 0 < length <= STEER_LIMIT else b""
            try:
                status, reply = answer_steer(steerer, body)
            except Exception as exc:  # noqa: BLE001 (any model failure is the request's error)
                status, reply = 500, {"error": str(exc)}
            self._json(status, reply)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    return Handler


# --- the live status file -----------------------------------------------------------------------


class LiveStatus:
    """The ``--live`` JSON file: remote status fields plus a ``generation`` the binary watches.

    Written whole and renamed into place, so the binary never reads half a file.
    """

    def __init__(self, path: Path, fields: dict) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fields = {key: value for key, value in dict(fields).items() if key != "generation"}
        self.generation = 0
        self._write()

    def update(self, **fields) -> None:
        with self._lock:
            self._fields.update({key: value for key, value in fields.items() if key != "generation"})
            self._write()

    def bump(self) -> None:
        with self._lock:
            self.generation += 1
            self._write()

    def payload(self) -> dict:
        return {**self._fields, "generation": self.generation}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(f".{self.path.name}.part")
        part.write_text(json.dumps(self.payload(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(part, self.path)


# --- the viewer handle --------------------------------------------------------------------------


class Viewer:
    """A running viewer. ``url`` ends in ``/``. ``backend`` is ``"rust"`` or ``"python"``."""

    url: str
    backend: str

    def reload(self) -> None:
        """Load the run folder again (live runs)."""
        raise NotImplementedError

    def update(self, **fields) -> None:
        """Change the live status shown on the page (live runs)."""
        raise NotImplementedError

    def alive(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def wait(self) -> None:
        """Serve until Ctrl+C or the server stops, then shut down."""
        try:
            while self.alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()


class RustViewer(Viewer):
    backend = "rust"

    def __init__(self, proc: subprocess.Popen, url: str, steer: SteerService | None, live: LiveStatus | None, scratch) -> None:
        self.proc = proc
        self.url = url
        self.steer = steer
        self.live = live
        self._scratch = scratch
        self._closed = False
        atexit.register(self.close)

    def reload(self) -> None:
        if self.live is not None:
            self.live.bump()

    def update(self, **fields) -> None:
        if self.live is not None:
            self.live.update(**fields)

    @property
    def generation(self) -> int:
        return 0 if self.live is None else self.live.generation

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        _stop(self.proc)
        if self.steer is not None:
            self.steer.close()
        if self._scratch is not None:
            self._scratch.cleanup()


class PythonViewer(Viewer):
    backend = "python"

    def __init__(self, model, host: str, port: int) -> None:
        from interpretableintelligence.view import bind_server  # noqa: PLC0415

        self.model = model
        self.httpd = bind_server(model, host, int(port))
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="ii-view", daemon=True)
        self.thread.start()
        bound_host, bound_port = self.httpd.server_address[:2]
        self.url = f"http://{bound_host}:{bound_port}/"
        self._closed = False

    def reload(self) -> None:
        if hasattr(self.model, "reload"):
            self.model.reload()

    def update(self, **fields) -> None:
        if hasattr(self.model, "update"):
            self.model.update(**fields)

    @property
    def generation(self) -> int:
        return int(getattr(self.model, "generation", 0))

    def alive(self) -> bool:
        return self.thread.is_alive()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# --- starting one -------------------------------------------------------------------------------


def start_viewer(
    run_dir: Path,
    clips_dir: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    steerer=None,
    live: dict | None = None,
    model=None,
    log=None,
) -> Viewer:
    """Start the viewer and return once ``/api/run`` answers.

    ``steerer`` turns on the playground. ``live`` (status fields) serves a run that is still filling
    in. ``model`` is an already loaded ``view.RunModel``, reused if the Python server is the one
    that starts. ``log`` gets one line whenever the Python server is used instead of the binary.
    """
    run_dir = Path(run_dir)
    clips_dir = None if clips_dir is None else Path(clips_dir)
    say = log or (lambda _line: None)
    reason = _rust_unavailable_reason()
    if reason is None:
        try:
            binary = server_bin.find_server()
        except server_bin.ServerNotFound as exc:
            reason = f"no viewer binary ({exc})"
        else:
            viewer = _start_rust(binary, run_dir, clips_dir, host, port, steerer, live)
            videos = _video_clips(viewer.url)
            if not videos:
                return viewer
            viewer.close()
            reason = f"{len(videos)} clip(s) are video files, which the Rust viewer does not decode yet"
    say(f"viewer: using the Python server: {reason}")
    return _start_python(run_dir, clips_dir, host, port, steerer, live, model)


def open_in_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except OSError:
        pass


def _rust_unavailable_reason() -> str | None:
    choice = os.environ.get("II_VIEWER", "").strip().lower()
    if choice == "python":
        return "II_VIEWER=python"
    return None


def _start_rust(binary: Path, run_dir: Path, clips_dir: Path | None, host: str, port: int, steerer, live) -> RustViewer:
    from interpretableintelligence.view import STATIC_DIR  # noqa: PLC0415

    command = [str(binary), "view", str(run_dir), "--host", host, "--port", str(int(port)), "--static", str(STATIC_DIR)]
    if clips_dir is not None:
        command += ["--clips", str(clips_dir)]
    service = None
    scratch = None
    status = None
    try:
        if steerer is not None:
            service = SteerService(steerer)
            command += ["--steer-url", service.url]
        if live is not None:
            scratch = tempfile.TemporaryDirectory(prefix="ii-live-")
            status = LiveStatus(Path(scratch.name) / "live.json", live)
            command += ["--live", str(status.path)]
        proc = subprocess.Popen(
            command,
            env=_local_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except BaseException:
        if service is not None:
            service.close()
        if scratch is not None:
            scratch.cleanup()
        raise
    tail: deque[str] = deque(maxlen=40)
    first: list[str] = []
    ready = threading.Event()

    def read_stdout() -> None:
        for line in proc.stdout:
            if not first:
                first.append(line)
                ready.set()
            tail.append(line.rstrip())
        ready.set()

    def read_stderr() -> None:
        for line in proc.stderr:
            tail.append(line.rstrip())

    threading.Thread(target=read_stdout, name="ii-view-out", daemon=True).start()
    threading.Thread(target=read_stderr, name="ii-view-err", daemon=True).start()
    ready.wait(timeout=_START_TIMEOUT_S)
    line = first[0] if first else ""
    if "http://" not in line:
        _stop(proc)
        if service is not None:
            service.close()
        if scratch is not None:
            scratch.cleanup()
        time.sleep(0.05)
        detail = "\n".join(item for item in tail if item) or "it printed nothing"
        message = detail.removeprefix("error: ") if detail.startswith("error: ") else detail
        raise ViewerError(f"the viewer did not start: {message}")
    url = "http://" + line.split("http://", 1)[1].strip().rstrip("/") + "/"
    viewer = RustViewer(proc, url, service, status, scratch)
    try:
        _wait_health(url)
    except Exception:
        viewer.close()
        raise
    return viewer


def _start_python(run_dir, clips_dir, host, port, steerer, live, model) -> PythonViewer:
    from interpretableintelligence.view import LiveRun, load_run  # noqa: PLC0415

    if live is not None:
        model = LiveRun(run_dir, clips_dir, live)
    elif model is None:
        model = load_run(run_dir, clips_dir)
    if steerer is not None:
        model.steerer = steerer
        model.run_payload = {**model.run_payload, "steer": steer_status(steerer)}
    return PythonViewer(model, host, port)


def _video_clips(url: str) -> list[str]:
    """Clip ids the run shows as video, read from the running viewer's ``/api/run``."""
    payload = _get_json(url + "api/run")
    clips = payload.get("clips") if isinstance(payload, dict) else None
    return [
        str(clip.get("clip_id"))
        for clip in clips or []
        if isinstance(clip, dict) and clip.get("media") == "video" and clip.get("pixels") == "ok"
    ]


def _wait_health(url: str) -> dict:
    target = url + "api/run"
    last = "the viewer did not answer"
    deadline = time.monotonic() + _START_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            payload = _get_json(target)
            if isinstance(payload, dict) and payload.get("run_id"):
                return payload
            last = "/api/run has no run_id"
        except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(0.05)
    raise ViewerError(f"/api/run health check failed: {last}")


# The viewer is local: never send its requests through an HTTP(S)_PROXY.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_json(url: str):
    with _LOCAL.open(url, timeout=10) as response:  # noqa: S310 (a local viewer URL)
        return json.loads(response.read().decode("utf-8"))


def _local_env() -> dict[str, str]:
    """The binary's environment, with the steering service's loopback address kept off any proxy."""
    env = dict(os.environ)
    for key in ("NO_PROXY", "no_proxy"):
        listed = [item for item in env.get(key, "").split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in listed:
                listed.append(host)
        env[key] = ",".join(listed)
    return env


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
