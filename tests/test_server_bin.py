"""Finding the Rust viewer binary: II_SERVER, a checkout build, or a checked download."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from interpretableintelligence import server_bin


@pytest.fixture
def release_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local stand-in for GitHub releases serving one asset, with the finder pointed at it."""
    served = tmp_path / "served"
    served.mkdir()
    asset = "interpretableintelligence-server-linux-x86_64"
    payload = b"#!/bin/sh\necho fake viewer\n"
    (served / asset).write_bytes(payload)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(served), **kwargs)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    release_file = tmp_path / "server_release.json"

    def pin(sha: str) -> None:
        release_file.write_text(json.dumps({"repo": "owner/repo", "tag": "server-v9", "sha256": {asset: sha}}))

    pin(hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(server_bin, "RELEASE_FILE", release_file)
    monkeypatch.setattr(server_bin, "DOWNLOAD", f"http://127.0.0.1:{httpd.server_address[1]}/{{asset}}")
    monkeypatch.setattr(server_bin, "platform_asset", lambda: asset)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("II_SERVER", raising=False)
    yield pin, payload
    httpd.shutdown()


def test_override_wins_and_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "server"
    binary.write_text("x")
    monkeypatch.setenv("II_SERVER", str(binary))
    assert server_bin.find_server() == binary
    monkeypatch.setenv("II_SERVER", str(tmp_path / "missing"))
    with pytest.raises(server_bin.ServerNotFound, match="not a file"):
        server_bin.find_server()


def test_download_is_checked_cached_and_executable(release_host) -> None:
    _pin, payload = release_host
    first = server_bin.find_server(build=False)
    assert first.read_bytes() == payload
    assert first.stat().st_mode & 0o111
    assert first.parent.name == "server-v9"
    assert server_bin.find_server(build=False) == first


def test_a_download_with_the_wrong_sum_is_refused_and_removed(release_host) -> None:
    pin, _payload = release_host
    pin("00" * 32)
    with pytest.raises(server_bin.ServerNotFound, match="refusing to run it"):
        server_bin.find_server(build=False)
    assert not list(server_bin.cache_dir().rglob("interpretableintelligence-server-*"))


def test_no_release_yet_says_what_to_do(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release_file = tmp_path / "server_release.json"
    release_file.write_text(json.dumps({"repo": "owner/repo", "tag": None, "sha256": {}}))
    monkeypatch.setattr(server_bin, "RELEASE_FILE", release_file)
    monkeypatch.delenv("II_SERVER", raising=False)
    with pytest.raises(server_bin.ServerNotFound, match="rustup"):
        server_bin.find_server(build=False)


def test_release_pin_lists_only_assets_the_workflow_builds() -> None:
    workflow = (server_bin.REPO_ROOT / ".github" / "workflows" / "release-server.yml").read_text()
    pinned = server_bin.release()
    assert pinned["repo"] == "jerrydjin/interpretableintelligence"
    for asset in pinned["sha256"]:
        assert asset in workflow
    for name in ("linux-x86_64", "linux-aarch64", "macos-arm64"):
        assert f"interpretableintelligence-server-{name}" in workflow
