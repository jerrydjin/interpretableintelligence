"""The Rust viewer (``interpretableintelligence-server view``) serves what the Python viewer serves.

Every route ``ii view`` answers is compared on a set of run folders that cover the viewer's rules:
the fake run's eight-channel clips, window placement from ``manifest.extra``, explicit window
starts, crops, status badges, hardest moments, candidates with pictures, clip resolution by name,
stem and sha256, and the clip-file failures. JSON floats may differ by at most 1e-9; PNG frames
must decode to the same pixels.
"""

from __future__ import annotations

import json
import subprocess
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path

import httpx
import numpy as np

from interpretableintelligence.artifacts import write_feature_acts, write_json, write_jsonl
from interpretableintelligence.view import STATIC_DIR, bind_server, load_run
from test_contract import _assert_json_equal, _server_bin, _wait_until
from test_view import _status_run, _tiny_run


@contextmanager
def _python_viewer(run: Path, clips: Path | None):
    httpd = bind_server(load_run(run, clips), "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@contextmanager
def _rust_viewer(run: Path, clips: Path | None):
    command = [str(_server_bin()), "view", str(run), "--port", "0", "--static", str(STATIC_DIR)]
    if clips is not None:
        command += ["--clips", str(clips)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        if "http://" not in line:
            proc.kill()
            raise AssertionError(f"rust viewer did not start: {line!r} {proc.stderr.read()[:2000]}")
        url = line.split("http://", 1)[1].strip().rstrip("/")
        _wait_until(f"http://{url}/api/run")
        yield f"http://{url}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _pixels(png: bytes) -> tuple[tuple[int, int], bytes]:
    assert png.startswith(b"\x89PNG\r\n\x1a\n"), png[:16]
    at, size, idat = 8, None, b""
    while at < len(png):
        length = int.from_bytes(png[at : at + 4], "big")
        tag = png[at + 4 : at + 8]
        data = png[at + 8 : at + 8 + length]
        if tag == b"IHDR":
            size = (int.from_bytes(data[0:4], "big"), int.from_bytes(data[4:8], "big"))
        elif tag == b"IDAT":
            idat += data
        at += 12 + length
    return size, zlib.decompress(idat)


def _compare_paths(left: str, right: str, paths: list[str], kind: str = "json", headers: dict | None = None) -> None:
    with httpx.Client(timeout=60.0) as client:
        for path in paths:
            a = client.get(left + path, headers=headers)
            b = client.get(right + path, headers=headers)
            assert a.status_code == b.status_code, (path, a.status_code, a.text[:300], b.status_code, b.text[:300])
            if a.headers.get("content-type", "").startswith("application/json") or kind == "json" and a.status_code != 200:
                _assert_json_equal(a.json(), b.json(), path)
            elif kind == "png":
                assert _pixels(a.content) == _pixels(b.content), path
            else:
                assert a.content == b.content, path
                for name in ("content-range", "accept-ranges"):
                    assert a.headers.get(name) == b.headers.get(name), (path, name)


def _compare_viewers(run: Path, clips: Path | None, *, png_frames: int = 3, extra_paths: list[str] = ()) -> None:
    with _python_viewer(run, clips) as left, _rust_viewer(run, clips) as right:
        with httpx.Client(timeout=60.0) as client:
            payload = client.get(left + "/api/run").json()
        clip_ids = [clip["clip_id"] for clip in payload["clips"]]
        feature_ids = [feature["id"] for feature in payload["features"]]
        json_paths = ["/api/run", "/api/clips/no-such-clip", "/api/features/99999", "/api/nope"]
        json_paths += [f"/api/clips/{clip_id}" for clip_id in clip_ids]
        json_paths += [f"/api/features/{feature_id}" for feature_id in feature_ids]
        png_paths, media_paths = [], []
        for clip in payload["clips"]:
            clip_id, n_frames = clip["clip_id"], clip["n_frames"]
            frames = sorted({0, n_frames // 2, max(n_frames - 1, 0)})
            for frame in frames + [n_frames]:
                for feature_id in feature_ids[:6]:
                    json_paths.append(f"/api/clips/{clip_id}/heatmap?frame={frame}&feature={feature_id}")
            json_paths.append(f"/api/clips/{clip_id}/heatmap?frame=0")
            png_paths += [f"/api/clips/{clip_id}/frames/{frame}.png" for frame in frames[:png_frames]]
            json_paths.append(f"/api/clips/{clip_id}/frames/{n_frames}.png")
            media_paths.append(f"/api/clips/{clip_id}/media")
        _compare_paths(left, right, json_paths)
        _compare_paths(left, right, png_paths, kind="png")
        _compare_paths(left, right, media_paths, kind="bytes")
        _compare_paths(left, right, media_paths, kind="bytes", headers={"Range": "bytes=2-5"})
        _compare_paths(left, right, [f"/api/features/{feature_id}/viz" for feature_id in feature_ids], kind="bytes")
        _compare_paths(left, right, ["/", "/static/app.js", "/static/style.css", "/static/../view.py"], kind="bytes")
        _compare_paths(left, right, list(extra_paths))


def test_fake_run_matches(demo_run: Path) -> None:
    _compare_viewers(demo_run, demo_run.parent.parent / "clips", png_frames=2)


def test_tiny_run_and_clip_file_rules_match(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    run, clips = _tiny_run(tmp_path / "a")
    _compare_viewers(run, clips, png_frames=4)
    run, clips = _tiny_run(tmp_path / "b", manifest_path=True)
    _compare_viewers(run, None)
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((4, 4, 4, 6), dtype=np.float32))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["clips"][0]["path"] = str(wrong)
    (run / "manifest.json").write_text(json.dumps(manifest))
    _compare_viewers(run, None)
    _compare_viewers(run, clips)


def test_status_badges_and_moments_match(tmp_path: Path) -> None:
    _compare_viewers(_status_run(tmp_path), None)


def _run(root: Path, manifest: dict, acts: dict, files: dict | None = None) -> Path:
    """A run folder: manifest, acts, and any other files as ``{relative path: payload}``."""
    run = root / manifest["run_id"]
    (run / "features").mkdir(parents=True)
    write_json(run / "manifest.json", manifest)
    write_feature_acts(run / "features" / "feature_acts.parquet", acts)
    for relative, payload in (files or {}).items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        elif relative.endswith(".jsonl"):
            write_jsonl(path, payload)
        else:
            write_json(path, payload)
    return run


def _acts(rows: list[tuple]) -> dict:
    columns = ("feature_id", "clip_id", "window", "t", "h", "w", "value")
    return {name: [row[index] for row in rows] for index, name in enumerate(columns)}


def test_window_placement_and_named_videos_match(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "droid_ep000.mp4").write_bytes(b"named-mp4-bytes")
    (clips / "0.mp4").write_bytes(b"stem-for-clip-id-0")
    (clips / "short.webm").write_bytes(b"short-webm")
    grid = {"t": 8, "h": 2, "w": 2, "tubelet": 2, "patch_px": 16}
    run = _run(
        tmp_path,
        {
            "run_id": "vjepa-real",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ab" * 32},
            "layer": 24,
            "grid": grid,
            "extra": {"frames": 16, "stride": 2, "windows": 3},
            "clips": [
                {"clip_id": 0, "name": "droid_ep000.mp4", "fps": 8.0, "n_frames": 167, "n_windows": 3, "split": "train", "hw": [292, 519]},
                {"clip_id": 1, "name": "short.webm", "fps": 8.0, "n_frames": 20, "n_windows": 3, "split": "train"},
                {"clip_id": 2, "fps": 8, "n_frames": 40, "split": "heldout", "stride": 1},
            ],
        },
        _acts([(7, "0", 1, 3, 1, 0, 3.0), (7, "1", 0, 3, 0, 1, 1.0), (8, "0", 2, 7, 0, 0, 0.5), (8, "2", 0, 0, 1, 1, 2.0)]),
        {"features/features.jsonl": [{"id": 7, "label": "grasp", "label_model": "hand-v1"}, {"id": 8, "label": "lift"}]},
    )
    _compare_viewers(run, clips)


def test_explicit_starts_crops_and_pixel_formats_match(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    rng = np.random.default_rng(0)
    arrays = {
        "gray": rng.random((6, 5, 7, 1), dtype=np.float32),
        "rgb8": rng.integers(0, 256, size=(6, 5, 7, 3), dtype=np.uint8),
        "rgba": (rng.random((6, 5, 7, 4)) * 4 - 1).astype(np.float64),
        "plain": rng.random((6, 5, 7), dtype=np.float32) * 300,
        "multi": rng.random((6, 5, 7, 5), dtype=np.float32) * 3,
        "short": rng.random((3, 5, 7, 1), dtype=np.float32),
        "flat": np.zeros((6,), dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(clips / f"{name}.npy", array)
    (clips / "notes.txt").write_text("not a clip")
    entries = [
        {"clip_id": "gray", "n_frames": 6, "fps": 8.0, "split": "train", "window_starts": [1, 3]},
        {"clip_id": "rgb8", "n_frames": 6, "fps": 8.0, "split": "train", "windows": [{"index": 1, "start_frame": 2}, {"start": 0}]},
        {"clip_id": "rgba", "n_frames": 6, "fps": 8.0, "split": "heldout", "window_start": 2, "stride": 2},
        {"clip_id": "plain", "n_frames": 6, "fps": 8.0, "split": "heldout", "hw": [50, 70]},
        {"clip_id": "multi", "n_frames": 6, "fps": 8.0, "split": "train", "windows": [4, 1]},
        {"clip_id": "short", "n_frames": 6, "fps": 8.0, "split": "train"},
        {"clip_id": "flat", "n_frames": 6, "fps": 8.0, "split": "train"},
        {"clip_id": "gone", "n_frames": 6, "fps": 8.0, "split": "train", "sha256": "00" * 32},
        {"clip_id": "notes", "name": "notes.txt", "n_frames": 6, "split": "train"},
    ]
    rows = []
    for index, entry in enumerate(entries):
        for t in range(2):
            rows.append((index % 3, entry["clip_id"], t % 2, t, t % 2, (t + 1) % 2, 0.25 + index * 0.1 + t))
    crops = ({"x": 1, "y": 2, "width": 3, "size": 4}, [0, 0, 5, 5], "centre", {"mode": "center"}, {"x": 1})
    for index, crop in enumerate(crops):
        run = _run(
            tmp_path / f"crop-{index}",
            {
                "run_id": "pixels",
                "model": {"adapter": "fake", "name": "fake", "sha": "ab" * 32},
                "layer": 2,
                "grid": {"t": 2, "h": 2, "w": 2, "tubelet": 1, "window_stride": 3},
                "extra": {"crop": crop},
                "clips_dir": str(clips),
                "clips": entries,
            },
            _acts(rows),
            {
                "scoreboard.json": {
                    "sae": {"usable": False, "ev_heldout": "0.5"},
                    "features": [
                        {"id": 0, "label": "zero", "real": True, "status": "Real"},
                        {"id": 2, "real": False, "status": "  "},
                    ],
                },
                "steering/scorecard.json": {"rows": [{"feature_id": 0, "effect": 0.1}]},
            },
        )
        _compare_viewers(run, None, png_frames=3)


def test_candidates_and_pictures_match(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        {
            "run_id": "cand",
            "model": {"name": "tiny", "sha": "abc"},
            "layer": 3,
            "grid": {"t": 2, "h": 2, "w": 2, "tubelet": 1},
            "clips": [{"clip_id": "c0", "n_frames": 2, "split": "train", "fps": 8}],
        },
        _acts([(0, "c0", 0, 0, 0, 0, 0.2), (2, "c0", 0, 0, 0, 1, 0.9), (5, "c0", 0, 1, 1, 1, 0.3)]),
    )
    write_jsonl(run / "features" / "features.jsonl", [{"id": 0, "label": "zero"}, {"id": 1, "label": "one"}])
    write_json(
        run / "features" / "candidates.json",
        {"ranked": [{"feature_id": 2, "viz": "pics/two.png"}, {"feature_id": 0}, {"feature_id": 2}]},
    )
    (run / "pics").mkdir()
    (run / "pics" / "two.png").write_bytes(b"\x89PNG two")
    (run / "features" / "viz").mkdir()
    (run / "features" / "viz" / "feature_00000.png").write_bytes(b"\x89PNG zero")
    (run / "features" / "viz" / "feature_1.png").write_bytes(b"\x89PNG one")
    _compare_viewers(run, None, extra_paths=["/api/features/5", "/api/features/1"])


def test_a_bad_run_folder_is_refused_with_its_neighbours_named(tmp_path: Path) -> None:
    (tmp_path / "real-run").mkdir()
    (tmp_path / "real-run" / "manifest.json").write_text("{}")
    proc = subprocess.run(
        [str(_server_bin()), "view", str(tmp_path / "typo"), "--port", "0"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "run folder not found" in proc.stderr
    assert "real-run" in proc.stderr


def test_viewer_answers_head_like_get(tmp_path: Path) -> None:
    run, clips = _tiny_run(tmp_path)
    with _rust_viewer(run, clips) as url, httpx.Client() as client:
        head = client.head(url + "/api/run")
        get = client.get(url + "/api/run")
        assert head.status_code == 200
        assert head.content == b""
        assert int(head.headers["content-length"]) == len(get.content)
