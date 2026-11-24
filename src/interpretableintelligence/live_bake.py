"""``ii bake-live``: the playground's first answer, worked out ahead of time.

A words pack (``ii export-live --readout words``) needs about 100 MB of model before the page can
say anything. This opens the playground itself in headless Chromium with ``?bake``: the page runs
its usual code on every example picture (what it sees, why it sees the top word, and switching
those features off beside the random control) and hands back the result, which is saved as
``instant.json`` next to ``pack.json``. The page then shows that answer at once and works it out
again for real when the model is ready, so the numbers a visitor reads are the same code's, on the
same pictures, on ONNX Runtime Web's CPU path.

The file records the digests of the model files it came from; the page ignores it when they
differ (a re-export), and ``ii export-live`` drops it from ``pack.json``. Exploration only.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

INSTANT = "instant.json"
ORT_URL = re.compile(r"https://cdn\.jsdelivr\.net/npm/onnxruntime-web@[^/]+/dist/(.+)$")


class BakeError(RuntimeError):
    """The pack cannot be baked."""


@dataclass
class BakeSummary:
    path: Path
    samples: int
    bytes: int
    backend: str


def find_chrome() -> str | None:
    """A Chrome or Chromium binary: ``II_CHROME``, an installed one, or one Playwright downloaded."""
    forced = os.environ.get("II_CHROME")
    if forced:
        return forced if Path(forced).is_file() else None
    for candidate in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if Path(candidate).is_file():
            return candidate
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), "/opt/pw-browsers"]
    if sys.platform == "darwin":
        roots.append(str(Path.home() / "Library" / "Caches" / "ms-playwright"))
    else:
        roots.append(str(Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ms-playwright"))
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for pattern in ("chromium-*/chrome-linux*/chrome", "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium", "chromium/chrome-linux*/chrome"):
            hits = sorted(Path(root).glob(pattern))
            if hits:
                return str(hits[-1])
    return None


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def bake_live(pack_dir: Path, *, page: Path | None = None, ort_dir: Path | None = None, timeout_s: float = 900, log=print) -> BakeSummary:
    """Bake ``instant.json`` for a words pack and list it in ``pack.json``.

    ``page`` is the playground folder (default: the folder holding ``packs/``, as in
    ``demo-site/live/packs/<name>``). ``ort_dir`` serves ONNX Runtime Web from a local ``dist``
    folder instead of cdn.jsdelivr.net (for machines that cannot reach it).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - dev dependency
        raise BakeError("ii bake-live needs Playwright (pip install playwright)") from exc

    pack_dir = Path(pack_dir).resolve()
    meta_path = pack_dir / "pack.json"
    if not meta_path.is_file():
        raise BakeError(f"{pack_dir} has no pack.json; make one with ii export-live")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("model") or {}).get("readout") != "words":
        raise BakeError("only a words pack (ii export-live --readout words) has a first answer to bake")
    page = Path(page).resolve() if page else pack_dir.parent.parent
    if not (page / "index.html").is_file() or not (page / "see.js").is_file():
        raise BakeError(f"{page} is not the playground folder (index.html and see.js)")
    try:
        rel_pack = pack_dir.relative_to(page).as_posix()
    except ValueError as exc:
        raise BakeError(f"{pack_dir} must sit inside the playground folder {page}") from exc
    chrome = find_chrome()
    if chrome is None:
        raise BakeError("no Chrome or Chromium found; set II_CHROME to one")

    # Serve the site root (the page loads ../static/), with the page at /<page name>/.
    root = page.parent
    handler = functools.partial(_Quiet, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/{page.name}/index.html?bake&cpu&pack={rel_pack}/"
    log(f"baking {len(meta.get('samples') or [])} pictures in headless Chromium (CPU)…")
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--disable-gpu"])
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            if ort_dir is not None:
                dist = Path(ort_dir).resolve()

                def serve_ort(route):
                    name = ORT_URL.match(route.request.url).group(1)
                    path = dist / name
                    if not path.is_file():
                        return route.fulfill(status=404, body="")
                    kind = "application/wasm" if name.endswith(".wasm") else "text/javascript"
                    return route.fulfill(status=200, body=path.read_bytes(), headers={"content-type": kind, "access-control-allow-origin": "*"})

                context.route(ORT_URL, serve_ort)
            tab = context.new_page()
            tab.on("pageerror", lambda exc: errors.append(str(exc)))
            tab.goto(url)
            tab.wait_for_function("() => window.__ii && window.__ii.baked", timeout=timeout_s * 1000)
            baked = tab.evaluate("() => window.__ii.baked")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    if errors:
        raise BakeError(f"the page failed while baking: {errors[0]}")
    if not baked.get("samples"):
        raise BakeError("the page baked no pictures")
    out = pack_dir / INSTANT
    out.write_text(json.dumps(baked, separators=(",", ":")) + "\n", encoding="utf-8")
    meta.setdefault("files", {})["instant"] = INSTANT
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return BakeSummary(path=out, samples=len(baked["samples"]), bytes=out.stat().st_size, backend=baked.get("backend", "?"))
