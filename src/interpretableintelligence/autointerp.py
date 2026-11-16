"""Auto-interp labels for real video: a vision model explains, held-out clips test.

The CLIP concept labeller can only name a feature with a word from a fixed list,
read off noisy dense CLIP maps. Features in a real model are often parts of
things, textures, positions or the robot itself, so on real video no label held.

This labeller follows explain-then-simulate (Bills et al. 2023), with the same
held-out gate as every other labeller:

1. **Explain** (train clips only). A vision-language model sees frames where the
   feature fires most strongly, each with the strongest cell boxed and the
   feature's activation map overlaid, and names what the boxed regions share.
2. **Simulate** (held-out clips only). The model sees held-out frames with one
   cell boxed and, given only the label, rates how well the boxed region matches
   it. Half the boxes are cells where the feature fires (its strongest held-out
   cells and a random sample of the rest), half are cells where it does not
   (in the same frames, and in random held-out frames).
3. The label holds only if those ratings separate firing from non-firing cells
   with ROC AUC >= ``LABEL_HOLD_MIN_AUROC`` (0.80), the same bar as CLIP.

The explainer never sees a held-out clip, and the simulator never sees an
activation. Only spatial grids are labelled (``grid.h > 1``); a GR00T action
token has no region to box.

Providers are read from the environment, so a run without a key keeps the CLIP
labeller: ``ANTHROPIC_API_KEY``, ``FIREWORKS_API_KEY``, or ``OPENAI_API_KEY``
(with ``II_LABEL_BASE_URL`` for any OpenAI-compatible endpoint). Set
``II_LABEL_PROVIDER`` to pick one when several keys are set, ``II_LABEL_MODEL``
to change the model, and ``II_LABEL_PROVIDER=none`` to turn this off. Responses
are cached on disk by request hash, so a rerun costs nothing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from interpretableintelligence.thresholds import DEAD_ACT_EPS, LABEL_HOLD_MIN_AUROC

AUTOINTERP_VERSION = "autointerp-v1"
# Frames the explainer sees, and firing / non-firing cells the simulator rates.
N_EXPLAIN = 8
N_SIMULATE = 10
SIMULATE_BATCH = 5
# Fewer held-out firings than this and the label cannot be tested, so it does not hold.
MIN_HELDOUT_POSITIVES = 4
# Without a prefilter shortlist, label at most this many features (the highest attribution).
MAX_FEATURES = 32
# Features asked about at once. The calls are network waits, so threads are enough.
LABEL_WORKERS = 8
IMAGE_PX = 320
# Reasoning models spend tokens before answering; too few and the answer comes back empty.
MAX_TOKENS = 4000

PROVIDERS = {
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": "claude-haiku-5-5", "url": "https://api.anthropic.com/v1/messages"},
    "fireworks": {
        "key": "FIREWORKS_API_KEY",
        "model": "accounts/fireworks/models/glm-5p3-flash",
        "url": "https://api.fireworks.ai/inference/v1",
    },
    "openai": {"key": "OPENAI_API_KEY", "model": "gpt-4o-mini", "url": "https://api.openai.com/v1"},
}

EXPLAIN_PROMPT = """These {n} images are frames from video clips. A vision model has a learned feature that responds to part of each frame.
In every image the red box marks where the feature responds most strongly, and the orange tint shows where else it responds.

What do the boxed regions have in common? Be concrete: an object, a part of an object, a material or texture, an edge or position in the frame, a robot part, or a kind of motion blur. Name it in at most six words a newcomer would understand, then give one sentence of description.

Reply with JSON only: {{"label": "...", "description": "..."}}"""

SIMULATE_PROMPT = """A feature of a vision model was described as: "{label}". {description}

Each of these {n} images has one red box. For each image, rate from 0 to 10 how well the region inside the red box matches that description (0 = not at all, 10 = clearly).

Reply with JSON only: a list of {n} integers in image order, for example [3, 9, 0]."""


class VisionClient(Protocol):
    name: str

    def complete(self, prompt: str, images: list[bytes]) -> str: ...


# ----------------------------------------------------------------- providers


@dataclass
class _Http:
    url: str
    headers: dict[str, str]
    timeout: float = 120.0

    def post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        delay = 2.0
        for attempt in range(5):
            request = urllib.request.Request(self.url, data=body, headers={"content-type": "application/json", **self.headers})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    if attempt < 4:
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise RuntimeError(f"{self.url} returned {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"{self.url} unreachable: {exc}") from exc
        raise RuntimeError(f"{self.url} failed after retries")


class OpenAICompatibleClient:
    """Chat completions with base64 PNG images (Fireworks, OpenAI, vLLM, ...)."""

    def __init__(self, base_url: str, api_key: str, model: str, provider: str = "openai") -> None:
        self.model = model
        self.name = f"{provider}/{model}"
        self._http = _Http(base_url.rstrip("/") + "/chat/completions", {"authorization": f"Bearer {api_key}"})

    def complete(self, prompt: str, images: list[bytes]) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            data = base64.b64encode(image).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        reply = self._http.post(
            {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": MAX_TOKENS}
        )
        return str(reply["choices"][0]["message"].get("content") or "")


class AnthropicClient:
    """Claude Messages API with base64 PNG images."""

    def __init__(self, api_key: str, model: str, url: str = PROVIDERS["anthropic"]["url"]) -> None:
        self.model = model
        self.name = f"anthropic/{model}"
        self._http = _Http(url, {"x-api-key": api_key, "anthropic-version": "2023-06-01"})

    def complete(self, prompt: str, images: list[bytes]) -> str:
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(image).decode()}}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        reply = self._http.post(
            {"model": self.model, "max_tokens": MAX_TOKENS, "temperature": 0, "messages": [{"role": "user", "content": content}]}
        )
        return "".join(part.get("text", "") for part in reply.get("content", []) if part.get("type") == "text")


class CachedClient:
    """Answers keyed by provider, model, prompt and image bytes, so reruns are free."""

    def __init__(self, inner: VisionClient, directory: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.directory = Path(directory)

    def complete(self, prompt: str, images: list[bytes]) -> str:
        digest = hashlib.sha256()
        digest.update(f"{AUTOINTERP_VERSION}\0{self.name}\0{prompt}".encode())
        for image in images:
            digest.update(hashlib.sha256(image).digest())
        path = self.directory / f"{digest.hexdigest()}.json"
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["text"])
        except (OSError, ValueError, KeyError):
            pass
        text = self.inner.complete(prompt, images)
        if not text.strip():
            return text  # an empty reply is a failure, not an answer: never cache it
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": self.name, "text": text}), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
        return text


def provider_from_env(env: dict[str, str] | None = None) -> tuple[str, str] | None:
    """``(provider, model)`` the environment selects, or ``None`` (keep the CLIP labeller)."""
    env = dict(os.environ if env is None else env)
    chosen = env.get("II_LABEL_PROVIDER", "").strip().lower()
    if chosen == "none":
        return None
    if chosen:
        if chosen not in PROVIDERS:
            raise ValueError(f"II_LABEL_PROVIDER must be one of {sorted(PROVIDERS)} or none, got {chosen!r}")
        if not env.get(PROVIDERS[chosen]["key"]):
            raise ValueError(f"II_LABEL_PROVIDER={chosen} needs {PROVIDERS[chosen]['key']}")
    else:
        chosen = next((name for name, spec in PROVIDERS.items() if env.get(spec["key"])), "")
        if not chosen:
            return None
    model = env.get("II_LABEL_MODEL", "").strip() or PROVIDERS[chosen]["model"]
    return chosen, model


def client_from_env(env: dict[str, str] | None = None) -> VisionClient | None:
    env = dict(os.environ if env is None else env)
    picked = provider_from_env(env)
    if picked is None:
        return None
    provider, model = picked
    key = env[PROVIDERS[provider]["key"]]
    if provider == "anthropic":
        inner: VisionClient = AnthropicClient(key, model)
    else:
        base = env.get("II_LABEL_BASE_URL", "").strip() or PROVIDERS[provider]["url"]
        inner = OpenAICompatibleClient(base, key, model, provider=provider)
    cache = env.get("II_LABEL_CACHE", "").strip() or str(Path.home() / ".cache" / "interpretableintelligence" / "autointerp")
    return CachedClient(inner, Path(cache))


def label_model_name(client: VisionClient) -> str:
    return f"{AUTOINTERP_VERSION}:{client.name}"


# ------------------------------------------------------------------ rendering


def png_bytes(image: np.ndarray) -> bytes:
    """Encode ``[h, w, 3]`` uint8 as PNG with the standard library."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _square(frame: np.ndarray, size: int = IMAGE_PX) -> np.ndarray:
    """Squash a frame to a square, as the hf adapter does, as float RGB in [0, 1]."""
    pixels = torch.from_numpy(np.ascontiguousarray(frame)).float()
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if pixels.shape[-1] == 1:
        pixels = pixels.expand(-1, -1, 3)
    if float(pixels.max()) > 1.5:
        pixels = pixels / 255.0
    pixels = pixels.permute(2, 0, 1)[None]
    pixels = torch.nn.functional.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return pixels[0].permute(1, 2, 0).clamp(0, 1).numpy()


def render(frame: np.ndarray, cell: tuple[int, int], grid_hw: tuple[int, int], heat: np.ndarray | None = None) -> bytes:
    """The frame with one grid cell boxed in red and, optionally, an activation map tinted orange."""
    image = _square(frame)
    size = image.shape[0]
    rows, cols = grid_hw
    if heat is not None:
        peak = float(np.max(heat))
        if peak > 0:
            weight = torch.from_numpy(np.asarray(heat, dtype=np.float32) / peak)[None, None]
            weight = torch.nn.functional.interpolate(weight, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
            alpha = 0.55 * weight[..., None]
            image = image * (1 - alpha) + np.array([1.0, 0.55, 0.0]) * alpha
    image = (image * 255).round().astype(np.uint8)
    row, col = cell
    cell_h, cell_w = size / rows, size / cols
    # One cell of context on every side, so the box shows a region, not a speck.
    top = int(max(0, (row - 1) * cell_h))
    bottom = int(min(size - 1, (row + 2) * cell_h - 1))
    left = int(max(0, (col - 1) * cell_w))
    right = int(min(size - 1, (col + 2) * cell_w - 1))
    red = np.array([230, 20, 20], dtype=np.uint8)
    thick = 3
    image[top : top + thick, left : right + 1] = red
    image[bottom - thick + 1 : bottom + 1, left : right + 1] = red
    image[top : bottom + 1, left : left + thick] = red
    image[top : bottom + 1, right - thick + 1 : right + 1] = red
    return png_bytes(image)


# ------------------------------------------------------------------- parsing


def parse_label(text: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            label = " ".join(str(data.get("label", "")).split())[:80]
            return label, " ".join(str(data.get("description", "")).split())[:300]
        except ValueError:
            pass
    line = text.strip().splitlines()[0] if text.strip() else ""
    return " ".join(line.split())[:80], ""


def parse_scores(text: str, n: int) -> list[float] | None:
    match = re.search(r"\[[^\[\]]*\]", text, flags=re.S)
    if not match:
        return None
    try:
        values = [float(value) for value in json.loads(match.group(0))]
    except (ValueError, TypeError):
        return None
    if len(values) != n:
        return None
    return [min(10.0, max(0.0, value)) for value in values]


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """ROC AUC with ties counted half (Mann-Whitney). 0.5 when a class is empty."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ------------------------------------------------------------------ sampling


@dataclass
class Tokens:
    """Where each captured token sits, and the chosen features' activations there."""

    clip: np.ndarray
    window: np.ndarray
    t: np.ndarray
    h: np.ndarray
    w: np.ndarray
    split: np.ndarray  # 0 train, 1 held-out
    acts: np.ndarray  # [n_tokens, n_chosen]


def gather_tokens(acts_dir: Path, sae, feature_ids: list[int], device) -> Tokens:
    from interpretableintelligence.store import RowSource

    ids = torch.as_tensor(list(feature_ids), dtype=torch.long)
    with RowSource(Path(acts_dir)) as source:
        parts = []
        for start, stop in source.chunks(65536):
            rows = torch.from_numpy(np.array(source.rows[start:stop], dtype=np.float32, copy=True)).to(device)
            with torch.no_grad():
                parts.append(sae.encode(rows)[:, ids.to(rows.device)].float().cpu().numpy())
        return Tokens(
            clip=np.array(source.clip_idx, dtype=np.int64, copy=True),
            window=np.array(source.window, dtype=np.int64, copy=True),
            t=np.array(source.t, dtype=np.int64, copy=True),
            h=np.array(source.h, dtype=np.int64, copy=True),
            w=np.array(source.w, dtype=np.int64, copy=True),
            split=np.array(source.split, dtype=np.uint8, copy=True),
            acts=np.concatenate(parts) if parts else np.zeros((0, len(feature_ids)), dtype=np.float32),
        )


def _frame_key(tokens: Tokens, row: int) -> tuple[int, int, int]:
    return int(tokens.clip[row]), int(tokens.window[row]), int(tokens.t[row])


def pick_top(tokens: Tokens, column: np.ndarray, rows: np.ndarray, k: int, per_clip: int) -> list[int]:
    """Strongest firing rows, one per frame and at most ``per_clip`` per clip."""
    firing = rows[column[rows] > DEAD_ACT_EPS]
    order = firing[np.argsort(-column[firing], kind="stable")]
    chosen: list[int] = []
    frames: set[tuple[int, int, int]] = set()
    per: dict[int, int] = {}
    for row in order:
        key = _frame_key(tokens, int(row))
        clip = key[0]
        if key in frames or per.get(clip, 0) >= per_clip:
            continue
        chosen.append(int(row))
        frames.add(key)
        per[clip] = per.get(clip, 0) + 1
        if len(chosen) >= k:
            break
    return chosen


def simulation_set(tokens: Tokens, column: np.ndarray, rng: np.random.Generator, n: int = N_SIMULATE) -> tuple[list[int], list[bool]]:
    """Held-out rows to rate: firing (half strongest, half random) and the same number not firing."""
    held = np.flatnonzero(tokens.split == 1)
    top = pick_top(tokens, column, held, k=(n + 1) // 2, per_clip=2)
    firing = held[column[held] > DEAD_ACT_EPS]
    rest = np.setdiff1d(firing, np.asarray(top, dtype=np.int64))
    extra = rng.choice(rest, size=min(n - len(top), rest.size), replace=False).tolist() if rest.size else []
    positives = top + [int(row) for row in extra]
    quiet = held[column[held] <= DEAD_ACT_EPS]
    if not positives or quiet.size == 0:
        return [], []
    # Half the negatives from the frames the positives are in, so content in the frame is not enough.
    positive_frames = {_frame_key(tokens, p) for p in positives}
    same_frame = [int(row) for row in quiet if _frame_key(tokens, int(row)) in positive_frames]
    n_same = min(len(positives) // 2, len(same_frame))
    negatives = rng.choice(np.asarray(same_frame, dtype=np.int64), size=n_same, replace=False).tolist() if n_same else []
    others = np.setdiff1d(quiet, np.asarray(negatives, dtype=np.int64))
    negatives += rng.choice(others, size=min(len(positives) - n_same, others.size), replace=False).tolist()
    rows = [int(row) for row in positives] + [int(row) for row in negatives]
    truth = [True] * len(positives) + [False] * len(negatives)
    order = rng.permutation(len(rows))
    return [rows[i] for i in order], [truth[i] for i in order]


# -------------------------------------------------------------------- driver


class _Frames:
    def __init__(self, adapter, meta) -> None:
        self.adapter = adapter
        self.meta = meta
        self._clip: int | None = None
        self._windows: dict[int, np.ndarray] = {}

    def get(self, clip_index: int, window_index: int, t: int) -> np.ndarray:
        from interpretableintelligence.clips import load_clip

        if self._clip != clip_index:
            clip = load_clip(self.meta[clip_index])
            self._windows = {int(window.index): np.asarray(window.frames) for window in self.adapter.windows(clip)}
            self._clip = clip_index
        frames = self._windows[int(window_index)]
        tubelet = int(self.adapter.grid().tubelet)
        return frames[min(int(t) * tubelet, frames.shape[0] - 1)]


def _heat(tokens: Tokens, column: np.ndarray, row: int, grid) -> np.ndarray:
    key = _frame_key(tokens, row)
    mask = (tokens.clip == key[0]) & (tokens.window == key[1]) & (tokens.t == key[2])
    heat = np.zeros((grid.h, grid.w), dtype=np.float32)
    heat[tokens.h[mask], tokens.w[mask]] = np.maximum(column[mask], 0.0)
    return heat


def label_features(
    *,
    client: VisionClient,
    adapter,
    meta: list[dict],
    acts_dir: Path,
    sae,
    feature_ids: list[int],
    device,
    seed: int,
    log=print,
) -> tuple[dict[int, dict], dict]:
    """Labels for ``feature_ids`` and the evidence behind each (written to ``features/autointerp.json``)."""
    grid = adapter.grid()
    if int(grid.h) <= 1 or not feature_ids:
        return {}, {"version": AUTOINTERP_VERSION, "model": client.name, "features": []}
    model_name = label_model_name(client)
    tokens = gather_tokens(acts_dir, sae, feature_ids, device)
    # Visit frames clip by clip so each clip is decoded about once per feature.
    frames = _Frames(adapter, meta)
    labels: dict[int, dict] = {}
    evidence: list[dict] = []
    # Render every image first (one decoded clip at a time), then ask the model
    # about the features side by side: the wait is the provider's, not ours.
    jobs: list[dict] = []
    for column_index, feature_id in enumerate(feature_ids):
        column = tokens.acts[:, column_index]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(feature_id), 29]))
        train = np.flatnonzero(tokens.split == 0)
        examples = sorted(pick_top(tokens, column, train, k=N_EXPLAIN, per_clip=2), key=lambda row: int(tokens.clip[row]))
        record: dict = {"id": int(feature_id), "label": "", "description": "", "auroc_heldout": None, "holds": False}
        job: dict = {"id": int(feature_id), "record": record, "explain": None}
        jobs.append(job)
        if len(examples) < 3:
            record["note"] = "fires in fewer than 3 training frames"
            continue
        job["explain"] = [
            render(frames.get(*_frame_key(tokens, row)), (int(tokens.h[row]), int(tokens.w[row])), (grid.h, grid.w), _heat(tokens, column, row, grid))
            for row in examples
        ]
        job["examples"] = [list(_frame_key(tokens, row)) for row in examples]
        rows, truth = simulation_set(tokens, column, rng)
        job["rows"], job["truth"] = rows, truth
        job["cells"] = [
            {"frame": list(_frame_key(tokens, row)), "cell": [int(tokens.h[row]), int(tokens.w[row])], "fires": bool(fires)}
            for row, fires in zip(rows, truth, strict=True)
        ]
        job["batches"] = []
        if int(sum(truth)) >= MIN_HELDOUT_POSITIVES:
            order = sorted(range(len(rows)), key=lambda i: _frame_key(tokens, rows[i]))
            for start in range(0, len(order), SIMULATE_BATCH):
                batch = order[start : start + SIMULATE_BATCH]
                images = [
                    render(frames.get(*_frame_key(tokens, rows[i])), (int(tokens.h[rows[i]]), int(tokens.w[rows[i]])), (grid.h, grid.w))
                    for i in batch
                ]
                job["batches"].append((batch, images))

    def ask(job: dict) -> None:
        record = job["record"]
        if job["explain"] is None:
            return
        label, description = parse_label(client.complete(EXPLAIN_PROMPT.format(n=len(job["explain"])), job["explain"]))
        record.update(label=label, description=description, explain_examples=job["examples"])
        truth = job["truth"]
        n_pos = int(sum(truth))
        scores: list[float] = []
        if label and n_pos >= MIN_HELDOUT_POSITIVES:
            by_index: dict[int, float] = {}
            for batch, images in job["batches"]:
                prompt = SIMULATE_PROMPT.format(label=label.replace('"', "'"), description=description, n=len(batch))
                parsed = parse_scores(client.complete(prompt, images), len(batch))
                for offset, i in enumerate(batch):
                    # An unreadable reply scores every image 0: it can only hurt the label.
                    by_index[i] = 0.0 if parsed is None else parsed[offset]
            scores = [by_index[i] for i in range(len(truth))]
            value = auroc(np.asarray(scores), np.asarray(truth))
            record["auroc_heldout"] = value
            record["holds"] = bool(value >= LABEL_HOLD_MIN_AUROC)
        elif label:
            record["note"] = f"only {n_pos} held-out firings; the label cannot be tested"
        record["simulation"] = [
            dict(cell, score=score) for cell, score in zip(job["cells"], scores or [None] * len(truth), strict=True)
        ]

    with ThreadPoolExecutor(max_workers=max(1, min(LABEL_WORKERS, len(jobs)))) as pool:
        # list() re-raises a provider error from any feature.
        list(pool.map(ask, jobs))
    for job in jobs:
        record = job["record"]
        labels[job["id"]] = {"label": record["label"], "label_model": model_name, "label_holds_heldout": bool(record["holds"])}
        evidence.append(record)
        if job["explain"] is None:
            continue
        auc_text = "-" if record["auroc_heldout"] is None else f"{record['auroc_heldout']:.2f}"
        log(f"autointerp feature {job['id']}: {record['label'] or '-'!r}  held-out AUROC {auc_text}  holds={record['holds']}")
    return labels, {
        "version": AUTOINTERP_VERSION,
        "model": client.name,
        "hold_min_auroc": LABEL_HOLD_MIN_AUROC,
        "features": evidence,
    }
