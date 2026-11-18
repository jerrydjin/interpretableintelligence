"""Bring your own model: build a playground pack for a Hugging Face vision model picked on the page.

A visitor names a model (``facebook/dinov2-base``); the backend checks it is something the
playground can run and that it fits the limits, then a build job does what a learner would do
by hand::

    ii run hf --model <id> --clips <bundled robot clips> --no-score --no-serve
    ii export-live <run> --out <builds>/packs/<pack id>   (backend only, no ONNX)

and the new pack is served like any other (``PackRegistry``). This module touches the pipeline
and model tensors, so it is Python (AGENTS.md, "Language"). It knows nothing about HTTP or
Modal: ``playground.http`` routes to ``Builds``, and a ``runner`` (a thread here, a GPU function
on Modal) decides where ``run_build`` executes.

Cost guards, all in ``Limits``:

- A model is checked on the Hub before anything is downloaded: public, not gated, a vision
  transformer, at most ``max_params`` parameters.
- A pack is keyed by model, commit and layer, so a second request for the same model reuses it
  and never builds twice.
- At most ``max_running`` builds at once, ``builds_per_day`` new builds a day (UTC), and each one
  gets ``timeout_s`` before it counts as failed (the GPU function's own timeout is the hard stop).

Nothing here scores features: every feature of a built pack is unverified and explorable, and
no response can make one verified. ``ii run`` without ``--no-score`` is the way to verify.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Bump when a build would make a different pack from the same model, so old packs are not reused.
RECIPE = 1
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
STATES = ("queued", "running", "done", "failed")
# What the page may fetch from a pack through the API: the files a backend pack has besides its weights.
SERVED_FILES = re.compile(r"^(pack\.json|features\.json|samples/[A-Za-z0-9._-]+\.jpg)$")
# Model types the live split is known to handle (an .embeddings module, then the blocks, then a norm).
# Others are tried; the export checks the split reproduces the full forward and refuses if not.
KNOWN_TYPES = frozenset({"vit", "dinov2", "dinov2_with_registers", "deit", "vit_msn", "vit_mae", "dinov3_vit"})


@dataclass(frozen=True)
class Limits:
    max_params: int = 350_000_000
    builds_per_day: int = 10
    max_running: int = 1
    timeout_s: int = 1200
    n_clips: int = 40
    allow_local: bool = False  # local model directories (tests, ii serve-live); never on a public deployment

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Limits":
        env = dict(os.environ if env is None else env)
        base = cls()

        def num(key: str, default: int) -> int:
            raw = env.get(key)
            return int(raw) if raw not in (None, "") else default

        return cls(
            max_params=num("II_BUILD_MAX_PARAMS", base.max_params),
            builds_per_day=num("II_BUILD_PER_DAY", base.builds_per_day),
            max_running=num("II_BUILD_MAX_RUNNING", base.max_running),
            timeout_s=num("II_BUILD_TIMEOUT", base.timeout_s),
            n_clips=num("II_BUILD_CLIPS", base.n_clips),
        )


class BuildError(Exception):
    """An error the API reports. ``status`` and ``code`` are part of the contract (docs/PLAYGROUND-API.md)."""

    status = 400
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotFound(BuildError):
    status = 404
    code = "model_not_found"


class ModelNotSupported(BuildError):
    code = "model_not_supported"


class ModelTooLarge(BuildError):
    code = "model_too_large"


class BuildLimit(BuildError):
    status = 429
    code = "build_limit"


class JobNotFound(BuildError):
    status = 404
    code = "job_not_found"


class BuildsOff(BuildError):
    status = 503
    code = "builds_off"


@dataclass(frozen=True)
class ModelPlan:
    """A model that passed the checks, and the pack it becomes."""

    model: str
    revision: str
    layer: int
    n_layers: int
    params: int | None
    model_type: str
    pack_id: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HubModel:
    """What the Hub says about a model, without downloading its weights."""

    revision: str
    params: int | None
    gated: bool
    private: bool
    config: dict


def hub_lookup(model: str) -> HubModel:
    """Metadata and ``config.json`` from the Hub; nothing else is downloaded."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        info = HfApi().model_info(model, files_metadata=True)
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise ModelNotFound(f"no public model called {model} on Hugging Face") from exc
    except GatedRepoError as exc:
        raise ModelNotSupported(f"{model} is gated; the playground only builds public models") from exc
    params = None
    if info.safetensors is not None:
        params = int(info.safetensors.total)
    else:
        # No safetensors index: estimate from the largest weights format at 4 bytes a parameter,
        # so an unknown size can never slip past the limit.
        sizes: dict[str, int] = {}
        for sibling in info.siblings or []:
            name = sibling.rfilename
            for ext in (".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack"):
                if name.endswith(ext) and "/" not in name:
                    sizes[ext] = sizes.get(ext, 0) + int(sibling.size or 0)
        params = max(sizes.values()) // 4 if sizes else None
    try:
        path = hf_hub_download(model, "config.json", revision=info.sha)
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001  (any failure means there is no usable config)
        raise ModelNotSupported(f"{model} has no config.json the playground can read") from exc
    return HubModel(revision=str(info.sha), params=params, gated=bool(info.gated), private=bool(info.private), config=config)


def local_lookup(model: str) -> HubModel:
    """The same for a local model directory (tests, ``ii serve-live --builds`` with ``allow_local``)."""
    import hashlib

    path = Path(model).expanduser()
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        digest.update(file.read_bytes())
    from safetensors import safe_open

    params = 0
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(str(file), "np") as handle:
            for key in handle.keys():
                params += _prod(handle.get_slice(key).get_shape())
    return HubModel(revision=digest.hexdigest()[:40], params=params or None, gated=False, private=False, config=config)


def _prod(shape) -> int:
    out = 1
    for n in shape:
        out *= int(n)
    return out


def pack_id_for(model: str, revision: str, layer: int) -> str:
    """``facebook--dinov2-base-l6-1a2b3c4d``: one pack per model, commit, layer and recipe."""
    name = re.sub(r"[^a-z0-9._-]+", "-", model.lower().replace("/", "--")).strip("-.")[-60:].strip("-.")
    return f"{name}-l{int(layer)}-{revision[:8].lower()}r{RECIPE}"


def check_model(model: object, layer: object, limits: Limits, lookup: Callable[[str], HubModel] | None = None) -> ModelPlan:
    """Refuse anything the playground cannot run or that breaks a limit, before any download."""
    if not isinstance(model, str) or not model.strip():
        raise BuildError("model must be a Hugging Face model id, such as facebook/dinov2-base")
    model = model.strip()
    is_local = limits.allow_local and Path(model).expanduser().is_dir()
    if not is_local and not MODEL_ID.match(model):
        raise BuildError(f"{model!r} is not a Hugging Face model id (owner/name)")
    if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 1):
        raise BuildError("layer must be a block number from 1, or left out for the middle block")
    found = (local_lookup if is_local else (lookup or hub_lookup))(model)
    if found.private or found.gated:
        raise ModelNotSupported(f"{model} is private or gated; the playground only builds public models")
    if found.params is None and not is_local:
        raise ModelNotSupported(f"{model} does not say how big it is (no weights file the Hub lists), so the playground cannot check it fits")
    if found.params is not None and found.params > limits.max_params:
        raise ModelTooLarge(f"{model} has {found.params / 1e6:.0f}M parameters; the limit here is {limits.max_params / 1e6:.0f}M")
    config = found.config
    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else config
    model_type = str(vision.get("model_type") or config.get("model_type") or "")
    if vision.get("patch_size") is None or vision.get("image_size") is None:
        raise ModelNotSupported(
            f"{model} is not a vision transformer with a fixed image size; the playground takes ViT, DINOv2 and DeiT-style models for now"
        )
    if vision is not config:
        raise ModelNotSupported(f"{model} is an image-text model; the playground takes plain vision transformers for now")
    n_layers = int(vision.get("num_hidden_layers") or 0)
    if n_layers < 2:
        raise ModelNotSupported(f"{model} does not say how many blocks it has (num_hidden_layers)")
    if layer is None:
        layer = max(1, n_layers // 2)
    if layer > n_layers - 1:
        raise BuildError(f"{model} has {n_layers} blocks; pick a layer from 1 to {n_layers - 1} so some of the model runs after the edit")
    return ModelPlan(
        model=model,
        revision=found.revision,
        layer=int(layer),
        n_layers=n_layers,
        params=found.params,
        model_type=model_type,
        pack_id=pack_id_for(model, found.revision, int(layer)),
    )


# ------------------------------------------------------------------ the build itself

# Pipeline log lines that move the progress bar: (pattern, stage, start, end of that stage's share).
_STAGES = (
    ("loading", 0.0, 0.08),
    ("capture", 0.08, 0.55),
    ("train", 0.55, 0.85),
    ("features", 0.85, 0.93),
    ("pack", 0.93, 1.0),
)
_SHARE = {name: (start, end) for name, start, end in _STAGES}
_COUNT = re.compile(r"(?:capture: (\d+)/(\d+) clips|step (\d+)/(\d+))")
STAGE_TEXT = {
    "queued": "Waiting for a GPU",
    "loading": "Downloading the model",
    "capture": "Reading the robot videos",
    "train": "Training the dictionary",
    "features": "Measuring every feature",
    "pack": "Packing it for the playground",
    "done": "Ready",
}


def progress_from_log(line: str, stage: str) -> tuple[str, float | None]:
    """The stage and overall fraction a pipeline log line implies (``None``: no new fraction)."""
    if line == "run capture" or line == "skip capture":
        return "capture", _SHARE["capture"][0]
    if line == "run sae" or line == "skip sae":
        return "train", _SHARE["train"][0]
    if line == "run features" or line == "skip features":
        return "features", _SHARE["features"][0]
    match = _COUNT.search(line)
    if match:
        done, total = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        which = "capture" if match.group(1) else "train"
        start, end = _SHARE[which]
        return which, start + (end - start) * int(done) / max(int(total), 1)
    return stage, None


def run_build(plan: ModelPlan, root: Path, clips_dir: Path, report: Callable[[str, float, str | None], None]) -> Path:
    """Build ``plan``'s pack under ``root/packs/<pack id>``; ``report(stage, fraction, line)`` as it goes.

    Runs where the runner puts it (a GPU on Modal). The run folder lives under ``root/work`` so a
    retried build resumes from the last finished stage instead of starting again."""
    import shutil

    from interpretableintelligence.adapters.hf import make_spec
    from interpretableintelligence.live import export_live
    from interpretableintelligence.pipeline import execute_pipeline

    root = Path(root)
    run_dir = root / "work" / plan.pack_id
    final = root / "packs" / plan.pack_id
    staging = root / "packs" / f".{plan.pack_id}.partial"
    stage = "loading"
    report(stage, 0.0, None)

    def log(line: str) -> None:
        nonlocal stage
        line = str(line)
        stage, fraction = progress_from_log(line, stage)
        report(stage, fraction if fraction is not None else -1.0, line)

    execute_pipeline(
        make_spec(plan.model),
        clips_dir,
        run_dir,
        layer=plan.layer,
        score=False,
        serve=False,
        log=log,
    )
    report("pack", _SHARE["pack"][0], None)
    if staging.exists():
        shutil.rmtree(staging)
    short = plan.model.rstrip("/").split("/")[-1]
    export_live(
        run_dir,
        staging,
        clips_dir=clips_dir,
        title=f"{short}, block {plan.layer} of {plan.n_layers}, dictionary trained on robot videos",
        log=lambda line: report("pack", -1.0, str(line)),
        browser=False,
    )
    pack = json.loads((staging / "pack.json").read_text(encoding="utf-8"))
    pack["built"] = {"model": plan.model, "revision": plan.revision, "recipe": RECIPE, "params": plan.params}
    (staging / "pack.json").write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    # The activations are the bulk of the run folder and are not needed once the pack exists.
    shutil.rmtree(run_dir / "acts", ignore_errors=True)
    return final


# ------------------------------------------------------------------ jobs on disk

def _now() -> float:
    return time.time()


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class Job:
    id: str
    pack_id: str
    model: str
    layer: int
    state: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str | None = None
    created: float = field(default_factory=_now)
    updated: float = field(default_factory=_now)
    plan: dict = field(default_factory=dict)

    def public(self, timeout_s: int) -> dict:
        elapsed = (self.updated if self.state in ("done", "failed") else _now()) - self.created
        return {
            "id": self.id,
            "pack": self.pack_id,
            "model": self.model,
            "layer": self.layer,
            "state": self.state,
            "stage": self.stage,
            "stage_text": STAGE_TEXT.get(self.stage, self.stage),
            "progress": round(float(self.progress), 3),
            "message": self.message,
            "elapsed_s": round(max(elapsed, 0.0), 1),
            "timeout_s": timeout_s,
        }


class JobStore:
    """Jobs as small JSON files under ``root/jobs`` (a Modal Volume when hosted), one per pack.

    ``sync()`` and ``persist()`` are hooks for the storage: on Modal they reload and commit the
    Volume, so the web containers and the GPU function see each other's writes."""

    def __init__(self, root: Path, sync: Callable[[], None] = lambda: None, persist: Callable[[], None] = lambda: None) -> None:
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        (self.root / "packs").mkdir(parents=True, exist_ok=True)
        self._sync = sync
        self._persist = persist
        self._lock = threading.Lock()

    def path(self, pack_id: str) -> Path:
        return self.root / "jobs" / f"{pack_id}.json"

    def load(self, pack_id: str) -> Job | None:
        try:
            data = json.loads(self.path(pack_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})

    def save(self, job: Job, persist: bool = True) -> None:
        job.updated = _now()
        path = self.path(job.pack_id)
        tmp = path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(asdict(job)) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        if persist:
            self._persist()

    def all(self) -> list[Job]:
        out = []
        for path in sorted((self.root / "jobs").glob("*.json")):
            job = self.load(path.stem)
            if job is not None:
                out.append(job)
        return out

    def by_id(self, job_id: str) -> Job | None:
        for job in self.all():
            if job.id == job_id:
                return job
        return None

    def pack_ready(self, pack_id: str) -> bool:
        return (self.root / "packs" / pack_id / "pack.json").is_file()

    def sync(self) -> None:
        self._sync()


class Builds:
    """What the HTTP layer calls: submit a model, read a job, find a built pack's files."""

    def __init__(self, store: JobStore, runner: Callable[[ModelPlan], None], limits: Limits, lookup: Callable[[str], HubModel] | None = None) -> None:
        self.store = store
        self.runner = runner
        self.limits = limits
        self.lookup = lookup
        self._lock = threading.Lock()
        self._plans: OrderedDict[tuple[str, int | None], tuple[float, ModelPlan]] = OrderedDict()

    def info(self) -> dict:
        return {
            "max_params": self.limits.max_params,
            "builds_per_day": self.limits.builds_per_day,
            "timeout_s": self.limits.timeout_s,
            "n_clips": self.limits.n_clips,
        }

    def _plan(self, model: object, layer: object) -> ModelPlan:
        # The Hub answer for a model is reused for ten minutes, so a page polling or a visitor
        # pressing the button twice does not ask the Hub again.
        key = (str(model).strip(), layer if isinstance(layer, int) else None)
        hit = self._plans.get(key)
        if hit is not None and _now() - hit[0] < 600:
            return hit[1]
        plan = check_model(model, layer, self.limits, self.lookup)
        self._plans[key] = (_now(), plan)
        while len(self._plans) > 256:
            self._plans.popitem(last=False)
        return plan

    def submit(self, model: object, layer: object = None) -> tuple[dict, bool]:
        """The job for this model (an existing one when the pack is built or building). ``True`` if new."""
        plan = self._plan(model, layer)
        with self._lock:
            self.store.sync()
            existing = self.store.load(plan.pack_id)
            if self.store.pack_ready(plan.pack_id):
                if existing is None or existing.state != "done":
                    existing = existing or Job(id=uuid.uuid4().hex[:16], pack_id=plan.pack_id, model=plan.model, layer=plan.layer, plan=plan.as_dict())
                    existing.state, existing.stage, existing.progress = "done", "done", 1.0
                    self.store.save(existing)
                return existing.public(self.limits.timeout_s), False
            if existing is not None:
                self._expire(existing)
                if existing.state in ("queued", "running"):
                    return existing.public(self.limits.timeout_s), False
            jobs = self.store.all()
            for job in jobs:
                self._expire(job)
            running = sum(1 for job in jobs if job.state in ("queued", "running"))
            if running >= self.limits.max_running:
                raise BuildLimit(
                    f"{running} model{'s are' if running != 1 else ' is'} already being built here; try again in a few minutes"
                )
            today = _day(_now())
            started_today = sum(1 for job in jobs if _day(job.created) == today)
            if started_today >= self.limits.builds_per_day:
                raise BuildLimit(f"this playground has built its {self.limits.builds_per_day} models for today (UTC); try again tomorrow")
            job = Job(id=uuid.uuid4().hex[:16], pack_id=plan.pack_id, model=plan.model, layer=plan.layer, plan=plan.as_dict())
            self.store.save(job)
        try:
            self.runner(plan)
        except Exception as exc:  # noqa: BLE001  (the job records why it never started)
            job.state, job.message = "failed", f"the build could not start: {exc}"
            self.store.save(job)
        return job.public(self.limits.timeout_s), True

    def job(self, job_id: str) -> dict:
        self.store.sync()
        job = self.store.by_id(job_id)
        if job is None:
            raise JobNotFound(f"no build job {job_id!r}")
        self._expire(job)
        return job.public(self.limits.timeout_s)

    def file(self, pack_id: str, name: str) -> Path:
        if not SERVED_FILES.match(name) or "/" in pack_id or pack_id.startswith("."):
            raise BuildError(f"{name!r} is not a file the playground serves")
        path = self.store.root / "packs" / pack_id / name
        if not path.is_file():
            self.store.sync()
        if not path.is_file():
            raise JobNotFound(f"pack {pack_id} has no {name}")
        return path

    def _expire(self, job: Job) -> None:
        """A build that went quiet for longer than its timeout failed (its container is gone)."""
        if job.state in ("queued", "running") and _now() - job.updated > self.limits.timeout_s + 120:
            job.state, job.message = "failed", "the build stopped answering (timed out); try again"
            self.store.save(job)


def execute_job(plan: ModelPlan, store: JobStore, clips_dir: Path, every_s: float = 5.0) -> None:
    """The runner's body: run the build and keep the job file up to date (written at most every ``every_s``)."""
    job = store.load(plan.pack_id) or Job(id=uuid.uuid4().hex[:16], pack_id=plan.pack_id, model=plan.model, layer=plan.layer, plan=plan.as_dict())
    job.state, job.message = "running", None
    store.save(job)
    last = [0.0]

    def report(stage: str, fraction: float, _line: str | None) -> None:
        changed = stage != job.stage
        job.stage = stage
        if fraction >= 0:
            job.progress = max(job.progress, min(float(fraction), 0.99))
        if changed or _now() - last[0] >= every_s:
            last[0] = _now()
            store.save(job)

    try:
        run_build(plan, store.root, clips_dir, report)
    except Exception as exc:  # noqa: BLE001  (every failure is reported to the page)
        job.state, job.message = "failed", _friendly(exc)
        store.save(job)
        raise
    job.state, job.stage, job.progress = "done", "done", 1.0
    store.save(job)


def _friendly(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    if "cannot be split" in text or "does not reproduce the full forward" in text or "image transformers" in text:
        return f"this model's layout is not one the playground can cut in two yet ({text})"
    return text[:500]


def thread_runner(store: JobStore, clips_dir: Path) -> Callable[[ModelPlan], None]:
    """Build in a background thread of this process: ``ii serve-live --builds`` and the tests."""

    def start(plan: ModelPlan) -> None:
        def body() -> None:
            try:
                execute_job(plan, store, clips_dir)
            except Exception:  # noqa: BLE001  (already recorded on the job)
                pass

        threading.Thread(target=body, name=f"build-{plan.pack_id}", daemon=True).start()

    return start
