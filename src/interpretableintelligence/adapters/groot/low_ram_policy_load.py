#!/usr/bin/env python3
"""Low-RAM loader for lerobot 0.7.0 GrootPolicy and PI05Policy.

lerobot's PreTrainedPolicy.from_pretrained ignores torch_dtype / low_cpu_mem_usage /
device_map: it constructs the full module (often fp32 on CPU) and then loads the
entire state dict. PI05Policy.from_pretrained additionally uses safetensors.torch.load_file
when use_native_pretrained_loader is False.

This loader:
  * fixes GrootConfig.base_model_path when a checkpoint hard-codes a stale local
    snapshot path (invalid as a Hugging Face repo id)
  * builds the module under torch.device("meta") without downloading GR00T base
    weights or Cosmos-Reason2-2B (architecture comes from the base config.json;
    the finetuned model.safetensors already has every weight)
  * streams tensors one at a time with safetensors.safe_open(..., device="cpu"),
    casts floating tensors to the target dtype, moves them to the target device,
    and load_state_dict(..., assign=True); shares tied weights; gc after each
  * prints RSS at every stage so a Colab RAM crash is diagnosable
"""

from __future__ import annotations

import dataclasses
import gc
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open


class CpuWeightArena:
    """File-backed storage for CPU param materialization.

    Anonymous fp16 weights for π0.5 are ~7.2 GiB; a 15 GiB box shared with an SAE
    job cannot hold that as anon RSS. Tensors are written into a shared mmap and
    earlier pages are dropped (msync + MADV_DONTNEED) so peak RSS stays a small
    window plus the current tensor. The model remains complete; pages fault back
    in from disk on first use. Not used when the target device is CUDA.
    """

    def __init__(self, nbytes: int, path: str | Path):
        import mmap as _mmap

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.nbytes = int(nbytes)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        try:
            os.ftruncate(fd, self.nbytes)
            self.mm = _mmap.mmap(fd, self.nbytes)
        finally:
            os.close(fd)
        # The mapping keeps the pages alive; unlinking now means a crash or a normal exit
        # never leaves a multi-GB file behind in /tmp (POSIX only; Windows keeps the file).
        try:
            self.path.unlink()
        except OSError:
            pass
        self.off = 0
        self._views: list[torch.Tensor] = []

    def _align(self, n: int, align: int = 64) -> int:
        return (n + align - 1) & ~(align - 1)

    def put(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.detach().contiguous()
        n = t.nelement() * t.element_size()
        start = self._align(self.off)
        if start + n > self.nbytes:
            raise RuntimeError(f"weight arena overflow: need {start+n}, have {self.nbytes}")
        dest = torch.frombuffer(memoryview(self.mm)[start : start + n], dtype=t.dtype).view(t.shape)
        dest.copy_(t)
        self.off = start + n
        self._views.append(dest)
        # Periodically flush and drop older pages so RSS does not accumulate.
        if self.off == n or self.off % (256 * 1024 * 1024) < n + 64:
            self.drop_old_pages()
        return dest

    def drop_old_pages(self) -> None:
        import ctypes
        import mmap as _mmap

        try:
            self.mm.flush()
        except Exception:
            pass
        # Keep the last 64 MiB of written data warm; drop the rest.
        keep = 64 * 1024 * 1024
        drop = (self.off - keep) & ~4095
        if drop <= 0:
            return
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # MADV_DONTNEED = 4; file-backed MAP_SHARED pages re-read from disk.
        ok = libc.madvise(ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.mm))), drop, 4)
        if ok != 0:
            # Fallback: posix_fadvise on the file.
            try:
                os.posix_fadvise(self.path.open("rb").fileno(), 0, drop, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
        try:
            libc.malloc_trim(0)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.mm.flush()
        except Exception:
            pass


def _use_cpu_arena(nbytes: int, log: Callable[[str], None] | None = None) -> bool:
    """File-backed weights only when /tmp can hold them.

    Writing through a shared mapping on a full disk kills the process with
    SIGBUS and no traceback, so with too little free space the weights stay in
    RAM instead. ``GROOT_CPU_ARENA=0`` or ``1`` forces the choice.
    """
    forced = os.environ.get("GROOT_CPU_ARENA", "").strip()
    if forced in {"0", "1"}:
        return forced == "1"
    import shutil

    free = shutil.disk_usage("/tmp").free
    if free >= nbytes + (1 << 30):
        return True
    message = f"/tmp has {free/1e9:.1f} GB free, under the {nbytes/1e9:.1f} GB weight arena; keeping weights in RAM"
    (log or (lambda m: print(m, flush=True)))(message)
    return False


def _arena_bytes_for_file(fh, dtype: torch.dtype) -> int:
    total = 64
    for key in fh.keys():
        # safe_open slice has get_shape / get_dtype
        try:
            sl = fh.get_slice(key)
            shape = tuple(sl.get_shape())
            src_dt = sl.get_dtype()
        except Exception:
            continue
        numel = 1
        for s in shape:
            numel *= int(s)
        # floats become `dtype`; ints keep 4 or 8 bytes roughly — over-allocate 2x for ints.
        if str(src_dt) in {"F32", "F16", "BF16", "F64"}:
            elem = torch.tensor([], dtype=dtype).element_size()
        elif str(src_dt) in {"I64", "U64"}:
            elem = 8
        else:
            elem = 4
        total += ((numel * elem + 63) & ~63) + 64
    return total


GROOT_BASE_HUB_ID = "nvidia/GR00T-N1.7-3B"
GROOT_CKPT_DEFAULT = "nvidia/gr00t17-lerobot-libero_10-640"
PI05_CKPT_DEFAULT = "lerobot/pi05_libero_finetuned_v044"


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            return 0


def rss_gb() -> float:
    return rss_bytes() / (1024**3)


def hwm_gb() -> float:
    """Peak RSS from /proc/self/status VmHWM (kB)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        return rss_gb()
    return rss_gb()


def log_rss(step: str, log: Callable[[str], None] | None = None) -> float:
    gb = rss_gb()
    msg = f"[rss] {step}: {gb:.3f} GiB"
    if log is not None:
        log(msg)
    else:
        print(msg, flush=True)
    return gb


def _is_existing_local_dir(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path).expanduser()
    return p.is_dir() and (p / "config.json").is_file()


def _looks_like_hub_id(path: str | None) -> bool:
    if not path:
        return False
    s = str(path).strip()
    if s.startswith("/") or s.startswith(".") or "\\" in s or "snapshots" in s or "models--" in s:
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts) and all(c.isascii() for c in s)


def fix_groot_base_model_path(config: Any, override: str | None = GROOT_BASE_HUB_ID) -> str:
    """Rewrite a stale local snapshot path to a hub id or a real local snapshot.

    nvidia/gr00t17-lerobot-libero_10-640 config.json sets
      base_model_path = /cache/huggingface/models--nvidia--GR00T-N1.7-3B/snapshots/<hash>
    which is not a valid repo id. lerobot then calls snapshot_download on it and
    raises OSError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'").
    """
    raw = getattr(config, "base_model_path", None)
    if _is_existing_local_dir(raw):
        return str(Path(raw).expanduser())
    if _looks_like_hub_id(raw) and override is None:
        return str(raw)
    fixed = override or GROOT_BASE_HUB_ID
    if hasattr(config, "base_model_path"):
        config.base_model_path = fixed
    return fixed


def _noop_module_to_if_real_device(orig):
    """Keep dtype-only .to() (e.g. PI05's to_bfloat16) but skip device moves during meta init."""

    def _to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        if device is None and args and not isinstance(args[0], torch.dtype):
            device = args[0]
        if device is not None:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type != "meta":
                return self
        return orig(self, *args, **kwargs)

    return _to


def _construct_on_meta(factory: Callable[[], Any], log: Callable[[str], None] | None = None) -> Any:
    orig_to = torch.nn.Module.to
    torch.nn.Module.to = _noop_module_to_if_real_device(orig_to)  # type: ignore[method-assign]
    try:
        try:
            from transformers.modeling_utils import no_init_weights
        except Exception:
            from contextlib import nullcontext as no_init_weights  # type: ignore
        with no_init_weights(), torch.device("meta"):
            obj = factory()
    finally:
        torch.nn.Module.to = orig_to  # type: ignore[method-assign]
    log_rss("after meta construct", log)
    return obj


def _named_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        out[name] = param
    for name, buf in model.named_buffers(remove_duplicate=False):
        if name not in out:
            out[name] = buf
    return out


def _remap_key(key: str, model_keys: set[str]) -> str | None:
    if key in model_keys:
        return key
    candidates = [key]
    for prefix in ("_groot_model.", "model."):
        if key.startswith(prefix):
            candidates.append(key[len(prefix) :])
        candidates.append(prefix + key)
    extra: list[str] = []
    for c in candidates:
        # transformers version drift: SiglipVisionModel wraps embeddings in .vision_model
        if ".vision_tower.vision_model." in c:
            extra.append(c.replace(".vision_tower.vision_model.", ".vision_tower."))
        if ".vision_tower." in c and ".vision_tower.vision_model." not in c:
            extra.append(c.replace(".vision_tower.", ".vision_tower.vision_model."))
    for c in candidates + extra:
        if c in model_keys:
            return c
    return None



def _apply_remaining_ties(model: torch.nn.Module, ties: dict[str, str]) -> int:
    """Assign tied partners that were not present as their own tensors in the file."""
    tensors = _named_tensors(model)
    model_keys = set(tensors)
    shared = 0
    for a, b in list(ties.items()):
        da, db = _remap_key(a, model_keys), _remap_key(b, model_keys)
        if da is None or db is None:
            continue
        ta, tb = tensors.get(da), tensors.get(db)
        if ta is None or tb is None:
            continue
        a_meta = ta.device.type == "meta"
        b_meta = tb.device.type == "meta"
        if a_meta and not b_meta:
            model.load_state_dict({da: tb}, strict=False, assign=True)
            shared += 1
        elif b_meta and not a_meta:
            model.load_state_dict({db: ta}, strict=False, assign=True)
            shared += 1
    return shared


def _is_meta(t: torch.Tensor | None) -> bool:
    return t is not None and getattr(t, "device", None) is not None and t.device.type == "meta"


def _rope_theta(cfg: Any) -> float:
    """Theta from this module's own config. No default: missing theta is a bug, not 10000."""
    if cfg is None:
        raise RuntimeError("RoPE module has no config; cannot determine rope_theta")
    rp = getattr(cfg, "rope_parameters", None)
    th = None
    if isinstance(rp, dict):
        th = rp.get("rope_theta")
    elif rp is not None:
        th = getattr(rp, "rope_theta", None)
    if th is None:
        th = getattr(cfg, "rope_theta", None)
    if th is None:
        raise RuntimeError(
            f"RoPE config {type(cfg).__name__} has no rope_theta (rope_parameters={rp!r})"
        )
    return float(th)


def _inv_freq_from_theta(theta: float, n: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Closed-form default/axial RoPE: inv_freq[i] = theta^{-i/n} (equiv. arange(0, 2n, 2)/(2n))."""
    i = torch.arange(0, n, device=device, dtype=dtype)
    return 1.0 / (theta ** (i / n))


def _theta_from_inv_freq(inv_freq: torch.Tensor) -> float:
    """Invert inv_freq[i] = theta^{-i/n} from a mid-to-high bin (i>0)."""
    x = inv_freq.detach().float().reshape(-1)
    n = x.numel()
    if n < 2:
        return float("nan")
    i = n - 1
    val = float(x[i])
    if val <= 0:
        return float("nan")
    return float(val ** (-n / i))


def _inv_freq_theta_diff(inv_freq: torch.Tensor, theta: float) -> float:
    n = int(inv_freq.numel())
    expected = _inv_freq_from_theta(theta, n, device=inv_freq.device, dtype=torch.float32)
    return float((inv_freq.detach().float() - expected).abs().max())


def verify_rotary_inv_freq(
    module: torch.nn.Module,
    name: str = "",
    *,
    inv_freq: torch.Tensor | None = None,
    atol: float = 1e-5,
) -> float:
    """Raise RuntimeError unless inv_freq matches this module's own config rope_theta.

    Qwen3-VL text uses rope_theta=5_000_000; vision uses its own (typically 10_000).
    """
    label = name or type(module).__name__
    inv = module.inv_freq if inv_freq is None else inv_freq
    if not isinstance(inv, torch.Tensor):
        raise RuntimeError(f"rope {label}: missing inv_freq")
    if _is_meta(inv):
        raise RuntimeError(f"rope {label}: inv_freq is still on the meta device")
    theta = _rope_theta(getattr(module, "config", None))
    max_diff = _inv_freq_theta_diff(inv, theta)
    if max_diff > atol:
        actual = _theta_from_inv_freq(inv)
        raise RuntimeError(
            f"rope {label}: inv_freq does not match config theta "
            f"(expected theta={theta}, actual theta≈{actual}, max |Δ inv_freq|={max_diff})"
        )
    return max_diff


def rebuild_rotary_inv_freq(module: torch.nn.Module, device: torch.device | str) -> tuple[torch.Tensor, float | None]:
    """Recompute a RoPE module's inv_freq exactly as its own __init__ does, then verify theta.

    v7 bug (fixed v8): the old code called `fn.__func__` on a staticmethod (AttributeError
    swallowed) and fell back to theta=getattr(config, "rope_theta", 10000). transformers 5
    keeps theta in config.rope_parameters, so Qwen3-VL's text RoPE (theta 5e6) was rebuilt
    with theta 1e4 -> wrong positional encoding in every backbone LLM layer.

    No silent fallback: construct a fresh real instance of the same class from
    module.config (CPU, a few floats) and take its buffer. Construction failure raises.
    The rebuilt buffer is then checked against this module's own config rope_theta
    (text 5e6, vision its own); a mismatch raises RuntimeError.
    """
    device = torch.device(device)
    cfg = getattr(module, "config", None)
    label = type(module).__name__
    if cfg is None:
        raise RuntimeError(f"rope {label}: module has no config; cannot rebuild inv_freq")
    try:
        with torch.device("cpu"):
            ref = type(module)(cfg)
    except Exception as e:
        raise RuntimeError(
            f"rope {label}: failed to construct a reference RoPE from {type(cfg).__name__}: {e}"
        ) from e
    inv = getattr(ref, "inv_freq", None)
    if not isinstance(inv, torch.Tensor) or inv.device.type == "meta":
        got = None if inv is None else f"{type(inv).__name__} on {getattr(inv, 'device', None)}"
        raise RuntimeError(f"rope {label}: class init did not produce a real inv_freq (got {got})")
    inv_freq = inv.detach().to(device=device, dtype=torch.float32)
    verify_rotary_inv_freq(module, label, inv_freq=inv_freq)
    return inv_freq, getattr(ref, "attention_scaling", None)


def repair_rotary_buffers(model: torch.nn.Module, log: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Post-load check/repair of every RoPE buffer against its class's own init (idempotent).

    Repairs an already-loaded policy in place, then verifies each buffer against that
    module's own config rope_theta. Verification failure raises; it is not swallowed.
    """
    out: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        inv = getattr(module, "inv_freq", None)
        if not isinstance(inv, torch.Tensor) or _is_meta(inv) or getattr(module, "config", None) is None:
            continue
        ref, scaling = rebuild_rotary_inv_freq(module, inv.device)
        if ref.shape != inv.shape:
            raise RuntimeError(
                f"rope {name}: shape {tuple(inv.shape)} != reference {tuple(ref.shape)} "
                f"(expected theta={_rope_theta(module.config)})"
            )
        diff = float((inv.float() - ref).abs().max())
        if diff > 0:
            module.inv_freq = torch.nn.Buffer(ref.to(inv.dtype), persistent=False)
            if isinstance(getattr(module, "original_inv_freq", None), torch.Tensor):
                module.original_inv_freq = torch.nn.Buffer(ref.clone().to(inv.dtype), persistent=False)
            if scaling is not None:
                module.attention_scaling = scaling
        # Raise if the installed buffer still does not match this module's config theta.
        verify_rotary_inv_freq(module, name)
        out.append({
            "module": name,
            "class": type(module).__name__,
            "theta": _rope_theta(module.config),
            "max_abs_diff_before": diff,
            "repaired": diff > 0,
        })
    msg = f"[rope] {out}"
    (log or print)(msg)
    return out


def materialize_computed_buffers(model: torch.nn.Module, device: torch.device | str) -> list[str]:
    """Rebuild tiny non-persistent buffers left on meta after a weight stream (RoPE, position ids, embed scale)."""
    device = torch.device(device)
    done: list[str] = []
    for name, module in model.named_modules():
        if hasattr(module, "embed_scale") and hasattr(module, "scalar_embed_scale") and _is_meta(getattr(module, "embed_scale", None)):
            module.embed_scale = torch.nn.Buffer(
                torch.tensor(module.scalar_embed_scale, device=device),
                persistent=False,
            )
            done.append(f"{name}.embed_scale")
        if hasattr(module, "position_ids") and hasattr(module, "num_positions") and _is_meta(getattr(module, "position_ids", None)):
            module.position_ids = torch.nn.Buffer(
                torch.arange(int(module.num_positions), device=device).expand((1, -1)),
                persistent=False,
            )
            done.append(f"{name}.position_ids")
        inv = getattr(module, "inv_freq", None)
        if _is_meta(inv):
            inv_freq, scaling = rebuild_rotary_inv_freq(module, device)
            if scaling is not None:
                module.attention_scaling = scaling
            module.inv_freq = torch.nn.Buffer(inv_freq, persistent=False)
            done.append(f"{name}.inv_freq")
            if _is_meta(getattr(module, "original_inv_freq", None)):
                module.original_inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
                done.append(f"{name}.original_inv_freq")
            verify_rotary_inv_freq(module, name)
    return done


def stream_safetensors_into(
    model: torch.nn.Module,
    safetensor_path: str | Path,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stream one tensor at a time: CPU open → cast → device → assign=True."""
    device = torch.device(device)
    path = str(safetensor_path)
    model_keys = set(_named_tensors(model))
    assigned: set[str] = set()
    unexpected: list[str] = []
    tied_shared = 0
    max_tensor_mb = 0.0

    log_rss(f"before safe_open {path}", log)
    with safe_open(path, framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
        raw_meta = {}
        try:
            raw_meta = fh.metadata() or {}
        except Exception:
            raw_meta = {}
        # Bidirectional tie map so we only materialize one of (embed, lm_head).
        ties: dict[str, str] = {}
        for a, b in raw_meta.items():
            if not a or not b or a == "__metadata__" or not isinstance(b, str):
                continue
            ties[a] = b
            ties[b] = a
        n = len(keys)
        arena = None
        if device.type == "cpu" and _use_cpu_arena(_arena_bytes_for_file(fh, dtype), log):
            arena_bytes = _arena_bytes_for_file(fh, dtype)
            arena_path = Path("/tmp") / f"weight_arena_{os.getpid()}.bin"
            arena = CpuWeightArena(arena_bytes, arena_path)
            if log:
                log(f"CPU weight arena {arena_bytes/1e9:.3f} GB at {arena_path}")
            else:
                print(f"CPU weight arena {arena_bytes/1e9:.3f} GB at {arena_path}", flush=True)
        if log:
            log(f"streaming {n} tensors from {path} → {device}/{dtype}")
        else:
            print(f"streaming {n} tensors from {path} → {device}/{dtype}", flush=True)

        for i, key in enumerate(keys, 1):
            dest = _remap_key(key, model_keys)
            if dest is None:
                unexpected.append(key)
                continue

            partner_name = ties.get(key) or ties.get(dest)
            partner_dest = _remap_key(partner_name, model_keys) if partner_name else None
            if partner_dest and partner_dest in assigned:
                partner = _named_tensors(model).get(partner_dest)
                if partner is not None and partner.device.type != "meta":
                    model.load_state_dict({dest: partner}, strict=False, assign=True)
                    assigned.add(dest)
                    tied_shared += 1
                    continue

            tensor = fh.get_tensor(key)
            nbytes = tensor.nelement() * tensor.element_size()
            max_tensor_mb = max(max_tensor_mb, nbytes / (1024**2))
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            tensor = tensor.to(device=device)
            if arena is not None:
                backed = arena.put(tensor)
                del tensor
                tensor = backed
            model.load_state_dict({dest: tensor}, strict=False, assign=True)
            assigned.add(dest)
            del tensor
            if i % 16 == 0 or i == n or nbytes >= 256 * 1024 * 1024:
                gc.collect()
                try:
                    import ctypes
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except Exception:
                    pass
                log_rss(f"stream {i}/{n} assigned={len(assigned)}", log)

    tied_shared += _apply_remaining_ties(model, ties)
    assigned = {k for k, t in _named_tensors(model).items() if t.device.type != "meta"}
    rebuilt = materialize_computed_buffers(model, device)
    if rebuilt:
        if log:
            log(f"materialized computed buffers: {rebuilt}")
        else:
            print(f"materialized computed buffers: {rebuilt}", flush=True)
    leftover_meta = [k for k, t in _named_tensors(model).items() if getattr(t, "device", None) and t.device.type == "meta"]
    missing = sorted(set(_named_tensors(model)) - assigned - set(rebuilt))
    # Buffers that stay unassigned may be non-persistent; report meta leftovers clearly.
    gc.collect()
    log_rss("after stream assign", log)
    if arena is not None:
        arena.drop_old_pages()
        model._weight_arena = arena  # noqa: SLF001 — keep mmap alive
    return {
        "n_file_tensors": n,
        "n_assigned": len(assigned),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "n_tied_shared": tied_shared,
        "missing_sample": missing[:12],
        "unexpected_sample": unexpected[:12],
        "leftover_meta_sample": leftover_meta[:12],
        "n_leftover_meta": len(leftover_meta),
        "rebuilt_buffers": rebuilt,
        "max_tensor_mb": round(max_tensor_mb, 1),
        "cpu_arena_bytes": None if arena is None else arena.nbytes,
        "rss_gb": round(rss_gb(), 3),
    }


def _set_config_device(config: Any, device: str) -> None:
    if hasattr(config, "device"):
        config.device = device


def load_pi05_low_ram(
    ckpt: str = PI05_CKPT_DEFAULT,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    log: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    log_rss("pi05 start", log)
    config = PI05Config.from_pretrained(ckpt)
    # Do not set config.dtype = float16: PI05Config.__post_init__ only allows bf16/fp32.
    # Weights are streamed as `dtype` regardless.
    _set_config_device(config, "meta")
    log_rss("pi05 config loaded", log)

    def factory():
        return PI05Policy(config)

    policy = _construct_on_meta(factory, log)
    model_file = hf_hub_download(ckpt, SAFETENSORS_SINGLE_FILE) if not Path(ckpt).is_dir() else str(Path(ckpt) / SAFETENSORS_SINGLE_FILE)
    if Path(ckpt).is_dir() and (Path(ckpt) / SAFETENSORS_SINGLE_FILE).is_file():
        model_file = str(Path(ckpt) / SAFETENSORS_SINGLE_FILE)
    else:
        model_file = hf_hub_download(ckpt, SAFETENSORS_SINGLE_FILE)
    stats = stream_safetensors_into(policy, model_file, dtype=dtype, device=device, log=log)
    _set_config_device(config, device)
    policy.eval()
    log_rss("pi05 done", log)
    stats["ckpt"] = ckpt
    stats["family"] = "pi05"
    stats["base_model_path_fix"] = None
    return policy, stats


def load_groot_low_ram(
    ckpt: str = GROOT_CKPT_DEFAULT,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    base_model_path: str = GROOT_BASE_HUB_ID,
    log: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config, _tie_unused_qwen_lm_head
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    log_rss("groot start", log)
    config = GrootConfig.from_pretrained(ckpt)
    raw_path = getattr(config, "base_model_path", None)
    fixed = fix_groot_base_model_path(config, override=base_model_path)
    if log:
        log(f"GrootConfig.base_model_path {raw_path!r} → {fixed!r}")
    else:
        print(f"GrootConfig.base_model_path {raw_path!r} → {fixed!r}", flush=True)
    # Do not set config.dtype = float16: GrootConfig only allows fp32/bf16.
    _set_config_device(config, "meta")
    log_rss("groot config loaded", log)

    # Only the base *config* is required to build the graph. The 12.6 GB finetuned
    # model.safetensors already contains backbone + action head. Do not
    # snapshot_download nvidia/GR00T-N1.7-3B (~6.9 GB shards) or Cosmos-Reason2-2B
    # (Qwen3Backbone._from_backbone_config uses a hard-coded config for that id).
    base_cfg = GR00TN17Config.from_pretrained(fixed)
    log_rss("groot base config.json loaded (no base weights)", log)

    orig_create = GrootPolicy._create_groot_model

    def _create_empty(self):
        model = GR00TN17(
            base_cfg,
            transformers_loading_kwargs={"trust_remote_code": True},
            load_backbone_weights=False,
        )
        backbone = getattr(model, "backbone", None)
        qwen_model = getattr(backbone, "model", None)
        if qwen_model is not None:
            _tie_unused_qwen_lm_head(qwen_model)
        return model

    GrootPolicy._create_groot_model = _create_empty  # type: ignore[method-assign]
    try:

        def factory():
            return GrootPolicy(config)

        policy = _construct_on_meta(factory, log)
    finally:
        GrootPolicy._create_groot_model = orig_create  # type: ignore[method-assign]

    if Path(ckpt).is_dir() and (Path(ckpt) / SAFETENSORS_SINGLE_FILE).is_file():
        model_file = str(Path(ckpt) / SAFETENSORS_SINGLE_FILE)
    else:
        model_file = hf_hub_download(ckpt, SAFETENSORS_SINGLE_FILE)
    stats = stream_safetensors_into(policy, model_file, dtype=dtype, device=device, log=log)
    _set_config_device(config, device)
    policy.eval()
    log_rss("groot done", log)
    stats["ckpt"] = ckpt
    stats["family"] = "groot"
    stats["base_model_path_raw"] = raw_path
    stats["base_model_path_fix"] = fixed
    return policy, stats


def load_policy_low_ram(
    family: str,
    ckpt: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    log: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    family = family.lower()
    if family in {"pi05", "pi0.5", "π0.5"}:
        return load_pi05_low_ram(ckpt, device=device, dtype=dtype, log=log)
    if family in {"groot", "gr00t"}:
        return load_groot_low_ram(ckpt, device=device, dtype=dtype, log=log)
    raise ValueError(f"unknown family {family}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["pi05", "groot"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--out", default=None, help="optional JSON report path")
    args = p.parse_args(argv)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    ckpt = args.ckpt or (PI05_CKPT_DEFAULT if args.family == "pi05" else GROOT_CKPT_DEFAULT)
    peak = {"gb": rss_gb()}

    def log(msg: str) -> None:
        gb = rss_gb()
        if gb > peak["gb"]:
            peak["gb"] = gb
        print(msg, flush=True)

    try:
        policy, stats = load_policy_low_ram(args.family, ckpt, device=args.device, dtype=dtype, log=log)
        n_params = sum(t.numel() for t in policy.parameters())
        n_bytes = sum(t.nelement() * t.element_size() for t in policy.parameters())
        stats["n_params"] = int(n_params)
        stats["param_bytes_gb"] = round(n_bytes / 1e9, 3)
        stats["peak_rss_gb_psutil"] = round(max(peak["gb"], hwm_gb()), 3)
        stats["peak_rss_gb_vmhwm"] = round(hwm_gb(), 3)
        print(f"PEAK_RSS_GB {stats['peak_rss_gb_vmhwm']:.3f} (VmHWM) psutil_peak={stats['peak_rss_gb_psutil']:.3f}", flush=True)
        print("LOAD_OK", stats, flush=True)
        if args.out:
            import json

            Path(args.out).write_text(json.dumps(stats, indent=2, default=str))
        # Drop the model so the process can exit cleanly; peak is already recorded.
        del policy
        gc.collect()
        return 0
    except Exception:
        print("LOAD_FAIL", flush=True)
        traceback.print_exc()
        log_rss("failed", log)
        return 1


if __name__ == "__main__":
    sys.exit(main())
