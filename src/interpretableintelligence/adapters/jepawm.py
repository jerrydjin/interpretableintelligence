"""Meta's JEPA-WMs world models (DINO-WM, JEPA-WM, retrained V-JEPA 2-AC) behind ``freeze-check``'s interface.

github.com/facebookresearch/jepa-wms ("What Drives Success in Physical Planning with Joint-Embedding
Predictive World Models?", Terver et al.) ships pretrained action-conditioned latent world models for
DROID and four simulated tasks, all loaded through its ``torch.hub`` entry points (``dino_wm_droid``,
``jepa_wm_droid``, ``vjepa2_ac_droid``, ...). Their DROID models take the same 7-D action as V-JEPA 2-AC
(the pose change to the next frame at 4 fps, built by the same ``poses_to_diffs``), so the clips from
``ii get-clips --with-states`` feed them unchanged.

Neither the code nor the weights are dependencies of this package: ``load_jepawm`` imports them from a
checkout (``code_dir``) the first time it is called. Only RGB-only models are supported (those whose
predictor does not read the arm pose); ``vjepa2_ac_oss`` reads it and is refused (use ``ii freeze-check
--model vjepa2-ac`` for Meta's public V-JEPA 2-AC checkpoint).
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.worldmodel.freeze_check import reset_weights

JEPAWMS_REPO = "https://github.com/facebookresearch/jepa-wms"


@dataclass
class JepaWorldModel:
    """``EncPredWM`` (jepa-wms' planning wrapper) as a ``freeze_check.WorldModel``. One frame of context."""

    wm: object  # EncPredWM: .encode(frames [B, T, C, H, W] 0-255) and .unroll(z [B, 1, V, H, W, D], act [T, B, A])
    crop: int
    patch: int
    device: torch.device
    name: str = "jepa-wm"
    info: dict = field(default_factory=dict)
    dtype: torch.dtype = torch.float32
    batch: int = 32

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.autocast("cpu", enabled=False)

    @torch.inference_mode()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        out = []
        for i in range(0, len(frames), self.batch):
            x = torch.from_numpy(np.ascontiguousarray(frames[i : i + self.batch])).permute(0, 3, 1, 2).float()
            with self._autocast():
                z = self.wm.encode(x.unsqueeze(1).to(self.device))  # [B, 1, V, H, W, D]
            out.append(z[:, 0].flatten(1, 3).float().cpu())  # [B, V * H * W, D]
        return torch.cat(out)

    @torch.inference_mode()
    def predict_next(self, z: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        side = self.crop // self.patch
        out = []
        for i in range(0, len(z), self.batch):
            zb = z[i : i + self.batch].to(self.device)
            b, p, d = zb.shape
            ctx = zb.view(b, 1, 1, side, side, d)
            act = actions[i : i + self.batch].to(self.device, torch.float32).unsqueeze(0)  # [T=1, B, A]
            with self._autocast():
                y = self.wm.unroll(ctx, act_suffix=act)  # [1 + 1, B, V, H, W, D]
            out.append(y[-1].reshape(b, p, d).float().cpu())
        return torch.cat(out)


def untrained_copy(model: JepaWorldModel, *, seed: int = 0) -> JepaWorldModel:
    """Same encoder (shared, not copied); predictor, action and proprio encoders at a seeded random init."""
    vw = model.wm.model
    keep = {"encoder": vw.encoder, "heads": getattr(vw, "heads", None)}
    vw.encoder, vw.heads = None, None
    try:
        fresh = copy.deepcopy(vw)
    finally:
        vw.encoder, vw.heads = keep["encoder"], keep["heads"]
    for name in ("predictor", "action_encoder", "proprio_encoder"):
        part = getattr(fresh, name, None)
        if isinstance(part, torch.nn.Module):
            reset_weights(part, seed=seed)
    fresh.encoder, fresh.heads = keep["encoder"], keep["heads"]
    wm = copy.copy(model.wm)
    if isinstance(wm, torch.nn.Module):  # a shallow copy shares the module dicts: give the copy its own
        for d in ("_modules", "_parameters", "_buffers"):
            setattr(wm, d, dict(getattr(model.wm, d)))
    wm.model = fresh
    for attr in ("encode_proprio", "encode_obs"):
        if hasattr(fresh, attr):
            setattr(wm, attr, getattr(fresh, attr))
    info = {**model.info, "untrained": "reset_weights(predictor, action_encoder, proprio_encoder)", "seed": seed}
    return JepaWorldModel(wm=wm, crop=model.crop, patch=model.patch, device=model.device, name=model.name + "-untrained",
                          info=info, dtype=model.dtype, batch=model.batch)


def _shim_eval_module() -> None:
    """jepa-wms' hub loader imports ``init_module`` from its planning-eval module, which pulls in every
    simulator (MuJoCo, Metaworld, D4RL...). Put in a module holding only that function, written as the
    original is: it hands the arguments to the model module's own ``init_module``."""
    name = "evals.simu_env_planning.eval"
    if name in sys.modules:
        return

    def init_module(folder, checkpoint, module_name, model_kwargs, device, cfgs_data=None, wrapper_kwargs=None,
                    action_dim=None, proprio_dim=None, preprocessor=None):
        return importlib.import_module(module_name).init_module(
            folder=folder, checkpoint=checkpoint, model_kwargs=model_kwargs, device=device, action_dim=action_dim,
            proprio_dim=proprio_dim, preprocessor=preprocessor, cfgs_data=cfgs_data, wrapper_kwargs=wrapper_kwargs,
        )

    shim = types.ModuleType(name)
    shim.init_module = init_module
    sys.modules[name] = shim


def load_jepawm(name: str, *, code_dir: Path | str, device: torch.device | str, dtype: torch.dtype | None = None) -> JepaWorldModel:
    """A pretrained JEPA-WMs model by its hub name (``dino_wm_droid``, ``jepa_wm_droid``, ``vjepa2_ac_droid``, ...)."""
    device = torch.device(device)
    code_dir = str(Path(code_dir).resolve())
    os.environ.setdefault("JEPAWM_HOME", str(Path(code_dir).parent))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    _shim_eval_module()
    cwd = os.getcwd()
    os.chdir(code_dir)  # the hub loader reads its configs by relative path
    try:
        wm, _pre = torch.hub.load(code_dir, name, source="local", pretrained=True, device=str(device))
    finally:
        os.chdir(cwd)
    vw = wm.model
    if getattr(vw, "use_proprio", False) and getattr(vw, "proprio_tokens", 0):
        raise ValueError(f"{name} reads the arm pose; only RGB-only JEPA-WMs models are supported")
    wm.eval() if hasattr(wm, "eval") else None
    crop = int(vw.img_size if isinstance(vw.img_size, int) else vw.img_size[0])
    patch = crop // int(vw.grid_size)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    info = {"hub_name": name, "repo": JEPAWMS_REPO, "enc_type": vw.enc_type, "pred_type": getattr(vw, "pred_type", None),
            "img_size": crop, "grid": int(vw.grid_size), "ctxt_window": getattr(wm, "ctxt_window", None)}
    return JepaWorldModel(wm=wm, crop=crop, patch=patch, device=device, name=name.replace("_", "-"), info=info, dtype=dtype)
