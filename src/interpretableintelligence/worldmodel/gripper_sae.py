"""``ii gripper-sae``: is there a gripper-command feature inside V-JEPA 2-AC's predictor?

``ii gripper-trace`` shows at which predictor layer the gripper command is strongest on the frame
tokens. This command trains a BatchTopK SAE (``sae_batchtopk``) on the frame-token residual at that
layer, on predictor runs over real DROID transitions from one half of the episodes (fold 0), and then
looks at the gripper events (gripper opens or closes while the arm stays still):

1. **Selection** (fold-0 events). For every latent, the change in its mean activation over the frame
   tokens when the gripper command flips from wrong to right, signed so that "toward the real
   gripper change" is positive. The latents with the largest mean change over its spread are picked.
2. **Held-out test** (fold-1 events, episodes the SAE and the selection never saw). ``consistency``
   is the share of events where the top latent moves the right way.
3. **Patching.** In the wrong-command run, only the picked latents are set to their right-command
   values at that layer (the change goes back through the decoder); ``recovered`` is the share of the
   final prediction gap between the two commands that this closes.

Controls: ``--random-saes`` untrained SAEs (research rule 4's untrained dictionary: the trainer's own
init, the same normalisation, a threshold for the same ``k``) go through the same selection and
patching, and so do as many randomly chosen live latents of the trained SAE. Verdict, fixed before
any real run: ``gripper_feature`` when the trained top latent's held-out consistency has a lower end
above 0.5 and is above every untrained SAE's top latent; ``none`` otherwise. Offline, on recorded
video; never a finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence import sae_batchtopk as sb
from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.gripper_probe import find_events
from interpretableintelligence.worldmodel.gripper_trace import COND, fold_of
from interpretableintelligence.worldmodel.plan_probe import GRIP_MAX, MAXNORM, _boot

SCHEMA = "gripper-sae/v1"
N_PICK = 3
N_BOOT = 2000


def _blocks(model: ACWorldModel):
    blocks = getattr(model.predictor, "predictor_blocks", None)
    if blocks is None:
        raise ValueError(f"{model.name} has no predictor_blocks")
    return list(blocks)


def _hook(model: ACWorldModel, layer: int, fn):
    """Call ``fn(x) -> x`` on the residual at ``layer`` (0 = the input, ``i`` = block ``i``'s output)."""
    blocks = _blocks(model)
    if not 0 <= layer <= len(blocks):
        raise ValueError(f"layer must be in 0..{len(blocks)}, got {layer}")
    if layer == 0:
        return blocks[0].register_forward_pre_hook(lambda _m, args, kw: ((fn(args[0]),) + tuple(args[1:]), kw), with_kwargs=True)
    return blocks[layer - 1].register_forward_hook(lambda _m, _a, out: fn(out))


def _run(model: ACWorldModel, z: torch.Tensor, states: np.ndarray, actions: np.ndarray, layer: int, fn) -> torch.Tensor:
    """One batch through the predictor with ``fn`` on the residual at ``layer``. Returns the layer-normed
    prediction on the CPU."""
    p = model.tokens_per_frame
    handle = _hook(model, layer, fn)
    try:
        zd = z.to(model.device)
        sd = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(model.device).unsqueeze(1)
        ad = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(model.device).unsqueeze(1)
        with torch.inference_mode(), model._autocast():
            y = model.predictor(zd, ad, sd)[:, -p:]
        return F.layer_norm(y.float(), (y.shape[-1],)).cpu()
    finally:
        handle.remove()


def collect_rows(model: ACWorldModel, trajectories: list[Trajectory], latents: list[torch.Tensor], layer: int, *, stride: int,
                 say=print) -> sb.Split:
    """Frame-token residuals at ``layer`` on the real transitions (every ``stride``-th) of ``trajectories``."""
    rows, eps = [], []
    for tr, z in zip(trajectories, latents):
        actions = poses_to_diff(tr.states)
        idx = list(range(0, len(actions), stride))
        for i in range(0, len(idx), model.predict_batch):
            sel = idx[i : i + model.predict_batch]
            got = []

            def grab(x, got=got):
                got.append(x[:, COND:].detach().to("cpu", torch.float16))
                return x

            _run(model, z[sel], tr.states[sel], actions[sel], layer, grab)
            rows.append(got[0].reshape(-1, got[0].shape[-1]))
            eps.append(torch.full((rows[-1].shape[0],), tr.episode, dtype=torch.long))
        say(f"     rows from {tr.clip_id}: {sum(len(r) for r in rows)} so far")
    acts = torch.cat(rows)
    return sb.Split(acts=acts, tok=torch.zeros(len(acts), dtype=torch.long), episode=torch.cat(eps))


def encode(weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """JumpReLU-at-threshold latents of raw rows ``x [..., D]``."""
    w = {k: v.to(x.device) for k, v in weights.items()}
    pre = sb.pre_activations(w, x.reshape(-1, x.shape[-1]))
    f = pre * (pre > w["threshold"])
    return f.reshape(*x.shape[:-1], -1)


def decode_delta(weights: dict[str, torch.Tensor], df: torch.Tensor) -> torch.Tensor:
    """Raw-unit residual change for a latent change ``df [..., M]``."""
    w = {k: v.to(df.device) for k, v in weights.items()}
    return (df @ w["W_dec"]) * w["norm_scale"].reshape(-1)


def event_inputs(tr: Trajectory, before: int, after: int):
    s0, s1 = tr.states[before], tr.states[after]
    sign = 1.0 if s1[6] > s0[6] else -1.0
    acts = np.zeros((2, 7), dtype=np.float32)
    acts[:, :3] = np.clip(s1[:3] - s0[:3], -MAXNORM, MAXNORM)
    acts[0, 6] = sign * GRIP_MAX  # right
    acts[1, 6] = -sign * GRIP_MAX  # wrong
    return acts, np.repeat(s0[None], 2, axis=0)


def latent_changes(model, z_before: torch.Tensor, acts, states, layer: int, dictionaries: dict) -> dict[str, torch.Tensor]:
    """Per dictionary: the change ``[M]`` in each latent's mean frame-token activation, right minus wrong."""
    got = []

    def grab(x):
        got.append(x[:, COND:].detach().float())
        return x

    _run(model, z_before.unsqueeze(0).expand(2, -1, -1), states, acts, layer, grab)
    h = got[0]
    out = {}
    for name, w in dictionaries.items():
        f = encode(w, h)
        out[name] = (f[0] - f[1]).mean(0).cpu()
    return out


def patch_recovery(model, z_before: torch.Tensor, acts, states, layer: int, patches: list[tuple[dict, list[int]]]) -> list[float]:
    """For each ``(dictionary, latents)``: the share of the final right-vs-wrong prediction gap closed by
    setting only those latents of the wrong run to their right-run values at ``layer``."""
    n = 2 + len(patches)
    rows_a = np.concatenate([acts, np.repeat(acts[1:2], len(patches), axis=0)])
    rows_s = np.repeat(states[:1], n, axis=0)

    def patch(x):
        x = x.clone()
        h = x[:, COND:].float()
        for r, (w, sel) in enumerate(patches, start=2):
            if not sel:
                continue
            f = encode(w, h[:2])
            df = torch.zeros_like(f[0])
            df[:, sel] = f[0][:, sel] - f[1][:, sel]
            x[r, COND:] = (h[1] + decode_delta(w, df)).to(x.dtype)
        return x

    y = _run(model, z_before.unsqueeze(0).expand(n, -1, -1), rows_s, rows_a, layer, patch)
    gap = float((y[1] - y[0]).norm())
    if gap == 0:
        return [float("nan")] * len(patches)
    return [1.0 - float((y[r] - y[0]).norm()) / gap for r in range(2, n)]


def pick(changes: np.ndarray, n: int = N_PICK) -> list[int]:
    """Latents with the largest mean signed change over its spread, from selection events ``[E, M]``."""
    mean = changes.mean(0)
    spread = changes.std(0) + 1e-9
    score = np.where(np.abs(mean) > 0, mean / spread, -np.inf)
    return [int(i) for i in np.argsort(-score)[:n]]


def verdict(trained: dict, randoms: list[dict]) -> str:
    if not np.isfinite(trained["lo"]):
        return "insufficient_data"
    if trained["lo"] > 0.5 and all(trained["value"] > r["value"] for r in randoms):
        return "gripper_feature"
    return "none"


def run_gripper_sae(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, layer: int, n_latents: int = 4096,
                    k: int = 32, steps: int = 3000, stride: int = 2, random_saes: int = 5, seed: int = 0,
                    n_boot: int = N_BOOT, say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents = []
    for i, tr in enumerate(trajectories):
        latents.append(model.encode_frames(tr.frames))
        say(f"     encoded {i + 1}/{len(trajectories)} {tr.clip_id}")
    fold = [fold_of(tr.episode, seed) for tr in trajectories]
    train_trs = [(tr, z) for tr, z, f in zip(trajectories, latents, fold) if f == 0]
    if not train_trs:
        raise ValueError("no fold-0 episodes to train the SAE on")
    split = collect_rows(model, [t for t, _ in train_trs], [z for _, z in train_trs], layer, stride=stride, say=say)
    device = model.device
    split = sb.Split(acts=split.acts.to(device), tok=split.tok.to(device), episode=split.episode.to(device))
    mu, scale = sb.norm_stats(split, None, "fit")
    config = sb.RetrainConfig(n_latents=n_latents, k=k, steps=steps, seed=seed, norm="fit")
    trained, curve = sb.train(split, mu, scale, config, log=lambda m: None)
    health = sb.evaluate(trained, split)
    sb.save_sae_dir(trained, out_dir / "sae", {"layer": layer, "n_latents": n_latents, "k": k, "model": model.name})
    dictionaries = {"trained": {key: v.to(device) for key, v in trained.items()}}
    for r in range(random_saes):
        w = {key: v.to(device) for key, v in sb.untrained_weights(trained, seed + 1000 + r).items()}
        threshold, _ = sb.batchtopk_threshold(w, split.chunks, k)
        w["threshold"] = torch.tensor(threshold, device=device)
        dictionaries[f"random:{seed + 1000 + r}"] = w
    n_rows = split.n
    del split
    say(f"     SAE: {n_latents} latents, k={k}, final FVU {curve[-1]['fvu']:.4f}")

    events = []
    for tr, z, f in zip(trajectories, latents, fold):
        for before, after in find_events(tr.states):
            acts, states = event_inputs(tr, before, after)
            ch = latent_changes(model, z[before], acts, states, layer, dictionaries)
            events.append({"clip_id": tr.clip_id, "episode": tr.episode, "before": before, "after": after, "fold": f,
                           "closing": bool(acts[0, 6] > 0), "z": z[before], "acts": acts, "states": states,
                           "changes": {k2: v.numpy() for k2, v in ch.items()}})
    sel_events = [e for e in events if e["fold"] == 0]
    test_events = [e for e in events if e["fold"] == 1]
    if not sel_events or not test_events:
        raise ValueError(f"need gripper events in both folds, got {len(sel_events)} and {len(test_events)}")

    rng = np.random.default_rng(seed)
    alive = np.flatnonzero(np.stack([e["changes"]["trained"] for e in sel_events]).std(0) > 0)
    picks = {}
    for name in dictionaries:
        picks[name] = pick(np.stack([e["changes"][name] for e in sel_events]))
    picks["trained-random-latents"] = [int(i) for i in rng.choice(alive, size=min(N_PICK, len(alive)), replace=False)] if len(alive) else []

    records = []
    patch_list = [(dictionaries[n if n != "trained-random-latents" else "trained"], picks[n]) for n in picks]
    for e in test_events:
        rec = patch_recovery(model, e["z"], e["acts"], e["states"], layer, patch_list)
        row = {"clip_id": e["clip_id"], "episode": e["episode"], "before": e["before"], "after": e["after"], "closing": e["closing"]}
        for (name, sel), r in zip(picks.items(), rec):
            src = "trained" if name == "trained-random-latents" else name
            row[f"top_moves_right/{name}"] = bool(sel and e["changes"][src][sel[0]] > 0)
            row[f"recovered/{name}"] = r
        records.append(row)

    def frac(key):
        return lambda rs: float(np.mean([r[key] for r in rs])) if rs else float("nan")

    def mean(key):
        return lambda rs: float(np.nanmean([r[key] for r in rs])) if rs else float("nan")

    arms = {}
    for name in picks:
        arms[name] = {
            "latents": picks[name],
            "consistency": _boot(records, frac(f"top_moves_right/{name}"), n_boot=n_boot, seed=seed),
            "recovered": _boot(records, mean(f"recovered/{name}"), n_boot=n_boot, seed=seed),
        }
    randoms = [arms[n]["consistency"] for n in picks if n.startswith("random:")]
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "layer": layer,
        "seed": seed,
        "n_boot": n_boot,
        "sae": {"n_latents": n_latents, "k": k, "steps": steps, "stride": stride, "rows": n_rows,
                "final_fvu": curve[-1]["fvu"], "health": health},
        "n_selection_events": len(sel_events),
        "n_test_events": len(test_events),
        "n_test_episodes": len({r["episode"] for r in records}),
        "arms": arms,
        "verdict": verdict(arms["trained"]["consistency"], randoms),
        "events": records,
    }
    (out_dir / "gripper_sae.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
