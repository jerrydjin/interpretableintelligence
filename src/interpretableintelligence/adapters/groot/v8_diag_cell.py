# === v8 DIAG CELL — paste into a live Colab runtime that already has `policy` (GR00T) loaded ===
# Works on a v7 runtime (uses `v7` / `V7`) or a v8 runtime (`v8` / `V8`). ~3-6 min on a T4.
#   (i)   OPEN-LOOP: frames from the checkpoint's training set (IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot,
#         per the checkpoint README `datasets:`), same preprocessing as the rollout path, predicted vs GT actions,
#         per-dim MAE vs zero- and mean-action baselines, explicit PASS/FAIL.
#         Run BEFORE and AFTER the RoPE repair (v7 loader bug: Qwen3-VL text RoPE theta 1e4 instead of 5e6).
#   (ii)  live env obs vs dataset frame: per-channel means, 180-flip correlation, state, preprocessed tensors.
#   (iii) 16-step timing: obs+preprocess / select_action / post / env.step (+render), predict/DiT/backbone call counts.
#   (iv)  optional bf16-autocast forward vs fp16 on the same frames + noise.
# Env knobs: DIAG_EPISODE (10 = moka), DIAG_FRAMES ("0,120,240"), DIAG_BF16 (1), DIAG_ENV (1), DIAG_REPAIR (1).
import os, sys, json, time, math, traceback
from pathlib import Path
import numpy as np
import torch

assert globals().get("policy") is not None, "no `policy` in this runtime; load GR00T first (cell 5)"
_P = globals()["policy"]
_CKPT = globals().get("CKPT") or "nvidia/gr00t17-lerobot-libero_10-640"
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS_REPO = "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot"
DS_EPISODE = int(os.environ.get("DIAG_EPISODE", "10"))           # ep 10 = "put both moka pots on the stove"
DS_FRAMES = [int(x) for x in os.environ.get("DIAG_FRAMES", "0,120,240").split(",")]
DO_BF16 = os.environ.get("DIAG_BF16", "1") == "1"
DO_ENV = os.environ.get("DIAG_ENV", "1") == "1"
DO_REPAIR = os.environ.get("DIAG_REPAIR", "1") == "1"
DIAG = {"ckpt": _CKPT, "dataset": DS_REPO, "episode": DS_EPISODE, "frames": DS_FRAMES}
_t_cell = time.time()
np.set_printoptions(precision=4, suppress=True, linewidth=160)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- helpers module / processors
_M = globals().get("v8") or globals().get("v7")
if _M is None:
    for _cand in ("/content/inproc_v8.py", "/content/physint_inproc_v7.py"):
        if Path(_cand).is_file():
            sys.path.insert(0, str(Path(_cand).parent))
            import importlib
            _M = importlib.import_module(Path(_cand).stem)
            break
_ctx = globals().get("V8") or globals().get("V7")
_procs = getattr(_ctx, "procs", None) if _ctx is not None else None
if _procs is None:
    assert _M is not None, "need inproc_v8 or physint_inproc_v7 for build_processors"
    _procs = _M.build_processors(_P, _CKPT, device=_DEV)
_pre, _post, _env_pre, _env_post = _procs
if "predict_action_chunk" in _P.__dict__:
    print("WARNING: policy.predict_action_chunk is patched (a capture hook is still installed?)")


def build_obs(canon, task):
    """Identical to v8.build_policy_obs / v7 run_episode: canonical -> prepare -> task -> env_pre -> pre."""
    from lerobot.envs.utils import prepare_canonical_observation
    obs = prepare_canonical_observation({k: np.asarray(v)[None] for k, v in canon.items()})
    obs["task"] = [task]
    return _pre(_env_pre(obs))


def post_chunk(a):
    from lerobot.utils.constants import ACTION
    return _env_post({ACTION: _post(a)})[ACTION]


def _rope_theta(cfg, default=10000.0):
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    return float(getattr(cfg, "rope_theta", None) or default)


def rope_report(repair=False):
    out = []
    for name, m in _P.named_modules():
        inv = getattr(m, "inv_freq", None)
        if not isinstance(inv, torch.Tensor) or getattr(m, "config", None) is None:
            continue
        with torch.device("cpu"):
            ref_m = type(m)(m.config)
        ref = ref_m.inv_freq.detach().float()
        diff = float((inv.detach().float().cpu() - ref).abs().max())
        if repair and diff > 0:
            new = ref.to(device=inv.device, dtype=inv.dtype)
            m.inv_freq = torch.nn.Buffer(new, persistent=False)
            if isinstance(getattr(m, "original_inv_freq", None), torch.Tensor):
                m.original_inv_freq = torch.nn.Buffer(new.clone(), persistent=False)
            m.attention_scaling = getattr(ref_m, "attention_scaling", getattr(m, "attention_scaling", 1.0))
        out.append({"module": name.split(".")[-2] + "." + name.split(".")[-1], "theta": _rope_theta(m.config),
                    "max_abs_diff": round(diff, 6), "repaired": bool(repair and diff > 0)})
    return out


# ---------------------------------------------------------------- dataset frames (tiny: 1 parquet + 2 mp4)
def load_dataset_frames(ep, frames):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    info = json.loads(Path(hf_hub_download(DS_REPO, "meta/info.json", repo_type="dataset")).read_text())
    eps = [json.loads(l) for l in Path(hf_hub_download(DS_REPO, "meta/episodes.jsonl", repo_type="dataset")).read_text().splitlines() if l.strip()]
    task = next(e for e in eps if int(e["episode_index"]) == ep)["tasks"][0]
    chunk = ep // int(info["chunks_size"])
    pq = hf_hub_download(DS_REPO, info["data_path"].format(episode_chunk=chunk, episode_index=ep), repo_type="dataset")
    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    need = max(frames)
    vids = {}
    for key in ("observation.images.image", "observation.images.wrist_image"):
        mp4 = hf_hub_download(DS_REPO, info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep),
                              repo_type="dataset")
        imgs = []
        import av
        with av.open(mp4) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i in frames:
                    imgs.append(f.to_ndarray(format="rgb24"))
                if i >= need:
                    break
        vids[key] = imgs
    out = []
    for j, t in enumerate(frames):
        if t + 16 > len(action):
            raise ValueError(f"frame {t}+16 > episode length {len(action)}")
        out.append({"t": t, "canon": {"observation.images.image": vids["observation.images.image"][j],
                                       "observation.images.wrist_image": vids["observation.images.wrist_image"][j],
                                       "observation.state": state[t]},
                    "gt": action[t:t + 16]})
    return task, out, {"n_frames_ep": len(df), "codebase_version": info.get("codebase_version")}


# ---------------------------------------------------------------- open-loop eval
_ah = _P._groot_model.action_head
_q = int(_P._action_queue_steps)
_g = torch.Generator().manual_seed(1234)
_NOISE = [torch.randn((1, int(_ah.config.action_horizon), int(_ah.action_dim)), generator=_g) for _ in DS_FRAMES]
_ACT_STATS = None
for _st in _post.steps:
    if getattr(_st, "stats", None) and "action" in _st.stats:
        _ACT_STATS = {k: np.asarray(torch.as_tensor(v).float().cpu()).reshape(-1) for k, v in _st.stats["action"].items()}


def _gt_to_env(gt):
    """dataset gripper is 0/1 (1 = open); LIBERO env wants -1 open / +1 close (postprocessor: -sign(2g-1))."""
    g = gt.copy()
    g[:, 6] = -np.sign(2.0 * g[:, 6] - 1.0)
    return g


def _backbone_layers():
    for n, m in _P.named_modules():
        if n.endswith("language_model.layers") and isinstance(m, torch.nn.ModuleList):
            return m
    return None


def open_loop(task, samples, label):
    layers = _backbone_layers()
    amax = {"v": 0.0, "nonfinite": False}

    def _h(_m, _i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        if isinstance(t, torch.Tensor):
            amax["v"] = max(amax["v"], float(t.detach().abs().max()))
            amax["nonfinite"] |= not bool(torch.isfinite(t).all())
    hs = [l.register_forward_hook(_h) for l in layers] if layers is not None else []
    preds, gts, secs = [], [], []
    try:
        for s, nz in zip(samples, _NOISE):
            batch = build_obs(s["canon"], task)
            _sync(); t0 = time.perf_counter()
            with torch.inference_mode():
                a = _P.predict_action_chunk(batch, noise=nz)
            _sync(); secs.append(time.perf_counter() - t0)
            pa = post_chunk(a).float().cpu().numpy().reshape(-1, 7)[:16]
            preds.append(pa)
            gts.append(_gt_to_env(s["gt"]))
    finally:
        for h in hs:
            h.remove()
    P, G = np.concatenate(preds), np.concatenate(gts)
    mae = np.abs(P - G).mean(0)
    zero = np.abs(G).mean(0)
    mean_vec = np.zeros(7, np.float32)
    if _ACT_STATS is not None and "mean" in _ACT_STATS:
        mean_vec[:6] = _ACT_STATS["mean"][:6]
        mean_vec[6] = -np.sign(2.0 * _ACT_STATS["mean"][6] - 1.0)
    meanb = np.abs(G - mean_vec).mean(0)
    grip_agree = float((np.sign(P[:, 6]) == np.sign(G[:, 6])).mean())
    arm_ok = bool((mae[:6] < 0.5 * zero[:6]).all())
    passed = arm_ok and grip_agree >= 0.9
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    print(f"\n--- OPEN-LOOP [{label}] task={task!r} frames={DS_FRAMES} ({len(P)} action steps), "
          f"predict {np.mean(secs):.3f}s/chunk, backbone |h|max={amax['v']:.1f} nonfinite={amax['nonfinite']}")
    print(f"{'dim':>5} {'MAE':>8} {'zeroMAE':>8} {'meanMAE':>8} {'ratio':>6}")
    for i, n in enumerate(names):
        print(f"{n:>5} {mae[i]:8.4f} {zero[i]:8.4f} {meanb[i]:8.4f} {mae[i] / max(zero[i], 1e-9):6.2f}")
    print(f"gripper sign agreement {grip_agree:.1%} (zero-baseline n/a; mean-baseline "
          f"{float((np.sign(mean_vec[6]) == np.sign(G[:, 6])).mean()):.1%})")
    print(f"first chunk step0 pred {P[0]}\n                 gt   {G[0]}")
    print(f"OPEN-LOOP {label}: {'PASS' if passed else 'FAIL'} (rule: MAE < 0.5x zero-baseline on all 6 arm dims "
          f"[{'ok' if arm_ok else 'no'}] AND gripper sign agreement >= 90% [{grip_agree:.1%}])")
    return {"label": label, "pass": passed, "mae": mae.tolist(), "zero_mae": zero.tolist(), "mean_mae": meanb.tolist(),
            "grip_sign_agree": grip_agree, "predict_s": float(np.mean(secs)), "backbone_absmax": amax["v"],
            "backbone_nonfinite": amax["nonfinite"], "pred": P, "gt": G}


try:
    print(f"[diag] action queue: _action_queue_steps={_q} n_action_steps={_P.config.n_action_steps} "
          f"deque maxlen={_P._action_queue.maxlen} chunk={_P.config.chunk_size} horizon={_ah.config.action_horizon} "
          f"denoise={_ah.num_inference_timesteps} use_bf16={_P.config.use_bf16} param dtype={next(_P.parameters()).dtype}")
    _r0 = rope_report(repair=False)
    print(f"[diag] RoPE before: {_r0}")
    DIAG["rope_before"] = _r0
    _task, _samples, _dsinfo = load_dataset_frames(DS_EPISODE, DS_FRAMES)
    DIAG["dataset_info"] = _dsinfo
    _bad_rope = any(r["max_abs_diff"] > 0 for r in _r0)
    OL = {}
    OL["fp16_as_loaded"] = open_loop(_task, _samples, "as loaded" + (" (RoPE WRONG)" if _bad_rope else ""))
    if _bad_rope and DO_REPAIR:
        DIAG["rope_repair"] = rope_report(repair=True)
        print(f"[diag] RoPE repaired: {DIAG['rope_repair']}  -> verify: {rope_report(False)}")
        OL["fp16_rope_fixed"] = open_loop(_task, _samples, "RoPE fixed")
    if DO_BF16:
        _old = _P.config.use_bf16
        try:
            _P.config.use_bf16 = True
            OL["bf16_autocast"] = open_loop(_task, _samples, "bf16 autocast")
        except Exception as e:
            print(f"[diag] bf16 forward failed: {e!r}")
        finally:
            _P.config.use_bf16 = _old
        _last_fp16 = OL.get("fp16_rope_fixed", OL["fp16_as_loaded"])
        if "bf16_autocast" in OL:
            d = np.abs(OL["bf16_autocast"]["pred"] - _last_fp16["pred"])
            print(f"[diag] bf16 vs fp16 (same noise): max|d|={d.max():.4f} mean|d| per dim={d.mean(0)}")
    DIAG["open_loop"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("pred", "gt")} for k, v in OL.items()}
except Exception as e:
    DIAG["open_loop_error"] = repr(e)
    traceback.print_exc()

# ---------------------------------------------------------------- (ii) live obs vs dataset + (iii) timing
def _corr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b) + 1e-12))


def _summ_batch(b):
    o = {}
    for k in ("state", "pixel_values", "image_grid_thw", "input_ids", "embodiment_id"):
        v = b.get(k) if isinstance(b, dict) else None
        if isinstance(v, torch.Tensor):
            vf = v.float()
            o[k] = {"shape": list(v.shape), "mean": round(float(vf.mean()), 4), "std": round(float(vf.std()), 4) if v.numel() > 1 else 0.0}
    if isinstance(b.get("state"), torch.Tensor):
        o["state_norm8"] = [round(float(x), 3) for x in b["state"].reshape(-1)[:8]]
    try:
        tok = next(st for st in _pre.steps if type(st).__name__ == "GrootN17VLMEncodeStep").proc.tokenizer
        txt = tok.decode(b["input_ids"][0].tolist())
        o["text"] = " ".join(t for t in txt.replace("<|image_pad|>", "").split())[:300]
    except Exception as e:
        o["text"] = f"(decode failed {e!r})"
    return o


if DO_ENV and _M is not None:
    _env = None
    try:
        from lerobot.sim.integrations.libero import canonical_observation
        _M.write_libero_config(globals().get("PLUS_DIR") or getattr(_ctx, "plus_dir"), log=lambda *_: None)
        _env = _M.make_libero_env(_M.vanilla_moka_suite(), 0)
        _env.init_state_id = 0
        _raw, _ = _env.reset(seed=0)
        _li = getattr(_env._env, "language_instruction", None)
        _li = " ".join(_li) if isinstance(_li, (list, tuple)) else _li
        _ltask = str(_li or _env.task_description).strip().strip('"')
        _canon = canonical_observation(_raw)
        print(f"\n--- LIVE vs DATASET (env vanilla moka init 0 vs dataset ep {DS_EPISODE} frame {DS_FRAMES[0]}) task={_ltask!r}")
        live = {}
        for key in ("observation.images.image", "observation.images.wrist_image"):
            L_, D_ = _canon[key], _samples[0]["canon"][key]
            live[key] = {"live_mean_rgb": L_.reshape(-1, 3).mean(0).round(1).tolist(),
                         "ds_mean_rgb": D_.reshape(-1, 3).mean(0).round(1).tolist(),
                         "corr_vs_ds": round(_corr(L_, D_), 3), "corr_vs_ds_rot180": round(_corr(L_, D_[::-1, ::-1]), 3),
                         "corr_vs_ds_vflip": round(_corr(L_, D_[::-1]), 3), "corr_vs_ds_hflip": round(_corr(L_, D_[:, ::-1]), 3)}
            ok = live[key]["corr_vs_ds"] > live[key]["corr_vs_ds_rot180"]
            print(f"{key}: {live[key]}  orientation {'OK' if ok else 'FLIPPED?'}")
        print(f"state live {np.asarray(_canon['observation.state'])}\nstate ds   {_samples[0]['canon']['observation.state']}")
        _bl = build_obs(_canon, _ltask)
        _bd = build_obs(_samples[0]["canon"], _task)
        print(f"preprocessed live: {json.dumps(_summ_batch(_bl))}\npreprocessed ds:   {json.dumps(_summ_batch(_bd))}")
        DIAG["live_vs_ds"] = {"images": live, "state_live": np.asarray(_canon["observation.state"]).tolist(),
                              "state_ds": _samples[0]["canon"]["observation.state"].tolist(),
                              "pre_live": _summ_batch(_bl), "pre_ds": _summ_batch(_bd)}

        # (iii) 16 env steps, rollout path, every-step preprocessing (FAST_PREPROCESS=0 worst case)
        from lerobot.processor import policy_output_to_transition, transition_to_policy_action
        cnt = {"predict": 0, "dit5": 0, "backbone_layer0": 0, "hook_s": 0.0}
        _blk = _P._groot_model.action_head.model.transformer_blocks[int(globals().get("LAYER") or 5)]

        def _cap_hook(_m, _i, o):  # same work as FinalDenoiseCapture._hook
            t0 = time.perf_counter()
            t = o[0] if isinstance(o, (tuple, list)) else o
            _ = t.detach()
            cnt["dit5"] += 1
            cnt["hook_s"] += time.perf_counter() - t0
        _layers = _backbone_layers()
        _h1 = _blk.register_forward_hook(_cap_hook)
        _h2 = _layers[0].register_forward_hook(lambda *a: cnt.__setitem__("backbone_layer0", cnt["backbone_layer0"] + 1)) if _layers is not None else None
        _inner = _P.predict_action_chunk

        def _counting_predict(batch, *a, **k):
            cnt["predict"] += 1
            return _inner(batch, *a, **k)
        _P.predict_action_chunk = _counting_predict
        T = {"obs_pre": [], "select_action": [], "post": [], "env_step": []}
        try:
            _P.reset(); _pre.reset(); _post.reset()
            _env.init_state_id = 0
            _raw, _ = _env.reset(seed=0)
            for t in range(16):
                _sync(); a0 = time.perf_counter()
                _obs = build_obs(canonical_observation(_raw), _ltask)
                _sync(); a1 = time.perf_counter()
                with torch.inference_mode():
                    _a = transition_to_policy_action(policy_output_to_transition(_P.select_action(_obs)))
                _sync(); a2 = time.perf_counter()
                _a = post_chunk(_a).float().cpu().numpy()[0]
                a3 = time.perf_counter()
                _raw, *_rest = _env.step(_a.astype(np.float32))
                a4 = time.perf_counter()
                for k_, v_ in zip(T, (a1 - a0, a2 - a1, a3 - a2, a4 - a3)):
                    T[k_].append(v_)
        finally:
            _P.__dict__.pop("predict_action_chunk", None)
            _h1.remove()
            if _h2 is not None:
                _h2.remove()
        counts16 = {k: cnt[k] for k in ("predict", "dit5", "backbone_layer0")}  # before the overhead probe below
        # render share of env.step
        base = _env._env.env
        r = []
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                base._get_observations(force_update=True)
            except TypeError:
                base._get_observations()
            r.append(time.perf_counter() - t0)
        # hook overhead on one predict (with vs without hook)
        _obs = build_obs(canonical_observation(_raw), _ltask)
        def _one(with_hook):
            hh = _blk.register_forward_hook(_cap_hook) if with_hook else None
            try:
                _sync(); t0 = time.perf_counter()
                with torch.inference_mode():
                    _P.predict_action_chunk(_obs)
                _sync(); return time.perf_counter() - t0
            finally:
                if hh is not None:
                    hh.remove()
        _one(False)
        t_no, t_yes = min(_one(False) for _ in range(2)), min(_one(True) for _ in range(2))
        tim = {k: {"sum_s": round(sum(v), 3), "step0_s": round(v[0], 3), "mean_rest_s": round(float(np.mean(v[1:])), 4)}
               for k, v in T.items()}
        tim.update({"render_obs_s": round(float(np.mean(r)), 4), "predict_no_hook_s": round(t_no, 3),
                    "predict_with_hook_s": round(t_yes, 3), "hook_body_s_total": round(cnt["hook_s"], 5),
                    "counts_16_steps": counts16})
        q_ok = counts16["predict"] == 1 and counts16["dit5"] == int(_ah.num_inference_timesteps)
        est = (520 * (tim["env_step"]["mean_rest_s"] + tim["post"]["mean_rest_s"] + tim["select_action"]["mean_rest_s"])
               + math.ceil(520 / _q) * (tim["select_action"]["step0_s"] + tim["obs_pre"]["mean_rest_s"]))
        tim["est_s_per_520_step_episode_fast_preprocess"] = round(est, 1)
        print(f"\n--- TIMING 16 env steps: {json.dumps(tim, indent=1)}")
        print(f"QUEUE CHECK: predict_action_chunk ran {counts16['predict']}x, DiT block hook {counts16['dit5']}x, backbone "
              f"layer0 {counts16['backbone_layer0']}x in 16 steps -> {'OK (1 inference per 16-step chunk)' if q_ok else 'UNEXPECTED'}")
        DIAG["timing"] = tim
        DIAG["queue_ok"] = q_ok
    except Exception as e:
        DIAG["env_error"] = repr(e)
        traceback.print_exc()
    finally:
        if _env is not None:
            _env.close()
        _P.reset()

DIAG["cell_s"] = round(time.time() - _t_cell, 1)
_out = Path(str(globals().get("RUN", "/content"))) / "smoke" / "diag_v8.json"
try:
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text(json.dumps(DIAG, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\n[diag] wrote {_out}")
except Exception as e:
    print(f"[diag] could not write {_out}: {e!r}")
_ol = DIAG.get("open_loop", {})
print("DIAG SUMMARY:", json.dumps({k: {"pass": v["pass"], "grip": round(v["grip_sign_agree"], 3),
                                        "mae_arm": [round(x, 3) for x in v["mae"][:6]]} for k, v in _ol.items()}),
      "| queue_ok:", DIAG.get("queue_ok"), "| cell_s:", DIAG["cell_s"])
