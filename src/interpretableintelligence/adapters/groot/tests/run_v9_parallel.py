"""Box test (CPU): v9 parallel env pool vs v8 sequential runner, stand-in policy, real LIBERO env.

  python tests/run_v9_parallel.py [equiv|capture|probe|batched|speed|all]
"""
import os, sys, json, time, shutil
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/workspace/groot-colab/tests")
from pathlib import Path
import numpy as np


def _setup(run):
    from v9_standin import StandIn, CK
    import physint_inproc_v8 as v8
    import physint_inproc_v9 as v9
    pol = StandIn()
    run = Path(run); shutil.rmtree(run, ignore_errors=True)
    ctx = v9.Ctx(run=run, run_id="box-v9-test", policy=pol, ckpt=CK, layer=5,
                 plus_dir=Path("/workspace/groot-colab/LIBERO-plus"), snapshot_tar=run.parent / (run.name + ".tar"))
    v8.setup(ctx, device="cpu")
    v9.install_noise_router()
    return v8, v9, pol, ctx


def _seq(v8, pol, ctx, jobs, capture=True, fresh=True):
    """v8 sequential reference: v8.run_episode (global torch.manual_seed per episode), unchanged.
    fresh=True builds a fresh env per episode (v9 env lifecycle); fresh=False reuses one env per task (v8)."""
    suite = v8.plus_suite()
    envs, out = {}, []
    t0 = time.time()
    try:
        for j in jobs:
            idx = j.key[1]
            if fresh and idx in envs:
                envs.pop(idx).close()
            if idx not in envs:
                envs[idx] = v8.make_libero_env(suite, idx)
            if capture:
                with v8.FinalDenoiseCapture(pol, ctx.layer) as cap:
                    ep = v8.run_episode(pol, ctx.procs, envs[idx], seed=j.seed, init_state_id=j.init_state_id,
                                        capture=cap, max_steps=j.max_steps)
            else:
                ep = v8.run_episode(pol, ctx.procs, envs[idx], seed=j.seed, init_state_id=j.init_state_id,
                                    max_steps=j.max_steps)
            out.append(ep)
    finally:
        for e in envs.values():
            e.close()
    return out, time.time() - t0


def _par(v9, ctx, jobs, n, capture=True, batch="0"):
    res = {}
    st = v9.run_pool(ctx, jobs, on_done=lambda j, ep: res.__setitem__(j.meta["i"], ep), capture=capture,
                     n_envs=n, batch_predict=batch, label=f"test-N{n}")
    return [res[i] for i in range(len(jobs))], st


def _cmp(a, b, *, exact=True, atol=0.0):
    """Return list of mismatches between two Episodes."""
    bad = []
    if (a.success, a.n_steps, a.n_chunks) != (b.success, b.n_steps, b.n_chunks):
        bad.append(("success/n_steps/n_chunks", (a.success, a.n_steps, a.n_chunks), (b.success, b.n_steps, b.n_chunks)))
    if a.actions.shape != b.actions.shape or not np.allclose(a.actions, b.actions, atol=atol, rtol=0):
        bad.append(("actions", float(np.abs(a.actions - b.actions).max()) if a.actions.shape == b.actions.shape else a.actions.shape))
    for k in a.sim:
        x, y = np.asarray(a.sim[k]), np.asarray(b.sim[k])
        if x.dtype.kind in "fc":
            ok = x.shape == y.shape and np.allclose(x, y, atol=atol, rtol=0, equal_nan=True)
        else:
            ok = x.shape == y.shape and (x == y).all()
        if not ok:
            bad.append(("sim." + k,))
    if [c[0] for c in a.chunks] != [c[0] for c in b.chunks]:
        bad.append(("chunk steps", [c[0] for c in a.chunks], [c[0] for c in b.chunks]))
    else:
        for (sa, xa), (sb, xb) in zip(a.chunks, b.chunks):
            if xa.shape != xb.shape or (xa.float() - xb.float()).abs().max() > atol:
                bad.append(("chunk acts", sa))
                break
    if a.first_action != b.first_action and not np.allclose(a.first_action, b.first_action, atol=atol):
        bad.append(("first_action",))
    if (a.language, a.language_suite, a.task_name, a.seed, a.init_state_id) != (b.language, b.language_suite, b.task_name, b.seed, b.init_state_id):
        bad.append(("meta", (a.language, a.task_name), (b.language, b.task_name)))
    return bad


def test_equiv(results):
    v8, v9, pol, ctx = _setup("/tmp/v9t_equiv")
    print("[noise] generator vs global max diff (cpu):", v9.check_generator_matches_global("cpu", dtype=torch_dtype()))
    lens = [40, 72, 23, 56, 90, 33]  # different lengths -> envs finish at different times; 23/33/90 not multiples of 16
    jobs = [v9.Job(key=("plus", 613), seed=10000 + k, init_state_id=k, max_steps=m, meta={"i": k}) for k, m in enumerate(lens)]
    seq, t_seq = _seq(v8, pol, ctx, jobs)
    out = {"seq_s": round(t_seq, 1), "seq": [(e.success, e.n_steps, e.n_chunks) for e in seq]}
    for n in (2, 4):
        par, st = _par(v9, ctx, jobs, n)
        mism = {k: _cmp(a, b) for k, (a, b) in enumerate(zip(seq, par)) if _cmp(a, b)}
        out[f"N{n}"] = {"mismatches": mism, "wall_s": st.as_dict()["wall_s"],
                        "workers_used": sorted({e.timing["worker"] for e in par}),
                        "per_ep": [(e.success, e.n_steps, e.n_chunks, e.timing["worker"]) for e in par]}
        assert not mism, mism
    results["equiv"] = out
    print("EQUIV OK", json.dumps(out, default=str))


def torch_dtype():
    import torch
    return torch.float32


def _cmp_runs(a, b, eps_exact):
    """Compare two capture run dirs. Layout (files, keys, dtypes, trailing shapes) always; values for eps_exact."""
    from safetensors.numpy import load_file
    fa = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".npz"))
    fb = sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".npz"))
    assert fa == fb, (fa, fb)
    diffs = []
    for rel in fa:
        ld = (lambda p: load_file(str(p))) if rel.endswith(".safetensors") else (lambda p: dict(np.load(p)))
        x, y = ld(a / rel), ld(b / rel)
        if sorted(x) != sorted(y):
            diffs.append((rel, "keys")); continue
        exact = any(f"{e:05d}" in rel for e in eps_exact)
        for k in x:
            if x[k].dtype != y[k].dtype or x[k].shape[1:] != y[k].shape[1:]:
                diffs.append((rel, k, "dtype/shape"))
            elif exact and (x[k].shape != y[k].shape or not np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f")):
                diffs.append((rel, k, "values"))
    ma, mb = json.loads((a / "manifest.json").read_text()), json.loads((b / "manifest.json").read_text())
    strip = lambda m: [{k: v for k, v in e.items() if k != "seconds"} for e in m["episodes"] if e["episode_idx"] in eps_exact]
    if strip(ma) != strip(mb):
        diffs.append(("manifest.episodes", strip(ma), strip(mb)))
    for k in ("n_tok", "d_model", "token_indices", "token_subsample", "hook_target", "n_init_states", "seeds", "format",
              "layer", "task_index", "task_id", "action_queue_steps", "action_horizon", "episode_seed_rule"):
        if ma.get(k) != mb.get(k):
            diffs.append((k, ma.get(k), mb.get(k)))
    if sorted(ma["episodes"][0]) != sorted(mb["episodes"][0]):
        diffs.append(("episode row keys",))
    return fa, diffs, sorted(set(mb) - set(ma)), mb


def test_capture(results):
    """v8.capture vs v9.capture on 3 eps: same files, keys, dtypes, shapes; values identical.
    (a) v9 N=1 with GROOT_FRESH_ENV=0 (v8 env reuse) == v8.capture byte-for-byte on all 3 eps;
    (b) v9 N=2 fresh envs: same layout, ep 0 identical (cold env in both), eps 1-2 differ only via v8's env reuse."""
    os.environ["GROOT_MAX_STEPS"] = "40"
    try:
        v8, v9, pol, ctx = _setup("/tmp/v9t_cap_v9a")
        c8 = v8.Ctx(run=Path("/tmp/v9t_cap_v8"), run_id="box-v9-test", policy=pol, ckpt=ctx.ckpt, layer=5,
                    plus_dir=ctx.plus_dir, snapshot_tar=Path("/tmp/v9t_cap_v8.tar"), capture_n=3, procs=ctx.procs)
        shutil.rmtree(c8.run, ignore_errors=True)
        v8.setup(c8, device="cpu")
        chosen = {"index": 613, "id": 614, "name": "x", "sr": 0.5, "episodes_done": 1}
        v8.atomic_json(c8.run / "probe" / "chosen.json", chosen)
        v8.capture(c8)
        out = {}
        for tag, n, fresh in (("N1_reuse_env", 1, "0"), ("N2_fresh_env", 2, "1")):
            run = Path(f"/tmp/v9t_cap_{tag}"); shutil.rmtree(run, ignore_errors=True)
            c9 = v9.Ctx(run=run, run_id="box-v9-test", policy=pol, ckpt=ctx.ckpt, layer=5, plus_dir=ctx.plus_dir,
                        snapshot_tar=Path(f"/tmp/v9t_cap_{tag}.tar"), capture_n=3, n_envs=n, procs=ctx.procs)
            v8.setup(c9, device="cpu")
            v8.atomic_json(c9.run / "probe" / "chosen.json", chosen)
            os.environ["GROOT_FRESH_ENV"] = fresh
            try:
                v9.capture(c9)
            finally:
                os.environ.pop("GROOT_FRESH_ENV", None)
            exact = [0, 1, 2] if fresh == "0" else [0]
            files, diffs, extra, mb = _cmp_runs(c8.run, run, exact)
            assert not diffs, (tag, diffs)
            out[tag] = {"identical_eps": exact, "files": files, "manifest_extra_keys_v9": extra,
                        "episodes": [(e["episode_idx"], e["n_steps"], e["fail_kind"], e["n_act_rows"]) for e in mb["episodes"]]}
    finally:
        os.environ.pop("GROOT_MAX_STEPS", None)
    from safetensors.numpy import load_file
    a = load_file("/tmp/v9t_cap_N2_fresh_env/acts/shard-00000.safetensors")
    out["acts_shard0"] = {k: [list(v.shape), str(v.dtype)] for k, v in a.items()}
    results["capture"] = out
    print("CAPTURE LAYOUT OK", json.dumps(out, default=str)[:2000])


def test_probe(results):
    os.environ["GROOT_MAX_STEPS"] = "24"
    try:
        v8, v9, pol, ctx = _setup("/tmp/v9t_probe")
        ctx.probe_n, ctx.n_envs = 3, 3
        # resume: a v8-style probe.json with ep 0 of 614 already done
        v8.atomic_json(ctx.run / "probe" / "probe.json", {"instances": {"613": {"index": 613, "id": 614, "name": "pre",
                       "results": [{"ep": 0, "seed": 10000, "init_state_id": 0, "success": True, "n_steps": 1, "s": 1.0}]}}})
        ch = v9.probe(ctx)
    finally:
        os.environ.pop("GROOT_MAX_STEPS", None)
    pr = json.loads((ctx.run / "probe" / "probe.json").read_text())
    i614, i1028 = pr["instances"]["613"], pr["instances"]["1027"]
    assert [r["ep"] for r in i614["results"]] == [0, 1, 2] and i614["results"][0]["n_steps"] == 1  # ep 0 kept, not rerun
    assert [r["ep"] for r in i1028["results"]] == [0, 1, 2]
    assert i614["successes"] == 1 and i1028["successes"] == 0
    assert ch["id"] == 1028 or ch["id"] == 614
    lo, hi = v9.wilson(10, 20)
    assert abs(lo - 0.2993) < 1e-3 and abs(hi - 0.7007) < 1e-3, (lo, hi)
    results["probe"] = {"chosen": ch, "614": i614["wilson95"], "1028": i1028["wilson95"], "wilson_10_of_20": [lo, hi]}
    print("PROBE OK", json.dumps(results["probe"], default=str))


def test_batched(results):
    """Batched predict path: verify vs single on real env obs (2 different init states), then pool run."""
    import torch
    v8, v9, pol, ctx = _setup("/tmp/v9t_batch")
    lens = [40, 50, 33, 40]
    jobs = [v9.Job(key=("plus", 613), seed=10000 + k, init_state_id=k, max_steps=m, meta={"i": k}) for k, m in enumerate(lens)]
    seq, _ = _seq(v8, pol, ctx, jobs)
    par, st = _par(v9, ctx, jobs, 4, batch="1")
    v = st.verify
    assert v and v["pass"] and st.batch_mode == "batched", v
    # batched B=4 CPU matmuls may differ in the last ulp; closed loop amplifies it, so compare with tolerance
    worst = max(float(np.abs(a.actions - b.actions).max()) for a, b in zip(seq, par) if a.actions.shape == b.actions.shape)
    same_len = all(a.n_steps == b.n_steps and a.success == b.success for a, b in zip(seq, par))
    results["batched"] = {"verify": {k: v[k] for k in ("B", "max_abs_diff_action", "max_abs_diff_hook", "pass")},
                          "pool_predict_calls": st.predict_calls, "pool_mode": st.batch_mode,
                          "closed_loop_max_abs_action_diff_vs_seq": worst, "same_success_steps": same_len}
    assert same_len
    print("BATCHED OK", json.dumps(results["batched"], default=str))


def test_speed(results):
    v8, v9, pol, ctx = _setup("/tmp/v9t_speed")
    n_eps, L = 8, 96
    jobs = [v9.Job(key=("plus", 613), seed=10000 + k, init_state_id=k, max_steps=L, meta={"i": k}) for k in range(n_eps)]
    seq, t_seq = _seq(v8, pol, ctx, jobs, capture=False)
    steps = sum(e.n_steps for e in seq)
    sp = {"episodes": n_eps, "steps_per_episode": L, "cpu_count": os.cpu_count(),
          "seq_v8": {"wall_s": round(t_seq, 1), "env_steps_per_s": round(steps / t_seq, 2),
                     "episodes_per_hour": round(3600 * n_eps / t_seq, 1)}}
    for n in (1, 2, 4):
        par, st = _par(v9, ctx, jobs, n, capture=False)
        mism = {k: _cmp(a, b) for k, (a, b) in enumerate(zip(seq, par)) if _cmp(a, b)}
        d = st.as_dict()
        sp[f"v9_N{n}"] = {"wall_s": d["wall_s"], "env_steps_per_s": d["env_steps_per_s"],
                          "episodes_per_hour": d["episodes_per_hour"], "speedup_vs_seq": round(t_seq / d["wall_s"], 2),
                          "identical_to_seq": not mism}
    results["speed"] = sp
    print("SPEED", json.dumps(sp))


def test_collate(results):
    """v9.collate_obs of two B=1 preprocessed obs == the checkpoint processor's own B=2 batch (bit-exact)."""
    import torch
    v8, v9, pol, ctx = _setup("/tmp/v9t_coll")
    from lerobot.sim.integrations.libero import canonical_observation
    from lerobot.envs.utils import prepare_canonical_observation
    suite = v8.plus_suite(); canons = []
    for k in (0, 1):
        e = v8.make_libero_env(suite, 613); e.init_state_id = k; raw, _ = e.reset(seed=10000 + k)
        canons.append(canonical_observation(raw)); e.close()
    task = "put both moka pots on the stove"
    mine = v9.collate_obs([v8.build_policy_obs(c, task, ctx.procs) for c in canons])
    pre, _p, env_pre, _ = ctx.procs
    obs = prepare_canonical_observation({k: np.stack([np.asarray(c[k]) for c in canons]) for k in canons[0]})
    obs["task"] = [task, task]
    native = pre(env_pre(obs))
    rep = {k: [list(v.shape), float((v.float() - mine[k].float()).abs().max()) if v.shape == mine[k].shape else "shape"]
           for k, v in native.items() if isinstance(v, torch.Tensor)}
    assert all(r[1] == 0.0 for r in rep.values()), rep
    results["collate"] = rep
    print("COLLATE OK", rep)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    allt = ["equiv", "capture", "probe", "batched", "collate", "speed"]
    todo = allt if "all" in which else which
    results = {}
    for t in todo:
        globals()["test_" + t](results)
    out = Path("/workspace/groot-colab/v9/box_test_results.json")
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=2, default=str))
    print("ALL V9 TESTS OK", todo)
