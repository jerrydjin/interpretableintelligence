"""Box test (CPU, stand-in policy, real LIBERO-plus env 1028): the ablation branch of v9_hotfix_cell.py end to end.

Exec the hotfix twice on a simulated live v8 runtime with GROOT_ABLATE_PLAN = a shrunken copy of the real plan
(baseline 4, recon-only 2, both candidates + 1 control each, cap 3) and a d_in=8 SAE subset for the stand-in's block:
  run 1 is interrupted after 7 episodes (keep_going -> False) and its last log line is torn;
  run 2 resumes and finishes. Checks: logs/summaries, seeds 40000+i / init i%50, recon-only == baseline actions
  (bit-identical on CPU), ablated conditions change actions, hook stats, no duplicate episodes, resume order.
/content -> /tmp/content9a.
"""
from box_paths import COLAB, RUNS  # noqa: E402 (box roots; see box_paths.py)
import os, sys, shutil, json, copy
from pathlib import Path

def make_mock(root: Path):
    """Shrunken copy of the real plan + a d_in=8 SAE subset (same latent ids) for the stand-in policy."""
    import torch
    from safetensors.torch import save_file
    import ablate_v9 as A
    plan = json.loads(Path(f"{RUNS}/groot-1028-v9-ablate/ablate_plan.json").read_text())
    ids = plan["sae"]["latent_ids"]
    g = torch.Generator().manual_seed(3)
    W = torch.randn(8, len(ids), generator=g); W = W / W.norm(dim=0, keepdim=True)
    sae_p = root / "mock_sae_subset.safetensors"
    save_file({"W_enc": W.contiguous(), "W_dec": W.t().contiguous(), "b_enc": torch.full((len(ids),), 0.3), "b_dec": torch.zeros(8),
               "norm_mu": torch.zeros(8), "norm_scale": torch.tensor([1.0]), "threshold": torch.tensor(0.05),
               "latent_ids": torch.tensor(ids, dtype=torch.int64)}, str(sae_p), metadata={"provenance": "mock d_in=8"})
    mp = copy.deepcopy(plan)
    mp["sae"].update({"subset_sha256": A.sha256_file(sae_p), "full_sha256": None, "d_in": 8})
    keep = ["baseline", "recon-only", "A-cand-720", "B-cand-7081+5852", "A-ctrl01-7774", "B-ctrl01-1159+7339"]
    mp["conditions"] = [c for c in mp["conditions"] if c["id"] in keep]
    for c in mp["conditions"]:
        c["n_max"] = {"baseline": 4, "recon_only": 2}.get(c["kind"], 3)
    mp["sprt"]["cap"] = 3
    mp["schedule"]["order"] = [c for c in mp["schedule"]["order"] if c in keep]
    plan_p = root / "mock_plan.json"; plan_p.write_text(json.dumps(mp))
    return plan_p, sae_p, mp


if __name__ == "__main__":
    os.environ.update({"MUJOCO_GL": "egl", "GROOT_MAX_STEPS": "20", "GROOT_N_ENVS": "3"})
    sys.path[:0] = [f"{COLAB}/tests", f"{COLAB}/v9", f"{COLAB}/v8"]
    from v9_standin import StandIn, CK
    import physint_inproc_v8 as v8
    root = Path("/tmp/content9a"); shutil.rmtree(root, ignore_errors=True); root.mkdir()
    plan_p, sae_p, mp = make_mock(root)
    os.environ.update({"GROOT_ABLATE_PLAN": str(plan_p), "GROOT_ABLATE_SAE": str(sae_p)})

    pol = StandIn()
    V8 = v8.Ctx(run=root / "physint/groot/colab-t4-libero-plus-v8", run_id="colab-t4-libero-plus-v8", policy=pol,
                ckpt=CK, layer=5, plus_dir=Path(f"{COLAB}/LIBERO-plus"), snapshot_tar=root / "physint_snapshot.tar")
    v8.setup(V8, device="cpu")
    src = open(f"{COLAB}/v9_hotfix_cell.py").read().replace("/content", str(root))
    out = root / "physint/groot/groot-1028-v9-ablate"

    # run 1: stop handing out work after 7 finished episodes (simulated disconnect), then tear the last log line
    g1 = {"__name__": "__main__", "policy": pol, "CKPT": CK, "FAMILY": "groot", "LAYER": 5,
          "PLUS_DIR": f"{COLAB}/LIBERO-plus", "V8": V8, "v8": v8}
    os.environ["GROOT_ABLATE_STOP_AFTER"] = "7"
    exec(compile(src.replace("ABLATE_SUMMARY = _run_ablate_from_env(V8)",
                             "_orig = v9.Ctx.hours_left\n"
                             "    def _hl(self):\n"
                             "        n = sum(1 for p in Path(" + repr(str(out)) + ").rglob('episodes.jsonl') for l in open(p) if '\"type\": \"episode\"' in l)\n"
                             "        return 0.0 if n >= int(os.environ['GROOT_ABLATE_STOP_AFTER']) else 99.0\n"
                             "    v9.Ctx.hours_left = _hl\n"
                             "    try:\n"
                             "        ABLATE_SUMMARY = _run_ablate_from_env(V8)\n"
                             "    finally:\n"
                             "        v9.Ctx.hours_left = _orig", 1), "v9_hotfix_cell.py#1", "exec"), g1)
    n1 = {p.parent.relative_to(out).as_posix(): sum('"type": "episode"' in l for l in open(p)) for p in out.rglob("episodes.jsonl")}
    print("RUN1 episodes per condition", n1, flush=True)
    assert 7 <= sum(n1.values()) <= 7 + 3, n1          # in-flight episodes at the stop still finish and are logged
    assert not g1["ABLATE_SUMMARY"]["all_finished"]
    torn = out / "arm-A/cand-720/episodes.jsonl"
    with open(torn, "a") as f:
        f.write('{"type": "episode", "cond": "A-cand-720", "ep": 2, "success": tr')

    # run 2: resume
    g2 = {"__name__": "__main__", "policy": pol, "CKPT": CK, "FAMILY": "groot", "LAYER": 5,
          "PLUS_DIR": f"{COLAB}/LIBERO-plus", "V8": V8, "v8": v8}
    exec(compile(src, "v9_hotfix_cell.py#2", "exec"), g2)
    top = g2["ABLATE_SUMMARY"]
    assert top["all_finished"], top["table"]
    rows = {}
    for p in out.rglob("episodes.jsonl"):
        lines = [json.loads(l) for l in open(p) if l.strip().startswith("{") and l.strip().endswith("}")]
        hdr = [r for r in lines if r["type"] == "header"]
        eps = [r for r in lines if r["type"] == "episode"]
        assert len(hdr) == 1 and hdr[0]["schedule_order"] == mp["schedule"]["order"], p
        idx = [r["ep"] for r in eps]
        assert len(idx) == len(set(idx)), (p, idx)       # resume never re-ran a logged episode
        rows[hdr[0]["cond"]] = {r["ep"]: r for r in eps}
    for cid, rs in rows.items():
        c = next(c for c in mp["conditions"] if c["id"] == cid)
        assert sorted(rs) == list(range(c["n_max"])), (cid, sorted(rs))
        for i, r in rs.items():
            assert r["seed"] == 40000 + i and r["init_state_id"] == i % 50 and r["fail_rule"] and "fail_kind_active" in r
            assert (out / c["dir"] / "sim" / f"ep-{i:05d}.npz").is_file()
            if c["latents"] is None:
                assert r["edit_stats"] is None
            else:
                es = r["edit_stats"]
                n = r["n_predict"]
                assert n == 2 and es["n_calls"] == n and es["n_tok_all"] == 41 * n and es["n_tok_stat"] == 32 * n, (cid, i, es)
                if c["latents"]:
                    assert es["max_abs_change"] > 0 and es["mean_rel_delta"] > 0 and min(es["fire_frac_all"]) > 0, (cid, es)
    base, rec = rows["baseline"], rows["recon-only"]
    assert all(rec[i]["actions_sha256"] == base[i]["actions_sha256"] for i in rec), "recon-only must reproduce baseline exactly"
    assert all(r["edit_stats"]["max_abs_change"] == 0.0 for r in rec.values())
    changed = {cid: sum(rows[cid][i]["actions_sha256"] != base[i]["actions_sha256"] for i in rows[cid] if i in base)
               for cid in rows if cid not in ("baseline", "recon-only")}
    assert all(v > 0 for v in changed.values()), changed  # every ablation changed the stand-in's actions on some seed
    s = json.loads((out / "baseline/recon-only/summary.json").read_text())
    assert s["recon_check"]["n_identical_actions"] == s["recon_check"]["n_compared_with_baseline"] == 2
    sa = json.loads((out / "arm-A/cand-720/summary.json").read_text())
    assert sa["n"] == 3 and sa["sprt"]["stop_reason"] == "cap" and sa["vs_baseline"]["paired_shared_seeds"]["n_shared"] == 3
    assert (root / "ablate_snapshot.tar").is_file() and "predict_action_chunk" not in pol.__dict__
    assert not pol._groot_model.action_head.model.transformer_blocks[5]._forward_hooks, "hooks left installed"
    print("MOCK V9 ABLATE OK", json.dumps(top["table"]), "| actions changed vs baseline:", changed,
          "| pool", top["pool"]["episodes"], "eps this session", flush=True)
