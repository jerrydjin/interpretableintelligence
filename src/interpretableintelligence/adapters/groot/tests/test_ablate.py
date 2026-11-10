"""SAE feature-ablation runner (v9/ablate_v9.py): hook arithmetic, identity path, SPRT boundaries, resumable logs."""
import json, math, sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = ["/workspace/groot-colab/v8", "/workspace/groot-colab/v9"]
import ablate_v9 as A  # noqa: E402

SAE_DIR = Path("/workspace/runs/groot-1028-v9-n100/sae")
PLAN = Path("/workspace/runs/groot-1028-v9-ablate/ablate_plan.json")
SUBSET = Path("/workspace/runs/groot-1028-v9-ablate/sae_subset.safetensors")


def synth_sae(D=16, M=64, seed=0, thr=0.05):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D, M, generator=g); W = W / W.norm(dim=0, keepdim=True)
    return A.SAESubset(list(range(M)), W.clone(), 0.02 * torch.randn(M, generator=g), W.t().contiguous(),
                       0.1 * torch.randn(D, generator=g), torch.randn(D, generator=g), 3.0, thr, {})


def fake_x(sae, B=3, T=41, seed=1, k=6):
    """Activations built from the SAE dictionary (so latents fire) + an off-dictionary error term."""
    g = torch.Generator().manual_seed(seed)
    M = sae.W_dec.shape[0]
    f = torch.zeros(B, T, M)
    idx = torch.randint(0, M, (B, T, k), generator=g)
    f.scatter_(2, idx, 0.5 + torch.rand(B, T, k, generator=g))
    err = 0.05 * torch.randn(B, T, sae.d_in, generator=g)
    return sae.norm_mu + sae.norm_scale * (f @ sae.W_dec + sae.b_dec + err)


def ablator(sae, rows, **kw):
    a = A.LatentAblator(sae, **kw)
    a.rows = rows
    return a


def decompose(sae, x):
    """x = mu + s * (sum_j f_j W_dec[j] + b_dec) + s * err   (f from x)."""
    f = sae.encode(x)
    recon = sae.norm_mu + sae.norm_scale * (f @ sae.W_dec + sae.b_dec)
    return f, x - recon


@pytest.mark.parametrize("target", [[5], [3, 17], [0, 1, 2, 63]])
def test_removes_exactly_targeted_contribution(target):
    sae = synth_sae()
    x = fake_x(sae).double().float()
    assert (sae.encode(x)[..., target] > 0).any(), "targets must fire for the test to mean anything"
    e = A.Edit("c", tuple(target), tuple(target))
    out, st = ablator(sae, [e, None, e]).edit_tensor(x)
    f, err = decompose(sae, x)
    keep = [j for j in range(sae.W_dec.shape[0]) if j not in target]
    # other latents' contributions (with x's codes) and the reconstruction error are all still there, unchanged
    want = sae.norm_mu + sae.norm_scale * (f[..., keep] @ sae.W_dec[keep] + sae.b_dec) + err
    for b in (0, 2):
        torch.testing.assert_close(out[b], want[b], rtol=0, atol=2e-5)
        removed = x[b] - out[b]
        torch.testing.assert_close(removed, sae.norm_scale * f[b][:, target] @ sae.W_dec[target], rtol=0, atol=2e-5)
        assert st[b]["max_abs_change"] > 0
    assert torch.equal(out[1], x[1]) and st[1] is None  # a baseline row in the same batch is untouched


def test_only_targeted_rows_and_tokens_change():
    sae = synth_sae()
    x = fake_x(sae)
    f = sae.encode(x)
    j = int(torch.nonzero(f[0].sum(0) > 0)[0])
    e = A.Edit("c", (j,), (j,))
    out, _ = ablator(sae, [e, None, None], tokens=[1, 2, 3]).edit_tensor(x)
    d = (out[0] - x[0]).abs().sum(-1)
    assert torch.equal(out[1:], x[1:])
    assert float(d[[0] + list(range(4, 41))].max()) == 0.0
    fired = f[0, 1:4, j] > 0
    assert bool(((d[1:4] > 0) == fired).all())  # a token changes iff the latent fired there


def test_real_sae_subset_matches_full_and_removes_contribution():
    if not (SAE_DIR / "weights.safetensors").is_file() or not SUBSET.is_file():
        pytest.skip("SAE run folder absent")
    full = A.load_sae(SAE_DIR / "weights.safetensors", None, SAE_DIR / "norm_stats.safetensors")
    sub = A.load_sae(SUBSET, None, SAE_DIR / "norm_stats.safetensors")
    ids = sub.latent_ids
    assert 720 in ids and 7081 in ids and 5852 in ids and len(ids) == 30
    assert torch.equal(full.W_enc[:, ids], sub.W_enc) and torch.equal(full.W_dec[ids], sub.W_dec)
    assert torch.equal(full.b_enc[ids], sub.b_enc) and torch.equal(full.b_dec, sub.b_dec) and torch.equal(full.norm_mu, sub.norm_mu)
    assert sub.norm_scale == pytest.approx(269.1237487792969) and sub.threshold == pytest.approx(0.016480766236782074)
    # fake activations on which 720 / 7081 / 5852 fire, in the model's fp16
    g = torch.Generator().manual_seed(0)
    fz = torch.zeros(2, 41, 8192)
    alive = torch.tensor([720, 7081, 5852, 1159, 7339, 2343, 100, 4000])
    fz[..., alive] = torch.rand(2, 41, len(alive), generator=g) + 0.3
    x = (full.norm_mu + full.norm_scale * (fz @ full.W_dec + full.b_dec + 0.02 * torch.randn(2, 41, 1536, generator=g))).half()
    x32 = x.float()
    ff = full.encode(x32)
    torch.testing.assert_close(sub.encode(x32), ff[..., ids], rtol=0, atol=1e-5)  # same columns; fp32 matmul order only
    for lat in ([720], [7081, 5852]):
        assert float((ff[..., lat] > 0).float().mean()) > 0.5  # targets fire on most fake tokens
        e = A.Edit("c", tuple(sub.pos(lat)), tuple(sub.pos(lat)))
        out, _ = ablator(sub, [e, e]).edit_tensor(x)
        assert out.dtype == torch.float16
        want = x32 - full.norm_scale * ff[..., lat] @ full.W_dec[lat]
        torch.testing.assert_close(out.float(), want.half().float(), rtol=0, atol=0.0313)  # <= 1 fp16 ulp at |x| < 64
        keep = torch.ones(8192, dtype=torch.bool); keep[lat] = False
        rest = full.norm_mu + full.norm_scale * (ff[..., keep] @ full.W_dec[keep] + full.b_dec) + (x32 - (full.norm_mu + full.norm_scale * (ff @ full.W_dec + full.b_dec)))
        torch.testing.assert_close(out.float(), rest, rtol=0, atol=0.25)  # fp16 output rounding (|x| ~ 30)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_empty_set_is_bit_identical(dtype):
    sae = synth_sae()
    x = fake_x(sae).to(dtype)
    e = A.Edit("recon-only", (), tuple(range(64)))
    out, st = ablator(sae, [e, e, None]).edit_tensor(x)
    assert torch.equal(out, x) and out.dtype == dtype
    assert st[0]["max_abs_change"] == 0.0 and sum(st[0]["fired_all"]) > 0


class TupleBlock(torch.nn.Module):
    def __init__(s, D):
        super().__init__(); s.l = torch.nn.Linear(D, D)
    def forward(s, x): return (torch.tanh(s.l(x)) * 3, None)


def test_hook_final_step_only_and_call_count():
    sae = synth_sae()
    torch.manual_seed(0)  # block init independent of test order
    blk = TupleBlock(16)
    a = A.LatentAblator(sae, denoise_steps="final").install_on(blk, 4)
    x = fake_x(sae, B=2) / 10
    ref = [blk(x)[0] for _ in range(4)]
    fired = (sae.encode(ref[0][0]) > 0).sum(0)
    top = tuple(int(j) for j in torch.argsort(fired, descending=True)[:3])
    assert int(fired[list(top)].min()) > 0
    e = A.Edit("c", top, top)
    a.begin([e, A.Edit("r", (), (0,))])
    with torch.inference_mode():
        got = [blk(x)[0] for _ in range(4)]
    st = a.end(); a.disarm()
    for i in range(3):
        assert torch.equal(got[i], ref[i])      # denoise steps 0..2 untouched
    assert not torch.equal(got[3][0], ref[3][0]) and torch.equal(got[3][1], ref[3][1])  # step 3: row 0 ablated, row 1 identity
    assert st[0]["max_abs_change"] > 0 and st[1]["max_abs_change"] == 0.0
    a.begin([e, None]); blk(x); blk(x)
    with pytest.raises(RuntimeError, match="fired 2x"):
        a.end()
    a.disarm()
    assert torch.equal(blk(x)[0], ref[0])        # disarmed hook is a no-op
    a.begin([None, None])
    for _ in range(4):
        assert torch.equal(blk(x)[0], ref[0])
    assert a.end() == [None, None]
    a.remove()
    aa = A.LatentAblator(sae, denoise_steps="all").install_on(blk, 4)
    aa.begin([e, None]); outs = [blk(x)[0] for _ in range(4)]; aa.end(); aa.disarm()
    assert all(not torch.equal(o[0], r[0]) for o, r in zip(outs, ref))


def test_sprt_boundaries_and_increments():
    lo, hi = A.sprt_bounds(0.05, 0.05)
    assert lo == pytest.approx(math.log(0.05 / 0.95)) and hi == pytest.approx(math.log(19.0))
    assert lo == pytest.approx(-2.944439, abs=1e-6) and hi == pytest.approx(2.944439, abs=1e-6)
    assert A.sprt_bounds(0.05, 0.2) == pytest.approx((math.log(0.2 / 0.95), math.log(0.8 / 0.05)))
    assert A.llr_step(True, 0.56, 0.86) == pytest.approx(math.log(0.86 / 0.56))
    assert A.llr_step(False, 0.56, 0.86) == pytest.approx(math.log(0.14 / 0.44))
    assert A.llr_step(True, 0.56, 0.26) == pytest.approx(math.log(0.26 / 0.56))
    assert A.llr_step(False, 0.56, 0.26) == pytest.approx(math.log(0.74 / 0.44))
    sp = dict(p0=0.56, p1_rise=0.86, p1_fall=0.26, alpha=0.05, beta=0.05, cap=75)
    # all successes: fall accepts H0 at n=4 (4 x -0.767 = -3.07), rise accepts H1 at n=7 (7 x 0.429 = 3.00; 6 x = 2.57)
    d = A.sequential_decision([True] * 20, sp)
    assert (d["verdict"], d["n_used"], d["tests"]["fall"]["n_decided"], d["tests"]["rise"]["n_decided"]) == ("rise", 7, 4, 7)
    # all failures: rise H0 at n=3 (3 x -1.145 = -3.44), fall H1 at n=6 (6 x 0.520 = 3.12; 5 x = 2.60)
    d = A.sequential_decision([False] * 20, sp)
    assert (d["verdict"], d["n_used"], d["tests"]["rise"]["n_decided"], d["tests"]["fall"]["n_decided"]) == ("fall", 6, 3, 6)
    # exact boundary arithmetic on a mixed sequence, checked against a direct sum
    seq = [True, False, True, True, False, True, True, True, False, True, True, True, True, True, True]
    d = A.sequential_decision(seq, sp)
    r = f = 0.0
    for n, y in enumerate(seq[: d["n_used"]], 1):
        r += A.llr_step(y, 0.56, 0.86) if d["tests"]["rise"]["n_decided"] in (None,) or n <= d["tests"]["rise"]["n_decided"] else 0
        f += A.llr_step(y, 0.56, 0.26) if d["tests"]["fall"]["n_decided"] in (None,) or n <= d["tests"]["fall"]["n_decided"] else 0
    assert d["tests"]["rise"]["llr"] == pytest.approx(r, abs=1e-6) and d["tests"]["fall"]["llr"] == pytest.approx(f, abs=1e-6)
    # undecided until the cap
    alt = [True, False] * 40
    d = A.sequential_decision(alt[:10], sp)
    assert not d["stopped"] and d["verdict"] == "running"
    d = A.sequential_decision([True, True, False] * 30, dict(sp, cap=5))
    assert d["stopped"] and d["stop_reason"] == "cap" and d["n_used"] == 5
    # the literal "stop when either test decides" can never return rise / fall with these parameters
    rng = np.random.default_rng(0)
    lit = dict(sp, stop_rule="either_test_decides")
    assert {A.sequential_decision(list(rng.random(75) < p), lit)["verdict"] for p in (0.1, 0.56, 0.95) for _ in range(300)} <= {"no_rise", "no_fall"}


def test_paired_delta_and_wilson():
    cond = {0: True, 1: True, 2: False, 3: True}
    base = {0: False, 1: True, 2: False, 3: False, 4: True}
    d = A.paired_delta(cond, base)
    assert d["n_shared"] == 4 and d["delta"] == 0.5 and d["n_cond_only_success"] == 2 and d["n_base_only_success"] == 0
    assert d["mcnemar_exact_p_two_sided"] == pytest.approx(0.5)
    import physint_inproc_v9 as v9
    assert A.wilson(56, 100) == [round(v, 4) for v in v9.wilson(56, 100)] == [0.4623, 0.6533]


def small_plan(tmp_path, cap=6, base_n=6, recon_n=2):
    sp = dict(p0=0.56, p1_rise=0.86, p1_fall=0.26, alpha=0.05, beta=0.05, cap=cap, stop_rule="h1_or_both_h0")
    conds = [{"id": "baseline", "arm": "baseline", "kind": "baseline", "latents": None, "n_max": base_n, "sprt": False, "dir": "baseline"},
             {"id": "recon-only", "arm": "baseline", "kind": "recon_only", "latents": [], "n_max": recon_n, "sprt": False, "dir": "baseline/recon-only"},
             {"id": "A-cand", "arm": "A", "kind": "candidate", "latents": [1], "n_max": cap, "sprt": True, "dir": "arm-A/cand"},
             {"id": "A-ctrl01", "arm": "A", "kind": "control", "latents": [2], "n_max": cap, "sprt": True, "dir": "arm-A/ctrl01"}]
    return {"schema": A.PLAN_SCHEMA, "sprt": sp, "seeds": {"base": 40000}, "conditions": conds,
            "schedule": {"rule": "lockstep", "order": [c["id"] for c in conds]}}


def ep_row(i, ok, **kw):
    return {"ep": i, "seed": 40000 + i, "init_state_id": i % 50, "success": ok, "fail_kind": None if ok else "never_grasped", **kw}


def test_schedule_is_lockstep_and_interleaved(tmp_path):
    st = A.AblationState(small_plan(tmp_path), tmp_path / "o")
    got = [st.next() for _ in range(9)]
    assert got == [("baseline", 0), ("recon-only", 0), ("A-cand", 0), ("A-ctrl01", 0), ("baseline", 1), ("recon-only", 1),
                   ("A-cand", 1), ("A-ctrl01", 1), ("baseline", 2)]
    hdr = json.loads((tmp_path / "o/arm-A/cand/episodes.jsonl").read_text().splitlines()[0])
    assert hdr["type"] == "header" and hdr["schedule_position"] == 2 and hdr["schedule_order"][:3] == ["baseline", "recon-only", "A-cand"]


def test_log_resumes(tmp_path):
    plan, out = small_plan(tmp_path), tmp_path / "o"
    st = A.AblationState(plan, out)
    seq = {"baseline": [True, False, True], "A-cand": [False, False, False, True], "A-ctrl01": [True]}
    for cid, ys in seq.items():
        for i, y in enumerate(ys):
            st.inflight[cid].add(i)
            st.record(cid, ep_row(i, y))
    st.inflight["A-ctrl01"].add(1)            # in flight when the runtime disconnects: must be re-run
    st.write_summaries()
    p = out / "arm-A/cand/episodes.jsonl"
    with open(p, "a") as f:
        f.write('{"type": "episode", "cond": "A-cand", "ep": 9, "succ')   # torn write
    before = {c: dict(st.rows[c]) for c in st.rows}
    st2 = A.AblationState(plan, out)
    assert {c: sorted(r) for c, r in st2.rows.items()} == {c: sorted(r) for c, r in before.items()}
    assert st2.prefix("A-cand") == [False, False, False, True]
    assert st2.decision("A-cand") == st.decision("A-cand")
    # A-cand: rise H0 at n=3 (3 failures), fall still open -> keeps running; its next index is 4
    assert st2.decision("A-cand")["tests"]["rise"]["decision"] == "H0" and not st2.finished("A-cand")
    assert st2.next_index("A-cand") == 4 and st2.next_index("A-ctrl01") == 1 and st2.next_index("baseline") == 3
    st2.inflight["A-cand"].add(4)
    st2.record("A-cand", ep_row(4, False))   # appended after the torn line, on its own line
    lines = p.read_text().splitlines()
    assert json.loads(lines[-1])["ep"] == 4
    st3 = A.AblationState(plan, out)
    assert sorted(st3.rows["A-cand"]) == [0, 1, 2, 3, 4]
    # re-recording an episode that is already logged is ignored (append-only, first wins)
    n = len(p.read_text().splitlines())
    st3.record("A-cand", ep_row(4, True))
    assert len(p.read_text().splitlines()) == n and st3.rows["A-cand"][4]["success"] is False
    # a different plan in the same out dir is refused
    other = small_plan(tmp_path, cap=7)
    with pytest.raises(RuntimeError, match="different plan"):
        A.AblationState(other, out)


def test_sprt_stop_on_prefix_and_summary(tmp_path):
    plan, out = small_plan(tmp_path, cap=75, base_n=10), tmp_path / "o"
    st = A.AblationState(plan, out)
    order = [6, 0, 1, 2, 3, 4, 5, 7]           # out-of-order completion; ep 6 first
    for i in order:
        st.inflight["A-cand"].add(i)
        st.record("A-cand", ep_row(i, True))
    for i in range(10):
        st.record("baseline", ep_row(i, i % 2 == 0))
    d = st.decision("A-cand")
    assert d["verdict"] == "rise" and d["n_used"] == 7 and st.finished("A-cand")
    assert st.next_index("A-cand") is None
    s = st.summary("A-cand")
    assert s["n"] == 7 and s["successes"] == 7 and s["n_logged"] == 8 and s["n_logged_not_used"] == 1
    assert s["rate"] == 1.0 and s["wilson95"] == A.wilson(7, 7)
    pd = s["vs_baseline"]["paired_shared_seeds"]
    assert pd["n_shared"] == 7 and pd["rate_base"] == round(4 / 7, 4) and pd["delta"] == round(1 - 4 / 7, 4)
    assert s["vs_baseline"]["unpaired"]["baseline_n"] == 10
    b = st.summary("baseline")
    assert b["n"] == 10 and b["fail_kind_counts"] == {"never_grasped": 5} and "vs_baseline" not in b
    top = st.write_summaries()
    assert [r["cond"] for r in top["table"]] == plan["schedule"]["order"]
    assert json.loads((out / "arm-A/cand/summary.json").read_text())["sprt"]["verdict"] == "rise"


def test_default_plan_matches_candidates():
    if not PLAN.is_file():
        pytest.skip("plan absent")
    p = json.loads(PLAN.read_text())
    cf = json.loads(Path("/workspace/runs/groot-1028-v9-n100/candidates_final.json").read_text())
    c = {x["id"]: x for x in p["conditions"]}
    assert c["A-cand-720"]["latents"] == [720] and c["B-cand-7081+5852"]["latents"] == [7081, 5852]
    a = [x for x in p["conditions"] if x["arm"] == "A" and x["kind"] == "control"]
    b = [x for x in p["conditions"] if x["arm"] == "B" and x["kind"] == "control"]
    assert [x["latents"][0] for x in a] == cf["matched_random_controls"]["720"]
    assert [x["latents"] for x in b] == [[u, v] for u, v in zip(cf["matched_random_controls"]["7081"], cf["matched_random_controls"]["5852"])]
    assert c["baseline"]["n_max"] == 75 and not c["baseline"]["sprt"] and c["baseline"]["latents"] is None
    assert c["recon-only"]["latents"] == [] and p["schedule"]["order"][:4] == ["baseline", "recon-only", "A-cand-720", "B-cand-7081+5852"]
    assert p["sprt"] | {} == p["sprt"] and (p["sprt"]["p0"], p["sprt"]["p1_rise"], p["sprt"]["p1_fall"], p["sprt"]["alpha"], p["sprt"]["beta"], p["sprt"]["cap"]) == (0.56, 0.86, 0.26, 0.05, 0.05, 75)
    assert p["hook"]["denoise_steps"] == "final" and p["hook"]["tokens"] == "all" and p["seeds"]["base"] == 40000
    assert p["task"]["id"] == 1028 and p["task"]["index"] == 1027 and p["task"]["n_init_states"] == 50
    assert A.sha256_file(SUBSET) == p["sae"]["subset_sha256"]
