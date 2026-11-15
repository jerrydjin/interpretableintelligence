"""``ii import --format groot-v9`` on a tiny synthetic rollcap-v0 capture."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.importers import import_run
from interpretableintelligence.validate import validate_run

D_MODEL = 8
N_LATENTS = 16
CHUNK = 4
HORIZON = 4  # DiT sequence = state token + 4 action tokens
TOKEN_INDICES = [0, 1, 3, 4]  # the capture keeps 4 of the 5 tokens
N_EPISODES = 24
HELDOUT = [0, 5, 9, 13, 17, 22]
NORM_SCALE = 3.5
THRESHOLD = 0.05
SHA = "ab" * 32


def _episode(idx: int) -> dict:
    success = idx % 3 != 0
    n_chunks = 2 + idx % 2  # 2 or 3 policy calls: clips have different window counts
    pot0 = {} if success else {"fail_kind": "never_grasped", "fail_t": 5, "fail_subkind": "close_attempt_failed"}
    return {
        "episode_idx": idx,
        "episode_id": f"ep-{idx:05d}",
        "task_id": "1028",
        "seed": 20000 + idx,
        "init_state_id": idx % 5,
        "success": success,
        "fail_kind": pot0.get("fail_kind"),
        "fail_t": pot0.get("fail_t"),
        "fail_subkind": pot0.get("fail_subkind"),
        "fail_rule": "pot0-v1: success" if success else "pot0-v1: close_attempt_failed",
        # The v9 active-object detector's labels; the importer must not read them.
        "fail_kind_v9_active": "drop",
        "fail_t_v9_active": 1,
        "n_steps": n_chunks * CHUNK - 1,
        "n_chunks": n_chunks,
        "status": "done",
        "split": "heldout" if idx in HELDOUT else "train",
    }


def make_v9_capture(root: Path, *, pot0: bool = True) -> Path:
    """A rollcap-v0 folder shaped like groot-1028-v9-n100, small enough for CI."""
    rng = np.random.default_rng(7)
    src = root / "tiny-v9"
    (src / "acts").mkdir(parents=True)
    (src / "sae").mkdir()
    episodes = [_episode(idx) for idx in range(N_EPISODES)]
    if not pot0:
        for ep in episodes:
            ep["fail_rule"] = None
    for ep in episodes:
        rows = ep["n_chunks"] * len(TOKEN_INDICES)
        steps = np.repeat(np.arange(ep["n_chunks"]) * CHUNK, len(TOKEN_INDICES)).astype(np.int32)
        toks = np.tile(np.asarray(TOKEN_INDICES), ep["n_chunks"]).astype(np.int16)
        order = rng.permutation(rows)  # the source does not promise row order
        save_file(
            {
                "acts": (rng.normal(0.5, 2.0, size=(rows, D_MODEL))).astype(np.float16)[order],
                "episode_idx": np.full(rows, ep["episode_idx"], dtype=np.int32),
                "step": steps[order],
                "tok": toks[order],
            },
            str(src / "acts" / f"shard-{ep['episode_idx']:05d}.safetensors"),
        )
    w_dec = rng.normal(size=(N_LATENTS, D_MODEL)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    save_file(
        {
            "W_enc": np.ascontiguousarray(w_dec.T * 1.3),
            "W_dec": w_dec,
            "b_enc": rng.normal(0.0, 0.05, size=N_LATENTS).astype(np.float32),
            "b_dec": rng.normal(0.0, 0.05, size=D_MODEL).astype(np.float32),
            "norm_mu": rng.normal(0.5, 0.1, size=D_MODEL).astype(np.float32),
            "norm_scale": np.asarray([NORM_SCALE], dtype=np.float32),
            "threshold": np.asarray(THRESHOLD, dtype=np.float32),
        },
        str(src / "sae" / "weights.safetensors"),
    )
    (src / "sae" / "config.json").write_text(
        json.dumps(
            {
                "arch": "batchtopk",
                "d_in": D_MODEL,
                "d_sae": N_LATENTS,
                "k": 4,
                "n_latents": N_LATENTS,
                "norm": {"pos_mean": None, "norm_scale": NORM_SCALE},
                "inference_threshold": THRESHOLD,
                "train": {"steps": 10, "batch": 32, "lr": 0.0004, "seed": 0},
            }
        ),
        encoding="utf-8",
    )
    (src / "sae" / "heldout_eval.json").write_text(
        json.dumps({"loss_recovered": None, "loss_recovered_note": "not computed: no action head in this capture"}),
        encoding="utf-8",
    )
    train = [idx for idx in range(N_EPISODES) if idx not in HELDOUT]
    # Indented, with a rule string and no ii seed: it must be copied byte for byte, not re-serialised.
    split_text = json.dumps({"rule": "default_rng(0).choice -> heldout", "heldout": HELDOUT, "train": train}, indent=1)
    (src / "split.json").write_text(split_text, encoding="utf-8")
    (src / "episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in episodes), encoding="utf-8")
    manifest = {
        "format": "rollcap-v0",
        "run_id": "tiny-v9",
        "model": {"adapter": "groot", "name": "nvidia/gr00t17-lerobot-libero_10-640", "family": "groot"},
        "layer": 5,
        "hook": {"module": "_groot_model.action_head.model.transformer_blocks.5", "block_index": 5, "denoise_step_index": 3},
        "d_model": D_MODEL,
        "tokens_per_step": len(TOKEN_INDICES),
        "token_indices": TOKEN_INDICES,
        "n_tok": 1 + HORIZON,
        "action_queue_steps": CHUNK,
        "action_horizon": HORIZON,
        "episodes": episodes,
        "fail_relabel": {"rule": "pot0-v1" if pot0 else None, "script": "tools/derive_pot0_fail.py", "units": "env steps"},
    }
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return src


def _source_encode(src: Path, x: np.ndarray) -> np.ndarray:
    """The source SAE's own inference: normalise, ReLU, JumpReLU at the learned threshold."""
    from safetensors.numpy import load_file

    w = load_file(str(src / "sae" / "weights.safetensors"))
    xn = (x - w["norm_mu"]) / float(w["norm_scale"][0])
    pre = np.maximum((xn - w["b_dec"]) @ w["W_enc"] + w["b_enc"], 0.0)
    return pre * (pre > float(w["threshold"]))


@pytest.fixture()
def offline(monkeypatch):
    """No Hub lookups in tests: the importer falls back to the declared-model digest."""
    import interpretableintelligence.adapters.groot.adapter as groot

    def refuse(_checkpoint):
        raise OSError("offline")

    monkeypatch.setattr(groot, "_checkpoint_fingerprint", refuse)


def test_groot_v9_import_validates_and_keeps_the_source(tmp_path, offline):
    src = make_v9_capture(tmp_path)
    out = tmp_path / "runs" / "tiny"
    manifest = import_run("groot-v9", src, out, log=lambda _line: None)

    assert validate_run(out) == []
    assert (out / "split.json").read_bytes() == (src / "split.json").read_bytes()

    # PR #17's GR00T layout: one clip per episode, one window per policy call, w = DiT token index.
    assert manifest["model"]["adapter"] == "groot:nvidia/gr00t17-lerobot-libero_10-640"
    assert manifest["grid"] == {"t": 1, "h": 1, "w": 1 + HORIZON, "tubelet": CHUNK, "patch_px": 1}
    assert manifest["import"]["cells"]["w"] == TOKEN_INDICES
    assert [row["n_windows"] for row in manifest["import"]["clips"]] == [2 + idx % 2 for idx in range(N_EPISODES)]
    assert manifest["import"]["notes"]["model_sha_source"].startswith("declared")
    held = {clip["clip_id"] for clip in manifest["clips"] if clip["split"] == "heldout"}
    assert held == {f"ep-{idx:05d}" for idx in HELDOUT}

    # Failure labels are the pot-0 detector's, never the v9 active-object ones.
    for row in manifest["import"]["clips"]:
        idx = row["source_id"]
        assert row["fail_rule"].startswith("pot0-v1")
        assert row["fail_kind"] == (None if idx % 3 != 0 else "never_grasped")
        assert "fail_kind_v9_active" not in row

    # Uncomputable metrics are null with a reason, never 0.0, and nothing is real.
    scoreboard = json.loads((out / "scoreboard.json").read_text())
    sae = scoreboard["sae"]
    assert sae["loss_recovered"] is None and sae["loss_sae"] is None and sae["loss_visible_mean"] is None
    assert sae["usable"] is None
    assert isinstance(sae["ev_heldout"], float) and isinstance(sae["dead_frac"], float)
    nulls = manifest["import"]["nulls"]
    for key in ("scoreboard.sae.loss_recovered", "scoreboard.sae.usable", "features/clip_effects.parquet"):
        assert nulls[key].strip()
    assert "no action head" in nulls["scoreboard.sae.loss_recovered"]
    assert pq.read_metadata(out / "features" / "clip_effects.parquet").num_rows == 0
    assert all(feature["real"] is False for feature in scoreboard["features"])
    assert all(feature["ablation_effect"] is None and feature["n_random"] == 0 for feature in scoreboard["features"])
    assert not (out / "acts" / "stream").exists()

    # The folded SAE reproduces the source's active set; values are norm_scale times the source's.
    from safetensors.numpy import load_file

    episode = load_file(str(src / "acts" / "shard-00003.safetensors"))
    order = np.lexsort((episode["tok"], episode["step"]))
    expected = _source_encode(src, episode["acts"].astype(np.float32)[order]) * NORM_SCALE
    acts = pq.read_table(out / "features" / "feature_acts.parquet").to_pylist()
    got = np.zeros_like(expected)
    token_pos = {tok: i for i, tok in enumerate(TOKEN_INDICES)}
    for row in acts:
        if row["clip_id"] == "ep-00003":
            got[row["window"] * len(TOKEN_INDICES) + token_pos[row["w"]], row["feature_id"]] = row["value"]
    assert (expected > 0).sum() > 10  # the fixture SAE actually fires
    assert np.array_equal(got > 0, expected > 0)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_cli_imports_a_tarball_then_validates(tmp_path):
    src = make_v9_capture(tmp_path)
    archive = tmp_path / "tiny-v9.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(src, arcname=src.name)
    out = tmp_path / "runs" / "tiny-tar"
    runner = CliRunner()
    result = runner.invoke(app, ["import", "--format", "groot-v9", str(archive), str(out), "--model-sha", SHA])
    assert result.exit_code == 0, result.output
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"]["sha"] == SHA
    assert manifest["import"]["source"]["archive"]["name"] == archive.name
    assert not [path for path in out.parent.iterdir() if path.name.startswith(".ii-import-")]


def test_refuses_a_capture_without_pot0_labels(tmp_path, offline):
    src = make_v9_capture(tmp_path, pot0=False)
    with pytest.raises(ValueError, match="pot0-v1"):
        import_run("groot-v9", src, tmp_path / "out", log=lambda _line: None)
    assert not (tmp_path / "out").exists()


def test_refuses_unknown_format_and_nonempty_output(tmp_path):
    with pytest.raises(ValueError, match="unknown import format"):
        import_run("nope", tmp_path, tmp_path / "out")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(FileExistsError):
        import_run("groot-v9", tmp_path, busy)


def test_validate_catches_imported_run_tampering(tmp_path, offline):
    src = make_v9_capture(tmp_path)
    out = tmp_path / "runs" / "tiny"
    import_run("groot-v9", src, out, log=lambda _line: None)
    manifest_path = out / "manifest.json"
    scoreboard_path = out / "scoreboard.json"
    pristine_manifest = manifest_path.read_text()
    pristine_scoreboard = scoreboard_path.read_text()

    def errors_after(edit_manifest=None, edit_scoreboard=None) -> list[str]:
        manifest = json.loads(pristine_manifest)
        scoreboard = json.loads(pristine_scoreboard)
        if edit_manifest:
            edit_manifest(manifest)
        if edit_scoreboard:
            edit_scoreboard(scoreboard)
        manifest_path.write_text(json.dumps(manifest))
        scoreboard_path.write_text(json.dumps(scoreboard))
        return validate_run(out)

    # A null without its reason.
    errors = errors_after(lambda m: m["import"]["nulls"].pop("scoreboard.sae.loss_recovered"))
    assert any("loss_recovered is null" in error for error in errors)
    # A placeholder number with the loss still unmeasured: usable must stay null.
    errors = errors_after(edit_scoreboard=lambda s: s["sae"].update(usable=False))
    assert any("usable must be null" in error for error in errors)
    # A reason for something that was measured.
    errors = errors_after(edit_scoreboard=lambda s: s["sae"].update(loss_recovered=0.0, loss_sae=1.0, loss_visible_mean=1.0, usable=False))
    assert any("not null" in error for error in errors)
    # A hand-set real flag.
    errors = errors_after(edit_scoreboard=lambda s: s["features"][0].update(real=True))
    assert any("four gates" in error for error in errors)
    # A window dropped from a clip's record.
    errors = errors_after(lambda m: m["import"]["clips"][0].update(n_windows=1))
    assert any("activation rows" in error for error in errors)
    assert errors_after() == []

    # split.json re-serialised (same content, different bytes) no longer matches its recorded sha.
    split_path = out / "split.json"
    split_path.write_text(json.dumps(json.loads(split_path.read_text())))
    assert any("sha256 of the split.json bytes" in error for error in validate_run(out))
