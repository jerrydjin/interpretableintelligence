"""groot-v9 import: synthetic rollcap fixture and split preservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.import_formats.groot_v9 import import_groot_v9
from interpretableintelligence.validate import validate_run

runner = CliRunner()


def _write_mini_rollcap(root: Path, *, n_ep: int = 22, d_model: int = 16, n_sae: int = 12, tokens: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "acts").mkdir()
    (root / "sae").mkdir()
    episodes = []
    heldout = sorted({1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21})
    train = [i for i in range(n_ep) if i not in heldout]
    for i in range(n_ep):
        n_chunks = 2 + (i % 3)
        steps = np.arange(n_chunks, dtype=np.int32) * 16
        toks = np.tile(np.arange(tokens, dtype=np.int16), n_chunks)
        acts = np.random.default_rng(i).standard_normal((n_chunks * tokens, d_model)).astype(np.float16)
        ep_steps = np.repeat(steps, tokens)
        save_file(
            {
                "acts": acts,
                "episode_idx": np.full(len(acts), i, dtype=np.int32),
                "step": ep_steps.astype(np.int32),
                "tok": toks,
            },
            str(root / "acts" / f"shard-{i:05d}.safetensors"),
        )
        episodes.append(
            {
                "episode_idx": i,
                "episode_id": f"ep-{i:05d}",
                "task_id": "1028",
                "seed": 20000 + i,
                "success": i % 3 == 0,
                "fail_kind": None if i % 3 == 0 else "never_grasped",
                "fail_t": None if i % 3 == 0 else 32,
                "fail_subkind": None if i % 3 == 0 else "no_close_attempt",
                "fail_rule": "pot0-v1: success" if i % 3 == 0 else "pot0-v1: no_close_attempt",
                "n_steps": int(n_chunks * 16 + 8),
                "n_chunks": n_chunks,
                "tokens_per_step": tokens,
            }
        )
    (root / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8")
    split = {
        "rule": "episode-level test fixture",
        "heldout": heldout,
        "train": train,
    }
    (root / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "rollcap-v0",
                "run_id": "mini-rollcap",
                "model": {"adapter": "groot", "name": "synthetic"},
                "layer": 0,
                "d_model": d_model,
                "tokens_per_step": tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    w_dec = torch.randn(d_model, n_sae)
    w_dec = w_dec / w_dec.norm(dim=0, keepdim=True)
    save_file(
        {
            "W_enc": w_dec.numpy().astype(np.float32),
            "W_dec": w_dec.T.contiguous().numpy().astype(np.float32),
            "b_enc": np.zeros((n_sae,), np.float32),
            "b_dec": np.zeros((d_model,), np.float32),
        },
        str(root / "sae" / "weights.safetensors"),
    )
    (root / "sae" / "config.json").write_text(
        json.dumps(
            {
                "arch": "test",
                "d_in": d_model,
                "d_sae": n_sae,
                "n_latents": n_sae,
                "train": {"lr": 1e-3, "steps": 10, "batch": 64, "seed": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "sae" / "heldout_eval.json").write_text(
        json.dumps({"heldout": {"ev_normalised": 0.9, "dead_latents": 2}, "rollout_io_load": {"sae_M": n_sae}})
        + "\n",
        encoding="utf-8",
    )


def test_import_preserves_split_bytes_and_validates(tmp_path: Path) -> None:
    src = tmp_path / "rollcap"
    out = tmp_path / "imported"
    _write_mini_rollcap(src)
    src_split = (src / "split.json").read_bytes()
    import_groot_v9(src=src, out=out)
    out_split = (out / "split.json").read_bytes()
    assert out_split == src_split
    assert hashlib.sha256(out_split).hexdigest() == json.loads((out / "manifest.json").read_text())["split"]["sha"]
    assert validate_run(out) == []


def test_import_cli(tmp_path: Path) -> None:
    src = tmp_path / "rollcap2"
    out = tmp_path / "imported2"
    _write_mini_rollcap(src, n_ep=24)
    result = runner.invoke(app, ["import", str(src), str(out), "--format", "groot-v9"])
    assert result.exit_code == 0, result.output
    assert (out / "scoreboard.json").is_file()
