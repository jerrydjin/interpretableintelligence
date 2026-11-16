"""SAE training logs progress about every tenth of the way with a time estimate, so a long run never looks stuck."""

from __future__ import annotations

import torch

from interpretableintelligence.sae import SAEConfig, train_sae


def test_training_logs_progress_with_time_left() -> None:
    torch.manual_seed(0)
    config = SAEConfig(d_model=8, n_features=4, steps=40, batch_size=16)
    lines: list[str] = []
    model = train_sae(torch.randn(64, 8), config, torch.device("cpu"), log=lines.append)
    assert lines[0].startswith("step 1/40 ") and "left" not in lines[0]
    assert len(lines) == 11  # every 4 steps, plus the last
    assert all("left" in line for line in lines[1:-1])
    assert lines[-1].startswith("step 40/40 ") and "left" not in lines[-1]
    assert model._train_log == lines[-1]
