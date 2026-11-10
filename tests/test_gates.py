"""The real-feature rule is conjunctive, and null never passes."""

import pytest

from physint.stats import feature_is_real, loss_recovered, spearman


def test_loss_recovered_scale():
    assert loss_recovered(loss_zero=10.0, loss_sae=10.0, loss_clean=0.0) == pytest.approx(0.0)
    assert loss_recovered(loss_zero=10.0, loss_sae=0.0, loss_clean=0.0) == pytest.approx(1.0)
    assert loss_recovered(loss_zero=10.0, loss_sae=3.0, loss_clean=0.0) == pytest.approx(0.7)


def test_loss_recovered_rejects_a_flat_scale():
    with pytest.raises(RuntimeError):
        loss_recovered(loss_zero=1.0, loss_sae=1.0, loss_clean=1.0)


def test_real_requires_every_gate():
    passing = dict(
        usable=True,
        distinct_clips_top20=5,
        label_holds_heldout=True,
        random_percentile=95.0,
        absent_in_random_init=True,
    )
    assert feature_is_real(**passing) is True
    failures = [
        {"usable": False},
        {"distinct_clips_top20": 4},
        {"label_holds_heldout": False},
        {"label_holds_heldout": None},
        {"random_percentile": 94.9},
        {"random_percentile": None},
        {"absent_in_random_init": False},
        {"absent_in_random_init": None},
    ]
    for override in failures:
        kwargs = {**passing, **override}
        assert feature_is_real(**kwargs) is False


def test_spearman_ranks_with_ties():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2], [1, 3, 3]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        spearman([1.0], [1.0])
