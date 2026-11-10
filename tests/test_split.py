from interpretableintelligence.split import assign_split, split_sha


def test_holdout_is_by_clip_and_hashed():
    ids = [f"clip-{i:03d}" for i in range(30)]
    first = assign_split(ids, seed=0, heldout_frac=0.2)
    again = assign_split(ids, seed=0, heldout_frac=0.2)
    assert first == again
    assert len(first.heldout) == 6
    assert len(first.train) == 24
    assert set(first.train).isdisjoint(first.heldout)
    assert set(first.train) | set(first.heldout) == set(ids)
    assert first.sha == split_sha(0, 0.2, first.train, first.heldout)
    assert len(first.sha) == 64

    other = assign_split(ids, seed=1, heldout_frac=0.2)
    assert other.sha != first.sha
    assert set(other.heldout) != set(first.heldout)
