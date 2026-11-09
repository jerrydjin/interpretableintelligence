"""Hold out a fraction of clips, never a fraction of frames inside a clip."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    seed: int
    heldout_frac: float
    train: list[str]
    heldout: list[str]
    sha: str

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "heldout_frac": self.heldout_frac,
            "train": list(self.train),
            "heldout": list(self.heldout),
            "sha": self.sha,
        }

    def membership(self) -> dict[str, str]:
        out = {clip_id: "train" for clip_id in self.train}
        out.update({clip_id: "heldout" for clip_id in self.heldout})
        return out


def split_payload(seed: int, heldout_frac: float, train: list[str], heldout: list[str]) -> dict:
    return {
        "seed": seed,
        "heldout_frac": heldout_frac,
        "train": list(train),
        "heldout": list(heldout),
    }


def split_sha(seed: int, heldout_frac: float, train: list[str], heldout: list[str]) -> str:
    """sha256 of the canonical partition, excluding the sha field itself."""
    raw = json.dumps(
        split_payload(seed, heldout_frac, train, heldout),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def n_heldout(n_clips: int, heldout_frac: float) -> int:
    if n_clips < 2:
        raise ValueError("need at least 2 clips to hold one out")
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")
    n_held = int(round(n_clips * heldout_frac))
    return min(max(n_held, 1), n_clips - 1)


def assign_split(clip_ids: list[str], seed: int, heldout_frac: float) -> Split:
    """Shuffle clips with ``random.Random(seed)`` and hold out ``round(n * frac)`` of them.

    Lists in the result are sorted so the file depends on the partition, not the shuffle order.
    """
    ids = list(clip_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("clip ids must be unique")
    ids.sort()
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_held = n_heldout(len(ids), heldout_frac)
    heldout = sorted(ids[:n_held])
    train = sorted(ids[n_held:])
    sha = split_sha(seed, heldout_frac, train, heldout)
    return Split(seed=seed, heldout_frac=heldout_frac, train=train, heldout=heldout, sha=sha)
