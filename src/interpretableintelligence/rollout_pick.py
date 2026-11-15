"""Pick rollout candidates from an SAE, the same way for a trained and an untrained one.

The random-weights SAE control (AGENTS.md research rule 4) only means something
if the random dictionary's features are chosen by the same procedure as the
trained dictionary's candidates. ``ii rollouts --pick N`` therefore picks the
candidates here, and ``--random-sae K`` sends each untrained dictionary through
the same call.

A policy can bring its own picker as ``pick_features(dictionary, n, *, kind,
value, seed)``; GR00T's is ``picker/pick_candidates.py``. Without one, the
default screens every feature offline: on ``PICK_DISCOVERY_EPISODES`` seeds that
no rollout arm uses, it runs each episode clean and with the edit, and scores
the feature by the summed squared change in the policy's actions. That is the
picker's offline-prune metric (action divergence) without a failure contrast.
The top ``n`` by score are picked; ties go to the lower feature id.
"""

from __future__ import annotations

import numpy as np

from interpretableintelligence.policies.base import Intervention
from interpretableintelligence.thresholds import PICK_DISCOVERY_EPISODES, PICK_SEED_OFFSET

DEFAULT_PICKER = "action_divergence"


def discovery_seeds(seed: int, episodes: int = PICK_DISCOVERY_EPISODES) -> list[int]:
    """Seeds the default picker screens on. Disjoint from rollout seeds ``seed .. seed + N``."""
    return [int(seed) + PICK_SEED_OFFSET + offset for offset in range(int(episodes))]


def picker_name(env) -> str:
    return "policy" if callable(getattr(env, "pick_features", None)) else DEFAULT_PICKER


def pick_features(
    env,
    dictionary: str,
    n: int,
    *,
    kind: str,
    value: float,
    layer: int,
    task_id: str,
    seed: int,
    episodes: int = PICK_DISCOVERY_EPISODES,
) -> list[int]:
    """The ``n`` features of ``dictionary`` this policy's picker ranks highest, sorted by id."""
    if n < 1:
        raise ValueError(f"pick needs n >= 1, got {n}")
    width = int(env.n_features)
    if n > width:
        raise ValueError(f"cannot pick {n} features from a {width}-feature dictionary")
    own = getattr(env, "pick_features", None)
    if callable(own):
        picked = [int(item) for item in own(dictionary, int(n), kind=kind, value=float(value), seed=int(seed))]
        if len(picked) != n or len(set(picked)) != n or any(not 0 <= item < width for item in picked):
            raise ValueError(f"{env.name}.pick_features returned {picked}, expected {n} distinct ids in 0..{width - 1}")
        return sorted(picked)
    scores = action_divergence(env, dictionary, kind=kind, value=value, layer=layer, task_id=task_id, seed=seed, episodes=episodes)
    order = sorted(range(width), key=lambda feature: (-scores[feature], feature))
    return sorted(order[:n])


def action_divergence(
    env,
    dictionary: str,
    *,
    kind: str,
    value: float,
    layer: int,
    task_id: str,
    seed: int,
    episodes: int = PICK_DISCOVERY_EPISODES,
) -> list[float]:
    """Mean over discovery episodes of ``sum_t ||a_edit(t) - a_clean(t)||^2``, per feature.

    Each edited episode runs closed loop from the same reset as its clean twin, so
    the score includes where the edit sends the policy, not only its first action.
    """
    width = int(env.n_features)
    horizon = int(getattr(env, "horizon", 10_000))
    totals = np.zeros(width, dtype=np.float64)
    seeds = discovery_seeds(seed, episodes)
    for episode_seed in seeds:
        clean = _actions(env, None, task_id, episode_seed, horizon)
        for feature in range(width):
            edit = Intervention(feature_ids=(feature,), layer=int(layer), kind=kind, value=float(value), dictionary=dictionary)
            edited = _actions(env, edit, task_id, episode_seed, horizon)
            totals[feature] += _divergence(clean, edited)
    return [float(item) for item in totals / max(1, len(seeds))]


def _actions(env, intervention, task_id: str, seed: int, horizon: int) -> list[np.ndarray]:
    observation = env.reset(task_id=task_id, seed=int(seed))
    actions: list[np.ndarray] = []
    with env.intervene(intervention):
        done = False
        while not done and len(actions) < horizon:
            action = np.asarray(env.act(observation), dtype=np.float64)
            actions.append(action)
            observation, _reward, done, _info = env.step(action)
    return actions


def _divergence(clean: list[np.ndarray], edited: list[np.ndarray]) -> float:
    """Squared action change, step by step. An episode that ends early counts its missing steps as zero action."""
    total = 0.0
    for step in range(max(len(clean), len(edited))):
        left = clean[step] if step < len(clean) else np.zeros_like(edited[step])
        right = edited[step] if step < len(edited) else np.zeros_like(clean[step])
        total += float(np.sum((right - left) ** 2))
    return total


def random_sae_seeds(seed: int, k: int) -> list[int]:
    """Seeds of the ``k`` untrained dictionaries. A larger ``k`` extends the same list."""
    if k < 0:
        raise ValueError(f"random-sae count must be >= 0, got {k}")
    if k == 0:
        return []
    state = np.random.SeedSequence([int(seed), 0x5A3D]).generate_state(int(k))
    return [int(item) for item in state]
