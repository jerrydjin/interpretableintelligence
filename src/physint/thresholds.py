"""Scoreboard thresholds.

This module is the definition of a usable SAE and a real feature.
Later scoreboard revisions should freeze these numbers, not edit them
in place to make a run pass.
"""

MIN_CLIPS = 20
DEFAULT_HELDOUT_FRAC = 0.2
DEFAULT_SEED = 0
DEFAULT_LAYER = 16

# Spliced SAE must recover at least this fraction of predictor loss.
# 0 = layer zeroed, 1 = untouched model.
USABLE_LOSS_RECOVERED = 0.70

# Distinct clips among the top-20 strictly positive activations.
DISTINCT_CLIPS_TOP20 = 5
TOP_K_EXAMPLES = 20

# Ablation effect must land at or above this percentile of matched random features.
RANDOM_PERCENTILE_MIN = 95.0
N_RANDOM_FEATURES = 100

# clip_effects rows with effect_kind="ablation": this many features per clip.
TOP_K_ABLATION_PER_CLIP = 10

# Default width of the attribution shortlist (`physint rank --prefilter`).
# `physint run` still ablates every feature unless --prefilter is set.
DEFAULT_PREFILTER_K = 10

# Labeller: assign a cue only when train correlation clears this,
# and require this held-out correlation for label_holds_heldout.
LABEL_ASSIGN_MIN_CORR = 0.50
LABEL_HOLD_MIN_CORR = 0.60

# absent_in_random_init is true when no random-init SAE feature
# correlates with this feature above this Pearson value.
PATTERN_ABSENT_MAX_CORR = 0.45

# Activations at or below this count as "did not fire".
DEAD_ACT_EPS = 1e-5

# Vector norms below this are treated as zero in cosine distance.
COSINE_EPS = 1e-6

# Rollout causal gate (schema v1). Independent of `real`.
# delta is baseline success minus intervention success on paired seeds.
# A feature is causal only when the drop clears this floor, the exact paired
# test clears the run's alpha, and the drop beats this percentile of controls.
# Null (no controls) never passes.
CAUSAL_DELTA_MIN = 0.15
CAUSAL_CONTROL_PERCENTILE = 95.0

DEFAULT_ROLLOUT_EPISODES = 32
DEFAULT_ROLLOUT_CONTROLS = 10
DEFAULT_ROLLOUT_ALPHA = 0.05
DEFAULT_ROLLOUT_MIN_EPISODES = 8
DEFAULT_ROLLOUT_BATCH = 4
DEFAULT_ROLLOUT_WORKERS = 1

# One-sided efficacy SPRT on discordant pairs (control succeeds, candidate fails).
# theta1 = (1 + SPRT_LAMBDA) / 2 = 0.75. The run's alpha is family-wise across
# the controls (Bonferroni). Family-wise power is SPRT_FAMILY_POWER, so a
# control is designed at beta = 1 - power ** (1/K), about 0.022 for K=10.
# There is no futility boundary.
SPRT_LAMBDA = 0.5
SPRT_FAMILY_POWER = 0.8
