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

# SPRT alternative for a discordant pair: P(control succeeds, candidate fails)
# among discordant pairs is (1 + SPRT_LAMBDA) / 2. Any fixed lambda in (0, 1)
# keeps the type I error of the stop rule at alpha; 0.6 powers the test for
# an 0.80 one-sided discordant probability. It is not fit to a seed.
SPRT_LAMBDA = 0.6
