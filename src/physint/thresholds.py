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

# Loss-based ablations. alpha matches the cosine percentile gate:
# real iff the Monte Carlo p-value is <= alpha (ties count against the candidate).
EFFECT_ALPHA = (100.0 - RANDOM_PERCENTILE_MIN) / 100.0
# |loss delta| below this is numerical dust and counts as zero.
EFFECT_POS_EPS = 1e-8
# Co-active group size for effect_kind="group_loss_video".
GROUP_ABLATION_SIZE = 3

EFFECT_KIND_ESTIMATE = "estimate"
EFFECT_KIND_ABLATION = "ablation"
EFFECT_KIND_LOSS_VIDEO = "loss_video"
EFFECT_KIND_LOSS_WINDOW = "loss_window"
EFFECT_KIND_GROUP_LOSS_VIDEO = "group_loss_video"
EFFECT_KIND_GROUP_LOSS_WINDOW = "group_loss_window"
LOSS_EFFECT_KINDS = (
    EFFECT_KIND_LOSS_VIDEO,
    EFFECT_KIND_LOSS_WINDOW,
    EFFECT_KIND_GROUP_LOSS_VIDEO,
    EFFECT_KIND_GROUP_LOSS_WINDOW,
)
KNOWN_EFFECT_KINDS = (
    EFFECT_KIND_ESTIMATE,
    EFFECT_KIND_ABLATION,
    *LOSS_EFFECT_KINDS,
)

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
