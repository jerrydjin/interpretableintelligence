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

# Default width of the attribution shortlist (`interpretableintelligence rank --prefilter`).
# `interpretableintelligence run` still ablates every feature unless --prefilter is set.
DEFAULT_PREFILTER_K = 10
# SAE width for a real model behind the hf adapter, per activation dimension.
HF_SAE_EXPANSION = 8
# TopK SAE for hf: active features per token, and its optimiser schedule.
HF_SAE_TOPK = 32
HF_SAE_LR = 1e-3
HF_SAE_STEPS = 3000
HF_SAE_BATCH = 1024
# Above this many SAE features, ``ii run`` scores an attribution shortlist by
# default: each scored feature costs a held-out ablation plus its matched
# random controls, so scoring thousands is not a laptop job.
SCORE_ALL_MAX_FEATURES = 64

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
# CLIP concept labels (hf): ROC AUC of the concept map for tokens where the
# feature fires against tokens where it does not. 0.5 is chance.
LABEL_ASSIGN_MIN_AUROC = 0.80
LABEL_HOLD_MIN_AUROC = 0.80

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

# Steering scorecard (`ii steer-scorecard`, steering/). Independent of `real` and of
# the rollout causal gate: it never rewrites scoreboard.json and is not a rollout
# finding. Each top feature is removed (scale 0) and boosted (scale
# STEER_BOOST_FACTOR) on held-out windows, next to STEER_CONTROLS matched random
# directions. A row "beats controls" only when its mean action change clears the
# Monte Carlo p-value at STEER_ALPHA (ties count against the feature) and the
# lower confidence bound of its excess over the controls is above zero.
STEER_TOP_FEATURES = 5
STEER_BOOST_FACTOR = 3.0
STEER_CONTROLS = 32
STEER_BOOTSTRAP = 2000
STEER_CONFIDENCE = 0.95
STEER_ALPHA = 0.05
# Relative changes at or below this are numerical dust when deciding whether the
# feature moved the actions more than every control on a window.
STEER_TIE_EPS = 1e-9
