"""Untrained GR00T SAEs for the random-weights SAE control (AGENTS.md research rule 4).

The GR00T capture SAE (``groot-1028-v9-n100/sae``) is a BatchTopK SAE, ``k`` = 32, run at inference as a JumpReLU
at a learned threshold on ``x_n = (x - norm_mu) / norm_scale`` (``ablate_v9.SAESubset.encode``). Its trainer's init,
recorded in ``sae/SAE_SPEC_for_random_weights_control.json``, is a ``randn(D, M)`` matrix with unit-norm columns used
as the encoder and, transposed, as the decoder, with zero biases.

An untrained control keeps everything that is not learned and redraws the rest:

- ``norm_mu`` and ``norm_scale`` are copied: they are data statistics fixed before training, not weights.
- ``W_enc`` / ``W_dec`` come from that init with ``torch.Generator().manual_seed(seed)``; ``b_enc`` and ``b_dec`` are 0.
- The threshold is the one learned parameter BatchTopK inference needs that the init does not give. An untrained SAE
  run at the trained threshold fires about 1,900 latents per token instead of 32, so it would not be the same
  architecture at the same sparsity. The threshold is set so the untrained SAE keeps exactly ``k * n`` activations
  over the ``n`` calibration tokens (the trained SAE's train-split tokens), the BatchTopK rule over one batch of all
  of them. Its mean L0 there is ``k``.

The result has the trained SAE's keys and shapes, so ``ablate_v9.load_sae``, ``ablate_v9.write_subset`` and the
picker's ``rollout_io.PolicySAE`` read it unchanged. Nothing is optimised.
"""

# The implementation lives in core (``interpretableintelligence.sae_batchtopk``) so any model's SAE can use it.
from interpretableintelligence.sae_batchtopk import (  # noqa: F401
    INIT_RULE,
    SAE_KEYS,
    batchtopk_threshold,
    build_untrained_sae,
    load_tensors,
    pre_activations,
    save_sae_dir,
    untrained_weights,
)
