"""V-JEPA 2 adapter. Plain Hugging Face transformers, no ViT-Prisma."""

from interpretableintelligence.adapters.vjepa2.adapter import (
    MODEL_ID,
    WEIGHT_ALLOW_PATTERNS,
    VJEPA2Adapter,
    fetch_checkpoint,
    generate_tiny_clips,
)

__all__ = [
    "MODEL_ID",
    "VJEPA2Adapter",
    "WEIGHT_ALLOW_PATTERNS",
    "fetch_checkpoint",
    "generate_tiny_clips",
]
