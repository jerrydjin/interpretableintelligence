"""What the remote container installs. Kept free of Modal so tests can check it without the client."""

from __future__ import annotations

PYTHON_VERSION = "3.12"

# Same list as [project.dependencies] in pyproject.toml; tests/test_remote.py keeps them equal.
# On Linux, PyPI's torch wheel is the CUDA build.
RUNTIME_DEPENDENCIES = (
    "av>=18.1.0",
    "duckdb>=1.1",
    "fastapi>=0.115",
    "numpy>=1.26,<3",
    "pyarrow>=15",
    "python-multipart>=0.0.9",
    "safetensors>=0.4",
    "torch>=2.2",
    "transformers>=5.22,<6",
    "typer>=0.12",
    "uvicorn>=0.32",
)
# The groot adapter's own requirement (adapters/groot/adapter.py).
GROOT_DEPENDENCIES = ("lerobot[groot]==0.7.0",)
