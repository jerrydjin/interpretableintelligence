"""Register Jeff's module names before tests that load v9 from disk.

``test_ablate.py`` imports ``ablate_v9`` after putting ``/workspace/groot-colab``
on ``sys.path``, and a few cases read ``/workspace/runs``. CI has neither tree.
Importing this package registers the ``ablate_v9`` and ``physint_inproc_v9``
aliases, so that bare import resolves here. Cases that open a missing run folder
skip themselves, the same way ``test_pot_events.py`` does. If the alias is not
in place and those paths are missing (``box_paths.py`` says how to move them), collection skips the module instead of
erroring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import interpretableintelligence.adapters.groot.v9  # noqa: F401
from box_paths import COLAB as _COLAB, RUNS as _RUNS


def pytest_pycollect_makemodule(module_path, parent):
    """Skip ``test_ablate.py`` when its Colab and run paths are missing and the alias cannot satisfy the import."""
    path = Path(module_path)
    if path.name != "test_ablate.py":
        return None
    if (_COLAB.is_dir() and _RUNS.is_dir()) or "ablate_v9" in sys.modules:
        return None

    class _SkipAblade(pytest.Module):
        def collect(self):
            pytest.skip(
                f"ablate_v9 is not importable and {_COLAB} or {_RUNS} is absent",
                allow_module_level=True,
            )

    return _SkipAblade.from_parent(parent, path=path)
