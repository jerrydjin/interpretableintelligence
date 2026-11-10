"""Load v9 against this repo's v8 module.

``inproc_v9`` imports ``physint_inproc_v8``. That file is ``v8/inproc_v8.py`` here.
Register that module under the name v9 expects before importing ``inproc_v9``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_V8_FILE = Path(__file__).resolve().parents[1] / "v8" / "inproc_v8.py"


def _register_v8():
    """One module object, importable as ``inproc_v8`` and ``physint_inproc_v8``."""
    repo = sys.modules.get("inproc_v8")
    alias = sys.modules.get("physint_inproc_v8")
    if repo is not None:
        sys.modules["physint_inproc_v8"] = repo
        return repo
    if alias is not None:
        sys.modules["inproc_v8"] = alias
        return alias
    spec = importlib.util.spec_from_file_location("inproc_v8", _V8_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_V8_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["inproc_v8"] = module
    sys.modules["physint_inproc_v8"] = module
    spec.loader.exec_module(module)
    return module


_register_v8()

from . import inproc_v9

# Same module object whether code imports the package path or Jeff's filename.
sys.modules["physint_inproc_v9"] = inproc_v9
verify_rotary_inv_freq = inproc_v9.verify_rotary_inv_freq

__all__ = ["verify_rotary_inv_freq"]
