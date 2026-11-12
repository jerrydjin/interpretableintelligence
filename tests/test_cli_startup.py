"""`ii --help` and `ii view` should not wait for torch to import."""

import subprocess
import sys


def test_cli_import_does_not_load_torch() -> None:
    code = "import sys, interpretableintelligence.cli; print('torch' in sys.modules, 'transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.split()
    assert out == ["False", "False"]


def test_view_module_does_not_load_torch() -> None:
    code = "import sys, interpretableintelligence.view; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip()
    assert out == "False"
