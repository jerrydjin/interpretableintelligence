"""The README quickstart, the CI job that runs it, and the smoke script stay the same commands."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ["uv sync", "uv run ii try"]


def _readme_quickstart() -> list[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Quickstart", 1)[1]
    block = re.search(r"```bash\n(.*?)```", section, re.S)
    assert block, "README Quickstart has no bash block"
    return [line.strip() for line in block.group(1).splitlines() if line.strip()]


def test_readme_quickstart_is_clone_then_sync_then_try():
    lines = _readme_quickstart()
    assert lines[0].startswith("git clone https://github.com/") and lines[0].endswith("interpretableintelligence.git")
    assert lines[1] == "cd interpretableintelligence"
    assert lines[2:] == QUICKSTART


def test_smoke_script_runs_exactly_the_readme_commands():
    script = (ROOT / "scripts" / "quickstart_smoke.sh").read_text(encoding="utf-8")
    run = [
        line.strip().split(">", 1)[0].rstrip(" &")
        for line in script.splitlines()
        if line.strip().startswith("uv ") and not line.strip().startswith("uv run --no-sync")
    ]
    assert run == QUICKSTART


def test_ci_runs_the_smoke_script_on_linux_and_macos_from_clean_caches():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = workflow.split("  quickstart:", 1)[1]
    assert "ubuntu-latest" in job and "macos-latest" in job
    assert "bash scripts/quickstart_smoke.sh" in job
    assert "setup-uv" not in job and "actions/cache" not in job, "the quickstart job must start from an empty cache"
