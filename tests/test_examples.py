"""The README quotes the example's output, so the example has to run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_runs_and_finds_both_leaks():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "2 of 3 column(s) leak" in result.stdout
    assert "balance_filled" in result.stdout
    assert "balance_z" in result.stdout
    # The trailing window is correct and must not be implicated.
    assert "balance_mean_3d" not in result.stdout


def test_readme_quotes_the_real_output():
    readme = (ROOT / "README.md").read_text()
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    headline = result.stdout.splitlines()[0]
    assert headline in readme, f"README is stale; example now prints:\n{headline}"
