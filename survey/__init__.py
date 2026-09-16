"""A reproducible survey of how often real pipelines read the future.

Not part of the installed package: this is repo-local tooling for the write-up.
"""

from __future__ import annotations

from .report import summarise, to_markdown, write
from .runner import run_all, run_target
from .spec import Result, Target, load, load_all

__all__ = [
    "Result",
    "Target",
    "load",
    "load_all",
    "run_all",
    "run_target",
    "summarise",
    "to_markdown",
    "write",
]
