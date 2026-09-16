"""nopeek -- verify that a pipeline never reads the future.

Three checks, each of which hides something the pipeline should not be able to
see and requires the output not to change:

* :func:`verify` hides the future, for point-in-time correctness.
* :func:`verify_isolation` hides the other entities, for window bleed.
* :func:`verify_split` compares two splits for entity, row and period overlap.

The approach is empirical rather than syntactic: it runs the real code on altered
inputs instead of pattern-matching the source, so it sees leaks through fitted
transformers, groupby transforms, compiled kernels and third-party libraries --
none of which a linter can reach.
"""

from __future__ import annotations

from ._types import Leak, LeakError, Report
from .isolation import verify_isolation
from .lookahead import verify
from .splits import verify_split
from .testing import assert_isolated, assert_point_in_time, assert_split_clean

__all__ = [
    "Leak",
    "LeakError",
    "Report",
    "assert_isolated",
    "assert_point_in_time",
    "assert_split_clean",
    "verify",
    "verify_isolation",
    "verify_split",
]

__version__ = "0.1.0"
