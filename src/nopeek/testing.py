"""Assertion wrappers, for use inside a test suite.

The ``verify_*`` functions return a report and never raise on a finding, which
suits a notebook or a CLI. In a test you want the opposite: a failure, with the
whole report in the message.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .isolation import verify_isolation
from .lookahead import Builder, verify
from .splits import verify_split


def assert_point_in_time(fn: Builder, data: pd.DataFrame, **kwargs: Any) -> None:
    """Fail unless ``fn`` is point-in-time correct. See :func:`nopeek.verify`."""
    verify(fn, data, **kwargs).raise_for_status()


def assert_isolated(fn: Builder, data: pd.DataFrame, **kwargs: Any) -> None:
    """Fail unless ``fn`` treats entities independently.

    See :func:`nopeek.verify_isolation`.
    """
    verify_isolation(fn, data, **kwargs).raise_for_status()


def assert_split_clean(train: pd.DataFrame, test: pd.DataFrame, **kwargs: Any) -> None:
    """Fail unless the two splits are disjoint. See :func:`nopeek.verify_split`."""
    verify_split(train, test, **kwargs).raise_for_status()
