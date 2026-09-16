"""pytest plugin.

Registered through the ``pytest11`` entry point, so installing nopeek is enough::

    def test_features_are_point_in_time(nopeek, stays):
        nopeek.point_in_time(build_features, stays, time="hour", group="patient_id")

The fixture exists so the check lands in somebody's CI rather than being run once
by hand and forgotten. That is the whole distribution strategy for a verifier.
"""

from __future__ import annotations

from typing import Any

import pytest

from .testing import assert_isolated, assert_point_in_time, assert_split_clean


class NoPeek:
    """Assertion helpers, bound to the ``nopeek`` fixture."""

    point_in_time = staticmethod(assert_point_in_time)
    isolated = staticmethod(assert_isolated)
    split = staticmethod(assert_split_clean)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "nopeek: marks a test that verifies temporal correctness of a pipeline",
    )


@pytest.fixture
def nopeek() -> NoPeek:
    """Temporal-correctness assertions: ``point_in_time``, ``isolated``, ``split``."""
    return NoPeek()
