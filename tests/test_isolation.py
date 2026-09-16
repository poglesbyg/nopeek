"""Entity isolation and order independence."""

from __future__ import annotations

import pytest

import nopeek
from conftest import (
    CORRECT,
    GROUP,
    TIME,
    expanding_mean,
    positional,
    ungrouped_window,
)


@pytest.mark.parametrize("builder", CORRECT, ids=lambda f: f.__name__)
def test_grouped_builders_are_isolated(builder, panel):
    report = nopeek.verify_isolation(builder, panel, group=GROUP, time=TIME)
    assert report.ok, str(report)
    assert report.checks > 0


def test_ungrouped_window_bleeds_between_entities(panel):
    report = nopeek.verify_isolation(ungrouped_window, panel, group=GROUP, time=TIME)
    assert not report.ok
    assert "x_roll" in report.leaking_columns
    assert any(leak.kind == "cross_group" for leak in report.leaks)


def test_order_dependence_is_caught(panel):
    """Only the reversal check can see this; each entity alone looks fine."""
    report = nopeek.verify_isolation(positional, panel, group=GROUP, time=TIME)
    assert not report.ok
    assert any("order" in leak.detail for leak in report.leaks)


def test_order_check_can_be_switched_off(panel):
    report = nopeek.verify_isolation(
        positional, panel, group=GROUP, time=TIME, check_order=False
    )
    assert not report.ok  # still caught per-entity, since positions shift
    clean = nopeek.verify_isolation(
        expanding_mean, panel, group=GROUP, time=TIME, check_order=False
    )
    assert clean.ok


def test_groups_are_sampled_deterministically(panel):
    first = nopeek.verify_isolation(
        ungrouped_window, panel, group=GROUP, time=TIME, max_groups=2
    )
    second = nopeek.verify_isolation(
        ungrouped_window, panel, group=GROUP, time=TIME, max_groups=2
    )
    assert first.checks == second.checks
    assert [leak.example for leak in first.leaks] == [leak.example for leak in second.leaks]


def test_explicit_groups(panel):
    report = nopeek.verify_isolation(
        expanding_mean, panel, group=GROUP, time=TIME, groups=["e000"], check_order=False
    )
    assert report.checks == 1
    assert report.ok


def test_missing_group_column_raises(panel):
    with pytest.raises(KeyError):
        nopeek.verify_isolation(expanding_mean, panel, group="nope", time=TIME)
