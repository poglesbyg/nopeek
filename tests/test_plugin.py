"""The pytest fixture, exercised through the real pytest11 entry point.

If this file passes, the plugin is discoverable by anyone who pip-installs the
package -- which is the whole point of shipping it as a plugin.
"""

from __future__ import annotations

import pytest

from conftest import GROUP, TIME, expanding_mean, next_value, ungrouped_window


def test_fixture_is_registered(nopeek):
    assert hasattr(nopeek, "point_in_time")
    assert hasattr(nopeek, "isolated")
    assert hasattr(nopeek, "split")


def test_point_in_time_passes(nopeek, panel):
    nopeek.point_in_time(expanding_mean, panel, time=TIME, group=GROUP)


def test_point_in_time_fails_loudly(nopeek, panel):
    with pytest.raises(AssertionError, match="x_next"):
        nopeek.point_in_time(next_value, panel, time=TIME, group=GROUP)


def test_isolated(nopeek, panel):
    nopeek.isolated(expanding_mean, panel, group=GROUP, time=TIME)
    with pytest.raises(AssertionError):
        nopeek.isolated(ungrouped_window, panel, group=GROUP, time=TIME)


def test_split(nopeek, panel):
    nopeek.split(panel[panel[GROUP] < "e002"], panel[panel[GROUP] >= "e002"], group=GROUP)
