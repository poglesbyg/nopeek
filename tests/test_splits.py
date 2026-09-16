"""Split auditing."""

from __future__ import annotations

import pandas as pd
import pytest

import nopeek
from conftest import GROUP, TIME, make_panel


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_panel(n_entities=6, n_steps=10)


def test_clean_entity_split_passes(frame):
    train = frame[frame[GROUP] < "e003"]
    test = frame[frame[GROUP] >= "e003"]
    report = nopeek.verify_split(train, test, group=GROUP)
    assert report.ok, str(report)
    assert report.checks == 2


def test_row_level_split_is_caught(frame):
    """Splitting rows rather than entities puts the same patient on both sides."""
    train = frame.iloc[::2]
    test = frame.iloc[1::2]
    report = nopeek.verify_split(train, test, group=GROUP)
    assert not report.ok
    leak = next(x for x in report.leaks if x.kind == "group_overlap")
    assert leak.n_rows == len(test)


def test_duplicate_rows_are_caught(frame):
    train = frame[frame[GROUP] < "e003"]
    test = pd.concat([frame[frame[GROUP] >= "e003"], train.head(4)], ignore_index=True)
    report = nopeek.verify_split(test.drop(columns=[]), train, group=None)
    assert not report.ok
    leak = next(x for x in report.leaks if x.kind == "duplicate_rows")
    assert leak.n_rows == 4


def test_chronological_overlap_is_opt_in(frame):
    train = frame[frame[GROUP] < "e003"]
    test = frame[frame[GROUP] >= "e003"]
    assert nopeek.verify_split(train, test, group=GROUP, time=TIME).ok
    report = nopeek.verify_split(train, test, group=GROUP, time=TIME, chronological=True)
    assert not report.ok
    assert any(leak.kind == "time_overlap" for leak in report.leaks)


def test_chronological_split_passes(frame):
    train = frame[frame[TIME] < 6]
    test = frame[frame[TIME] >= 6]
    report = nopeek.verify_split(
        train, test, time=TIME, chronological=True, check_duplicates=False
    )
    assert report.ok, str(report)


def test_chronological_needs_a_time_column(frame):
    with pytest.raises(ValueError, match="needs a time column"):
        nopeek.verify_split(frame, frame, chronological=True)


def test_subset_narrows_the_duplicate_check(frame):
    train = frame[frame[GROUP] < "e003"]
    test = frame[frame[GROUP] >= "e003"]
    # Every split shares timestamps, so hashing on time alone must flag it.
    report = nopeek.verify_split(train, test, subset=[TIME])
    assert not report.ok
