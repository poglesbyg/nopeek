"""Does verify() agree with us about which builders are correct?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nopeek
from conftest import (
    CORRECT,
    GROUP,
    LEAKY,
    TIME,
    backfill,
    centred_window,
    drops_final_row,
    expanding_mean,
    make_panel,
    next_value,
    row_count_feature,
    whole_series_zscore,
)


@pytest.mark.parametrize("builder", CORRECT, ids=lambda f: f.__name__)
def test_correct_builders_are_clean(builder, panel):
    report = nopeek.verify(builder, panel, time=TIME, group=GROUP)
    assert report.ok, str(report)
    assert report.checks > 0
    assert report.leaks == []


@pytest.mark.parametrize("builder", LEAKY, ids=lambda f: f.__name__)
def test_leaky_builders_are_caught(builder, panel):
    report = nopeek.verify(builder, panel, time=TIME, group=GROUP)
    assert not report.ok
    assert not report  # Report is falsy when it found something
    assert all(leak.kind in ("lookahead", "missing_rows") for leak in report.leaks)


def test_report_names_the_offending_column(panel):
    report = nopeek.verify(backfill, panel, time=TIME, group=GROUP)
    assert report.leaking_columns == ("x_filled",)
    assert "x_filled" in str(report)


def test_clean_columns_are_not_blamed(panel):
    """A leak in one feature must not implicate the ones beside it."""

    def mixed(df):
        out = expanding_mean(df)
        out["x_leak"] = df.groupby(GROUP)["x"].shift(-1)
        return out

    report = nopeek.verify(mixed, panel, time=TIME, group=GROUP)
    assert report.leaking_columns == ("x_leak",)


def test_ignore_excludes_a_column(panel):
    def with_label(df):
        return expanding_mean(df).assign(future_label=df.groupby(GROUP)["label"].shift(-1))

    assert not nopeek.verify(with_label, panel, time=TIME, group=GROUP)
    assert nopeek.verify(with_label, panel, time=TIME, group=GROUP, ignore=["future_label"])


def test_reach_reflects_the_window_span(panel):
    """A centred window of width 5 reaches two steps ahead, so two rows are dirty.

    The row at the cut reads t+1 and t+2; the row before it reads t+1. The row
    two before the cut reaches only as far as the cut itself and stays clean, so
    the contamination stops at offset 1 -- half the span, minus the cut row.
    """
    report = nopeek.verify(centred_window, panel, time=TIME, group=GROUP, strategy="poison")
    reaches = {leak.farthest_offset for leak in report.leaks}
    assert reaches, str(report)
    assert max(reaches) == 1
    assert {leak.nearest_offset for leak in report.leaks} == {0}


def test_whole_series_statistic_reaches_the_beginning(panel):
    """A dataset-wide mean contaminates every row, not just those near the cut."""
    report = nopeek.verify(whole_series_zscore, panel, time=TIME, group=GROUP, cuts=[6])
    leak = next(x for x in report.leaks if x.column == "x_z")
    assert leak.farthest_offset == 6
    assert leak.nearest_offset == 0
    assert leak.max_abs_diff is not None and leak.max_abs_diff > 0


def test_poison_misses_what_truncation_catches(panel):
    """The two strategies are complementary; row_count_feature proves it."""
    truncate = nopeek.verify(
        row_count_feature, panel, time=TIME, group=GROUP, strategy="truncate"
    )
    poison = nopeek.verify(row_count_feature, panel, time=TIME, group=GROUP, strategy="poison")
    assert not truncate, "truncation should see the row count change"
    assert poison, "poisoning holds shape, so it cannot see a row count"
    assert not nopeek.verify(row_count_feature, panel, time=TIME, group=GROUP)


def test_missing_rows_are_reported_not_ignored(panel):
    report = nopeek.verify(drops_final_row, panel, time=TIME, group=GROUP)
    kinds = {leak.kind for leak in report.leaks}
    assert "missing_rows" in kinds
    assert any(leak.column == "<rows>" for leak in report.leaks)


def test_example_shows_both_values(panel):
    report = nopeek.verify(next_value, panel, time=TIME, group=GROUP, cuts=[5])
    leak = next(x for x in report.leaks if x.column == "x_next")
    assert set(leak.example) == {GROUP, TIME, "with_future", "without_future"}


def test_missingness_alone_counts_as_a_leak():
    """Backfill often reproduces the same finite numbers; what moves is the nulls."""
    frame = pd.DataFrame({GROUP: "e0", TIME: [0, 1, 2, 3], "x": [np.nan, np.nan, 5.0, 5.0]})

    def fill(df):
        return df[[GROUP, TIME]].assign(x_filled=df.groupby(GROUP)["x"].transform("bfill"))

    report = nopeek.verify(fill, frame, time=TIME, group=GROUP, cuts=[1])
    assert not report.ok
    assert report.leaking_columns == ("x_filled",)


def test_no_group_column_is_fine():
    frame = pd.DataFrame({TIME: np.arange(20), "x": np.arange(20, dtype=float)})

    def cumulative(df):
        return df[[TIME]].assign(total=df["x"].cumsum())

    def reversed_cumulative(df):
        return df[[TIME]].assign(total=df["x"][::-1].cumsum()[::-1])

    assert nopeek.verify(cumulative, frame, time=TIME)
    assert not nopeek.verify(reversed_cumulative, frame, time=TIME)


def test_datetime_time_column():
    n = 24
    frame = pd.DataFrame(
        {
            GROUP: "e0",
            TIME: pd.date_range("2026-01-01", periods=n, freq="h"),
            "x": np.arange(n, dtype=float),
        }
    )
    assert nopeek.verify(expanding_mean, frame, time=TIME, group=GROUP)
    report = nopeek.verify(next_value, frame, time=TIME, group=GROUP)
    assert not report.ok
    assert isinstance(report.leaks[0].nearest_offset, pd.Timedelta)


def test_fn_may_mutate_its_input(panel):
    """The builder gets a copy, so an in-place edit must not corrupt later checks."""

    def vandal(df):
        df["x"] = 0.0
        return expanding_mean(df)

    assert nopeek.verify(vandal, panel, time=TIME, group=GROUP)


def test_cuts_are_spread_over_the_series():
    frame = make_panel(n_entities=1, n_steps=50)
    report = nopeek.verify(
        whole_series_zscore, frame, time=TIME, group=GROUP, cuts=4, strategy="truncate"
    )
    assert report.checks == 4
    assert len({leak.cut for leak in report.leaks}) == 4


def test_explicit_cuts_are_honoured(panel):
    report = nopeek.verify(
        whole_series_zscore, panel, time=TIME, group=GROUP, cuts=[3, 7], strategy="poison"
    )
    assert report.checks == 2
    assert {leak.cut for leak in report.leaks} == {3, 7}
