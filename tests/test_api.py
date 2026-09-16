"""Argument validation and report behaviour.

A verifier that silently checks nothing is worse than no verifier, so the cases
where nopeek cannot do its job must raise rather than return a clean report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nopeek
from conftest import GROUP, TIME, expanding_mean, next_value


def test_unknown_time_column(panel):
    with pytest.raises(KeyError, match="time column"):
        nopeek.verify(expanding_mean, panel, time="nope", group=GROUP)


def test_unknown_group_column(panel):
    with pytest.raises(KeyError, match="group column"):
        nopeek.verify(expanding_mean, panel, time=TIME, group="nope")


def test_key_must_contain_time(panel):
    with pytest.raises(ValueError, match="must include the time column"):
        nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP, key=[GROUP])


def test_non_unique_key_is_refused(panel):
    """Without a unique key the alignment join would fan out and compare noise."""

    def duplicated(df):
        return pd.concat([expanding_mean(df), expanding_mean(df)], ignore_index=True)

    with pytest.raises(ValueError, match="does not uniquely identify"):
        nopeek.verify(duplicated, panel, time=TIME, group=GROUP)


def test_output_missing_the_key_is_refused(panel):
    with pytest.raises(KeyError, match="missing key column"):
        nopeek.verify(lambda df: df[["x"]], panel, time=TIME, group=GROUP)


def test_non_frame_input_is_refused():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        nopeek.verify(expanding_mean, {"t": [1, 2, 3]}, time=TIME)


def test_fn_must_return_a_frame(panel):
    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        nopeek.verify(lambda df: df["x"], panel, time=TIME, group=GROUP)


def test_bad_strategy(panel):
    with pytest.raises(ValueError, match="strategy must be"):
        nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP, strategy="guess")


def test_unknown_requested_column(panel):
    with pytest.raises(KeyError, match="not in the output"):
        nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP, columns=["nope"])


def test_series_too_short_to_cut():
    frame = pd.DataFrame({GROUP: "e0", TIME: [0, 1], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 3 distinct"):
        nopeek.verify(expanding_mean, frame, time=TIME, group=GROUP)


def test_empty_cuts(panel):
    with pytest.raises(ValueError, match="must not be empty"):
        nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP, cuts=[])


def test_nothing_left_to_compare_is_a_note_not_a_pass(panel):
    report = nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP, ignore=["x_mean"])
    assert report.ok
    assert report.checks == 0
    assert any("no columns left" in note for note in report.notes)


def test_a_crashing_builder_is_noted(panel):
    """If poisoning makes fn blow up, say so rather than quietly claiming a pass."""

    def picky(df):
        if (df["x"].dropna() > 1000).any():
            raise ValueError("x out of range")
        return expanding_mean(df)

    report = nopeek.verify(picky, panel, time=TIME, group=GROUP, strategy="poison")
    assert report.checks == 0
    assert report.notes
    assert all("x out of range" in note for note in report.notes)


def test_unpoisonable_dtype_is_noted(panel):
    frame = panel.assign(weird=[complex(i, 1) for i in range(len(panel))])
    report = nopeek.verify(
        expanding_mean, frame, time=TIME, group=GROUP, strategy="poison", cuts=[5]
    )
    assert any("weird" in note for note in report.notes)


def test_preserve_keeps_a_column_intact(panel):
    """Some columns are constant per entity and known up front -- age, site, sex.

    A pipeline may legitimately read those from any row of the stay, which
    poisoning would otherwise flag: it rewrote a later row of a column that in
    reality never changes. ``preserve`` is how the caller says so. Truncation is
    unaffected either way, which is why only the poison strategy is checked here.
    """
    frame = panel.assign(age=panel[GROUP].map(lambda e: 60.0 + int(e[1:])))

    def uses_static(df):
        age = df.groupby(GROUP)["age"].transform("max")
        return df[[GROUP, TIME]].assign(scaled=df["x"] / age)

    assert not nopeek.verify(uses_static, frame, time=TIME, group=GROUP, strategy="poison"), (
        "poisoning a static column should look like a leak until preserve= says otherwise"
    )
    assert nopeek.verify(
        uses_static, frame, time=TIME, group=GROUP, strategy="poison", preserve=["age"]
    )
    assert nopeek.verify(uses_static, frame, time=TIME, group=GROUP, strategy="truncate")


def test_raise_for_status(panel):
    nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP).raise_for_status()
    with pytest.raises(nopeek.LeakError, match="x_next"):
        nopeek.verify(next_value, panel, time=TIME, group=GROUP).raise_for_status()


def test_assert_helpers(panel):
    nopeek.assert_point_in_time(expanding_mean, panel, time=TIME, group=GROUP)
    nopeek.assert_isolated(expanding_mean, panel, group=GROUP, time=TIME)
    nopeek.assert_split_clean(
        panel[panel[GROUP] < "e002"], panel[panel[GROUP] >= "e002"], group=GROUP
    )
    with pytest.raises(nopeek.LeakError):
        nopeek.assert_point_in_time(next_value, panel, time=TIME, group=GROUP)


def test_report_str_is_readable(panel):
    report = nopeek.verify(next_value, panel, time=TIME, group=GROUP)
    text = str(report)
    assert text.startswith("nopeek:")
    assert "x_next" in text
    assert "column(s) leak" in text
    # One mistake, found at every cut by both strategies: ten findings, one line.
    assert len(report.leaks) > len(text.splitlines())
    assert len(report.details().splitlines()) == len(report.leaks)
    clean = str(nopeek.verify(expanding_mean, panel, time=TIME, group=GROUP))
    assert "clean" in clean


def test_report_does_not_print_numpy_wrappers(panel):
    text = str(nopeek.verify(next_value, panel, time=TIME, group=GROUP, cuts=[5]))
    assert "np.int64" not in text
    assert (
        "np.int64"
        not in nopeek.verify(next_value, panel, time=TIME, group=GROUP, cuts=[5]).details()
    )


def test_all_exports_exist():
    for name in nopeek.__all__:
        assert hasattr(nopeek, name), name


def test_nan_only_column_compares_equal():
    frame = pd.DataFrame({GROUP: "e0", TIME: np.arange(10), "x": np.full(10, np.nan)})
    assert nopeek.verify(expanding_mean, frame, time=TIME, group=GROUP)


def test_categorical_and_boolean_columns_are_poisoned():
    """Non-numeric dtypes must be poisoned too, or leaks through them go unseen."""
    n = 12
    frame = pd.DataFrame(
        {
            GROUP: "e0",
            TIME: np.arange(n),
            "ward": pd.Categorical(["icu"] * (n // 2) + ["hdu"] * (n // 2)),
            # False throughout the early hours, so a max over the group is
            # decided by the future and not by anything already observed.
            "ventilated": np.arange(n) >= n - 3,
            "note": [f"n{i}" for i in range(n)],
        }
    )

    def reads_the_last_ward(df):
        last = df.groupby(GROUP)["ward"].transform("last")
        return df[[GROUP, TIME]].assign(ward_at_end=last.astype(str))

    report = nopeek.verify(
        reads_the_last_ward, frame, time=TIME, group=GROUP, strategy="poison"
    )
    assert not report.ok, "a categorical column must be poisoned, not skipped"
    assert not report.notes, f"nothing should have been left unpoisoned: {report.notes}"

    def reads_the_last_flag(df):
        return df[[GROUP, TIME]].assign(
            ever=df.groupby(GROUP)["ventilated"].transform("max"),
            final_note=df.groupby(GROUP)["note"].transform("last"),
        )

    bad = nopeek.verify(reads_the_last_flag, frame, time=TIME, group=GROUP, strategy="poison")
    assert set(bad.leaking_columns) == {"ever", "final_note"}


def test_version_is_not_duplicated():
    """__version__ comes from installed metadata, so it cannot drift from pyproject.

    Read with a regex rather than tomllib, which is not stdlib on the oldest
    Python this project supports.
    """
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    table = text.split("[project]", 1)[1].split("\n[", 1)[0]
    declared = re.search(r'^version = "([^"]+)"', table, re.M)
    assert declared is not None, "no version in the [project] table"
    assert nopeek.__version__ == declared.group(1)
