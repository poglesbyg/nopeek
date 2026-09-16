"""Dtypes and float tolerance.

Both of the bugs this file guards against were found by running nopeek against a
real pipeline rather than by writing tests for it, and both were invisible from
inside the suite: one made poisoning silently do nothing on the dtype feature
matrices actually use, the other reported float32 round-off as leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nopeek
from conftest import GROUP, TIME
from nopeek._poison import poison_future

DTYPES = [
    "float64",
    "float32",
    "float16",
    "Float32",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "Int64",
    "bool",
    "boolean",
    "string",
    "object",
]


def _frame(dtype: str, n: int = 8) -> pd.DataFrame:
    if dtype in ("string", "object"):
        values: object = [f"v{i}" for i in range(n)]
    elif dtype in ("bool", "boolean"):
        values = [i % 2 == 0 for i in range(n)]
    else:
        values = list(range(1, n + 1))
    return pd.DataFrame({GROUP: "e0", TIME: np.arange(n), "x": pd.Series(values, dtype=dtype)})


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_common_dtype_is_actually_poisoned(dtype):
    """A dtype that cannot be poisoned is a silent hole, so none may be skipped."""
    frame = _frame(dtype)
    poisoned, skipped = poison_future(frame, time=TIME, cut=3, protect=[GROUP, TIME])

    assert skipped == [], f"{dtype} was left unpoisoned: {skipped}"
    future = frame[TIME] > 3
    assert not poisoned.loc[future, "x"].equals(frame.loc[future, "x"]), (
        f"{dtype} came back unchanged"
    )
    assert poisoned.loc[~future, "x"].equals(frame.loc[~future, "x"]), (
        f"{dtype}: the past was modified"
    )
    assert poisoned["x"].dtype == frame["x"].dtype


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_poison_stays_finite(dtype):
    """float16 tops out near 65504, so a fixed large poison value would be inf."""
    poisoned, _ = poison_future(_frame(dtype), time=TIME, cut=3, protect=[GROUP, TIME])
    assert np.isfinite(poisoned["x"].astype("float64")).all()


def test_a_float32_leak_is_caught_by_poisoning():
    """The regression that matters: float32 is what feature matrices are made of."""
    frame = _frame("float32", n=10)

    def reads_the_last_value(df):
        return df[[GROUP, TIME]].assign(final=df.groupby(GROUP)["x"].transform("last"))

    report = nopeek.verify(
        reads_the_last_value, frame, time=TIME, group=GROUP, strategy="poison"
    )
    assert not report.ok
    assert report.leaking_columns == ("final",)


def test_a_skipped_column_says_why():
    """'Unsupported dtype' once hid a bug where the dtype was fine."""
    frame = _frame("float32").assign(weird=[complex(i, 1) for i in range(8)])
    _, skipped = poison_future(frame, time=TIME, cut=3, protect=[GROUP, TIME])
    assert len(skipped) == 1
    assert "weird" in skipped[0]
    assert "complex" in skipped[0]
    assert "Error" in skipped[0] or "TypeError" in skipped[0]


def _rounding_frame() -> pd.DataFrame:
    """A float32 column whose builder differs from itself by about one ulp."""
    n = 12
    return pd.DataFrame({GROUP: "e0", TIME: np.arange(n), "x": np.ones(n, dtype="float32")})


def _ulp_noise_builder(scale: float):
    """Adds a difference of roughly one float32 ulp once the frame is truncated."""

    def build(df: pd.DataFrame) -> pd.DataFrame:
        wobble = 0.0 if len(df) == 12 else float(np.finfo(np.float32).eps) * scale
        return df[[GROUP, TIME]].assign(feature=np.full(len(df), wobble, dtype="float32"))

    return build


def test_round_off_is_a_note_not_a_finding():
    report = nopeek.verify(
        _ulp_noise_builder(1.4), _rounding_frame(), time=TIME, group=GROUP, cuts=[6]
    )
    assert report.ok, str(report)
    assert any("within float tolerance" in note for note in report.notes)
    assert any("round-off, not a leak" in note for note in report.notes)


def test_exact_comparison_is_still_available():
    """Round-off is tolerated by default, not ignored: atol=0 still reports it."""
    report = nopeek.verify(
        _ulp_noise_builder(1.4),
        _rounding_frame(),
        time=TIME,
        group=GROUP,
        cuts=[6],
        atol=0.0,
    )
    assert not report.ok
    assert report.leaking_columns == ("feature",)


def test_a_difference_far_above_round_off_is_still_a_finding():
    report = nopeek.verify(
        _ulp_noise_builder(1e6), _rounding_frame(), time=TIME, group=GROUP, cuts=[6]
    )
    assert not report.ok


def test_the_floor_follows_the_dtype():
    """The same absolute gap is noise in float32 and enormous in float64."""
    from nopeek._compare import differs, resolve_atol

    gap = 1.6e-07
    small = pd.Series([gap], dtype="float32"), pd.Series([0.0], dtype="float32")
    large = pd.Series([gap], dtype="float64"), pd.Series([0.0], dtype="float64")

    assert resolve_atol(*small, None) > gap
    assert resolve_atol(*large, None) < gap
    assert not differs(*small).any()
    assert differs(*large).any()


def test_non_float_columns_get_no_tolerance():
    """An integer off by one is off by one, whatever the float floor happens to be."""
    from nopeek._compare import resolve_atol

    left = pd.Series([1, 2], dtype="int64")
    right = pd.Series([1, 3], dtype="int64")
    assert resolve_atol(left, right, None) == 0.0
