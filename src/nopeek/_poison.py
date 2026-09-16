"""Replace the future with garbage, leaving its shape intact.

Truncation and poisoning fail on different bugs, which is why nopeek runs both:

* Truncation deletes the future. It catches anything that reads future *values*
  and anything that depends on how many rows exist.
* Poisoning keeps every row and destroys the values in them. It catches value
  leaks while holding shape constant, so a pipeline that legitimately needs a
  fixed-length input still runs.

A pipeline that passes poisoning but fails truncation is reading the *extent* of
the series rather than its contents -- normalising by total length, say. That
distinction is in the report, because the two have very different fixes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api import types as pdt

_SENTINEL = "__nopeek_future__"
_FAR_LOC = 1.0e6
_FAR_SCALE = 1.0e3


def poison_future(
    data: pd.DataFrame,
    *,
    time: str,
    cut: object,
    protect: Iterable[str],
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of ``data`` with post-``cut`` values replaced.

    Also returns the columns left untouched, whose dtype had no poison recipe.
    """
    protected = set(protect) | {time}
    future = np.asarray(data[time] > cut)
    out = data.copy()
    skipped: list[str] = []

    if not future.any():
        return out, skipped

    rng = np.random.default_rng(seed)
    for column in data.columns:
        if column in protected:
            continue
        try:
            out[column] = _poison_column(data[column], future, rng)
        except Exception:  # noqa: BLE001 - an exotic dtype is a coverage gap, not a crash
            skipped.append(column)
    return out, skipped


def _poison_column(
    series: pd.Series, future: np.ndarray, rng: np.random.Generator
) -> pd.Series:
    n = int(future.sum())
    dtype = series.dtype
    out = series.copy()

    if pdt.is_bool_dtype(dtype):
        flipped = ~series.to_numpy(dtype="bool", na_value=False)[future]
        out.iloc[future] = flipped
        return out.astype(dtype)

    if pdt.is_float_dtype(dtype):
        out.iloc[future] = rng.normal(_FAR_LOC, _FAR_SCALE, n)
        return out.astype(dtype)

    if pdt.is_integer_dtype(dtype):
        info = np.iinfo(
            np.asarray(series.dropna().to_numpy()).dtype if series.notna().any() else np.int64
        )
        lo = max(info.min // 2, -(2**31))
        hi = min(info.max // 2, 2**31)
        out.iloc[future] = rng.integers(lo, hi, n)
        return out.astype(dtype)

    if pdt.is_datetime64_any_dtype(dtype):
        offsets = pd.to_timedelta(rng.integers(3650, 7300, n), unit="D")
        out.iloc[future] = series.iloc[future].to_numpy() + offsets.to_numpy()
        return out

    if pdt.is_timedelta64_dtype(dtype):
        out.iloc[future] = pd.to_timedelta(rng.integers(3650, 7300, n), unit="D").to_numpy()
        return out

    if isinstance(dtype, pd.CategoricalDtype):
        # Adding a category rather than reusing an existing one keeps the poison
        # distinguishable from any legitimate value.
        widened = series.cat.add_categories([_SENTINEL])
        # .where rather than an .iloc assignment: it keeps the categorical dtype
        # and does not need a scalar written through a positional indexer.
        return widened.where(~future, _SENTINEL)

    if pdt.is_object_dtype(dtype) or pdt.is_string_dtype(dtype):
        out.iloc[future] = _SENTINEL
        return out

    raise TypeError(f"no poison recipe for dtype {dtype!r}")
