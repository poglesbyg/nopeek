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

Replacement values are generated in the column's own dtype. Handing pandas a
float64 array for a float32 column raises rather than silently downcasting, and
a poison value outside a narrow dtype's range is not representable at all -- so
a naive implementation quietly poisons nothing on exactly the dtypes a feature
matrix is most likely to use.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

_SENTINEL = "__nopeek_future__"
_FAR_LOC = 1.0e6


def poison_future(
    data: pd.DataFrame,
    *,
    time: str,
    cut: object,
    protect: Iterable[str],
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of ``data`` with post-``cut`` values replaced.

    Also returns a note per column left untouched, saying why, so a gap in
    coverage is visible rather than silent.
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
        except Exception as exc:  # noqa: BLE001 - a gap in coverage, not a crash
            # Naming the failure matters: reporting every one of these as an
            # "unsupported dtype" once hid a bug where the dtype was fine and
            # the replacement values were the problem.
            skipped.append(f"{column!r} ({data[column].dtype}): {type(exc).__name__}: {exc}")
    return out, skipped


def _poison_column(
    series: pd.Series, future: np.ndarray, rng: np.random.Generator
) -> pd.Series:
    n = int(future.sum())
    dtype = series.dtype
    out = series.copy()

    if pdt.is_bool_dtype(dtype):
        flipped = ~series.to_numpy(dtype="bool", na_value=False)[future]
        out.iloc[future] = _as_dtype(flipped, dtype)
        return out

    if pdt.is_float_dtype(dtype):
        out.iloc[future] = _as_dtype(_far_floats(dtype, n, rng), dtype)
        return out

    if pdt.is_integer_dtype(dtype):
        out.iloc[future] = _as_dtype(_far_integers(dtype, n, rng), dtype)
        return out

    if pdt.is_datetime64_any_dtype(dtype):
        offsets = pd.to_timedelta(rng.integers(3650, 7300, n), unit="D")
        out.iloc[future] = series.iloc[future].to_numpy() + offsets.to_numpy()
        return out

    if pdt.is_timedelta64_dtype(dtype):
        out.iloc[future] = pd.to_timedelta(rng.integers(3650, 7300, n), unit="D").to_numpy()
        return out

    if isinstance(dtype, pd.CategoricalDtype):
        # Adding a category rather than reusing an existing one keeps the poison
        # distinguishable from any legitimate value. .where rather than an .iloc
        # assignment: it keeps the categorical dtype and does not need a scalar
        # written through a positional indexer.
        widened = series.cat.add_categories([_SENTINEL])
        return widened.where(~future, _SENTINEL)

    if pdt.is_object_dtype(dtype) or pdt.is_string_dtype(dtype):
        out.iloc[future] = _SENTINEL
        return out

    raise TypeError(f"no poison recipe for dtype {dtype!r}")


def _far_floats(dtype: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Implausible but representable values.

    float16 tops out around 65504, so a fixed 1e6 would poison it with inf and
    destroy the dtype. The magnitude is therefore capped by what the dtype can
    hold.
    """
    try:
        ceiling = float(np.finfo(_numpy_dtype(dtype)).max)
    except (TypeError, ValueError):  # pragma: no cover - exotic float dtype
        ceiling = float(np.finfo(np.float64).max)
    loc = min(_FAR_LOC, ceiling / 1e3)
    return rng.normal(loc, max(loc / 1e3, 1.0), n)


def _far_integers(dtype: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Values spread over the dtype's own range, so int8 gets int8-sized poison."""
    try:
        info = np.iinfo(_numpy_dtype(dtype))
    except (TypeError, ValueError):  # pragma: no cover - exotic integer dtype
        info = np.iinfo(np.int64)
    lo = max(info.min // 2, -(2**31))
    hi = min(info.max // 2, 2**31)
    if lo >= hi:  # pragma: no cover - a dtype too narrow to have a range
        lo, hi = info.min, info.max
    return rng.integers(lo, hi, n)


def _numpy_dtype(dtype: Any) -> Any:
    """The numpy dtype behind a pandas dtype, nullable extension types included."""
    backing = getattr(dtype, "numpy_dtype", None)
    return backing if backing is not None else dtype


def _as_dtype(values: np.ndarray, dtype: Any) -> Any:
    """Cast replacement values to the column's dtype before they are assigned.

    pandas refuses an assignment that would lose precision rather than
    downcasting silently, so this is what makes float32 and the narrow integer
    types poisonable at all.
    """
    try:
        return values.astype(dtype)
    except TypeError:
        return pd.array(values, dtype=dtype)
