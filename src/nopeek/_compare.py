"""Dtype-aware, NaN-aware elementwise comparison.

Two subtleties that matter for leak detection and are easy to get wrong:

* Missingness is part of the answer. A builder that fills a gap by reaching
  forward produces the *same* finite values it would have produced anyway in
  most rows; what changes is which cells are null. Comparing only the finite
  values would miss it, so a null-vs-non-null mismatch counts as a difference.
* Float comparison needs a tolerance, because a correct pipeline may sum the
  same numbers in a different order when the frame is a different length.
  Integers, booleans, strings and timestamps get exact equality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt


def differs(
    left: pd.Series, right: pd.Series, *, rtol: float = 1e-7, atol: float = 0.0
) -> np.ndarray:
    """Boolean mask, positionally aligned, that is True where the two differ."""
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right):  # pragma: no cover - callers align first
        raise ValueError("series must be the same length to compare")

    left_null = np.asarray(left.isna())
    right_null = np.asarray(right.isna())
    out: np.ndarray = np.asarray(left_null != right_null, dtype=bool)

    both_present = ~left_null & ~right_null
    if not both_present.any():
        return out

    where = np.flatnonzero(both_present)
    lv = left.to_numpy()[where]
    rv = right.to_numpy()[where]

    if _float_like(left) or _float_like(right):
        same = np.isclose(_as_float(lv), _as_float(rv), rtol=rtol, atol=atol, equal_nan=True)
    else:
        same = np.asarray(lv == rv, dtype=bool)

    out[where[~same]] = True
    return out


def abs_delta(left: pd.Series, right: pd.Series, mask: np.ndarray) -> float | None:
    """Largest absolute numeric difference among masked rows, or None if not numeric."""
    if not (_numeric_like(left) and _numeric_like(right)):
        return None
    where = np.flatnonzero(mask)
    if where.size == 0:
        return None
    lv = _as_float(left.reset_index(drop=True).to_numpy()[where])
    rv = _as_float(right.reset_index(drop=True).to_numpy()[where])
    delta = np.abs(lv - rv)
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return None
    return float(finite.max())


def _float_like(series: pd.Series) -> bool:
    return bool(pdt.is_float_dtype(series.dtype))


def _numeric_like(series: pd.Series) -> bool:
    return bool(pdt.is_numeric_dtype(series.dtype)) and not bool(
        pdt.is_bool_dtype(series.dtype)
    )


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="float64")
