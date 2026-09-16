"""Dtype-aware, NaN-aware elementwise comparison.

Two subtleties that matter for leak detection and are easy to get wrong:

* Missingness is part of the answer. A builder that fills a gap by reaching
  forward produces the *same* finite values it would have produced anyway in
  most rows; what changes is which cells are null. Comparing only the finite
  values would miss it, so a null-vs-non-null mismatch counts as a difference.
* Float comparison needs a tolerance, because a correct pipeline may sum the
  same numbers in a different order when the frame is a different length.
  Integers, booleans, strings and timestamps get exact equality.
* That tolerance needs an absolute floor as well as a relative one. A rolling
  standard deviation over a constant window is 0 in exact arithmetic and about
  1e-7 in float32, and no relative tolerance can ever call those two close. With
  ``atol=0`` a verifier reports arithmetic as leakage, and a verifier that cries
  wolf gets switched off. The floor is derived from the dtype's own epsilon, so
  float32 data is judged at float32 resolution and float64 at float64.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

# A rolling statistic accumulates round-off roughly in proportion to its window,
# so the floor allows a few tens of ULPs rather than one. Anything a real leak
# does is orders of magnitude larger: filling a gap from the future moves a value
# by its own magnitude, not by an ulp.
_ULP_ALLOWANCE = 32


def resolve_atol(left: pd.Series, right: pd.Series, atol: float | None) -> float:
    """The absolute floor to compare with: explicit if given, else dtype-derived."""
    if atol is not None:
        return atol
    epsilons = [eps for eps in (_epsilon(left), _epsilon(right)) if eps is not None]
    if not epsilons:
        return 0.0
    return _ULP_ALLOWANCE * max(epsilons)


def differs(
    left: pd.Series,
    right: pd.Series,
    *,
    rtol: float = 1e-7,
    atol: float | None = None,
) -> np.ndarray:
    """Boolean mask, positionally aligned, that is True where the two differ.

    ``atol=None`` derives the absolute floor from the dtype; pass ``atol=0.0``
    for exact comparison.
    """
    atol = resolve_atol(left, right, atol)
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


def differs_at_all(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Exact difference, no tolerance at all.

    Used to tell "identical" apart from "different, but below the numerical
    resolution of the data" -- the second is worth a note even though it is not
    worth a finding.
    """
    return differs(left, right, rtol=0.0, atol=0.0)


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


def _epsilon(series: pd.Series) -> float | None:
    """Machine epsilon for a float column, or None if it is not one.

    Nullable float dtypes carry their numpy dtype behind ``numpy_dtype``;
    anything numpy cannot describe contributes no floor rather than raising.
    """
    if not _float_like(series):
        return None
    backing: Any = getattr(series.dtype, "numpy_dtype", series.dtype)
    try:
        return float(np.finfo(backing).eps)
    except (TypeError, ValueError):  # pragma: no cover - exotic float dtype
        return None


def _float_like(series: pd.Series) -> bool:
    return bool(pdt.is_float_dtype(series.dtype))


def _numeric_like(series: pd.Series) -> bool:
    return bool(pdt.is_numeric_dtype(series.dtype)) and not bool(
        pdt.is_bool_dtype(series.dtype)
    )


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="float64")
