"""Entity isolation: does one entity's data change another's features?

A rolling window that forgets to group is the second-most common temporal bug
after lookahead, and point-in-time verification will not catch it -- the window
only reads the *past*, just somebody else's past. The check here is the same
shape as the lookahead one: compute an entity's rows alongside everyone else's,
compute them alone, and require the two to agree.

Row order is checked too. A pipeline whose output depends on the order entities
happen to arrive in is not wrong at every call, which is worse than being wrong
at every call.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._compare import abs_delta, differs
from ._types import Leak, Report
from .lookahead import (
    Builder,
    _call,
    _compare_columns,
    _default_key,
    _require_columns,
    _require_frame,
)

_LEFT = "__nopeek_all"
_RIGHT = "__nopeek_solo"


def verify_isolation(
    fn: Builder,
    data: pd.DataFrame,
    *,
    group: str,
    time: str | None = None,
    key: Sequence[str] | None = None,
    groups: Sequence[Any] | None = None,
    max_groups: int = 5,
    columns: Sequence[str] | None = None,
    ignore: Iterable[str] = (),
    check_order: bool = True,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> Report:
    """Check that ``fn`` treats each entity independently.

    Parameters
    ----------
    group:
        Column identifying the entity.
    time:
        Optional; joins the alignment key when given.
    groups:
        Specific entities to test. Defaults to ``max_groups`` spread evenly over
        the sorted entities, which is deterministic run to run.
    check_order:
        Also run ``fn`` on the same rows with the entity blocks reversed, and
        require identical output.
    """
    frame = _require_frame(data, "data")
    if group not in frame.columns:
        raise KeyError(f"group column {group!r} is not in data")
    if time is not None and time not in frame.columns:
        raise KeyError(f"time column {time!r} is not in data")

    key_cols = (
        tuple(key) if key is not None else _default_key(time, group) if time else (group,)
    )
    if group not in key_cols:
        raise ValueError(f"key {key_cols!r} must include the group column {group!r}")

    baseline = _call(fn, frame, "baseline")
    _require_columns(baseline, key_cols, "output of fn")

    compare = _compare_columns(baseline, key_cols, columns, ignore)
    report = Report(columns_checked=tuple(compare))
    if not compare:
        report.notes.append("no columns left to compare after key and ignore")
        return report

    for entity in _choose_groups(frame[group], groups, max_groups):
        solo = frame.loc[frame[group] == entity].copy()
        try:
            observed = _call(fn, solo, f"group {entity!r} alone")
        except Exception as exc:  # noqa: BLE001
            report.notes.append(
                f"group {entity!r}: fn raised {type(exc).__name__}: {exc} -- not checked"
            )
            continue
        report.checks += 1
        report.leaks.extend(
            _diff(
                baseline.loc[baseline[group] == entity],
                observed,
                key_cols=key_cols,
                compare=compare,
                kind="cross_group",
                detail=f"value changed when entity {entity!r} was built alone",
                rtol=rtol,
                atol=atol,
            )
        )

    if check_order:
        report.extend(
            _check_order(
                fn,
                frame,
                baseline,
                group=group,
                key_cols=key_cols,
                compare=compare,
                rtol=rtol,
                atol=atol,
            )
        )
    return report


def _check_order(
    fn: Builder,
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    group: str,
    key_cols: tuple[str, ...],
    compare: list[str],
    rtol: float,
    atol: float,
) -> Report:
    order = list(dict.fromkeys(frame[group]))[::-1]
    rank = {entity: i for i, entity in enumerate(order)}
    shuffled = (
        frame.assign(__nopeek_rank=frame[group].map(rank))
        .sort_values("__nopeek_rank", kind="stable")
        .drop(columns="__nopeek_rank")
        .reset_index(drop=True)
    )
    report = Report()
    try:
        observed = _call(fn, shuffled, "entities in reverse order")
    except Exception as exc:  # noqa: BLE001
        report.notes.append(
            f"reverse order: fn raised {type(exc).__name__}: {exc} -- not checked"
        )
        return report
    report.checks += 1
    report.leaks.extend(
        _diff(
            baseline,
            observed,
            key_cols=key_cols,
            compare=compare,
            kind="cross_group",
            detail="value depends on the order entities appear in",
            rtol=rtol,
            atol=atol,
        )
    )
    return report


def _diff(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    key_cols: tuple[str, ...],
    compare: list[str],
    kind: Any,
    detail: str,
    rtol: float,
    atol: float,
) -> list[Leak]:
    leaks: list[Leak] = []
    present = [c for c in compare if c in right_frame.columns]
    if not present:
        return leaks
    wanted = list(key_cols) + present
    left = left_frame[wanted]
    merged = left.merge(
        right_frame[wanted], on=list(key_cols), how="inner", suffixes=(_LEFT, _RIGHT)
    ).reset_index(drop=True)
    dropped = len(left) - len(merged)
    if dropped > 0:
        leaks.append(
            Leak(
                column="<rows>",
                kind="missing_rows",
                detail=f"{dropped} row(s) went missing; {detail}",
                n_rows=dropped,
            )
        )
    if merged.empty:
        return leaks

    for column in present:
        mask = differs(merged[column + _LEFT], merged[column + _RIGHT], rtol=rtol, atol=atol)
        if not mask.any():
            continue
        first = int(np.flatnonzero(mask)[0])
        leaks.append(
            Leak(
                column=column,
                kind=kind,
                detail=detail,
                n_rows=int(mask.sum()),
                max_abs_diff=abs_delta(merged[column + _LEFT], merged[column + _RIGHT], mask),
                example={
                    **{k: merged.at[first, k] for k in key_cols},
                    "together": merged.at[first, column + _LEFT],
                    "apart": merged.at[first, column + _RIGHT],
                },
            )
        )
    return leaks


def _choose_groups(
    values: pd.Series, groups: Sequence[Any] | None, max_groups: int
) -> list[Any]:
    if groups is not None:
        return list(groups)
    if max_groups < 1:
        raise ValueError("max_groups must be >= 1")
    unique = np.sort(np.asarray(pd.unique(values.dropna())))
    if unique.size == 0:
        raise ValueError("group column has no non-null values")
    if max_groups >= unique.size:
        return list(unique)
    picks = np.unique(np.linspace(0, unique.size - 1, max_groups).round().astype(int))
    return list(unique[picks])
