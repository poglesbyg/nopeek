"""Point-in-time verification: does a pipeline read rows it could not have seen yet?

The property under test is narrow and entirely mechanical::

    fn(data)[time <= t]  ==  fn(data with everything after t hidden)[time <= t]

for every cut point ``t``. It is a claim about *implementation timing*, not about
the world: it says the code could have produced these rows at time ``t``, nothing
more. That is exactly the claim a backtest, a live scoring service and an offline
evaluation all silently assume, and it is the one nobody checks.

Verification is empirical rather than syntactic, which is the point. A linter
that greps for ``bfill`` cannot see leakage through a scaler fitted on the whole
frame, a target encoder, a ``groupby().transform()``, a Numba kernel or anything
implemented in C. Hiding the future and re-running the real code sees all of them,
and needs no list of patterns to match.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._compare import abs_delta, differs, differs_at_all
from ._poison import poison_future
from ._types import Leak, Report, Strategy

Builder = Callable[[pd.DataFrame], pd.DataFrame]

_LEFT = "__nopeek_full"
_RIGHT = "__nopeek_var"


def verify(
    fn: Builder,
    data: pd.DataFrame,
    *,
    time: str,
    group: str | None = None,
    key: Sequence[str] | None = None,
    cuts: int | Sequence[Any] = 5,
    strategy: Strategy | str = "both",
    columns: Sequence[str] | None = None,
    ignore: Iterable[str] = (),
    preserve: Iterable[str] = (),
    rtol: float = 1e-7,
    atol: float | None = None,
    seed: int = 0,
) -> Report:
    """Check that ``fn`` is point-in-time correct on ``data``.

    Parameters
    ----------
    fn:
        The pipeline under test: any callable taking a frame and returning a frame.
        It is handed a defensive copy, so it may mutate its argument freely.
    data:
        Input rows, one per entity-timestamp.
    time:
        Column holding the timestamp or sequence position. Must survive into the
        output, since rows are aligned on it.
    group:
        Column identifying the entity (patient, account, sensor). When given, it
        joins the alignment key, and windows are additionally checked for spilling
        across entities by :func:`nopeek.verify_isolation`.
    key:
        Columns uniquely identifying an output row. Defaults to ``(group, time)``,
        or ``(time,)`` when ``group`` is None. Must include ``time``.
    cuts:
        How many cut points to test (spread evenly over the observed timestamps),
        or an explicit sequence of cut values.
    strategy:
        ``"truncate"``, ``"poison"``, or ``"both"`` (the default). See
        :mod:`nopeek._poison` for why the two differ.
    columns:
        Output columns to check. Defaults to every output column outside the key.
    ignore:
        Output columns to skip -- the label, say, which legitimately depends on
        the future and is not a model input.
    preserve:
        Input columns that poisoning must leave alone, for values the pipeline is
        entitled to treat as static (age, site, admission time).
    rtol, atol:
        Float comparison tolerance. ``atol=None`` derives an absolute floor from
        the data's own dtype, so float32 is judged at float32 resolution; pass
        ``atol=0.0`` to demand bit-identical output. Differences smaller than the
        tolerance are reported in ``report.notes`` rather than as findings.

    Returns
    -------
    Report
        Falsy when a leak was found, so ``assert nopeek.verify(...)`` reads well.
    """
    frame = _require_frame(data, "data")
    strategies = _resolve_strategies(strategy)

    for name, value in (("time", time), ("group", group)):
        if value is not None and value not in frame.columns:
            raise KeyError(f"{name} column {value!r} is not in data")

    key_cols = tuple(key) if key is not None else _default_key(time, group)
    if time not in key_cols:
        raise ValueError(f"key {key_cols!r} must include the time column {time!r}")

    baseline = _call(fn, frame, "baseline")
    _require_columns(baseline, key_cols, "output of fn")
    duplicated = baseline.duplicated(list(key_cols))
    if bool(duplicated.any()):
        raise ValueError(
            f"key {key_cols!r} does not uniquely identify output rows "
            f"({int(duplicated.sum())} duplicate(s)); pass a key= that does"
        )

    compare = _compare_columns(baseline, key_cols, columns, ignore)
    report = Report(columns_checked=tuple(compare))
    if not compare:
        report.notes.append("no columns left to compare after key and ignore")
        return report

    protect = set(key_cols) | set(preserve) | ({group} if group else set())

    for cut in _choose_cuts(frame[time], cuts):
        for strat in strategies:
            if strat == "truncate":
                variant = frame.loc[frame[time] <= cut].copy()
            else:
                variant, skipped = poison_future(
                    frame, time=time, cut=cut, protect=protect, seed=seed
                )
                for column in skipped:
                    report.notes.append(
                        f"poison: left {column!r} unchanged (unsupported dtype), "
                        "so a leak through it would not be seen"
                    )
            try:
                observed = _call(fn, variant, f"{strat} at cut {cut!r}")
            except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a failure
                report.notes.append(
                    f"{strat} at cut {cut!r}: fn raised "
                    f"{type(exc).__name__}: {exc} -- not checked"
                )
                continue

            report.checks += 1
            leaks, notes = _compare_at_cut(
                baseline,
                observed,
                cut=cut,
                time=time,
                key_cols=key_cols,
                compare=compare,
                strategy=strat,
                rtol=rtol,
                atol=atol,
            )
            report.leaks.extend(leaks)
            report.notes.extend(notes)
    return report


def _compare_at_cut(
    baseline: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    cut: Any,
    time: str,
    key_cols: tuple[str, ...],
    compare: list[str],
    strategy: Strategy,
    rtol: float,
    atol: float | None,
) -> tuple[list[Leak], list[str]]:
    leaks: list[Leak] = []
    notes: list[str] = []
    present = [c for c in compare if c in observed.columns]
    for column in compare:
        if column not in observed.columns:
            leaks.append(
                Leak(
                    column=column,
                    kind="missing_rows",
                    detail="column disappeared when the future was hidden",
                    strategy=strategy,
                    cut=cut,
                )
            )
    if not present:
        return leaks, notes

    wanted = list(key_cols) + present
    left = baseline.loc[baseline[time] <= cut, wanted]
    right = observed.loc[observed[time] <= cut, wanted]
    merged = left.merge(
        right, on=list(key_cols), how="inner", suffixes=(_LEFT, _RIGHT)
    ).reset_index(drop=True)

    dropped = len(left) - len(merged)
    if dropped > 0:
        leaks.append(
            Leak(
                column="<rows>",
                kind="missing_rows",
                detail=(
                    f"{dropped} row(s) at or before the cut could not be produced "
                    "without the future"
                ),
                strategy=strategy,
                cut=cut,
                n_rows=dropped,
            )
        )
    if merged.empty:
        return leaks, notes

    times = merged[time]
    for column in present:
        mask = differs(merged[column + _LEFT], merged[column + _RIGHT], rtol=rtol, atol=atol)
        if not mask.any():
            # Different, but by less than the data's numerical resolution: an
            # artefact of float arithmetic rather than a leak. Worth saying,
            # not worth failing over.
            note = _residual_note(column, merged[column + _LEFT], merged[column + _RIGHT])
            if note is not None:
                notes.append(note)
            continue
        hit = times[mask]
        first = int(np.flatnonzero(mask)[0])
        leaks.append(
            Leak(
                column=column,
                kind="lookahead",
                detail="value changed when the future was hidden",
                strategy=strategy,
                cut=cut,
                n_rows=int(mask.sum()),
                nearest_offset=cut - hit.max(),
                farthest_offset=cut - hit.min(),
                max_abs_diff=abs_delta(merged[column + _LEFT], merged[column + _RIGHT], mask),
                example={
                    **{k: merged.at[first, k] for k in key_cols},
                    "with_future": merged.at[first, column + _LEFT],
                    "without_future": merged.at[first, column + _RIGHT],
                },
            )
        )
    return leaks, notes


def _residual_note(column: str, left: pd.Series, right: pd.Series) -> str | None:
    """Describe a difference too small to be a leak, or None if there is none."""
    residual = differs_at_all(left, right)
    if not residual.any():
        return None
    delta = abs_delta(left, right, residual)
    size = "" if delta is None else f", max |delta| {delta:.3g}"
    return (
        f"{column}: differs only within float tolerance "
        f"({int(residual.sum())} row(s){size}) -- round-off, not a leak"
    )


def _default_key(time: str, group: str | None) -> tuple[str, ...]:
    return (group, time) if group else (time,)


def _compare_columns(
    baseline: pd.DataFrame,
    key_cols: tuple[str, ...],
    columns: Sequence[str] | None,
    ignore: Iterable[str],
) -> list[str]:
    skip = set(key_cols) | set(ignore)
    if columns is not None:
        missing = [c for c in columns if c not in baseline.columns]
        if missing:
            raise KeyError(f"columns not in the output of fn: {missing}")
        return [c for c in columns if c not in skip]
    return [c for c in baseline.columns if c not in skip]


def _choose_cuts(times: pd.Series, cuts: int | Sequence[Any]) -> list[Any]:
    if not isinstance(cuts, int):
        chosen = list(cuts)
        if not chosen:
            raise ValueError("cuts must not be empty")
        return chosen
    if cuts < 1:
        raise ValueError("cuts must be >= 1")

    unique = np.sort(np.asarray(pd.unique(times.dropna())))
    if unique.size < 3:
        raise ValueError(
            f"need at least 3 distinct values of the time column to cut, got {unique.size}"
        )
    # Endpoints are useless: the first has no history, the last has no future.
    inner = unique[1:-1]
    if cuts >= inner.size:
        return list(inner)
    picks = np.unique(np.linspace(0, inner.size - 1, cuts).round().astype(int))
    return list(inner[picks])


def _resolve_strategies(strategy: Strategy | str) -> tuple[Strategy, ...]:
    if strategy == "both":
        return ("truncate", "poison")
    if strategy in ("truncate", "poison"):
        return (strategy,)  # type: ignore[return-value]
    raise ValueError(f"strategy must be 'truncate', 'poison' or 'both', got {strategy!r}")


def _require_frame(data: object, what: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{what} must be a pandas DataFrame, got {type(data).__name__}")
    return data


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{what} is missing key column(s) {missing}")


def _call(fn: Builder, frame: pd.DataFrame, what: str) -> pd.DataFrame:
    out = fn(frame.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError(
            f"fn must return a pandas DataFrame ({what} returned {type(out).__name__})"
        )
    return out
