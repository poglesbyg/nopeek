"""Split auditing: is the test set actually held out?

Three failures, in rough order of how often they go unnoticed:

* The same entity appears in both splits. Rows within an entity are strongly
  autocorrelated, so the model can recognise the patient rather than the illness,
  and the test score measures memorisation. Splitting at the row level is the
  usual cause.
* Identical rows appear in both splits -- a duplicated export, a merge that
  fanned out, an augmentation applied before splitting.
* The test period overlaps the training period, when the split was meant to be
  chronological.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ._types import Leak, Report
from .lookahead import _require_frame

_MAX_EXAMPLES = 5


def verify_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group: str | None = None,
    time: str | None = None,
    chronological: bool = False,
    subset: Sequence[str] | None = None,
    check_duplicates: bool = True,
) -> Report:
    """Check that ``train`` and ``test`` are disjoint in the ways that matter.

    Parameters
    ----------
    group:
        Entity column. Any entity present in both splits is reported.
    time:
        Timestamp column, needed for ``chronological``.
    chronological:
        Require every test timestamp to fall strictly after every training one.
        Off by default, because a grouped random split legitimately overlaps in
        time -- turn it on when the split was meant to be forward in time.
    subset:
        Columns used for the duplicate-row check. Defaults to the columns the two
        frames have in common.
    """
    left = _require_frame(train, "train")
    right = _require_frame(test, "test")
    report = Report()

    if group is not None:
        for name, frame in (("train", left), ("test", right)):
            if group not in frame.columns:
                raise KeyError(f"group column {group!r} is not in {name}")
        report.checks += 1
        shared = pd.Index(pd.unique(left[group])).intersection(
            pd.Index(pd.unique(right[group]))
        )
        if len(shared):
            n_rows = int(right[group].isin(shared).sum())
            report.leaks.append(
                Leak(
                    column=group,
                    kind="group_overlap",
                    detail=(
                        f"{len(shared)} entit(y/ies) appear in both splits, "
                        f"covering {n_rows} test row(s)"
                    ),
                    n_rows=n_rows,
                    example={"shared": list(shared[:_MAX_EXAMPLES])},
                )
            )

    if check_duplicates:
        columns = (
            list(subset)
            if subset is not None
            else [c for c in left.columns if c in right.columns]
        )
        if columns:
            report.checks += 1
            left_hash = pd.util.hash_pandas_object(left[columns], index=False)
            right_hash = pd.util.hash_pandas_object(right[columns], index=False)
            overlap = right_hash.isin(set(left_hash))
            n_rows = int(overlap.sum())
            if n_rows:
                report.leaks.append(
                    Leak(
                        column="<rows>",
                        kind="duplicate_rows",
                        detail=f"{n_rows} test row(s) are byte-identical to a training row",
                        n_rows=n_rows,
                        example={"test_index": list(right.index[overlap][:_MAX_EXAMPLES])},
                    )
                )

    if chronological:
        if time is None:
            raise ValueError("chronological=True needs a time column")
        for name, frame in (("train", left), ("test", right)):
            if time not in frame.columns:
                raise KeyError(f"time column {time!r} is not in {name}")
        report.checks += 1
        if not left.empty and not right.empty:
            boundary = left[time].max()
            early = right[time] <= boundary
            n_rows = int(early.sum())
            if n_rows:
                report.leaks.append(
                    Leak(
                        column=time,
                        kind="time_overlap",
                        detail=(
                            f"{n_rows} test row(s) at or before the last training "
                            f"timestamp ({boundary!r})"
                        ),
                        n_rows=n_rows,
                        example={"earliest_test": right[time].min()},
                    )
                )
    return report
