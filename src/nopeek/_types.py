"""Result types: what a check found, and how to read it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal[
    "lookahead",
    "missing_rows",
    "cross_group",
    "group_overlap",
    "time_overlap",
    "duplicate_rows",
]
Strategy = Literal["truncate", "poison"]


def plain(value: Any) -> Any:
    """Unwrap a numpy scalar for display, so a cut prints as 10 rather than np.int64(10)."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (AttributeError, ValueError):  # pragma: no cover - exotic scalars
            return value
    return value


class LeakError(AssertionError):
    """Raised by :meth:`Report.raise_for_status` when a check found a leak.

    Subclasses ``AssertionError`` so it reads naturally in a test suite.
    """


@dataclass(frozen=True)
class Leak:
    """One way a pipeline was observed to depend on information it should not have.

    ``nearest_offset`` and ``farthest_offset`` are the distance, in units of the
    time column, from the cut point back to the closest and furthest contaminated
    row. They describe the *reach* of the leak: a centred 24-step window leaks
    about 12 steps back, an expanding whole-series statistic leaks to the start.
    """

    column: str
    kind: Kind
    detail: str
    strategy: Strategy | None = None
    cut: Any = None
    n_rows: int = 0
    nearest_offset: Any = None
    farthest_offset: Any = None
    max_abs_diff: float | None = None
    example: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        bits = [f"{self.column}: {self.detail}"]
        if self.cut is not None:
            bits.append(f"cut={plain(self.cut)}")
        if self.strategy is not None:
            bits.append(f"via {self.strategy}")
        if self.n_rows:
            bits.append(f"{self.n_rows} row(s)")
        if self.farthest_offset is not None:
            bits.append(f"reach={plain(self.nearest_offset)}..{plain(self.farthest_offset)}")
        if self.max_abs_diff is not None:
            bits.append(f"max|delta|={self.max_abs_diff:.6g}")
        return "  ".join(bits)


@dataclass
class Report:
    """The outcome of a check. Falsy when a leak was found, so ``assert report`` works."""

    leaks: list[Leak] = field(default_factory=list)
    checks: int = 0
    columns_checked: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.leaks

    def __bool__(self) -> bool:
        return self.ok

    @property
    def leaking_columns(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for leak in self.leaks:
            seen.setdefault(leak.column, None)
        return tuple(seen)

    def raise_for_status(self) -> None:
        """Raise :class:`LeakError` if anything leaked. No-op otherwise."""
        if self.leaks:
            raise LeakError(str(self))

    def __str__(self) -> str:
        """One line per leaking column, not per cut point.

        A whole-dataset statistic leaks at every cut under both strategies, which
        is ten findings about one mistake. The findings stay in :attr:`leaks` for
        code to read; what a person needs first is the list of columns to fix.
        """
        if self.ok:
            lines = [
                f"nopeek: clean -- {self.checks} check(s) over "
                f"{len(self.columns_checked)} column(s)"
            ]
            return "\n".join(lines + [f"  note: {n}" for n in self.notes])

        lines = [
            f"nopeek: {len(self.leaking_columns)} of {len(self.columns_checked)} "
            f"column(s) leak -- {len(self.leaks)} finding(s) over {self.checks} check(s)"
        ]
        for column in self.leaking_columns:
            lines.append(f"  - {self._summarise(column)}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def _summarise(self, column: str) -> str:
        found = [leak for leak in self.leaks if leak.column == column]
        kinds = sorted({leak.kind for leak in found})
        strategies = sorted({leak.strategy for leak in found if leak.strategy})
        cuts = {leak.cut for leak in found if leak.cut is not None}
        rows = max((leak.n_rows for leak in found), default=0)
        deltas = [leak.max_abs_diff for leak in found if leak.max_abs_diff is not None]
        reaches = [leak.farthest_offset for leak in found if leak.farthest_offset is not None]

        bits = [f"{column}  {'/'.join(kinds)}"]
        bits.append(found[0].detail)
        if strategies:
            bits.append("via " + "+".join(strategies))
        if cuts:
            bits.append(f"{len(cuts)} cut(s)")
        if rows:
            bits.append(f"up to {rows} row(s)")
        if reaches:
            bits.append(f"reach<={plain(max(reaches))}")
        if deltas:
            bits.append(f"max|delta|={max(deltas):.4g}")
        return "  ".join(bits)

    def details(self) -> str:
        """Every finding on its own line, cut by cut. The long form of ``str(report)``."""
        return "\n".join(str(leak) for leak in self.leaks)

    def extend(self, other: Report) -> None:
        self.leaks.extend(other.leaks)
        self.checks += other.checks
        self.notes.extend(other.notes)
        merged = dict.fromkeys(self.columns_checked)
        merged.update(dict.fromkeys(other.columns_checked))
        self.columns_checked = tuple(merged)
