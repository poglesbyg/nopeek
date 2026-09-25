"""Turn results into the table that goes in the write-up.

The headline number is a leak *rate*: of the pipelines that ran, how many read
the future. Targets that failed to run are reported separately and excluded from
the denominator, because counting a failed clone as "clean" would flatter the
ecosystem and counting it as "leaks" would slander it.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from .spec import COUNTED, Result

STATUS_MARK = {
    "clean": "clean",
    "leaks": "**LEAKS**",
    "error": "did not run",
    "timeout": "timed out",
}


def summarise(results: Sequence[Result]) -> dict[str, float | int]:
    """Counts over the categories that answer "does real code leak?".

    Usage targets are excluded because they are demonstrations, built to leak;
    counting them as evidence that pipelines leak would be circular. Calibration
    targets are excluded because they say nothing except whether the harness
    still works. Both are still reported, just not in this number.
    """
    counted = [r for r in results if r.category in COUNTED]
    ran = [r for r in counted if r.status in ("clean", "leaks")]
    leaking = [r for r in ran if r.status == "leaks"]
    return {
        "targets": len(results),
        "counted": len(counted),
        "ran": len(ran),
        "did_not_run": len(counted) - len(ran),
        "leaking": len(leaking),
        "leak_rate": round(len(leaking) / len(ran), 4) if ran else 0.0,
        "columns_checked": sum(r.columns for r in ran),
        "columns_leaking": sum(r.leaking for r in ran),
    }


def by_category(results: Sequence[Result]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        row = out.setdefault(r.category, {"clean": 0, "leaks": 0, "did_not_run": 0})
        row["did_not_run" if r.status in ("error", "timeout") else r.status] += 1
    return out


def to_markdown(results: Sequence[Result]) -> str:
    stats = summarise(results)
    lines = ["# nopeek survey", ""]
    if stats["ran"]:
        lines += [
            f"**{stats['leaking']} of {stats['ran']} pipelines that ran read the "
            f"future ({stats['leak_rate']:.0%}).** "
            f"{stats['columns_leaking']} of {stats['columns_checked']} features "
            "affected.",
            "",
            "That rate covers libraries and real projects only. Usage targets are "
            "demonstrations, built to leak, so counting them would be circular; "
            "calibration targets only say whether the harness still works. Both "
            "are listed below, outside the number.",
            "",
        ]
    if stats["did_not_run"]:
        lines += [
            f"{stats['did_not_run']} counted target(s) could not be run and are "
            "excluded from the rate; they are listed with their errors.",
            "",
        ]

    counts = by_category(results)
    lines += ["| category | clean | leaks | did not run |", "|---|---|---|---|"]
    for category in sorted(counts):
        row = counts[category]
        lines.append(
            f"| {category} | {row['clean']} | {row['leaks']} | {row['did_not_run']} |"
        )
    lines.append("")

    for category in sorted(counts):
        members = [r for r in results if r.category == category]
        lines += [
            f"## {category}",
            "",
            "| target | verdict | features | leaking | seconds |",
            "|---|---|---|---|---|",
        ]
        for result in sorted(members, key=lambda r: (r.status != "leaks", r.name)):
            name = f"[{result.name}]({result.url})" if result.url else result.name
            columns = str(result.columns) if result.status in ("clean", "leaks") else "-"
            leaking = (
                ", ".join(result.leaking_columns[:4]) or "-"
                if result.status == "leaks"
                else "-"
            )
            lines.append(
                f"| {name} | {STATUS_MARK.get(result.status, result.status)} "
                f"| {columns} | {leaking} | {result.seconds:.1f} |"
            )
        lines.append("")

    failures = [r for r in results if r.status in ("error", "timeout")]
    if failures:
        lines += ["## Targets that did not run", ""]
        lines += [f"- **{r.name}**: {r.message}" for r in failures]
        lines.append("")

    mismatched = [r for r in results if r.matched_expectation is False]
    if mismatched:
        lines += ["## Expectation mismatches", ""]
        lines += [f"- **{r.name}**: expected {r.expect}, got {r.status}" for r in mismatched]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write(results: Sequence[Result], directory: str | Path) -> dict[str, Path]:
    """Write results.json, results.csv and REPORT.md; return the paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summarise(results),
        "by_category": by_category(results),
        "results": [r.to_dict() for r in results],
    }
    paths = {
        "json": out / "results.json",
        "csv": out / "results.csv",
        "markdown": out / "REPORT.md",
    }
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "status", "columns", "leaking", "leaking_columns", "seconds"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.status,
                    result.columns,
                    result.leaking,
                    ";".join(result.leaking_columns),
                    round(result.seconds, 2),
                ]
            )
    paths["markdown"].write_text(to_markdown(results), encoding="utf-8")
    return paths
