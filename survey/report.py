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

from .spec import Result

STATUS_MARK = {
    "clean": "clean",
    "leaks": "**LEAKS**",
    "error": "did not run",
    "timeout": "timed out",
}


def summarise(results: Sequence[Result]) -> dict[str, float | int]:
    ran = [r for r in results if r.status in ("clean", "leaks")]
    leaking = [r for r in ran if r.status == "leaks"]
    return {
        "targets": len(results),
        "ran": len(ran),
        "did_not_run": len(results) - len(ran),
        "leaking": len(leaking),
        "leak_rate": round(len(leaking) / len(ran), 4) if ran else 0.0,
        "columns_checked": sum(r.columns for r in ran),
        "columns_leaking": sum(r.leaking for r in ran),
    }


def to_markdown(results: Sequence[Result]) -> str:
    stats = summarise(results)
    lines = ["# nopeek survey", ""]
    if stats["ran"]:
        lines += [
            f"**{stats['leaking']} of {stats['ran']} pipelines that ran read the future "
            f"({stats['leak_rate']:.0%}).** "
            f"{stats['columns_leaking']} of {stats['columns_checked']} features affected.",
            "",
        ]
    if stats["did_not_run"]:
        lines += [
            f"{stats['did_not_run']} target(s) could not be run and are excluded "
            "from the rate; they are listed below.",
            "",
        ]

    lines += ["| target | verdict | features | leaking | seconds |", "|---|---|---|---|---|"]
    for result in sorted(results, key=lambda r: (r.status != "leaks", r.name)):
        name = f"[{result.name}]({result.url})" if result.url else result.name
        columns = str(result.columns) if result.status in ("clean", "leaks") else "-"
        leaking = (
            ", ".join(result.leaking_columns[:4]) or "-" if result.status == "leaks" else "-"
        )
        lines.append(
            f"| {name} | {STATUS_MARK.get(result.status, result.status)} | {columns} "
            f"| {leaking} | {result.seconds:.1f} |"
        )

    failures = [r for r in results if r.status in ("error", "timeout")]
    if failures:
        lines += ["", "## Targets that did not run", ""]
        lines += [f"- **{r.name}**: {r.message}" for r in failures]

    mismatched = [r for r in results if r.matched_expectation is False]
    if mismatched:
        lines += ["", "## Expectation mismatches", ""]
        lines += [f"- **{r.name}**: expected {r.expect}, got {r.status}" for r in mismatched]
    return "\n".join(lines) + "\n"


def write(results: Sequence[Result], directory: str | Path) -> dict[str, Path]:
    """Write results.json, results.csv and REPORT.md; return the paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summarise(results),
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
