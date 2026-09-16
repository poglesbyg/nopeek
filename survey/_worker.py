"""Runs one target, in its own process, and prints a JSON result.

Separate process because the pipelines being surveyed are third-party code: they
import what they like, mutate global state, and are entitled to crash. A crash
here costs one row of the table.

Invoked as ``python -m survey._worker <spec.json>``; not meant to be called by hand.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from typing import Any

import nopeek


def resolve(reference: str) -> Any:
    """Turn ``package.module:attribute`` into the attribute."""
    if ":" not in reference:
        raise ValueError(f"expected 'module:attribute', got {reference!r}")
    module_name, _, attribute = reference.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"{module_name} has no attribute {attribute!r}") from exc


def run(spec: dict[str, Any]) -> dict[str, Any]:
    roles = spec["roles"]
    started = time.perf_counter()

    pipeline = resolve(spec["pipeline"])
    data = resolve(spec["data"])()

    report = nopeek.verify(
        pipeline,
        data,
        time=roles["time"],
        group=roles.get("group"),
        cuts=roles.get("cuts", 5),
        ignore=tuple(roles.get("ignore", ())),
        preserve=tuple(roles.get("preserve", ())),
    )
    return {
        "name": spec["name"],
        "status": "clean" if report.ok else "leaks",
        "columns": len(report.columns_checked),
        "leaking": len(report.leaking_columns),
        "leaking_columns": list(report.leaking_columns),
        "notes": list(report.notes),
        "seconds": time.perf_counter() - started,
        "url": spec.get("url", ""),
        "expect": spec.get("expect"),
        "summary": str(report),
    }


def main() -> int:
    spec = json.loads(sys.argv[1])
    try:
        payload = run(spec)
    except Exception as exc:  # noqa: BLE001 - any failure becomes a row, not a stack trace
        payload = {
            "name": spec.get("name", "?"),
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "url": spec.get("url", ""),
            "expect": spec.get("expect"),
        }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
