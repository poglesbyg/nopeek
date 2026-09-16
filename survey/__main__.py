"""``python -m survey`` -- run every target and write the report.

WARNING: targets execute third-party code fetched from the internet. Run this in
a container, not on your laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import summarise, write
from .runner import run_target
from .spec import load_all

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="survey", description=__doc__)
    parser.add_argument(
        "--targets", default=str(HERE / "targets"), help="directory of target TOML files"
    )
    parser.add_argument("--out", default="survey-results", help="output directory")
    parser.add_argument("--only", nargs="*", help="run only these target names")
    args = parser.parse_args(argv)

    targets = load_all(args.targets)
    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t.name in wanted]
        if not targets:
            parser.error(f"no target matched {sorted(wanted)}")

    results = []
    for target in targets:
        print(f"-> {target.name}", flush=True)
        result = run_target(target)
        flag = (
            "" if result.matched_expectation is not False else f"  [EXPECTED {target.expect}]"
        )
        print(f"   {result.status} ({result.seconds:.1f}s){flag}", flush=True)
        results.append(result)

    paths = write(results, args.out)
    stats = summarise(results)
    print()
    if stats["ran"]:
        print(
            f"{stats['leaking']}/{stats['ran']} pipelines leak "
            f"({stats['leak_rate']:.0%}); {stats['did_not_run']} did not run"
        )
    print(f"wrote {paths['markdown']}")

    # A calibration target coming back wrong means the harness is broken, and a
    # broken harness reports a reassuring leak rate of zero.
    broken = [r for r in results if r.matched_expectation is False]
    if broken:
        print("\nCALIBRATION FAILED: " + ", ".join(r.name for r in broken))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
