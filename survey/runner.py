"""Fetch a target's code, run the worker against it, collect a Result.

SAFETY: this executes third-party code downloaded from the internet. Run the
survey inside a container or a throwaway VM, never on a machine you care about.
Nothing here sandboxes anything -- process isolation is for crashes, not malice.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from .spec import Result, Source, Target

ROOT = Path(__file__).resolve().parents[1]


class PreparationError(RuntimeError):
    """The target's code could not be fetched or installed."""


def prepare(source: Source, workdir: Path, base_dir: Path | None = None) -> dict[str, str]:
    """Make the target importable; return environment overrides for the worker."""
    if source.kind == "none":
        return {"PYTHONPATH": str(ROOT)}

    if source.kind == "local":
        assert source.path is not None
        candidate = Path(source.path).expanduser()
        if not candidate.is_absolute() and base_dir is not None:
            candidate = base_dir / candidate
        base = candidate.resolve()
        if not base.exists():
            raise PreparationError(f"local path does not exist: {base}")
        return {"PYTHONPATH": str(base / source.subdir if source.subdir else base)}

    if source.kind == "git":
        assert source.url is not None
        clone = workdir / "clone"
        command = ["git", "clone", "--depth", "1"]
        if source.revision:
            command += ["--branch", source.revision]
        command += [source.url, str(clone)]
        _run(command, "git clone")
        return {"PYTHONPATH": str(clone / source.subdir if source.subdir else clone)}

    assert source.package is not None
    target_dir = workdir / "site"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--target",
            str(target_dir),
            source.package,
        ],
        "pip install",
    )
    return {"PYTHONPATH": str(target_dir)}


def _run(command: list[str], what: str) -> None:
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise PreparationError(f"{what} failed: {' / '.join(tail)}")


def run_target(target: Target, *, keep: bool = False) -> Result:
    """Run one target end to end. Never raises: every failure becomes a Result."""
    started = time.perf_counter()
    spec = {
        "name": target.name,
        "pipeline": target.pipeline,
        "data": target.data,
        "url": target.url,
        "expect": target.expect,
        "roles": {
            "time": target.roles.time,
            "group": target.roles.group,
            "ignore": list(target.roles.ignore),
            "preserve": list(target.roles.preserve),
            "cuts": target.roles.cuts,
        },
    }

    # NOTE: TemporaryDirectory(delete=...) is 3.12+, and the floor here is 3.10.
    workdir = Path(tempfile.mkdtemp(prefix="nopeek-survey-"))
    try:
        try:
            overrides = prepare(target.source, workdir, target.base_dir)
        except PreparationError as exc:
            return Result(
                name=target.name,
                status="error",
                message=str(exc),
                url=target.url,
                expect=target.expect,
                seconds=time.perf_counter() - started,
            )

        env = dict(os.environ)
        # The harness itself must stay importable inside the worker.
        paths = [overrides["PYTHONPATH"], str(ROOT)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)

        try:
            done = subprocess.run(
                [sys.executable, "-m", "survey._worker", json.dumps(spec)],
                capture_output=True,
                text=True,
                env=env,
                timeout=target.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return Result(
                name=target.name,
                status="timeout",
                message=f"exceeded {target.timeout_seconds}s",
                url=target.url,
                expect=target.expect,
                seconds=time.perf_counter() - started,
            )

    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)

    if done.returncode != 0 or not done.stdout.strip():
        tail = (done.stderr or "no output").strip().splitlines()[-3:]
        return Result(
            name=target.name,
            status="error",
            message=" / ".join(tail),
            url=target.url,
            expect=target.expect,
            seconds=time.perf_counter() - started,
        )

    payload = json.loads(done.stdout.strip().splitlines()[-1])
    result = Result.from_dict(payload)
    result.seconds = time.perf_counter() - started
    return result


def run_all(targets: Iterable[Target]) -> list[Result]:
    return [run_target(target) for target in targets]
