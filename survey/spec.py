"""Target specifications: one TOML file per pipeline to survey.

A survey of "how often do real pipelines leak?" is only worth reporting if the
method is reproducible by someone who doubts the answer. That means every target
is a checked-in file naming the exact source, the exact callable, and the exact
column roles -- not a screenshot of a notebook. Adding a target is a pull request.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the oldest-supported leg of CI
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "the survey harness reads TOML, which needs Python 3.11+ for tomllib "
            "or `uv pip install tomli` on 3.10"
        ) from exc

KINDS = ("none", "local", "git", "pypi")


@dataclass(frozen=True)
class Source:
    kind: str
    path: str | None = None
    url: str | None = None
    revision: str | None = None
    package: str | None = None
    subdir: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"source.kind must be one of {KINDS}, got {self.kind!r}")
        # "none" means the harness itself already provides the code, as the
        # calibration targets do; there is nothing to fetch.
        required = {"local": "path", "git": "url", "pypi": "package"}.get(self.kind)
        if required is not None and getattr(self, required) is None:
            raise ValueError(f"source.kind={self.kind!r} needs source.{required}")


@dataclass(frozen=True)
class Roles:
    time: str
    group: str | None = None
    ignore: tuple[str, ...] = ()
    preserve: tuple[str, ...] = ()
    cuts: int = 5


@dataclass(frozen=True)
class Target:
    name: str
    pipeline: str
    data: str
    roles: Roles
    source: Source
    description: str = ""
    url: str = ""
    timeout_seconds: int = 600
    expect: str | None = None
    base_dir: Path | None = None
    """Directory of the TOML file, which relative local paths resolve against."""

    @property
    def slug(self) -> str:
        return self.name.replace("/", "-").replace(" ", "-").lower()


def load(path: str | Path) -> Target:
    """Read one target TOML file."""
    file = Path(path)
    raw = tomllib.loads(file.read_text(encoding="utf-8"))
    return from_dict(raw, origin=str(file), base_dir=file.resolve().parent)


def from_dict(
    raw: dict[str, Any], *, origin: str = "<dict>", base_dir: Path | None = None
) -> Target:
    try:
        entry = raw["entry"]
        roles_raw = dict(raw["roles"])
        source_raw = dict(raw["source"])
    except KeyError as exc:  # pragma: no cover - message is the point
        raise ValueError(f"{origin}: missing section {exc}") from exc

    for field_name in ("ignore", "preserve"):
        if field_name in roles_raw:
            roles_raw[field_name] = tuple(roles_raw[field_name])

    limits = raw.get("limits", {})
    expect = raw.get("expect")
    if expect not in (None, "clean", "leaks"):
        raise ValueError(f"{origin}: expect must be 'clean' or 'leaks', got {expect!r}")

    return Target(
        name=raw["name"],
        description=raw.get("description", ""),
        url=raw.get("url", ""),
        pipeline=entry["pipeline"],
        data=entry["data"],
        roles=Roles(**roles_raw),
        source=Source(**source_raw),
        timeout_seconds=int(limits.get("timeout_seconds", 600)),
        expect=expect,
        base_dir=base_dir,
    )


def load_all(directory: str | Path) -> list[Target]:
    """Every ``*.toml`` in a directory, sorted by name."""
    found = sorted(Path(directory).glob("*.toml"))
    if not found:
        raise FileNotFoundError(f"no target files in {directory}")
    return sorted((load(p) for p in found), key=lambda t: t.name)


@dataclass
class Result:
    """What running one target produced."""

    name: str
    status: str  # clean | leaks | error | timeout
    columns: int = 0
    leaking: int = 0
    leaking_columns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    message: str = ""
    seconds: float = 0.0
    url: str = ""
    expect: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def matched_expectation(self) -> bool | None:
        if self.expect is None or self.status in ("error", "timeout"):
            return None
        return self.status == self.expect

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "columns": self.columns,
            "leaking": self.leaking,
            "leaking_columns": list(self.leaking_columns),
            "notes": list(self.notes),
            "message": self.message,
            "seconds": round(self.seconds, 2),
            "url": self.url,
            "expect": self.expect,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Result:
        return cls(
            name=raw["name"],
            status=raw["status"],
            columns=raw.get("columns", 0),
            leaking=raw.get("leaking", 0),
            leaking_columns=tuple(raw.get("leaking_columns", ())),
            notes=tuple(raw.get("notes", ())),
            message=raw.get("message", ""),
            seconds=raw.get("seconds", 0.0),
            url=raw.get("url", ""),
            expect=raw.get("expect"),
        )
