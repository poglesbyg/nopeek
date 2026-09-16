"""Integration test against a real research pipeline.

Unit tests written alongside a verifier prove it agrees with its author about
toy code. This one runs it against a feature builder written independently, for
a real dataset -- 40k ICU admissions of hourly vitals and labs -- with seven
blocks of carried-forward values, recency counters, rolling windows and derived
clinical scores. That pipeline maintains a hand-written no-lookahead test of its
own; this checks that nopeek reaches the same verdict without being told how.

Skipped unless the ``sepsis`` package is importable, so it is a local and
downstream check rather than a CI dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nopeek

sepsis_features = pytest.importorskip(
    "sepsis.features", reason="needs the sepsis-early-warning package on the path"
)
sepsis_config = pytest.importorskip("sepsis.config")

build_features = sepsis_features.build_features
CHANNELS = sepsis_config.CHANNELS

# Constant for the whole stay and known at admission, so the pipeline is entitled
# to read them from any row.
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]


def toy_stays(n_patients: int = 4, n_hours: int = 36, seed: int = 0) -> pd.DataFrame:
    """Frames shaped like the real cohort: dense vitals, very sparse labs."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_patients):
        row: dict[str, object] = {}
        for position, channel in enumerate(CHANNELS):
            values = rng.normal(50, 15, n_hours).astype("float32")
            values[rng.random(n_hours) < (0.15 if position < 8 else 0.92)] = np.nan
            row[channel] = values
        row["Age"] = np.full(n_hours, rng.uniform(20, 90), dtype="float32")
        row["Gender"] = np.full(n_hours, rng.integers(0, 2), dtype="float32")
        row["Unit1"] = np.full(n_hours, rng.integers(0, 2), dtype="float32")
        row["Unit2"] = np.full(n_hours, rng.integers(0, 2), dtype="float32")
        row["HospAdmTime"] = np.full(n_hours, rng.uniform(-40, 0), dtype="float32")
        row["ICULOS"] = np.arange(1, n_hours + 1)
        label = np.zeros(n_hours, dtype=np.int8)
        if rng.random() < 0.5:
            label[int(rng.integers(n_hours // 3, n_hours)) :] = 1
        row["SepsisLabel"] = label
        frame = pd.DataFrame(row)
        frame["patient_id"] = f"p{i:05d}"
        frame["hospital"] = "A"
        frame["hour"] = np.arange(n_hours)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def stays() -> pd.DataFrame:
    return toy_stays()


def test_real_builder_is_point_in_time(stays):
    report = nopeek.verify(
        build_features,
        stays,
        time="hour",
        group="patient_id",
        ignore=["SepsisLabel"],
        preserve=STATIC,
    )
    assert report.ok, str(report)
    assert len(report.columns_checked) > 50


def test_real_builder_isolates_admissions(stays):
    report = nopeek.verify_isolation(
        build_features,
        stays,
        group="patient_id",
        time="hour",
        ignore=["SepsisLabel"],
    )
    assert report.ok, str(report)


def test_a_planted_leak_in_the_real_builder_is_found(stays):
    """The suite above only means something if this one fails without the fix."""

    def sabotaged(df: pd.DataFrame) -> pd.DataFrame:
        built = build_features(df)
        built["HR_bfill"] = (
            df.sort_values(["patient_id", "hour"], ignore_index=True)
            .groupby("patient_id")["HR"]
            .transform(lambda s: s.bfill())
        )
        return built

    report = nopeek.verify(
        sabotaged,
        stays,
        time="hour",
        group="patient_id",
        ignore=["SepsisLabel"],
        preserve=STATIC,
    )
    assert not report.ok
    assert report.leaking_columns == ("HR_bfill",)
