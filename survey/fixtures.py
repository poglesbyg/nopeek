"""Pipelines and data makers that targets can point at.

Two of these are calibration targets: one known-clean, one known-leaky, both
declaring what they expect. A survey whose harness is silently broken reports a
leak rate of zero, which looks like good news. These two rows are how you tell
the difference between "nothing leaks" and "nothing ran".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = "entity"
TIME = "t"


def panel(n_entities: int = 8, n_steps: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        value = rng.normal(100.0, 12.0, n_steps)
        value[rng.random(n_steps) < 0.25] = np.nan
        # Always open with a two-step gap: the entity is enrolled before its
        # first reading. One step is not enough. The earliest cut point falls at
        # t=1, so a hole only at t=0 gets filled from data already available and
        # leaks nothing -- correctly. Pushing the first observation to t=2 puts
        # it strictly after that cut, which is what makes .bfill() reach across
        # it every run rather than whenever the seed cooperates.
        value[:2] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    GROUP: f"e{i:03d}",
                    TIME: np.arange(n_steps),
                    "value": value,
                    "other": rng.normal(0.0, 1.0, n_steps),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(GROUP)["value"]
    return df[[GROUP, TIME]].assign(
        carried=grouped.transform(lambda s: s.ffill()),
        mean_5=grouped.transform(lambda s: s.rolling(5, min_periods=1).mean()),
        seen=grouped.transform(lambda s: s.notna().cumsum()),
    )


def leaky_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(GROUP)["value"]
    return df[[GROUP, TIME]].assign(
        carried=grouped.transform(lambda s: s.ffill()),
        filled=grouped.transform(lambda s: s.ffill().bfill()),
        scaled=(df["value"] - df["value"].mean()) / df["value"].std(),
    )


def sepsis_stays() -> pd.DataFrame:
    """Toy frame shaped like the PhysioNet 2019 cohort, for the sepsis target."""
    from sepsis.config import CHANNELS

    rng = np.random.default_rng(0)
    n_hours = 36
    frames = []
    for i in range(4):
        row: dict[str, object] = {}
        for position, channel in enumerate(CHANNELS):
            values = rng.normal(50, 15, n_hours).astype("float32")
            values[rng.random(n_hours) < (0.15 if position < 8 else 0.92)] = np.nan
            row[channel] = values
        for static in ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime"):
            row[static] = np.full(n_hours, rng.uniform(0, 80), dtype="float32")
        row["ICULOS"] = np.arange(1, n_hours + 1)
        row["SepsisLabel"] = np.zeros(n_hours, dtype=np.int8)
        frame = pd.DataFrame(row)
        frame["patient_id"] = f"p{i:05d}"
        frame["hospital"] = "A"
        frame["hour"] = np.arange(n_hours)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
