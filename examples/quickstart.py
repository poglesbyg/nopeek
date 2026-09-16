"""A feature builder with two leaks in it, and what nopeek says about them.

Run with ``python examples/quickstart.py``. The output is pasted into the README,
so if this changes, that changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import nopeek


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(3):
        balance = rng.normal(1000.0, 80.0, 12)
        balance[rng.random(12) < 0.3] = np.nan
        balance[0] = np.nan  # the account opens before the first statement
        frames.append(
            pd.DataFrame({"account": f"acct-{i}", "day": np.arange(12), "balance": balance})
        )
    return pd.concat(frames, ignore_index=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["account", "day"]].copy()
    grouped = df.groupby("account")["balance"]

    # Leak 1: .bfill() reaches forward to fill the opening gap.
    out["balance_filled"] = grouped.transform(lambda s: s.ffill().bfill())

    # Leak 2: standardised against statistics of the whole dataset, future included.
    out["balance_z"] = (df["balance"] - df["balance"].mean()) / df["balance"].std()

    # Correct: a trailing window, within the account.
    out["balance_mean_3d"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
    return out


if __name__ == "__main__":
    report = nopeek.verify(features, make_data(), time="day", group="account")
    print(report)
    print()
    print("first three findings in full:")
    print("\n".join(report.details().splitlines()[:3]))
