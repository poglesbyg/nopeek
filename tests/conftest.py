"""Panel fixtures and a rogues' gallery of pipelines.

Each builder below is either point-in-time correct or wrong in one specific,
named way. The test suite is mostly the assertion that nopeek agrees about which
is which -- a verifier that cannot be shown to catch a real bug is decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

GROUP = "entity"
TIME = "t"


def make_panel(
    n_entities: int = 4, n_steps: int = 12, seed: int = 0, missing: float = 0.25
) -> pd.DataFrame:
    """Entity-by-time panel with a sparse numeric channel, like charted labs."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        x = rng.normal(10.0, 2.0, n_steps)
        x[rng.random(n_steps) < missing] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    GROUP: f"e{i:03d}",
                    TIME: np.arange(n_steps),
                    "x": x,
                    "y": rng.normal(0.0, 1.0, n_steps),
                    "label": rng.integers(0, 2, n_steps),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    return df[[GROUP, TIME]].copy()


# --- correct -----------------------------------------------------------------


def expanding_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Running mean over everything seen so far, within the entity."""
    feature = df.groupby(GROUP)["x"].transform(lambda s: s.expanding().mean())
    return _keys(df).assign(x_mean=feature)


def causal_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Previous value and the gap since it was last observed."""
    grouped = df.groupby(GROUP)["x"]
    return _keys(df).assign(
        x_lag1=grouped.shift(1),
        x_ffill=grouped.transform(lambda s: s.ffill()),
    )


def trailing_window(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking rolling window, correctly grouped."""
    feature = df.groupby(GROUP)["x"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    return _keys(df).assign(x_roll=feature)


# --- broken, one way each ----------------------------------------------------


def backfill(df: pd.DataFrame) -> pd.DataFrame:
    """Fills a gap from the next observation. The classic."""
    feature = df.groupby(GROUP)["x"].transform(lambda s: s.bfill())
    return _keys(df).assign(x_filled=feature)


def whole_series_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardised against statistics of the entire dataset, future included."""
    feature = (df["x"] - df["x"].mean()) / df["x"].std()
    return _keys(df).assign(x_z=feature)


def centred_window(df: pd.DataFrame) -> pd.DataFrame:
    """A window centred on the current step reads half its span from the future."""
    feature = df.groupby(GROUP)["x"].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).mean()
    )
    return _keys(df).assign(x_centred=feature)


def next_value(df: pd.DataFrame) -> pd.DataFrame:
    """Reads one step ahead outright."""
    return _keys(df).assign(x_next=df.groupby(GROUP)["x"].shift(-1))


def drops_final_row(df: pd.DataFrame) -> pd.DataFrame:
    """Needs a following row to exist, so the last step of each entity vanishes."""
    out = _keys(df).assign(x_next=df.groupby(GROUP)["x"].shift(-1))
    return out.dropna(subset=["x_next"]).reset_index(drop=True)


def row_count_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Depends on how long the series is, but not on any future value.

    Poisoning cannot see this one -- the frame keeps its shape -- and truncation
    can. The pair is why nopeek runs both.
    """
    return _keys(df).assign(n_rows=float(len(df)))


def ungrouped_window(df: pd.DataFrame) -> pd.DataFrame:
    """A rolling window that forgot to group, so entities bleed into each other."""
    feature = df["x"].rolling(3, min_periods=1).mean()
    return _keys(df).assign(x_roll=feature)


def positional(df: pd.DataFrame) -> pd.DataFrame:
    """Depends on where the row happens to sit in the frame."""
    return _keys(df).assign(pos=np.arange(len(df), dtype=float))


CORRECT = [expanding_mean, causal_lag, trailing_window]
LEAKY = [backfill, whole_series_zscore, centred_window, next_value]


def make_gappy_panel(n_entities: int = 2, n_steps: int = 8, seed: int = 0) -> pd.DataFrame:
    """A panel with a guaranteed gap at step 1 and a value after it.

    Some leaks are conditional on the data: ``bfill`` on a column with no holes
    is a no-op and leaks nothing, which is correct behaviour and not something a
    verifier should report. Tests that assert a leak *is* found need a frame
    where the leak actually exists.
    """
    frame = make_panel(n_entities=n_entities, n_steps=n_steps, seed=seed, missing=0.0)
    frame.loc[frame[TIME] == 1, "x"] = np.nan
    return frame
