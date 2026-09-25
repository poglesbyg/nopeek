"""Adapters for third-party pipelines.

Each function here presents somebody else's library to nopeek as a plain
``DataFrame -> DataFrame`` callable. Two rules keep the survey honest:

* **Use the library the way its own documentation does.** A survey that
  misconfigures a tool and then reports it as leaking is worthless, and worse
  than worthless if the tool's authors read it.
* **Say which claim is under test.** A *library* target asks whether a
  transformer designed to be point-in-time actually is. A *usage* target asks
  whether a pattern people write every day leaks -- the library is behaving
  exactly as documented, and the finding is about the code around it. Reporting
  those two as one number would be dishonest in both directions.

Imports are inside the functions because the target's own code is only on the
path inside the worker process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME = "t"
GROUP = "entity"


# --- data ---------------------------------------------------------------------


def series(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A single time series. What a forecasting library expects to be handed."""
    rng = np.random.default_rng(seed)
    value = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame({TIME: np.arange(n), "value": value, "other": rng.normal(0, 1, n)})


def panel(n_entities: int = 5, n_steps: int = 30, seed: int = 0) -> pd.DataFrame:
    """Several entities stacked, sorted by entity then time."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        frames.append(
            pd.DataFrame(
                {
                    GROUP: f"e{i:02d}",
                    TIME: np.arange(n_steps),
                    "value": np.cumsum(rng.normal(0, 1, n_steps)) + 100,
                    "bucket": rng.choice(list("abc"), n_steps),
                    "label": rng.integers(0, 2, n_steps),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- feature-engine, used as documented ---------------------------------------


def feature_engine_lags(df: pd.DataFrame) -> pd.DataFrame:
    from feature_engine.timeseries.forecasting import LagFeatures

    tr = LagFeatures(variables=["value"], periods=[1, 3], missing_values="ignore")
    out = tr.fit_transform(df[["value"]])
    return df[[TIME]].join(out.drop(columns=["value"]))


def feature_engine_window(df: pd.DataFrame) -> pd.DataFrame:
    from feature_engine.timeseries.forecasting import WindowFeatures

    tr = WindowFeatures(
        variables=["value"],
        window=5,
        functions=["mean", "std"],
        missing_values="ignore",
    )
    out = tr.fit_transform(df[["value"]])
    return df[[TIME]].join(out.drop(columns=["value"]))


def feature_engine_expanding(df: pd.DataFrame) -> pd.DataFrame:
    from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

    tr = ExpandingWindowFeatures(
        variables=["value"], functions=["mean", "max"], missing_values="ignore"
    )
    out = tr.fit_transform(df[["value"]])
    return df[[TIME]].join(out.drop(columns=["value"]))


# --- skforecast, used as documented -------------------------------------------


def skforecast_lag_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The training matrix a recursive forecaster builds from a series."""
    from skforecast.recursive import ForecasterRecursive
    from sklearn.linear_model import LinearRegression

    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    X, _y = forecaster.create_train_X_y(y=pd.Series(df["value"].to_numpy()))
    built = pd.DataFrame(X).reset_index(drop=True)
    built[TIME] = df[TIME].to_numpy()[-len(built) :]
    return built


# --- usage patterns: the library is fine, the code around it is the question ---


def target_encode_before_splitting(df: pd.DataFrame) -> pd.DataFrame:
    """Target encoding fitted on the whole frame, label and all.

    feature-engine's MeanEncoder does exactly what it says. The mistake is
    fitting it once over every row that will ever exist, which is how it is
    written in a great many notebooks.
    """
    from feature_engine.encoding import MeanEncoder

    tr = MeanEncoder(variables=["bucket"], unseen="encode")
    encoded = tr.fit_transform(df[["bucket"]], df["label"])
    return df[[GROUP, TIME]].assign(bucket_te=encoded["bucket"].to_numpy())


def scale_on_the_whole_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A scaler fitted over every row, including rows from the future.

    Nothing about StandardScaler is wrong. Calling .fit on the full frame before
    splitting is the single most common way a pipeline reads the future, and no
    linter can see it because the call looks identical either way.
    """
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(df[["value"]])
    return df[[GROUP, TIME]].assign(value_z=scaled[:, 0])


def rolling_without_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """A trailing window over a stacked panel, with the grouping left out."""
    return df[[GROUP, TIME]].assign(value_mean5=df["value"].rolling(5, min_periods=1).mean())
