"""Property tests: no false alarms, no silent misses, across panel shapes.

A verifier has exactly two ways to be useless. It can cry leak on correct code,
in which case people switch it off; or it can pass leaking code, in which case it
is worse than nothing. These two properties are the whole contract, so they are
checked over generated panels rather than one hand-picked frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import nopeek
from conftest import GROUP, TIME, make_gappy_panel, make_panel

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

CAUSAL_OPS = {
    "shift1": lambda g: g.shift(1),
    "ffill": lambda g: g.transform(lambda s: s.ffill()),
    "roll3": lambda g: g.transform(lambda s: s.rolling(3, min_periods=1).mean()),
    "expanding_max": lambda g: g.transform(lambda s: s.expanding().max()),
    "cumsum": lambda g: g.transform(lambda s: s.cumsum()),
    "diff": lambda g: g.diff(),
}

LEAKY_OPS = {
    "shift_minus_1": lambda g: g.shift(-1),
    "bfill": lambda g: g.transform(lambda s: s.bfill()),
    "centred_roll": lambda g: g.transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean()
    ),
    "reverse_cumsum": lambda g: g.transform(lambda s: s[::-1].cumsum()[::-1]),
}


def pipeline(names: list[str]) -> object:
    """Compose named per-entity operations into a feature builder."""
    ops = [CAUSAL_OPS.get(name) or LEAKY_OPS[name] for name in names]

    def fn(df: pd.DataFrame) -> pd.DataFrame:
        values = df["x"]
        for op in ops:
            values = op(df.assign(_v=values).groupby(GROUP)["_v"])
        return df[[GROUP, TIME]].assign(feature=np.asarray(values))

    return fn


panels = st.builds(
    make_panel,
    n_entities=st.integers(min_value=1, max_value=4),
    n_steps=st.integers(min_value=5, max_value=20),
    seed=st.integers(min_value=0, max_value=2**16),
    missing=st.floats(min_value=0.0, max_value=0.5),
)


@SETTINGS
@given(
    frame=panels,
    names=st.lists(st.sampled_from(sorted(CAUSAL_OPS)), min_size=1, max_size=3),
)
def test_causal_pipelines_never_raise_an_alarm(frame, names):
    report = nopeek.verify(pipeline(names), frame, time=TIME, group=GROUP)
    assert report.ok, f"false alarm on {names}\n{report}"


gappy_panels = st.builds(
    make_gappy_panel,
    n_entities=st.integers(min_value=1, max_value=4),
    n_steps=st.integers(min_value=5, max_value=20),
    seed=st.integers(min_value=0, max_value=2**16),
)


@SETTINGS
@given(frame=gappy_panels, name=st.sampled_from(sorted(LEAKY_OPS)))
def test_leaky_pipelines_are_always_caught(frame, name):
    """Every operation in LEAKY_OPS reads ahead, and none of them may slip through.

    The panel is the gappy one on purpose: a hole is what makes ``bfill`` reach
    forward at all. On a dense column it is a no-op, and reporting nothing there
    is the right answer, not a miss.
    """
    report = nopeek.verify(pipeline([name]), frame, time=TIME, group=GROUP)
    assert not report.ok, f"missed a leak from {name}"
    assert "feature" in report.leaking_columns


@SETTINGS
@given(
    frame=panels,
    names=st.lists(st.sampled_from(sorted(CAUSAL_OPS)), min_size=1, max_size=2),
)
def test_causal_pipelines_are_entity_isolated(frame, names):
    report = nopeek.verify_isolation(pipeline(names), frame, group=GROUP, time=TIME)
    assert report.ok, f"false alarm on {names}\n{report}"


@SETTINGS
@given(frame=panels, cuts=st.integers(min_value=1, max_value=8))
def test_any_cut_count_is_accepted(frame, cuts):
    report = nopeek.verify(pipeline(["shift1"]), frame, time=TIME, group=GROUP, cuts=cuts)
    assert report.ok
    assert 1 <= report.checks <= 2 * cuts


@SETTINGS
@given(frame=gappy_panels, name=st.sampled_from(sorted(LEAKY_OPS)))
def test_reports_are_deterministic(frame, name):
    fn = pipeline([name])
    first = nopeek.verify(fn, frame, time=TIME, group=GROUP)
    second = nopeek.verify(fn, frame, time=TIME, group=GROUP)
    assert [str(x) for x in first.leaks] == [str(x) for x in second.leaks]


@SETTINGS
@given(frame=panels, at=st.integers(min_value=1, max_value=18))
def test_a_chronological_split_is_clean_and_a_random_one_is_not(frame, at):
    train = frame[frame[TIME] < at]
    test = frame[frame[TIME] >= at]
    if train.empty or test.empty:
        return
    assert nopeek.verify_split(
        train, test, time=TIME, chronological=True, check_duplicates=False
    ).ok
    shuffled = frame.sample(frac=1.0, random_state=0)
    half = len(shuffled) // 2
    if frame[GROUP].nunique() > 1:
        report = nopeek.verify_split(shuffled.iloc[:half], shuffled.iloc[half:], group=GROUP)
        assert not report.ok
