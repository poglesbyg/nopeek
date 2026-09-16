# nopeek

**Verify that a data pipeline never reads the future.**

Temporal leakage is the bug that does not look like a bug. The tests pass, the
code reviews fine, the offline score goes *up*, and the model quietly fails the
moment it meets data arriving in real time. It is the reason a backtest beats the
fund, a clinical early-warning model beats the clinician on paper and not at the
bedside, and a churn model's AUC drops eight points in production.

nopeek checks for it the only way that generalises: it hides information the
pipeline should not have, runs the real code again, and demands the same answer.

```python
def features(df):
    out = df[["account", "day"]].copy()
    grouped = df.groupby("account")["balance"]
    out["balance_filled"] = grouped.transform(lambda s: s.ffill().bfill())
    out["balance_z"] = (df["balance"] - df["balance"].mean()) / df["balance"].std()
    out["balance_mean_3d"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
    return out

import nopeek
print(nopeek.verify(features, data, time="day", group="account"))
```

```
nopeek: 2 of 3 column(s) leak -- 12 finding(s) over 10 check(s)
  - balance_filled  lookahead  value changed when the future was hidden  via poison+truncate  1 cut(s)  up to 4 row(s)  reach<=1  max|delta|=9.991e+05
  - balance_z  lookahead  value changed when the future was hidden  via poison+truncate  5 cut(s)  up to 19 row(s)  reach<=9  max|delta|=5.743
```

Two of those three features leak. The `.bfill()` reaches one step forward to fill
an opening gap; the z-score is standardised against a mean that includes every
future row, so it contaminates everything back to day 0. The trailing window is
fine, and is not blamed. That example is [`examples/quickstart.py`](examples/quickstart.py) —
the output above is what it prints.

## Install

```bash
pip install nopeek
```

Requires Python 3.10+, numpy and pandas. Nothing else.

## The three checks

```python
import nopeek

# 1. Point in time: does a feature at t depend on anything after t?
nopeek.verify(build_features, data, time="hour", group="patient_id")

# 2. Isolation: does one entity's data change another's features?
nopeek.verify_isolation(build_features, data, group="patient_id", time="hour")

# 3. Splits: is the test set actually held out?
nopeek.verify_split(train, test, group="patient_id")
```

Each returns a `Report`, which is falsy when it found something, carries the
findings in `.leaks`, and prints one line per offending column. `report.details()`
gives the long form, cut by cut. `report.raise_for_status()` turns it into an
exception.

**`verify`** hides everything after a cut point and requires the rows at or before
it to come back bit-identical, for a spread of cut points across the series. Each
finding names the column, how many rows it contaminated, and how far back the
contamination reached — a centred window of width 5 reaches one row back, a
whole-dataset statistic reaches the beginning.

**`verify_isolation`** rebuilds one entity's rows alone and requires them to match
what was produced alongside everyone else. This catches the rolling window that
forgot to group, which point-in-time checking cannot see: the window only reads
the past, just somebody else's past. It also reorders the entities and re-runs, to
catch output that depends on the order rows happen to arrive in.

**`verify_split`** looks for entities appearing on both sides, byte-identical rows
in both, and — when you ask for a chronological split — test rows dated at or
before the last training row.

## Why differential rather than static

The alternative approach is to lint the source for suspicious patterns: `bfill`,
`shift(-1)`, `center=True`. That finds the easy half. It cannot see:

| Leak | Static linter | nopeek |
|---|---|---|
| `df.bfill()` | ✅ | ✅ |
| `StandardScaler` fitted on the full frame | ❌ | ✅ |
| target encoding fitted before the split | ❌ | ✅ |
| `groupby().transform()` over the whole series | ❌ | ✅ |
| a Numba or Cython kernel | ❌ | ✅ |
| a third-party feature library | ❌ | ✅ |
| leakage through a resample or a join | ❌ | ✅ |

nopeek runs your actual code, so it does not need to recognise the construct that
leaked. The cost is that it needs data and a callable rather than a source file,
and that it can only find leaks the data exercises: a `bfill` on a column with no
gaps is a no-op and is correctly reported as clean.

## Truncation and poisoning

Two ways to hide the future, which fail on different bugs, and both run by default:

- **Truncation** deletes the future. It catches anything reading future values,
  and anything depending on how many rows exist.
- **Poisoning** keeps every row and destroys the values in them. It catches value
  leaks while holding the frame's shape constant, so a pipeline that legitimately
  needs a fixed-length input still runs.

A pipeline that passes poisoning and fails truncation is reading the *extent* of
the series rather than its contents — normalising by total length, say. The report
tells you which, because the two have different fixes.

Columns that are genuinely static and known up front — age, site, sex — can be
held back from poisoning with `preserve=[...]`, and a label that is allowed to
depend on the future belongs in `ignore=[...]`.

## In your test suite

Installing nopeek registers a pytest fixture, so the check runs in CI rather than
once by hand:

```python
def test_features_are_point_in_time(nopeek, stays):
    nopeek.point_in_time(build_features, stays, time="hour", group="patient_id")

def test_features_do_not_cross_admissions(nopeek, stays):
    nopeek.isolated(build_features, stays, group="patient_id", time="hour")
```

## The survey

[`survey/`](survey/) is a reproducible harness for one question: how often do
real pipelines leak? Each pipeline surveyed is a checked-in TOML file naming the
source, the callable, the data maker and the column roles, so a result can be
disputed with a pull request rather than an argument.

```bash
python -m survey            # writes REPORT.md, results.csv, results.json
```

Two targets are calibration rows — one must come back clean, one must come back
leaking — and the run exits non-zero if either is wrong. A harness that silently
fails to run anything reports that nothing leaks, which reads like good news.

## What this does not claim

- **It proves timing, not causality.** A clean report says the code *could* have
  produced those rows at that moment. It says nothing about whether the features
  are sensible, whether the target is defined correctly, or whether the thing you
  are predicting is predictable.
- **It only sees leaks the data exercises.** No gaps, no backfill leak. Run it on
  data with the missingness your real data has.
- **A crash is not a pass.** If poisoning makes your pipeline raise, nopeek says
  so in `report.notes` and does not count the check. Read the notes.
- **Dtypes it cannot poison are reported, not skipped silently** — the note names
  the column, so you know coverage is reduced.
- **It is single-frame.** Leakage across a join with a separate table, or through
  a feature store, is out of scope for now.

## Status

Early. The API above is what exists and is tested; expect it to grow rather than
change. Polars support, a CLI, and scikit-learn `Pipeline` adapters are the next
things.

Contributions and bug reports are welcome — especially a leak it *missed*, which
is the most useful bug report this project can get.

MIT licensed.
