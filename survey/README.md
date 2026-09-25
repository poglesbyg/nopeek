# The survey

Tooling for one question: **how often do real time-series pipelines read the
future?**

A claim like "most public pipelines leak" is only worth making if someone who
disbelieves it can re-run it. So every pipeline surveyed is a checked-in TOML
file naming the exact source, the exact callable, the exact data maker and the
exact column roles. Adding one is a pull request, and disputing a result means
editing a file rather than arguing about a screenshot.

```bash
python -m survey                      # every target
python -m survey --only calibration/clean
python -m survey --out results/
```

Writes `REPORT.md`, `results.csv` and `results.json`.

## Safety

Targets execute third-party code fetched from the internet. **Run this in a
container or a throwaway VM.** The worker subprocess exists so that a crashing
pipeline costs one row of the table; it is not a sandbox and does not pretend to
be one.

## Categories

Every target says which claim it tests, because mixing them into one number
would be dishonest in both directions:

| category | the question |
|---|---|
| `library` | does this library's transformer do what it says? |
| `project` | does this real project's pipeline leak? |
| `usage` | does this everyday pattern leak? (the library is behaving correctly) |
| `calibration` | does the harness still work? |

**The headline rate covers `library` and `project` only.** A usage target is
built to leak, so counting it as evidence that pipelines leak is circular; a
calibration target says nothing except whether the harness is alive. Both are
still reported, in their own sections, outside the number.

The distinction matters most for the usage targets, because each of them uses
a perfectly correct library. `StandardScaler` is not broken. Fitting it on the
whole frame before splitting is, and the call looks identical either way --
which is exactly why a linter cannot see it and re-running the code can.

## Adding a target

```toml
name = "org/project"
url = "https://github.com/org/project"
expect = "clean"          # optional: "clean" or "leaks", for calibration rows

[source]
kind = "git"              # local | git | pypi
url = "https://github.com/org/project"
subdir = "src"            # added to PYTHONPATH

[entry]
pipeline = "project.features:build"      # module:callable, frame -> frame
data = "survey.fixtures:panel"           # module:callable, () -> frame

[roles]
time = "timestamp"
group = "entity_id"
ignore = ["target"]       # columns allowed to depend on the future
preserve = ["age"]        # entity-static columns, exempt from poisoning
```

The hard part of a target is never the TOML — it is finding a sample frame the
pipeline accepts. Prefer a small synthetic frame with the same columns, dtypes
and missingness as the real data over a downloaded dataset: it is faster, it is
redistributable, and leaks do not need volume to show up.

## Calibration

Two targets are deliberate: `calibration/clean` must come back clean and
`calibration/leaky` must come back leaking. If either is wrong, the harness is
broken, and `python -m survey` exits non-zero rather than reporting a leak rate.

This matters more than it looks. A broken harness — wrong import, empty frame,
silently swallowed exception — reports that nothing leaks, which reads as good
news. The calibration rows are how "nothing leaks" is told apart from "nothing
ran".

## Reporting honestly

- Targets that fail to run are excluded from the denominator and listed by name
  with their error. A failed clone is not evidence either way.
- A leak found is a leak *in the configuration surveyed*: the sample frame, the
  entry point, the roles declared. Roles are a judgement call — a column marked
  `ignore` is a claim that it may legitimately see the future.
- No leak found is not proof of correctness. It means nothing was found with the
  data given, and a pipeline with no gaps in it cannot demonstrate a backfill bug.
- Report the rate over pipelines, not over columns. One pipeline with forty
  leaking features is one leaking pipeline.

Anyone whose project appears here should be able to read the row, disagree, and
send a PR that changes it.
