"""The survey harness: specs, running, reporting.

The calibration targets are tested hardest. A harness that silently fails to run
anything reports that nothing leaks, which reads like good news, so the tests
that matter most are the ones asserting it still catches a planted leak.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from survey import fixtures, report, runner, spec

TARGETS = Path(__file__).resolve().parents[1] / "survey" / "targets"


def test_every_shipped_target_parses():
    targets = spec.load_all(TARGETS)
    assert {t.name for t in targets} >= {"calibration/clean", "calibration/leaky"}
    for target in targets:
        assert ":" in target.pipeline
        assert ":" in target.data


def test_source_validation():
    with pytest.raises(ValueError, match=r"source\.kind must be one of"):
        spec.Source(kind="telepathy")
    with pytest.raises(ValueError, match=r"needs source\.url"):
        spec.Source(kind="git")
    with pytest.raises(ValueError, match=r"needs source\.path"):
        spec.Source(kind="local")
    assert spec.Source(kind="none").kind == "none"


def test_expect_validation():
    raw = {
        "name": "x",
        "expect": "probably",
        "entry": {"pipeline": "a:b", "data": "c:d"},
        "roles": {"time": "t"},
        "source": {"kind": "none"},
    }
    with pytest.raises(ValueError, match="expect must be"):
        spec.from_dict(raw)


def test_missing_section_names_itself():
    with pytest.raises(ValueError, match="missing section"):
        spec.from_dict({"name": "x"})


@pytest.mark.parametrize(
    ("name", "expected"), [("calibration/clean", "clean"), ("calibration/leaky", "leaks")]
)
def test_calibration_targets_run_and_agree_with_themselves(name, expected):
    target = next(t for t in spec.load_all(TARGETS) if t.name == name)
    result = runner.run_target(target)
    assert result.status == expected, result.message
    assert result.matched_expectation is True


def test_the_leaky_calibration_target_catches_both_planted_leaks():
    """Both mistakes, every time -- not whichever one the seed happened to expose."""
    target = next(t for t in spec.load_all(TARGETS) if t.name == "calibration/leaky")
    result = runner.run_target(target)
    assert set(result.leaking_columns) == {"filled", "scaled"}
    assert "carried" not in result.leaking_columns


def test_panel_always_opens_with_a_gap():
    """The gap has to outlast the earliest cut, or the planted bfill leaks nothing."""
    frame = fixtures.panel()
    opening = frame[frame[fixtures.TIME] < 2]
    assert opening["value"].isna().all()
    arrived = frame[frame[fixtures.TIME] == 2].groupby(fixtures.GROUP)["value"].count()
    assert (arrived > 0).all()


def test_a_broken_target_becomes_a_row_not_an_exception():
    target = spec.from_dict(
        {
            "name": "broken",
            "entry": {
                "pipeline": "survey.fixtures:does_not_exist",
                "data": "survey.fixtures:panel",
            },
            "roles": {"time": "t", "group": "entity"},
            "source": {"kind": "none"},
        }
    )
    result = runner.run_target(target)
    assert result.status == "error"
    assert "does_not_exist" in result.message


def test_a_missing_local_path_becomes_a_row():
    target = spec.from_dict(
        {
            "name": "nowhere",
            "entry": {"pipeline": "a:b", "data": "c:d"},
            "roles": {"time": "t"},
            "source": {"kind": "local", "path": "/definitely/not/here"},
        }
    )
    result = runner.run_target(target)
    assert result.status == "error"
    assert "does not exist" in result.message


def test_summary_excludes_targets_that_did_not_run():
    """A failed clone is evidence of nothing and must not move the rate."""
    results = [
        spec.Result(name="a", status="leaks", columns=4, leaking=2),
        spec.Result(name="b", status="clean", columns=4),
        spec.Result(name="c", status="error", message="clone failed"),
        spec.Result(name="d", status="timeout"),
    ]
    stats = report.summarise(results)
    assert stats["ran"] == 2
    assert stats["did_not_run"] == 2
    assert stats["leak_rate"] == 0.5
    assert stats["columns_checked"] == 8


def test_summary_of_nothing_does_not_divide_by_zero():
    assert report.summarise([])["leak_rate"] == 0.0


def test_markdown_lists_failures_and_mismatches():
    results = [
        spec.Result(
            name="a",
            status="leaks",
            columns=3,
            leaking=1,
            leaking_columns=("x",),
            expect="clean",
        ),
        spec.Result(name="b", status="error", message="clone failed"),
    ]
    text = report.to_markdown(results)
    assert "Targets that did not run" in text
    assert "clone failed" in text
    assert "Expectation mismatches" in text
    assert "expected clean, got leaks" in text


def test_write_produces_all_three_artifacts(tmp_path):
    results = [spec.Result(name="a", status="clean", columns=2)]
    paths = report.write(results, tmp_path)
    assert set(paths) == {"json", "csv", "markdown"}
    assert all(p.exists() for p in paths.values())
    payload = json.loads(paths["json"].read_text())
    assert payload["summary"]["ran"] == 1
    assert payload["results"][0]["name"] == "a"
    assert "name,status" in paths["csv"].read_text()


def test_cli_runs_the_calibration_targets(tmp_path, capsys):
    from survey.__main__ import main

    code = main(["--only", "calibration/clean", "calibration/leaky", "--out", str(tmp_path)])
    assert code == 0
    assert "1/2 pipelines leak" in capsys.readouterr().out
    assert (tmp_path / "REPORT.md").exists()


def test_cli_rejects_an_unknown_target(tmp_path):
    from survey.__main__ import main

    with pytest.raises(SystemExit):
        main(["--only", "no/such/target", "--out", str(tmp_path)])
