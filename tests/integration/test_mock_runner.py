from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.workflow import runner as runner_module
from gpu_shortmd.workflow.runner import (
    RunConfigurationError,
    RunExecutionError,
    run_single_replica,
)


def test_mock_single_replica_stage_progression_and_outputs(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    result = run_single_replica(
        load_config(config_path),
        config_path=config_path,
        env=environment,
    )
    assert result.status == "COMPLETED"
    assert result.score.md_score_angstrom == 4.2
    run_dir = result.run_dir
    required = {
        "resolved_config.yaml",
        "input_manifest.json",
        "environment.json",
        "pose_summary.csv",
        "replica_summary.csv",
        "artifact_manifest.csv",
        "checksums.sha256",
        "run_report.md",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    input_manifest = json.loads(
        (run_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    external_dependencies = [
        item
        for item in input_manifest["files"]
        if item["kind"] == "external_topology_dependency"
    ]
    assert external_dependencies
    assert {item["destination"] for item in external_dependencies} == {
        "<EXTERNAL_GROMACS_DATA>"
    }
    assert str(config_path.parent) not in json.dumps(input_manifest)
    with (run_dir / "replica_summary.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["velocity_seed"] == "2026073001"
    assert row["status"] == "COMPLETED"
    assert row["max_rmsd_angstrom"] == "4.2"
    log_summary = next((run_dir / "logs").glob("*_summary.json"))
    summary = json.loads(log_summary.read_text(encoding="utf-8"))
    assert summary["overall_status"] == "COMPLETED"
    events = next((run_dir / "logs").glob("*_events.jsonl"))
    event_payloads = [
        json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
    ]
    rmsd_started = next(
        event for event in event_payloads if event["code"] == "RMSD_COMMAND_STARTED"
    )
    assert rmsd_started["params"]["mass_weighted"] is False
    assert "-nomw" in rmsd_started["params"]["args"]
    assert not (run_dir / "state.sqlite3").exists()
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix in {
            ".json",
            ".jsonl",
            ".log",
            ".md",
            ".txt",
            ".xvg",
            ".yaml",
        }:
            text = path.read_text(encoding="utf-8")
            assert str(run_dir) not in text


def test_mock_failure_is_nonzero_and_retains_structured_issue(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    environment["FAKE_GMX_FAIL_STAGE"] = "npt:mdrun"
    with pytest.raises(RunExecutionError) as caught:
        run_single_replica(
            load_config(config_path),
            config_path=config_path,
            env=environment,
        )
    assert caught.value.exit_code == 5
    run_dir = caught.value.run_dir
    issues = next((run_dir / "logs").glob("*_issues.jsonl"))
    assert "RUN_EXECUTION_FAILED" in issues.read_text(encoding="utf-8")
    with (run_dir / "pose_summary.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "FAILED"
    assert row["md_score_angstrom"] == ""
    with (run_dir / "replica_summary.csv").open(encoding="utf-8") as handle:
        replica = next(csv.DictReader(handle))
    assert replica["stage_reached"] == "NPT"
    summary = json.loads(
        next((run_dir / "logs").glob("*_summary.json")).read_text(encoding="utf-8")
    )
    assert summary["failed_steps"] == ["npt.mdrun"]


def test_mock_missing_required_output_is_failure(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    environment["FAKE_GMX_OMIT_OUTPUT"] = "production:.xtc"

    with pytest.raises(RunExecutionError) as caught:
        run_single_replica(
            load_config(config_path),
            config_path=config_path,
            env=environment,
        )

    assert caught.value.exit_code == 5
    with (caught.value.run_dir / "replica_summary.csv").open(
        encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "FAILED"
    assert row["stage_reached"] == "PRODUCTION"
    assert row["max_rmsd_angstrom"] == ""


def test_mock_invalid_rmsd_is_analysis_exit_6(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    environment["FAKE_GMX_INVALID_XVG"] = "1"

    with pytest.raises(RunExecutionError) as caught:
        run_single_replica(
            load_config(config_path),
            config_path=config_path,
            env=environment,
        )

    assert caught.value.exit_code == 6
    issues = next((caught.value.run_dir / "logs").glob("*_issues.jsonl"))
    assert "RUN_EXECUTION_FAILED" in issues.read_text(encoding="utf-8")


def test_unexpected_internal_error_returns_exit_9_and_bug_issue(
    single_replica_config: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, environment = single_replica_config

    def fail_invariant(**_: object) -> None:
        raise AssertionError("injected invariant violation")

    monkeypatch.setattr(runner_module, "_execute_stages", fail_invariant)
    with pytest.raises(RunExecutionError) as caught:
        run_single_replica(
            load_config(config_path),
            config_path=config_path,
            env=environment,
        )

    assert caught.value.exit_code == 9
    issues = next((caught.value.run_dir / "logs").glob("*_issues.jsonl"))
    payload = json.loads(issues.read_text(encoding="utf-8"))
    assert payload["code"] == "INTERNAL_SOFTWARE_ERROR"
    assert payload["severity"] == "CRITICAL"


@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        ("run", {"resume": True}, "resume is not enabled"),
        ("restart", {"retry_failed": True}, "retry_failed is not enabled"),
        ("scheduler", {"work_stealing": True}, "work stealing is not enabled"),
        (
            "scheduler",
            {"tasks_per_gpu": 2},
            "scheduler.tasks_per_gpu = 1",
        ),
    ],
)
def test_later_chunk_controls_are_not_silently_ignored(
    single_replica_config: tuple[Path, dict[str, str]],
    section: str,
    updates: dict[str, object],
    message: str,
) -> None:
    config_path, environment = single_replica_config
    config = load_config(config_path)
    section_value = getattr(config, section).model_copy(update=updates)
    config = config.model_copy(update={section: section_value})

    with pytest.raises(RunConfigurationError, match=message):
        run_single_replica(config, config_path=config_path, env=environment)
