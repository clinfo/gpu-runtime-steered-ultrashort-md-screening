from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.runtime.scheduler import (
    ExecutionResult,
    ReplicaExecutor,
    TaskControl,
    TaskInterrupted,
)
from gpu_shortmd.runtime.state import RuntimeState, TaskClaim
from gpu_shortmd.util.checksums import verify_checksum_file
from gpu_shortmd.util.logging import RunLogger
from gpu_shortmd.workflow.ensemble import resume_run, run_fresh
from gpu_shortmd.workflow.filesystem_identity import pose_filesystem_key
from gpu_shortmd.workflow.inspection import InspectionReport
from gpu_shortmd.workflow.prepared_input import PreparedSystem
from gpu_shortmd.workflow.runner import RunConfigurationError
from gpu_shortmd.workflow.stage_marker import STAGE_COMPLETION_MARKER

EXPECTED_EXTERNAL_VALIDATION_REFERENCE = "SEE_RELEASE_DOCUMENTATION"
EXPECTED_EXTERNAL_VALIDATION_NOTICE = (
    "External validation is documented at the software-release level. This run "
    "records environment and source provenance but does not independently certify "
    "the current checkout. See the repository validation documentation for the "
    "validated commit and scope."
)


def with_replicas(
    config: AppConfig,
    *,
    replicas: int,
    gpu_ids: list[int],
    pruning_threshold: float | None = None,
) -> AppConfig:
    payload = config.model_dump(mode="python")
    payload["trajectory"]["replicas"] = replicas
    payload["trajectory"]["seeds"] = [2026073001 + index for index in range(replicas)]
    payload["scheduler"]["gpu_ids"] = gpu_ids
    payload["scheduler"]["work_stealing"] = True
    if pruning_threshold is not None:
        payload["pruning"]["enabled"] = True
        payload["pruning"]["threshold_angstrom"] = pruning_threshold
        payload["pruning"]["grace_period_seconds"] = 0.0
    return AppConfig.model_validate(payload)


def test_five_replica_mock_run_has_transactional_contract(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=5,
        gpu_ids=[0, 1],
    )
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    result = run_fresh(
        config,
        config_path=config_path,
        env=environment,
    )
    assert result.status == "COMPLETED"
    assert result.md_score_angstrom == 4.2
    state = RuntimeState(result.run_dir / "state.sqlite3")
    replicas = state.rows("replicas")
    assert len(replicas) == 5
    assert len({row["velocity_seed"] for row in replicas}) == 5
    assert all(row["status"] == "COMPLETED" for row in replicas)
    assert all(row["trajectory_time_completed_ps"] == 20.0 for row in replicas)
    assert all(row["max_rmsd_angstrom"] == 4.2 for row in replicas)
    assert all(row["attempt_count"] == 1 for row in state.rows("tasks"))
    assert state.rows("artifacts")
    assert (
        verify_checksum_file(
            checksum_file=result.run_dir / "checksums.sha256",
            root=result.run_dir,
        )
        == []
    )
    required = {
        "resolved_config.yaml",
        "input_manifest.json",
        "environment.json",
        "state.sqlite3",
        "pose_summary.csv",
        "replica_summary.csv",
        "artifact_manifest.csv",
        "checksums.sha256",
        "run_report.md",
        "execution_plan.json",
    }
    assert required <= {path.name for path in result.run_dir.iterdir()}
    environment_payload = json.loads(
        (result.run_dir / "environment.json").read_text(encoding="utf-8")
    )
    assert (
        environment_payload["external_gpu_validation"]
        == EXPECTED_EXTERNAL_VALIDATION_REFERENCE
    )
    assert environment_payload["external_gpu_validation"] not in {
        "PASS",
        "VALIDATED",
        "EXTERNAL_VALIDATION_COMPLETED",
    }
    assert environment_payload["external_validation_notice"] == (
        EXPECTED_EXTERNAL_VALIDATION_NOTICE
    )
    assert set(environment_payload["source_revision"]) == {
        "checkout_clean",
        "commit_sha",
    }
    serialized_environment = json.dumps(environment_payload, sort_keys=True)
    obsolete_gate = "_".join(("DEFERRED", "EXTERNAL", "VALIDATION"))
    assert obsolete_gate not in serialized_environment
    assert "EXTERNAL_VALIDATION_PASS" not in serialized_environment
    assert "EXTERNAL_VALIDATION_COMPLETED" not in serialized_environment
    with (result.run_dir / "pose_summary.csv").open(encoding="utf-8") as handle:
        pose = next(csv.DictReader(handle))
    assert pose["n_replicas_requested"] == "5"
    assert pose["n_replicas_completed"] == "5"
    assert pose["md_score_angstrom"] == "4.2"
    pose_id = str(state.rows("poses")[0]["pose_id"])
    pose_root = result.run_dir / "poses" / pose_filesystem_key(pose_id)
    markers = list(pose_root.glob(f"replica_*/*/{STAGE_COMPLETION_MARKER}"))
    assert len(markers) == 15
    marker_payload = json.loads(markers[0].read_text(encoding="utf-8"))
    assert marker_payload["pose_id"] == pose_id
    assert marker_payload["velocity_seed"] in {
        row["velocity_seed"] for row in state.rows("replicas")
    }
    assert pose_filesystem_key(pose_id) in result.run_dir.name


@pytest.mark.parametrize(
    ("peak_time", "expected_maximum_time_ps"),
    [("0", 0.0), ("10", 10.0)],
)
def test_completed_duration_is_independent_of_rmsd_peak_time(
    single_replica_config: tuple[Path, dict[str, str]],
    peak_time: str,
    expected_maximum_time_ps: float,
) -> None:
    config_path, environment = single_replica_config
    environment["FAKE_GMX_RMSD_PEAK_TIME_PS"] = peak_time
    config = with_replicas(
        load_config(config_path),
        replicas=1,
        gpu_ids=[0],
    )

    result = run_fresh(
        config,
        config_path=config_path,
        env=environment,
    )

    assert result.status == "COMPLETED"
    replica = RuntimeState(result.run_dir / "state.sqlite3").rows("replicas")[0]
    assert replica["status"] == "COMPLETED"
    assert replica["trajectory_time_completed_ps"] == 20.0
    assert replica["max_rmsd_nm"] == 0.42
    assert replica["max_rmsd_angstrom"] == 4.2
    event_payloads = [
        json.loads(line)
        for path in (result.run_dir / "logs").glob("*_events.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rmsd_event = next(
        event for event in event_payloads if event["code"] == "RMSD_CALCULATED"
    )
    assert rmsd_event["metrics"]["max_rmsd_time_ps"] == expected_maximum_time_ps


def test_failed_stage_never_receives_completion_marker(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=1,
        gpu_ids=[0],
    )
    environment["FAKE_GMX_FAIL_STAGE"] = "npt:mdrun"
    result = run_fresh(
        config,
        config_path=config_path,
        env=environment,
    )
    assert result.status == "FAILED"
    pose_id = str(
        RuntimeState(result.run_dir / "state.sqlite3").rows("poses")[0]["pose_id"]
    )
    replica_root = (
        result.run_dir / "poses" / pose_filesystem_key(pose_id) / "replica_01"
    )
    assert (replica_root / "nvt" / STAGE_COMPLETION_MARKER).is_file()
    assert not (replica_root / "npt" / STAGE_COMPLETION_MARKER).exists()


def test_dry_run_persists_plan_and_launches_no_replica(
    single_replica_config: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gpu_shortmd.workflow.ensemble.available_cpu_count",
        lambda: 12,
    )
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=5,
        gpu_ids=[0, 1, 2],
    )
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    result = run_fresh(
        config,
        config_path=config_path,
        env=environment,
        dry_run=True,
    )
    assert result.status == "DRY_RUN"
    environment_text = (result.run_dir / "environment.json").read_text(encoding="utf-8")
    run_report = (result.run_dir / "run_report.md").read_text(encoding="utf-8")
    for artifact_text in (environment_text, run_report):
        obsolete_gate = "_".join(("DEFERRED", "EXTERNAL", "VALIDATION"))
        assert obsolete_gate not in artifact_text
        assert "EXTERNAL_VALIDATION_PASS" not in artifact_text
        assert "EXTERNAL_VALIDATION_COMPLETED" not in artifact_text
        assert EXPECTED_EXTERNAL_VALIDATION_NOTICE in artifact_text
    environment_payload = json.loads(environment_text)
    assert (
        environment_payload["external_gpu_validation"]
        == EXPECTED_EXTERNAL_VALIDATION_REFERENCE
    )
    assert environment_payload["external_gpu_validation"] not in {
        "PASS",
        "VALIDATED",
        "EXTERNAL_VALIDATION_COMPLETED",
    }
    assert set(environment_payload["source_revision"]) == {
        "checkout_clean",
        "commit_sha",
    }
    plan = json.loads(
        (result.run_dir / "execution_plan.json").read_text(encoding="utf-8")
    )
    assert plan["launches_external_processes"] is False
    assert len(plan["tasks"]) == 5
    assert not (result.run_dir / "poses").exists()
    assert all(
        row["status"] == "PENDING"
        for row in RuntimeState(result.run_dir / "state.sqlite3").rows("tasks")
    )


def test_mock_pruning_has_null_completed_score(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=5,
        gpu_ids=[0],
        pruning_threshold=4.0,
    )
    result = run_fresh(
        config,
        config_path=config_path,
        env=environment,
    )
    assert result.status == "PRUNED"
    assert result.md_score_angstrom is None
    state = RuntimeState(result.run_dir / "state.sqlite3")
    pose = state.rows("poses")[0]
    replica = state.rows("replicas")[0]
    assert pose["trigger_replica_id"] == "replica_01"
    assert pose["trigger_simulation_time_ps"] == 10.0
    assert pose["observed_max_rmsd_angstrom"] == 4.2
    assert replica["status"] == "PRUNED"
    assert replica["trajectory_time_completed_ps"] == 20.0
    assert [row["status"] for row in state.rows("tasks")] == [
        "PRUNED",
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
    ]
    pose_root = result.run_dir / "poses" / pose_filesystem_key(str(pose["pose_id"]))
    assert not (
        pose_root / "replica_01" / "production" / STAGE_COMPLETION_MARKER
    ).exists()
    assert not (pose_root / "replica_01" / ".pbc-rmsd-work").exists()


class StopAfterFirstExecutor:
    def __call__(
        self,
        claim: TaskClaim,
        control: TaskControl,
    ) -> ExecutionResult:
        if claim.replica_id == "replica_02":
            control.request_run_stop()
            raise TaskInterrupted("controlled test interruption")
        return ExecutionResult(
            status="COMPLETED",
            stage_reached="PRODUCTION",
            trajectory_time_completed_ps=1000.0,
            max_rmsd_nm=0.2,
            max_rmsd_angstrom=2.0,
            exit_code=0,
        )


def stop_executor_factory(
    _config: AppConfig,
    _report: InspectionReport,
    _frozen: PreparedSystem,
    _run_dir: Path,
    _environment: Mapping[str, str],
    _logger: RunLogger,
) -> ReplicaExecutor:
    return StopAfterFirstExecutor()


class CompleteExecutor:
    def __call__(
        self,
        _claim: TaskClaim,
        _control: TaskControl,
    ) -> ExecutionResult:
        return ExecutionResult(
            status="COMPLETED",
            stage_reached="PRODUCTION",
            trajectory_time_completed_ps=1000.0,
            max_rmsd_nm=0.3,
            max_rmsd_angstrom=3.0,
            exit_code=0,
        )


def complete_executor_factory(
    _config: AppConfig,
    _report: InspectionReport,
    _frozen: PreparedSystem,
    _run_dir: Path,
    _environment: Mapping[str, str],
    _logger: RunLogger,
) -> ReplicaExecutor:
    return CompleteExecutor()


def test_controlled_stop_and_resume_preserve_completed_work_and_seeds(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=3,
        gpu_ids=[0],
    )
    first = run_fresh(
        config,
        config_path=config_path,
        env=environment,
        executor_factory=stop_executor_factory,
    )
    assert first.status == "INTERRUPTED"
    state = RuntimeState(first.run_dir / "state.sqlite3")
    seeds_before = [row["velocity_seed"] for row in state.rows("replicas")]
    attempts_before = {
        row["replica_id"]: row["attempt_count"] for row in state.rows("tasks")
    }
    assert attempts_before["replica_01"] == 1

    resumed = resume_run(
        first.run_dir,
        retry_failed=False,
        env=environment,
        executor_factory=complete_executor_factory,
    )
    assert resumed.status == "COMPLETED"
    assert resumed.md_score_angstrom == 3.0
    assert [row["velocity_seed"] for row in state.rows("replicas")] == seeds_before
    attempts_after = {
        row["replica_id"]: row["attempt_count"] for row in state.rows("tasks")
    }
    assert attempts_after["replica_01"] == 1
    assert attempts_after["replica_02"] == 2
    assert attempts_after["replica_03"] == 1
    assert "RUN_RESUMED" in state.event_codes()


def test_resume_rejects_tampered_frozen_input(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=3,
        gpu_ids=[0],
    )
    first = run_fresh(
        config,
        config_path=config_path,
        env=environment,
        executor_factory=stop_executor_factory,
    )
    frozen_index = first.run_dir / "inputs" / "p2_index.ndx"
    frozen_index.write_text(
        frozen_index.read_text(encoding="utf-8") + "\n; tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RunConfigurationError,
        match="frozen input checksum changed",
    ) as caught:
        resume_run(
            first.run_dir,
            retry_failed=False,
            env=environment,
            executor_factory=complete_executor_factory,
        )
    assert caught.value.exit_code == 7


def test_resume_rejects_tampered_registered_artifact(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = with_replicas(
        load_config(config_path),
        replicas=3,
        gpu_ids=[0],
    )
    first = run_fresh(
        config,
        config_path=config_path,
        env=environment,
        executor_factory=stop_executor_factory,
    )
    report = first.run_dir / "run_report.md"
    report.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        RunConfigurationError,
        match="registered artifact size changed",
    ) as caught:
        resume_run(
            first.run_dir,
            retry_failed=False,
            env=environment,
            executor_factory=complete_executor_factory,
        )
    assert caught.value.exit_code == 7
