from __future__ import annotations

import csv
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.runtime.scheduler import (
    ExecutionResult,
    ReplicaExecutor,
    TaskControl,
    TaskInterrupted,
)
from gpu_shortmd.runtime.state import RuntimeState, TaskClaim
from gpu_shortmd.util.logging import RunLogger
from gpu_shortmd.workflow.ensemble import resume_run, run_fresh
from gpu_shortmd.workflow.filesystem_identity import derive_pose_filesystem_keys
from gpu_shortmd.workflow.inspection import InspectionReport
from gpu_shortmd.workflow.prepared_input import PreparedSystem

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"


def multi_pose_config(
    config_path: Path,
    *,
    gpu_ids: list[int],
) -> AppConfig:
    config = load_config(config_path)
    payload = config.model_dump(mode="python")
    payload["run"]["name"] = "two-pose-steering"
    payload["scheduler"]["gpu_ids"] = gpu_ids
    payload["scheduler"]["work_stealing"] = True
    payload["gromacs"]["ntomp"] = "auto"
    return AppConfig.model_validate(payload)


def write_manifest(path: Path) -> None:
    prepared = str(EXAMPLE / "prepared_input")
    value: dict[str, Any] = {
        "schema_version": 1,
        "poses": [
            {
                "pose_id": "pose-a-pruned",
                "prepared_system_dir": prepared,
                "start_structure": "p2_em.gro",
                "topology": "p2_topol.top",
                "index": "p2_index.ndx",
                "ligand_resname": "LIG",
                "overrides": {
                    "trajectory": {
                        "replicas": 5,
                        "seeds": [101, 102, 103, 104, 105],
                    },
                    "pruning": {
                        "enabled": True,
                        "threshold_angstrom": 4.0,
                        "grace_period_seconds": 0.0,
                    },
                },
            },
            {
                "pose_id": "pose-b-completed",
                "prepared_system_dir": prepared,
                "start_structure": "p2_em.gro",
                "topology": "p2_topol.top",
                "index": "p2_index.ndx",
                "ligand_resname": "LIG",
                "overrides": {
                    "trajectory": {
                        "replicas": 5,
                        "seeds": [201, 202, 203, 204, 205],
                    },
                    "pruning": {
                        "enabled": False,
                        "threshold_angstrom": None,
                        "grace_period_seconds": 0.0,
                    },
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def completed(value: float = 2.0) -> ExecutionResult:
    return ExecutionResult(
        status="COMPLETED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=1000.0,
        max_rmsd_nm=value / 10.0,
        max_rmsd_angstrom=value,
        exit_code=0,
    )


class SteeringFactory:
    def __init__(self) -> None:
        self.pose_a_running = threading.Barrier(2)
        self.pruned = threading.Event()
        self.stolen = threading.Event()
        self.executed: list[str] = []
        self.lock = threading.Lock()

    def __call__(
        self,
        _config: AppConfig,
        _report: InspectionReport,
        _frozen: PreparedSystem,
        _run_dir: Path,
        _environment: Mapping[str, str],
        _logger: RunLogger,
    ) -> ReplicaExecutor:
        return self.execute

    def execute(
        self,
        claim: TaskClaim,
        control: TaskControl,
    ) -> ExecutionResult:
        with self.lock:
            self.executed.append(claim.task_id)
        if claim.pose_id == "pose-a-pruned":
            self.pose_a_running.wait(timeout=3)
            if claim.replica_id == "replica_01":
                assert control.observe_rmsd(
                    simulation_time_ps=50.0,
                    rmsd_angstrom=4.5,
                )
                self.pruned.set()
                return completed(4.5)
            self.pruned.wait(timeout=3)
            control.checkpoint()
        if claim.stolen_from_gpu_id is not None:
            self.stolen.set()
        if (
            claim.pose_id == "pose-b-completed"
            and claim.worker_id == "gpu-1"
            and claim.replica_id == "replica_01"
        ):
            self.stolen.wait(timeout=3)
        return completed()


def test_two_pose_five_replica_pruning_and_cross_pose_stealing(
    single_replica_config: tuple[Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    config_path, environment = single_replica_config
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    manifest_path = tmp_path / "poses.yaml"
    write_manifest(manifest_path)
    factory = SteeringFactory()
    result = run_fresh(
        multi_pose_config(config_path, gpu_ids=[0, 1]),
        config_path=config_path,
        pose_manifest_path=manifest_path,
        env=environment,
        executor_factory=factory,
    )
    assert result.status == "COMPLETED"
    assert result.md_score_angstrom is None
    expected_keys = derive_pose_filesystem_keys(("pose-a-pruned", "pose-b-completed"))
    assert {path.name for path in (result.run_dir / "inputs").iterdir()} == set(
        expected_keys.values()
    )
    state = RuntimeState(result.run_dir / "state.sqlite3")
    tasks = state.rows("tasks")
    assert len(tasks) == 10
    assert len(factory.executed) == len(set(factory.executed))
    assert all(row["attempt_count"] <= 1 for row in tasks)
    assert all(
        row["attempt_count"] == 1 for row in tasks if row["task_id"] in factory.executed
    )
    assert all(int(row["resolved_ntomp"]) > 0 for row in tasks)
    by_pose = {
        pose_id: [row for row in tasks if row["pose_id"] == pose_id]
        for pose_id in ("pose-a-pruned", "pose-b-completed")
    }
    assert [row["status"] for row in by_pose["pose-a-pruned"]].count("PRUNED") == 1
    assert [row["status"] for row in by_pose["pose-a-pruned"]].count("INTERRUPTED") == 1
    assert [row["status"] for row in by_pose["pose-a-pruned"]].count("SKIPPED") == 3
    assert {row["status"] for row in by_pose["pose-b-completed"]} == {"COMPLETED"}
    assert all(not row["stop_requested"] for row in by_pose["pose-b-completed"])
    assert factory.stolen.is_set()

    poses = {row["pose_id"]: row for row in state.rows("poses")}
    assert poses["pose-a-pruned"]["status"] == "PRUNED"
    assert poses["pose-a-pruned"]["md_score_angstrom"] is None
    assert poses["pose-b-completed"]["status"] == "COMPLETED"
    events = state.rows("events")
    prune_sequence = next(
        row["sequence"] for row in events if row["code"] == "POSE_PRUNED"
    )
    assert any(
        row["sequence"] > prune_sequence
        and row["pose_id"] == "pose-b-completed"
        and row["code"] in {"TASK_CLAIMED", "TASK_STOLEN"}
        for row in events
    )
    assert all(row["pose_id"] for row in events)
    plan = json.loads(
        (result.run_dir / "execution_plan.json").read_text(encoding="utf-8")
    )
    assert plan["pose_ids"] == ["pose-a-pruned", "pose-b-completed"]
    assert len(plan["tasks"]) == 10
    assert all(task["resolved_ntomp"] > 0 for task in plan["tasks"])
    with (result.run_dir / "pose_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        pose_summary = list(csv.DictReader(handle))
    with (result.run_dir / "replica_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        replica_summary = list(csv.DictReader(handle))
    assert {row["pose_id"] for row in pose_summary} == {
        "pose-a-pruned",
        "pose-b-completed",
    }
    assert {row["pose_id"] for row in replica_summary} == {
        "pose-a-pruned",
        "pose-b-completed",
    }
    run_manifest = json.loads(
        (result.run_dir / "logs" / f"{result.run_id}_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["pose_ids"] == [
        "pose-a-pruned",
        "pose-b-completed",
    ]
    assert {pose["pose_id"] for pose in run_manifest["poses"]} == {
        "pose-a-pruned",
        "pose-b-completed",
    }
    event_rows = [
        json.loads(line)
        for line in (result.run_dir / "logs" / f"{result.run_id}_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert event_rows
    assert all(row["pose_id"] for row in event_rows)


class StopAfterPruneFactory:
    def __call__(
        self,
        _config: AppConfig,
        _report: InspectionReport,
        _frozen: PreparedSystem,
        _run_dir: Path,
        _environment: Mapping[str, str],
        _logger: RunLogger,
    ) -> ReplicaExecutor:
        return self.execute

    def execute(
        self,
        claim: TaskClaim,
        control: TaskControl,
    ) -> ExecutionResult:
        if claim.pose_id == "pose-a-pruned":
            assert claim.replica_id == "replica_01"
            assert control.observe_rmsd(
                simulation_time_ps=25.0,
                rmsd_angstrom=4.5,
            )
            return completed(4.5)
        if claim.replica_id == "replica_02":
            control.request_run_stop()
            raise TaskInterrupted("deterministic crash/restart boundary")
        return completed(2.5)


class ResumeCompleteFactory:
    def __call__(
        self,
        _config: AppConfig,
        _report: InspectionReport,
        _frozen: PreparedSystem,
        _run_dir: Path,
        _environment: Mapping[str, str],
        _logger: RunLogger,
    ) -> ReplicaExecutor:
        return self.execute

    def execute(
        self,
        _claim: TaskClaim,
        _control: TaskControl,
    ) -> ExecutionResult:
        return completed(3.0)


def test_multi_pose_crash_resume_preserves_pose_state_seeds_and_pruning(
    single_replica_config: tuple[Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    config_path, environment = single_replica_config
    manifest_path = tmp_path / "poses.yaml"
    write_manifest(manifest_path)
    first = run_fresh(
        multi_pose_config(config_path, gpu_ids=[0]),
        config_path=config_path,
        pose_manifest_path=manifest_path,
        env=environment,
        executor_factory=StopAfterPruneFactory(),
    )
    assert first.status == "INTERRUPTED"
    state = RuntimeState(first.run_dir / "state.sqlite3")
    seeds_before = {
        (row["pose_id"], row["replica_id"]): row["velocity_seed"]
        for row in state.rows("replicas")
    }
    prune_events_before = state.event_codes().count("POSE_PRUNED")

    resumed = resume_run(
        first.run_dir,
        retry_failed=False,
        env=environment,
        executor_factory=ResumeCompleteFactory(),
    )
    assert resumed.status == "COMPLETED"
    seeds_after = {
        (row["pose_id"], row["replica_id"]): row["velocity_seed"]
        for row in state.rows("replicas")
    }
    assert seeds_after == seeds_before
    poses = {row["pose_id"]: row for row in state.rows("poses")}
    assert poses["pose-a-pruned"]["status"] == "PRUNED"
    assert poses["pose-b-completed"]["status"] == "COMPLETED"
    assert state.event_codes().count("POSE_PRUNED") == prune_events_before
    assert "RUN_RESUMED" in state.event_codes()
