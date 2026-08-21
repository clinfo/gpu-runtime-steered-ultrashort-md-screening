from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from gpu_shortmd.analysis.xvg import XvgSample, XvgSeries
from gpu_shortmd.runtime.monitoring import MonitoringOutcome
from gpu_shortmd.runtime.scheduler import (
    ExecutionResult,
    TaskControl,
    TaskInterrupted,
    build_dry_run_plan,
    resolve_ntomp,
    run_local_scheduler,
)
from gpu_shortmd.runtime.state import RuntimeState, TaskClaim
from gpu_shortmd.workflow.replica import _monitor_target

RUN_ID = "RUN-20260731-000000-scheduler"
POSE_ID = "pose-1"


def create_state(tmp_path: Path, *, pruning: bool = False) -> RuntimeState:
    return RuntimeState.create(
        tmp_path / "state.sqlite3",
        run_id=RUN_ID,
        pose_id=POSE_ID,
        seeds=(101, 102, 103, 104, 105, 106, 107),
        gpu_ids=(0, 1, 2),
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        pruning_enabled=pruning,
        pruning_threshold_angstrom=4.0 if pruning else None,
    )


def completed_result(value: float = 2.0) -> ExecutionResult:
    return ExecutionResult(
        status="COMPLETED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=1000.0,
        max_rmsd_nm=value / 10.0,
        max_rmsd_angstrom=value,
        exit_code=0,
    )


def test_dry_run_plan_is_deterministic_and_launch_free() -> None:
    plan = build_dry_run_plan(
        run_id=RUN_ID,
        pose_id=POSE_ID,
        seeds=(11, 12, 13, 14),
        gpu_ids=(2, 5),
        stages=("nvt", "npt", "production"),
        pruning_enabled=True,
        pruning_threshold_angstrom=4.0,
        work_stealing=True,
        input_checksums={"topol.top": "b", "start.gro": "a"},
    )
    assert [task["initial_gpu_id"] for task in plan["tasks"]] == [2, 5, 2, 5]
    assert plan["launches_external_processes"] is False
    assert plan["pruning"]["strict_operator"] == ">"
    assert list(plan["input_checksums"]) == ["start.gro", "topol.top"]


def test_ntomp_resolution_is_safe_and_deterministic() -> None:
    assert resolve_ntomp("auto", gpu_ids=(0, 1, 2), ntmpi=1, cpu_count=10) == 3
    assert resolve_ntomp(2, gpu_ids=(0, 1, 2), ntmpi=1, cpu_count=6) == 2
    with pytest.raises(ValueError, match="oversubscribe"):
        resolve_ntomp(3, gpu_ids=(0, 1, 2), ntmpi=1, cpu_count=8)


def test_work_stealing_has_no_duplicate_claims(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    release_slow_worker = threading.Event()

    def executor(claim: TaskClaim, control: TaskControl) -> ExecutionResult:
        control.checkpoint()
        if claim.worker_id == "gpu-0":
            release_slow_worker.wait(timeout=2)
        if claim.worker_id != "gpu-0" and claim.assigned_gpu_id == 0:
            release_slow_worker.set()
        time.sleep(0.002)
        return completed_result()

    result = run_local_scheduler(
        state=state,
        run_id=RUN_ID,
        gpu_ids=(0, 1, 2),
        work_stealing=True,
        pruning_threshold_angstrom=None,
        executor=executor,
    )
    release_slow_worker.set()
    assert len(result.claimed_task_ids) == 7
    assert len(set(result.claimed_task_ids)) == 7
    assert result.failed_task_ids == ()
    assert result.stolen_task_ids
    assert all(row["attempt_count"] == 1 for row in state.rows("tasks"))
    assert "TASK_STOLEN" in state.event_codes()
    assert state.finalize_pose(run_id=RUN_ID, pose_id=POSE_ID) == "COMPLETED"
    assert state.rows("poses")[0]["md_score_angstrom"] == 2.0


def test_pruning_coordinates_running_and_pending_siblings(tmp_path: Path) -> None:
    state = create_state(tmp_path, pruning=True)
    running = threading.Barrier(3)

    def executor(claim: TaskClaim, control: TaskControl) -> ExecutionResult:
        running.wait(timeout=2)
        if claim.replica_id == "replica_01":
            assert control.observe_rmsd(
                simulation_time_ps=50.0,
                rmsd_angstrom=4.5,
            )
            return completed_result(4.5)
        time.sleep(0.01)
        control.checkpoint()
        return completed_result()

    result = run_local_scheduler(
        state=state,
        run_id=RUN_ID,
        gpu_ids=(0, 1, 2),
        work_stealing=True,
        pruning_threshold_angstrom=4.0,
        executor=executor,
    )
    assert len(set(result.claimed_task_ids)) == len(result.claimed_task_ids)
    statuses = [row["status"] for row in state.rows("tasks")]
    assert statuses.count("PRUNED") == 1
    assert statuses.count("INTERRUPTED") == 2
    assert statuses.count("SKIPPED") == 4
    assert state.finalize_pose(run_id=RUN_ID, pose_id=POSE_ID) == "PRUNED"
    pose = state.rows("poses")[0]
    assert pose["md_score_angstrom"] is None
    assert pose["trigger_replica_id"] == "replica_01"


def test_monitor_thread_sibling_pruning_is_interrupted_not_failed(
    tmp_path: Path,
) -> None:
    state = RuntimeState.create(
        tmp_path / "state.sqlite3",
        run_id=RUN_ID,
        pose_id=POSE_ID,
        seeds=(101, 102),
        gpu_ids=(0, 1),
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        pruning_enabled=True,
        pruning_threshold_angstrom=4.0,
    )
    executors_started = threading.Barrier(2)
    monitor_polling = threading.Event()
    pruning_complete = threading.Event()

    def executor(claim: TaskClaim, control: TaskControl) -> ExecutionResult:
        executors_started.wait(timeout=2)
        if claim.replica_id == "replica_01":
            assert monitor_polling.wait(timeout=2)
            assert control.observe_rmsd(
                simulation_time_ps=20.0,
                rmsd_angstrom=4.5,
            )
            pruning_complete.set()
            return completed_result(4.5)

        finished = threading.Event()
        cancel = threading.Event()
        outcomes: list[MonitoringOutcome] = []
        interruptions: list[TaskInterrupted] = []
        errors: list[BaseException] = []

        def snapshot_supplier() -> XvgSeries:
            monitor_polling.set()
            assert pruning_complete.wait(timeout=2)
            return XvgSeries(
                path=tmp_path / "online.xvg",
                samples=(XvgSample(time_ps=0.0, rmsd=0.1),),
            )

        monitor = threading.Thread(
            target=_monitor_target,
            kwargs={
                "finished": finished,
                "poll_interval_seconds": 0.001,
                "snapshot_supplier": snapshot_supplier,
                "control": control,
                "outcomes": outcomes,
                "interruptions": interruptions,
                "errors": errors,
                "cancel": cancel,
            },
        )
        monitor.start()
        monitor.join(timeout=3)
        assert not monitor.is_alive()
        assert cancel.is_set()
        assert outcomes == []
        assert errors == []
        assert len(interruptions) == 1
        raise interruptions[0]

    result = run_local_scheduler(
        state=state,
        run_id=RUN_ID,
        gpu_ids=(0, 1),
        work_stealing=False,
        pruning_threshold_angstrom=4.0,
        executor=executor,
    )
    assert result.failed_task_ids == ()
    assert len(result.interrupted_task_ids) == 1
    statuses = {row["replica_id"]: row["status"] for row in state.rows("tasks")}
    assert statuses == {
        "replica_01": "PRUNED",
        "replica_02": "INTERRUPTED",
    }
