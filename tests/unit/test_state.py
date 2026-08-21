from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from gpu_shortmd.runtime import state as state_module
from gpu_shortmd.runtime.processes import (
    ProcessIdentityResult,
    ProcessIdentityStatus,
    process_start_token,
)
from gpu_shortmd.runtime.pruning import PruningTrigger
from gpu_shortmd.runtime.state import (
    ResumeValidationError,
    RuntimeState,
    validate_sqlite_filesystem,
)

RUN_ID = "RUN-20260731-000000-test"
POSE_ID = "pose-1"


def create_state(
    tmp_path: Path,
    *,
    replicas: int = 3,
    pruning: bool = False,
) -> RuntimeState:
    return RuntimeState.create(
        tmp_path / "state.sqlite3",
        run_id=RUN_ID,
        pose_id=POSE_ID,
        seeds=tuple(range(101, 101 + replicas)),
        gpu_ids=(0, 1),
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        pruning_enabled=pruning,
        pruning_threshold_angstrom=4.0 if pruning else None,
    )


def test_sqlite_filesystem_capability_checks_locking(tmp_path: Path) -> None:
    capability = validate_sqlite_filesystem(tmp_path)
    assert capability.locking_validated is True
    assert capability.journal_mode in {"WAL", "DELETE"}
    assert not list(tmp_path.glob(".gpu-shortmd-sqlite-probe-*"))


def test_state_persists_unique_seeds_and_initial_tasks(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    replicas = state.rows("replicas")
    tasks = state.rows("tasks")
    assert [row["velocity_seed"] for row in replicas] == [101, 102, 103]
    assert [row["status"] for row in tasks] == ["PENDING"] * 3
    assert [row["assigned_gpu_id"] for row in tasks] == [0, 1, 0]
    with sqlite3.connect(state.path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_integrity_upgrades_pre_durable_lifecycle_task_schema(tmp_path: Path) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(claim, pid=999_984, start_token="pre-patch-owner")
    with sqlite3.connect(state.path) as connection:
        for column in (
            "heartbeat_write_count",
            "process_step_id",
            "process_finished_at",
            "process_returncode",
            "process_state",
        ):
            connection.execute(f"ALTER TABLE tasks DROP COLUMN {column}")
        connection.execute("PRAGMA user_version=1")

    RuntimeState(state.path).integrity_check()
    row = state.rows("tasks")[0]
    assert row["process_state"] == "RUNNING"
    assert row["process_returncode"] is None
    assert row["process_finished_at"] is None
    assert row["process_step_id"] is None
    assert row["heartbeat_write_count"] == 0


def test_integrity_upgrades_pre_command_step_schema_without_guessing(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(
        claim,
        pid=999_983,
        start_token="pre-command-step-owner",
        process_step_id="replica_01.nvt.grompp",
    )
    with sqlite3.connect(state.path) as connection:
        connection.execute("ALTER TABLE tasks DROP COLUMN process_step_id")
        connection.execute("PRAGMA user_version=2")

    RuntimeState(state.path).integrity_check()
    row = state.rows("tasks")[0]
    assert row["process_state"] == "RUNNING"
    assert row["process_step_id"] is None


def test_pruning_is_pose_atomic_and_stops_or_skips_siblings(tmp_path: Path) -> None:
    state = create_state(tmp_path, replicas=4, pruning=True)
    trigger_claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    sibling_claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-1",
        gpu_id=1,
        work_stealing=False,
    )
    assert trigger_claim is not None
    assert sibling_claim is not None

    outcome = state.trigger_pruning(
        trigger_claim,
        trigger=PruningTrigger(25.0, trigger_claim.replica_id, 4.5),
    )
    assert len(outcome.skipped_task_ids) == 2
    assert outcome.stop_requested_task_ids == (sibling_claim.task_id,)
    assert state.heartbeat(sibling_claim) is True

    state.finish_task(
        trigger_claim,
        status="PRUNED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=25.0,
        max_rmsd_nm=0.45,
        max_rmsd_angstrom=4.5,
        exit_code=0,
    )
    state.finish_task(
        sibling_claim,
        status="INTERRUPTED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=20.0,
        max_rmsd_nm=0.3,
        max_rmsd_angstrom=3.0,
        exit_code=8,
    )
    assert state.finalize_pose(run_id=RUN_ID, pose_id=POSE_ID) == "PRUNED"
    pose = state.rows("poses")[0]
    assert pose["md_score_angstrom"] is None
    assert pose["observed_max_rmsd_angstrom"] == 4.5
    assert pose["trigger_replica_id"] == trigger_claim.replica_id
    assert pose["trigger_simulation_time_ps"] == 25.0


def test_concurrent_pruning_appends_exactly_one_pose_pruned_event(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=2, pruning=True)
    claims = [
        state.claim_task(
            run_id=RUN_ID,
            worker_id=f"gpu-{gpu_id}",
            gpu_id=gpu_id,
            work_stealing=False,
        )
        for gpu_id in (0, 1)
    ]
    assert all(claim is not None for claim in claims)
    claimed = [claim for claim in claims if claim is not None]
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def trigger(index: int, time_ps: float, rmsd_angstrom: float) -> None:
        try:
            barrier.wait(timeout=2)
            claim = claimed[index]
            state.trigger_pruning(
                claim,
                trigger=PruningTrigger(
                    time_ps,
                    claim.replica_id,
                    rmsd_angstrom,
                ),
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=trigger, args=(0, 30.0, 5.0)),
        threading.Thread(target=trigger, args=(1, 20.0, 4.5)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert not failures
    assert state.event_codes().count("POSE_PRUNED") == 1
    assert state.event_codes().count("PRUNING_TRIGGER_OBSERVED") == 1
    pose = state.rows("poses")[0]
    assert pose["trigger_replica_id"] == claimed[1].replica_id
    assert pose["trigger_simulation_time_ps"] == 20.0
    assert pose["trigger_observed_rmsd_angstrom"] == 4.5
    assert pose["observed_max_rmsd_angstrom"] == 5.0


def test_resume_preserves_terminal_work_and_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_state(tmp_path)
    completed = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    running = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-1",
        gpu_id=1,
        work_stealing=False,
    )
    assert completed is not None
    assert running is not None
    state.finish_task(
        completed,
        status="COMPLETED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=1000.0,
        max_rmsd_nm=0.2,
        max_rmsd_angstrom=2.0,
        exit_code=0,
    )
    original_seeds = [row["velocity_seed"] for row in state.rows("replicas")]
    state.record_process(running, pid=999_990, start_token="exited-owner")
    monkeypatch.setattr(
        state_module,
        "inspect_process_identity",
        lambda **_: ProcessIdentityResult(
            pid=999_990,
            status=ProcessIdentityStatus.EXITED,
            reason="test exited process",
        ),
    )

    state.verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=False,
    )
    tasks = {row["task_id"]: row for row in state.rows("tasks")}
    assert tasks[completed.task_id]["status"] == "COMPLETED"
    assert tasks[running.task_id]["status"] == "PENDING"
    assert [row["velocity_seed"] for row in state.rows("replicas")] == original_seeds
    assert "RUN_RESUMED" in state.event_codes()


def test_resume_rejects_changed_immutable_inputs(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    with pytest.raises(ResumeValidationError, match="configuration checksum"):
        state.verify_resume(
            config_sha256="c" * 64,
            input_manifest_sha256="b" * 64,
            retry_failed=False,
        )


def test_resume_uses_complete_durable_exit_without_os_identity_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(claim, pid=999_989, start_token="confirmed-owner")
    state.record_process_exit(claim, returncode=0)
    monkeypatch.setattr(
        state_module,
        "inspect_process_identity",
        lambda **_: pytest.fail("durable EXITED state must not inspect the OS PID"),
    )

    state.verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=False,
    )
    row = state.rows("tasks")[0]
    assert row["status"] == "PENDING"
    assert row["process_state"] == "NONE"
    assert row["process_pid"] is None
    assert row["process_start_token"] is None
    assert row["process_returncode"] is None
    assert row["process_finished_at"] is None


def test_new_process_registration_atomically_replaces_prior_exit(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(
        claim,
        pid=999_987,
        start_token="first-owner",
        process_step_id="replica_01.nvt.grompp",
    )
    state.record_process_exit(claim, returncode=0)
    state.record_process(
        claim,
        pid=999_988,
        start_token="second-owner",
        process_step_id="replica_01.nvt.mdrun",
    )

    row = state.rows("tasks")[0]
    assert row["process_pid"] == 999_988
    assert row["process_start_token"] == "second-owner"
    assert row["process_state"] == "RUNNING"
    assert row["process_returncode"] is None
    assert row["process_finished_at"] is None
    assert row["process_step_id"] == "replica_01.nvt.mdrun"


def test_finish_task_clears_complete_process_lifecycle(tmp_path: Path) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(claim, pid=999_986, start_token="finished-owner")
    state.record_process_exit(claim, returncode=0)
    state.finish_task(
        claim,
        status="COMPLETED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=1000.0,
        max_rmsd_nm=0.2,
        max_rmsd_angstrom=2.0,
        exit_code=0,
    )

    row = state.rows("tasks")[0]
    assert row["status"] == "COMPLETED"
    assert row["process_state"] == "NONE"
    assert row["process_pid"] is None
    assert row["process_start_token"] is None
    assert row["process_returncode"] is None
    assert row["process_finished_at"] is None
    assert row["process_step_id"] is None


def test_resume_refuses_matching_live_owned_process(tmp_path: Path) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        state.record_process(
            claim,
            pid=process.pid,
            start_token=process_start_token(process.pid),
        )
        with pytest.raises(ResumeValidationError, match="still owns a live process"):
            state.verify_resume(
                config_sha256="a" * 64,
                input_manifest_sha256="b" * 64,
                retry_failed=False,
            )
        row = state.rows("tasks")[0]
        assert row["status"] == "RUNNING"
        assert row["process_pid"] == process.pid
    finally:
        process.terminate()
        process.wait(timeout=3)


def test_resume_recovers_confirmed_exited_process_and_clears_identity(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    token = process_start_token(process.pid)
    state.record_process(claim, pid=process.pid, start_token=token)
    process.terminate()
    process.wait(timeout=3)

    state.verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=False,
    )
    row = state.rows("tasks")[0]
    assert row["status"] == "PENDING"
    assert row["process_pid"] is None
    assert row["process_start_token"] is None


def test_resume_recovers_pid_reuse_without_touching_new_process(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    replacement = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        state.record_process(
            claim,
            pid=replacement.pid,
            start_token="token-from-exited-original-owner",
        )
        state.verify_resume(
            config_sha256="a" * 64,
            input_manifest_sha256="b" * 64,
            retry_failed=False,
        )
        row = state.rows("tasks")[0]
        assert row["status"] == "PENDING"
        assert row["process_pid"] is None
        assert row["process_start_token"] is None
        assert replacement.poll() is None
    finally:
        replacement.terminate()
        replacement.wait(timeout=3)


def test_resume_refuses_unverifiable_process_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(claim, pid=999_992, start_token="unknown-owner")
    monkeypatch.setattr(
        state_module,
        "inspect_process_identity",
        lambda **_: ProcessIdentityResult(
            pid=999_992,
            status=ProcessIdentityStatus.UNVERIFIABLE,
            reason="permission denied",
        ),
    )

    with pytest.raises(ResumeValidationError, match="cannot be verified"):
        state.verify_resume(
            config_sha256="a" * 64,
            input_manifest_sha256="b" * 64,
            retry_failed=False,
        )
    assert state.rows("tasks")[0]["status"] == "RUNNING"


def test_resume_refuses_running_task_without_complete_process_identity(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    with pytest.raises(ResumeValidationError, match="no complete process ownership"):
        state.verify_resume(
            config_sha256="a" * 64,
            input_manifest_sha256="b" * 64,
            retry_failed=False,
        )
    assert state.rows("tasks")[0]["status"] == "RUNNING"


def test_stop_marks_pending_and_requests_running_stop(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    assert state.request_stop(run_id=RUN_ID) == (claim.task_id,)
    assert state.heartbeat(claim) is True
    assert state.run_row()["status"] == "INTERRUPTED"
    assert all(
        row["status"] == "INTERRUPTED"
        for row in state.rows("tasks")
        if row["task_id"] != claim.task_id
    )


def test_stop_with_durable_exit_remains_resumable_without_database_edit(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    state.record_process(claim, pid=999_985, start_token="exited-before-stop")
    state.record_process_exit(claim, returncode=0)

    assert state.request_stop(run_id=RUN_ID) == (claim.task_id,)
    assert state.owned_processes(run_id=RUN_ID) == []
    state.verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=False,
    )
    row = state.rows("tasks")[0]
    assert row["status"] == "PENDING"
    assert row["process_state"] == "NONE"


def test_heartbeat_write_count_is_persisted_for_external_measurement(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=1)
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    assert state.heartbeat(claim) is False
    assert state.heartbeat(claim) is False
    state.update_progress(claim, stage_reached="NVT")
    assert state.rows("tasks")[0]["heartbeat_write_count"] == 3
