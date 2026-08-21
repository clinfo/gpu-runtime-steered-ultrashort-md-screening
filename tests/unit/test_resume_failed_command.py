from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_shortmd.runtime.pruning import PruningTrigger
from gpu_shortmd.runtime.state import RuntimeState, TaskClaim

RUN_ID = "RUN-20260801-000000-failed-command"
POSE_ID = "P00533_EGFR|p2|model4"


def create_state(
    tmp_path: Path,
    *,
    replicas: int = 1,
    pruning: bool = False,
) -> RuntimeState:
    return RuntimeState.create(
        tmp_path / "state.sqlite3",
        run_id=RUN_ID,
        pose_id=POSE_ID,
        seeds=tuple(range(2026080101, 2026080101 + replicas)),
        gpu_ids=(0, 1),
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        pruning_enabled=pruning,
        pruning_threshold_angstrom=4.0 if pruning else None,
    )


def claim_task(state: RuntimeState, *, gpu_id: int = 0) -> TaskClaim:
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id=f"gpu-{gpu_id}",
        gpu_id=gpu_id,
        work_stealing=False,
    )
    assert claim is not None
    return claim


def record_durable_exit(
    state: RuntimeState,
    claim: TaskClaim,
    *,
    returncode: int,
    process_step_id: str = "replica_01.nvt.grompp",
) -> None:
    state.record_process(
        claim,
        pid=999_980 + claim.assigned_gpu_id,
        start_token=f"owner-{claim.task_id}",
        process_step_id=process_step_id,
    )
    state.record_process_exit(claim, returncode=returncode)


def event_payloads(state: RuntimeState, code: str) -> list[dict[str, Any]]:
    return [
        json.loads(str(row["payload_json"]))
        for row in state.rows("events")
        if row["code"] == code
    ]


def resume(state: RuntimeState, *, retry_failed: bool) -> None:
    state.verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=retry_failed,
    )


def test_durable_successful_exit_resumes_as_pending(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    claim = claim_task(state)
    record_durable_exit(state, claim, returncode=0)

    resume(state, retry_failed=False)

    task = state.rows("tasks")[0]
    assert task["status"] == "PENDING"
    assert task["process_state"] == "NONE"
    assert task["process_returncode"] is None
    assert task["process_step_id"] is None


def test_durable_failed_exit_stays_failed_without_retry(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    claim = claim_task(state)
    record_durable_exit(state, claim, returncode=1)

    resume(state, retry_failed=False)

    task = state.rows("tasks")[0]
    replica = state.rows("replicas")[0]
    assert task["status"] == "FAILED"
    assert task["process_state"] == "EXITED"
    assert task["process_returncode"] == 1
    assert task["process_step_id"] == "replica_01.nvt.grompp"
    assert "without durable cooperative cancellation evidence" in task["last_error"]
    assert replica["status"] == "FAILED"
    assert replica["exit_code"] == 1
    payload = event_payloads(state, "DURABLE_COMMAND_EXIT_FAILED")[0]
    assert payload["returncode"] == 1
    assert payload["process_step_id"] == "replica_01.nvt.grompp"
    assert (
        state.claim_task(
            run_id=RUN_ID,
            worker_id="gpu-0-retry",
            gpu_id=0,
            work_stealing=False,
        )
        is None
    )


def test_durable_failed_exit_requeues_only_with_explicit_retry(tmp_path: Path) -> None:
    state = create_state(tmp_path)
    claim = claim_task(state)
    record_durable_exit(
        state,
        claim,
        returncode=1,
        process_step_id="replica_01.production.mdrun",
    )

    resume(state, retry_failed=True)

    task = state.rows("tasks")[0]
    replica = state.rows("replicas")[0]
    assert task["status"] == "PENDING"
    assert task["process_state"] == "NONE"
    assert task["process_pid"] is None
    assert task["process_start_token"] is None
    assert task["process_returncode"] is None
    assert task["process_finished_at"] is None
    assert task["process_step_id"] is None
    assert replica["status"] == "PENDING"
    assert replica["exit_code"] is None
    failed = event_payloads(state, "DURABLE_COMMAND_EXIT_FAILED")[0]
    retried = event_payloads(state, "FAILED_TASK_REQUEUED")[0]
    assert failed["returncode"] == 1
    assert retried["returncode"] == 1
    assert retried["process_step_id"] == "replica_01.production.mdrun"
    assert (
        "without durable cooperative cancellation evidence" in retried["failure_reason"]
    )


def test_nonzero_exit_with_user_stop_is_cooperative_interruption(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path)
    claim = claim_task(state)
    record_durable_exit(
        state,
        claim,
        returncode=143,
        process_step_id="replica_01.production.mdrun",
    )
    assert state.request_stop(run_id=RUN_ID) == (claim.task_id,)

    resume(state, retry_failed=False)

    task = state.rows("tasks")[0]
    assert task["status"] == "PENDING"
    assert "DURABLE_COMMAND_EXIT_FAILED" not in state.event_codes()
    payload = event_payloads(state, "DURABLE_COMMAND_EXIT_INTERRUPTED")[0]
    assert payload["returncode"] == 143
    assert payload["cooperative_evidence"] == "user stop requested"


def test_nonzero_sibling_exit_on_pruned_pose_is_not_failed_or_requeued(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path, replicas=2, pruning=True)
    trigger = claim_task(state, gpu_id=0)
    sibling = claim_task(state, gpu_id=1)
    state.record_process(
        sibling,
        pid=999_981,
        start_token="pruned-sibling-owner",
        process_step_id="replica_02.production.mdrun",
    )
    state.trigger_pruning(
        trigger,
        trigger=PruningTrigger(25.0, trigger.replica_id, 4.5),
    )
    state.record_process_exit(sibling, returncode=143)
    state.finish_task(
        trigger,
        status="PRUNED",
        stage_reached="PRODUCTION",
        trajectory_time_completed_ps=25.0,
        max_rmsd_nm=0.45,
        max_rmsd_angstrom=4.5,
        exit_code=0,
    )

    resume(state, retry_failed=True)

    tasks = {row["task_id"]: row for row in state.rows("tasks")}
    replicas = {row["replica_id"]: row for row in state.rows("replicas")}
    sibling_task = tasks[sibling.task_id]
    sibling_replica = replicas[sibling.replica_id]
    assert sibling_task["status"] == "INTERRUPTED"
    assert sibling_task["process_state"] == "NONE"
    assert sibling_replica["status"] == "INTERRUPTED"
    assert sibling_replica["exit_code"] == 143
    assert "DURABLE_COMMAND_EXIT_FAILED" not in state.event_codes()
    assert "FAILED_TASK_REQUEUED" not in state.event_codes()
    payload = event_payloads(state, "DURABLE_COMMAND_EXIT_INTERRUPTED")[0]
    assert payload["cooperative_evidence"] == "sibling triggered pruning"
    assert state.rows("poses")[0]["status"] == "PRUNED"


def test_signal_like_returncode_without_stop_evidence_is_failed(
    tmp_path: Path,
) -> None:
    state = create_state(tmp_path)
    claim = claim_task(state)
    record_durable_exit(
        state,
        claim,
        returncode=-15,
        process_step_id="replica_01.analysis.rmsd",
    )

    resume(state, retry_failed=False)

    task = state.rows("tasks")[0]
    assert task["status"] == "FAILED"
    assert task["process_returncode"] == -15
    assert state.rows("replicas")[0]["exit_code"] == -15
    payload = event_payloads(state, "DURABLE_COMMAND_EXIT_FAILED")[0]
    assert payload["returncode"] == -15
    assert payload["process_step_id"] == "replica_01.analysis.rmsd"
