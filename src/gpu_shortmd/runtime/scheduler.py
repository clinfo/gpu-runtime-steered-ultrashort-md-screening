"""Single-node worker-per-GPU scheduler with transactional work stealing."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from gpu_shortmd.runtime.pruning import evaluate_pruning
from gpu_shortmd.runtime.state import RuntimeState, StateError, TaskClaim


class TaskInterrupted(RuntimeError):
    """Raised by an executor after a cooperative stop/prune request."""


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    stage_reached: str
    trajectory_time_completed_ps: float
    max_rmsd_nm: float | None
    max_rmsd_angstrom: float | None
    exit_code: int


class ReplicaExecutor(Protocol):
    def __call__(
        self,
        claim: TaskClaim,
        control: TaskControl,
    ) -> ExecutionResult: ...


@dataclass(frozen=True)
class SchedulerResult:
    claimed_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    interrupted_task_ids: tuple[str, ...]
    stolen_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class DryRunPoseSpec:
    pose_id: str
    seeds: tuple[int, ...]
    pruning_enabled: bool
    pruning_threshold_angstrom: float | None
    input_checksums: dict[str, str]


def available_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
        except OSError:
            count = 0
        if count > 0:
            return count
    return os.cpu_count() or 1


def resolve_ntomp(
    value: int | str,
    *,
    gpu_ids: Sequence[int],
    ntmpi: int,
    cpu_count: int | None = None,
) -> int:
    if not gpu_ids:
        raise ValueError("ntomp resolution requires at least one GPU")
    if ntmpi < 1:
        raise ValueError("ntmpi must be positive")
    available = cpu_count if cpu_count is not None else available_cpu_count()
    if available < 1:
        raise ValueError("available CPU count must be positive")
    process_count = len(gpu_ids) * ntmpi
    if value == "auto":
        resolved = available // process_count
        if resolved < 1:
            raise ValueError(
                "selected GPUs and ntmpi exceed the available CPU allocation"
            )
        return resolved
    if not isinstance(value, int) or value < 1:
        raise ValueError("ntomp must be a positive integer or auto")
    if process_count * value > available:
        raise ValueError(
            "configured ntomp would oversubscribe the available CPU allocation"
        )
    return value


class TaskControl:
    """Narrow executor API for heartbeat, stop, and pruning observations."""

    def __init__(
        self,
        *,
        state: RuntimeState,
        claim: TaskClaim,
        pruning_threshold_angstrom: float | None,
    ) -> None:
        self._state = state
        self.claim = claim
        self.pruning_threshold_angstrom = pruning_threshold_angstrom
        self.triggered_pruning = False

    def stop_requested(self) -> bool:
        return self._state.heartbeat(self.claim)

    def checkpoint(self) -> None:
        if self.stop_requested():
            raise TaskInterrupted("task stop was requested")

    def update_progress(
        self,
        *,
        stage_reached: str,
        trajectory_time_completed_ps: float = 0.0,
        max_rmsd_nm: float | None = None,
        max_rmsd_angstrom: float | None = None,
    ) -> None:
        self._state.update_progress(
            self.claim,
            stage_reached=stage_reached,
            trajectory_time_completed_ps=trajectory_time_completed_ps,
            max_rmsd_nm=max_rmsd_nm,
            max_rmsd_angstrom=max_rmsd_angstrom,
        )

    def record_process(
        self,
        *,
        pid: int,
        start_token: str,
        process_step_id: str,
    ) -> None:
        self._state.record_process(
            self.claim,
            pid=pid,
            start_token=start_token,
            process_step_id=process_step_id,
        )

    def record_process_exit(self, returncode: int) -> None:
        self._state.record_process_exit(self.claim, returncode=returncode)

    def request_run_stop(self) -> tuple[str, ...]:
        return self._state.request_stop(run_id=self.claim.run_id)

    def observe_rmsd(
        self,
        *,
        simulation_time_ps: float,
        rmsd_angstrom: float,
    ) -> bool:
        """Persist one caller-selected pruning crossing.

        Callers scan samples in memory and checkpoint once at their polling or
        analysis boundary before invoking this method.
        """
        if self.pruning_threshold_angstrom is None:
            return False
        trigger = evaluate_pruning(
            replica_id=self.claim.replica_id,
            simulation_time_ps=simulation_time_ps,
            observed_rmsd_angstrom=rmsd_angstrom,
            threshold_angstrom=self.pruning_threshold_angstrom,
        )
        if trigger is None:
            return False
        self._state.trigger_pruning(self.claim, trigger=trigger)
        self.triggered_pruning = True
        return True


def build_dry_run_plan(
    *,
    run_id: str,
    pose_id: str,
    seeds: Sequence[int],
    gpu_ids: Sequence[int],
    stages: Sequence[str],
    pruning_enabled: bool,
    pruning_threshold_angstrom: float | None,
    work_stealing: bool,
    input_checksums: dict[str, str],
    resolved_ntomp: int = 1,
) -> dict[str, object]:
    if not gpu_ids:
        raise ValueError("dry-run plan requires at least one GPU")
    tasks = [
        {
            "task_id": f"{pose_id}:replica_{index:02d}",
            "replica_id": f"replica_{index:02d}",
            "velocity_seed": seed,
            "resolved_ntomp": resolved_ntomp,
            "initial_gpu_id": gpu_ids[(index - 1) % len(gpu_ids)],
            "eligible_for_work_stealing": work_stealing,
        }
        for index, seed in enumerate(seeds, start=1)
    ]
    return {
        "schema_version": 1,
        "plan_type": "local_single_node",
        "run_id": run_id,
        "pose_id": pose_id,
        "gpu_ids": list(gpu_ids),
        "worker_count": len(gpu_ids),
        "worker_model": "one_persistent_worker_per_gpu",
        "resolved_ntomp_per_task": resolved_ntomp,
        "tasks": tasks,
        "stages": list(stages),
        "pruning": {
            "enabled": pruning_enabled,
            "threshold_angstrom": pruning_threshold_angstrom,
            "strict_operator": ">",
        },
        "work_stealing": work_stealing,
        "input_checksums": dict(sorted(input_checksums.items())),
        "launches_external_processes": False,
    }


def build_multi_pose_dry_run_plan(
    *,
    run_id: str,
    poses: Sequence[DryRunPoseSpec],
    gpu_ids: Sequence[int],
    stages: Sequence[str],
    work_stealing: bool,
    resolved_ntomp: int,
    available_cpus: int,
) -> dict[str, object]:
    if not poses:
        raise ValueError("dry-run plan requires at least one pose")
    if not gpu_ids:
        raise ValueError("dry-run plan requires at least one GPU")
    tasks: list[dict[str, object]] = []
    task_index = 0
    pose_payload: list[dict[str, object]] = []
    for pose in poses:
        pose_payload.append(
            {
                "pose_id": pose.pose_id,
                "replica_count": len(pose.seeds),
                "pruning": {
                    "enabled": pose.pruning_enabled,
                    "threshold_angstrom": pose.pruning_threshold_angstrom,
                    "strict_operator": ">",
                },
                "input_checksums": dict(sorted(pose.input_checksums.items())),
            }
        )
        for index, seed in enumerate(pose.seeds, start=1):
            tasks.append(
                {
                    "task_id": f"{pose.pose_id}:replica_{index:02d}",
                    "pose_id": pose.pose_id,
                    "replica_id": f"replica_{index:02d}",
                    "velocity_seed": seed,
                    "resolved_ntomp": resolved_ntomp,
                    "initial_gpu_id": gpu_ids[task_index % len(gpu_ids)],
                    "eligible_for_work_stealing": work_stealing,
                }
            )
            task_index += 1
    return {
        "schema_version": 2,
        "plan_type": "local_single_node_multi_pose",
        "run_id": run_id,
        "pose_ids": [pose.pose_id for pose in poses],
        "gpu_ids": list(gpu_ids),
        "worker_count": len(gpu_ids),
        "worker_model": "one_persistent_worker_per_gpu",
        "tasks_per_gpu": 1,
        "resolved_ntomp_per_task": resolved_ntomp,
        "available_cpu_count": available_cpus,
        "tasks": tasks,
        "poses": pose_payload,
        "stages": list(stages),
        "work_stealing": work_stealing,
        "launches_external_processes": False,
    }


def run_local_scheduler(
    *,
    state: RuntimeState,
    run_id: str,
    gpu_ids: Sequence[int],
    work_stealing: bool,
    pruning_threshold_angstrom: float | None,
    executor: ReplicaExecutor,
) -> SchedulerResult:
    """Run one persistent thread per GPU; SQLite serializes all task claims."""
    if not gpu_ids:
        raise ValueError("scheduler requires at least one GPU")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("scheduler GPU IDs must be unique")

    lock = threading.Lock()
    claimed: list[str] = []
    completed: list[str] = []
    failed: list[str] = []
    interrupted: list[str] = []
    stolen: list[str] = []
    worker_errors: list[BaseException] = []

    def worker(gpu_id: int) -> None:
        worker_id = f"gpu-{gpu_id}"
        while True:
            try:
                claim = state.claim_task(
                    run_id=run_id,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    work_stealing=work_stealing,
                )
                if claim is None:
                    return
                with lock:
                    claimed.append(claim.task_id)
                    if claim.stolen_from_gpu_id is not None:
                        stolen.append(claim.task_id)
                control = TaskControl(
                    state=state,
                    claim=claim,
                    pruning_threshold_angstrom=(
                        claim.pruning_threshold_angstrom
                        if claim.pruning_threshold_angstrom is not None
                        else pruning_threshold_angstrom
                    ),
                )
                try:
                    result = executor(claim, control)
                    status = "PRUNED" if control.triggered_pruning else result.status
                    state.finish_task(
                        claim,
                        status=status,
                        stage_reached=result.stage_reached,
                        trajectory_time_completed_ps=(
                            result.trajectory_time_completed_ps
                        ),
                        max_rmsd_nm=result.max_rmsd_nm,
                        max_rmsd_angstrom=result.max_rmsd_angstrom,
                        exit_code=result.exit_code,
                    )
                    with lock:
                        if status in {"COMPLETED", "PRUNED"}:
                            completed.append(claim.task_id)
                        elif status == "INTERRUPTED":
                            interrupted.append(claim.task_id)
                        else:
                            failed.append(claim.task_id)
                except TaskInterrupted as exc:
                    state.finish_task(
                        claim,
                        status="INTERRUPTED",
                        stage_reached="INTERRUPTED",
                        trajectory_time_completed_ps=0.0,
                        max_rmsd_nm=None,
                        max_rmsd_angstrom=None,
                        exit_code=8,
                        error=str(exc),
                    )
                    with lock:
                        interrupted.append(claim.task_id)
                except Exception as exc:
                    state.finish_task(
                        claim,
                        status="FAILED",
                        stage_reached="FAILED",
                        trajectory_time_completed_ps=0.0,
                        max_rmsd_nm=None,
                        max_rmsd_angstrom=None,
                        exit_code=5,
                        error=f"{type(exc).__name__}: execution failed",
                    )
                    with lock:
                        failed.append(claim.task_id)
            except BaseException as exc:
                with lock:
                    worker_errors.append(exc)
                return

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id,),
            name=f"gpu-shortmd-worker-{gpu_id}",
        )
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if worker_errors:
        first = worker_errors[0]
        if isinstance(first, StateError):
            raise first
        raise RuntimeError(
            f"scheduler worker failed: {type(first).__name__}: {first}"
        ) from first
    return SchedulerResult(
        claimed_task_ids=tuple(claimed),
        completed_task_ids=tuple(completed),
        failed_task_ids=tuple(failed),
        interrupted_task_ids=tuple(interrupted),
        stolen_task_ids=tuple(stolen),
    )
