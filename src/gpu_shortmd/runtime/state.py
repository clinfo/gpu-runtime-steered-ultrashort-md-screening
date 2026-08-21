"""Authoritative transactional SQLite state for one local run."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpu_shortmd.runtime.processes import (
    ProcessIdentityStatus,
    inspect_process_identity,
)
from gpu_shortmd.runtime.pruning import (
    PruningTrigger,
    canonical_trigger,
)
from gpu_shortmd.util.logging import utc_now

TERMINAL_TASK_STATUSES = {"COMPLETED", "PRUNED", "SKIPPED"}
RETRYABLE_TASK_STATUSES = {"FAILED", "INTERRUPTED"}
PROCESS_STATES = {"NONE", "RUNNING", "EXITED"}
STATE_SCHEMA_VERSION = 3
TASK_STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "INTERRUPTED",
    "PRUNED",
    "SKIPPED",
}
RUN_SCOPE_POSE_ID = "__RUN__"
POSE_STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "INTERRUPTED",
    "PRUNED",
    "INCOMPLETE",
}


class StateError(RuntimeError):
    """Raised when persistent state violates the runtime contract."""


class StateCompatibilityError(StateError):
    """Raised when the output filesystem cannot safely host the state DB."""


class ResumeValidationError(StateError):
    """Raised when immutable run inputs do not match on resume."""


@dataclass(frozen=True)
class SqliteCapability:
    journal_mode: str
    locking_validated: bool
    warning: str | None


@dataclass(frozen=True)
class TaskClaim:
    task_id: str
    run_id: str
    pose_id: str
    replica_id: str
    velocity_seed: int
    resolved_ntomp: int
    pruning_threshold_angstrom: float | None
    assigned_gpu_id: int
    worker_id: str
    claim_token: str
    stolen_from_gpu_id: int | None
    attempt: int


@dataclass(frozen=True)
class PruningOutcome:
    canonical_trigger: PruningTrigger
    skipped_task_ids: tuple[str, ...]
    stop_requested_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class PoseTaskSpec:
    pose_id: str
    seeds: tuple[int, ...]
    pruning_enabled: bool
    pruning_threshold_angstrom: float | None
    resolved_ntomp: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    pose_manifest_sha256 TEXT NOT NULL,
    input_manifest_sha256 TEXT NOT NULL,
    journal_mode TEXT NOT NULL,
    stop_requested INTEGER NOT NULL DEFAULT 0 CHECK (stop_requested IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resume_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS poses (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    pose_id TEXT NOT NULL,
    status TEXT NOT NULL,
    n_replicas_requested INTEGER NOT NULL CHECK (n_replicas_requested > 0),
    pruning_enabled INTEGER NOT NULL CHECK (pruning_enabled IN (0, 1)),
    pruning_threshold_angstrom REAL,
    trigger_replica_id TEXT,
    trigger_simulation_time_ps REAL,
    trigger_observed_rmsd_angstrom REAL,
    observed_max_rmsd_angstrom REAL,
    md_score_angstrom REAL,
    started_at TEXT,
    finished_at TEXT,
    PRIMARY KEY (run_id, pose_id)
);

CREATE TABLE IF NOT EXISTS replicas (
    run_id TEXT NOT NULL,
    pose_id TEXT NOT NULL,
    replica_id TEXT NOT NULL,
    velocity_seed INTEGER NOT NULL CHECK (velocity_seed > 0),
    resolved_ntomp INTEGER NOT NULL CHECK (resolved_ntomp > 0),
    status TEXT NOT NULL,
    gpu_id INTEGER,
    stage_reached TEXT NOT NULL DEFAULT 'NONE',
    trajectory_time_completed_ps REAL NOT NULL DEFAULT 0,
    max_rmsd_nm REAL,
    max_rmsd_angstrom REAL,
    triggered_pruning INTEGER NOT NULL DEFAULT 0
        CHECK (triggered_pruning IN (0, 1)),
    exit_code INTEGER,
    started_at TEXT,
    finished_at TEXT,
    PRIMARY KEY (run_id, pose_id, replica_id),
    UNIQUE (run_id, velocity_seed),
    FOREIGN KEY (run_id, pose_id)
        REFERENCES poses(run_id, pose_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    pose_id TEXT NOT NULL,
    replica_id TEXT NOT NULL,
    status TEXT NOT NULL,
    assigned_gpu_id INTEGER NOT NULL,
    resolved_ntomp INTEGER NOT NULL CHECK (resolved_ntomp > 0),
    claimed_by TEXT,
    claim_token TEXT,
    claimed_at TEXT,
    heartbeat_at TEXT,
    stop_requested INTEGER NOT NULL DEFAULT 0 CHECK (stop_requested IN (0, 1)),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    process_pid INTEGER,
    process_start_token TEXT,
    process_state TEXT NOT NULL DEFAULT 'NONE'
        CHECK (process_state IN ('NONE', 'RUNNING', 'EXITED')),
    process_returncode INTEGER,
    process_finished_at TEXT,
    process_step_id TEXT,
    heartbeat_write_count INTEGER NOT NULL DEFAULT 0
        CHECK (heartbeat_write_count >= 0),
    FOREIGN KEY (run_id, pose_id, replica_id)
        REFERENCES replicas(run_id, pose_id, replica_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tasks_claim
ON tasks(run_id, status, assigned_gpu_id, task_id);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    pose_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    category TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, artifact_path),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""


def validate_sqlite_filesystem(directory: Path) -> SqliteCapability:
    """Validate creation, transactions, locking, and a documented journal mode."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".gpu-shortmd-sqlite-probe-",
            dir=directory,
        ) as temporary:
            database = Path(temporary) / "probe.sqlite3"
            first = sqlite3.connect(database, timeout=0.0, isolation_level=None)
            second = sqlite3.connect(database, timeout=0.0, isolation_level=None)
            try:
                requested_mode = first.execute("PRAGMA journal_mode=WAL").fetchone()
                journal_mode = str(requested_mode[0]).upper()
                warning: str | None = None
                if journal_mode != "WAL":
                    fallback = first.execute("PRAGMA journal_mode=DELETE").fetchone()
                    journal_mode = str(fallback[0]).upper()
                    if journal_mode != "DELETE":
                        raise StateCompatibilityError(
                            "SQLite WAL and DELETE journal modes are unavailable"
                        )
                    warning = (
                        "SQLite WAL was unavailable; validated DELETE journal mode "
                        "will be used with serialized writers."
                    )
                first.execute("CREATE TABLE probe (value INTEGER NOT NULL)")
                first.execute("BEGIN IMMEDIATE")
                first.execute("INSERT INTO probe VALUES (1)")
                locked = False
                try:
                    second.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    locked = "locked" in str(exc).lower()
                finally:
                    if second.in_transaction:
                        second.rollback()
                first.commit()
                visible = second.execute("SELECT value FROM probe").fetchone()
                if not locked or visible != (1,):
                    raise StateCompatibilityError(
                        "SQLite transaction locking could not be validated"
                    )
                return SqliteCapability(
                    journal_mode=journal_mode,
                    locking_validated=True,
                    warning=warning,
                )
            finally:
                first.close()
                second.close()
    except (OSError, sqlite3.Error) as exc:
        if isinstance(exc, StateCompatibilityError):
            raise
        raise StateCompatibilityError(
            f"SQLite filesystem capability check failed: {exc}"
        ) from exc


class RuntimeState:
    """Small transactional repository around the authoritative SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _ensure_schema_compatibility(self) -> None:
        """Upgrade the legacy task table without guessing ownership."""
        connection = self._connect()
        try:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if not columns:
                return
            additions = {
                "process_state": (
                    "ALTER TABLE tasks ADD COLUMN process_state TEXT NOT NULL "
                    "DEFAULT 'NONE' CHECK (process_state IN "
                    "('NONE', 'RUNNING', 'EXITED'))"
                ),
                "process_returncode": (
                    "ALTER TABLE tasks ADD COLUMN process_returncode INTEGER"
                ),
                "process_finished_at": (
                    "ALTER TABLE tasks ADD COLUMN process_finished_at TEXT"
                ),
                "process_step_id": (
                    "ALTER TABLE tasks ADD COLUMN process_step_id TEXT"
                ),
                "heartbeat_write_count": (
                    "ALTER TABLE tasks ADD COLUMN heartbeat_write_count INTEGER "
                    "NOT NULL DEFAULT 0 CHECK (heartbeat_write_count >= 0)"
                ),
            }
            missing = [name for name in additions if name not in columns]
            if not missing:
                return
            connection.execute("BEGIN IMMEDIATE")
            for name in missing:
                connection.execute(additions[name])
            connection.execute(
                """
                UPDATE tasks
                SET process_state='RUNNING'
                WHERE status='RUNNING'
                  AND process_pid IS NOT NULL
                  AND process_start_token IS NOT NULL
                  AND process_state='NONE'
                """
            )
            connection.execute(f"PRAGMA user_version={STATE_SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        run_id: str,
        pose_id: str,
        seeds: Sequence[int],
        gpu_ids: Sequence[int],
        config_sha256: str,
        input_manifest_sha256: str,
        pruning_enabled: bool,
        pruning_threshold_angstrom: float | None,
        pose_manifest_sha256: str | None = None,
        resolved_ntomp: int = 1,
        capability: SqliteCapability | None = None,
    ) -> RuntimeState:
        return cls.create_batch(
            path,
            run_id=run_id,
            poses=(
                PoseTaskSpec(
                    pose_id=pose_id,
                    seeds=tuple(seeds),
                    pruning_enabled=pruning_enabled,
                    pruning_threshold_angstrom=pruning_threshold_angstrom,
                    resolved_ntomp=resolved_ntomp,
                ),
            ),
            gpu_ids=gpu_ids,
            config_sha256=config_sha256,
            input_manifest_sha256=input_manifest_sha256,
            pose_manifest_sha256=pose_manifest_sha256,
            capability=capability,
        )

    @classmethod
    def create_batch(
        cls,
        path: Path,
        *,
        run_id: str,
        poses: Sequence[PoseTaskSpec],
        gpu_ids: Sequence[int],
        config_sha256: str,
        input_manifest_sha256: str,
        pose_manifest_sha256: str | None = None,
        capability: SqliteCapability | None = None,
    ) -> RuntimeState:
        if path.exists():
            raise StateError("state.sqlite3 already exists")
        if not poses:
            raise StateError("at least one pose is required")
        if not gpu_ids:
            raise StateError("at least one GPU ID is required")
        pose_ids = [pose.pose_id for pose in poses]
        if len(set(pose_ids)) != len(pose_ids):
            raise StateError("pose IDs must be unique within a run")
        all_seeds = [seed for pose in poses for seed in pose.seeds]
        if any(not pose.seeds for pose in poses):
            raise StateError("every pose requires at least one replica seed")
        if len(set(all_seeds)) != len(all_seeds):
            raise StateError("velocity seeds must be unique across the entire run")
        if any(pose.resolved_ntomp < 1 for pose in poses):
            raise StateError("resolved ntomp must be positive for every task")
        capability = capability or validate_sqlite_filesystem(path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = cls(path)
        connection = state._connect()
        try:
            selected_mode = connection.execute(
                f"PRAGMA journal_mode={capability.journal_mode}"
            ).fetchone()
            if str(selected_mode[0]).upper() != capability.journal_mode:
                raise StateCompatibilityError(
                    "selected SQLite journal mode could not be applied"
                )
            connection.executescript(SCHEMA)
            connection.execute(f"PRAGMA user_version={STATE_SCHEMA_VERSION}")
        finally:
            connection.close()

        now = utc_now()
        pose_manifest_digest = pose_manifest_sha256 or config_sha256
        with state._transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO runs (
                    run_id, status, config_sha256, pose_manifest_sha256,
                    input_manifest_sha256, journal_mode, created_at, updated_at
                ) VALUES (?, 'RUNNING', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config_sha256,
                    pose_manifest_digest,
                    input_manifest_sha256,
                    capability.journal_mode,
                    now,
                    now,
                ),
            )
            task_index = 0
            for pose in poses:
                transaction.execute(
                    """
                    INSERT INTO poses (
                        run_id, pose_id, status, n_replicas_requested,
                        pruning_enabled, pruning_threshold_angstrom, started_at
                    ) VALUES (?, ?, 'RUNNING', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        pose.pose_id,
                        len(pose.seeds),
                        int(pose.pruning_enabled),
                        pose.pruning_threshold_angstrom,
                        now,
                    ),
                )
                for index, seed in enumerate(pose.seeds, start=1):
                    replica_id = f"replica_{index:02d}"
                    task_id = f"{pose.pose_id}:{replica_id}"
                    gpu_id = gpu_ids[task_index % len(gpu_ids)]
                    task_index += 1
                    transaction.execute(
                        """
                        INSERT INTO replicas (
                            run_id, pose_id, replica_id, velocity_seed,
                            resolved_ntomp, status
                        ) VALUES (?, ?, ?, ?, ?, 'PENDING')
                        """,
                        (
                            run_id,
                            pose.pose_id,
                            replica_id,
                            seed,
                            pose.resolved_ntomp,
                        ),
                    )
                    transaction.execute(
                        """
                        INSERT INTO tasks (
                            task_id, run_id, pose_id, replica_id, status,
                            assigned_gpu_id, resolved_ntomp
                        ) VALUES (?, ?, ?, ?, 'PENDING', ?, ?)
                        """,
                        (
                            task_id,
                            run_id,
                            pose.pose_id,
                            replica_id,
                            gpu_id,
                            pose.resolved_ntomp,
                        ),
                    )
                state._append_event(
                    transaction,
                    run_id=run_id,
                    pose_id=pose.pose_id,
                    category="STATE",
                    code="POSE_REGISTERED",
                    payload={
                        "pose_id": pose.pose_id,
                        "replica_count": len(pose.seeds),
                        "resolved_ntomp": pose.resolved_ntomp,
                    },
                )
            state._append_event(
                transaction,
                run_id=run_id,
                pose_id=RUN_SCOPE_POSE_ID,
                category="STATE",
                code="STATE_CREATED",
                payload={
                    "pose_ids": pose_ids,
                    "pose_count": len(poses),
                    "replica_count": len(all_seeds),
                    "journal_mode": capability.journal_mode,
                },
            )
        return state

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        pose_id: str,
        category: str,
        code: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO events (
                run_id, pose_id, ts, category, code, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                pose_id,
                utc_now(),
                category,
                code,
                json.dumps(payload, sort_keys=True),
            ),
        )

    def integrity_check(self) -> None:
        try:
            self._ensure_schema_compatibility()
            connection = self._connect()
            try:
                row = connection.execute("PRAGMA integrity_check").fetchone()
                if row is None or row[0] != "ok":
                    raise StateError("SQLite integrity check failed")
                required = {"runs", "poses", "replicas", "tasks", "events", "artifacts"}
                existing = {
                    str(item[0])
                    for item in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if not required <= existing:
                    raise StateError("state database schema is incomplete")
                task_columns = {
                    str(item[1])
                    for item in connection.execute("PRAGMA table_info(tasks)")
                }
                required_task_columns = {
                    "process_pid",
                    "process_start_token",
                    "process_state",
                    "process_returncode",
                    "process_finished_at",
                    "process_step_id",
                    "heartbeat_write_count",
                }
                if not required_task_columns <= task_columns:
                    raise StateError("task lifecycle schema is incomplete")
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise StateError(f"cannot read state database: {exc}") from exc

    def run_row(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM runs").fetchone()
            if row is None:
                raise StateError("state database has no run")
            return dict(row)
        finally:
            connection.close()

    def set_run_status(self, *, run_id: str, status: str) -> None:
        with self._transaction() as transaction:
            changed = transaction.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                (status, utc_now(), run_id),
            ).rowcount
            if changed != 1:
                raise StateError(f"unknown run_id: {run_id}")
            self._append_event(
                transaction,
                run_id=run_id,
                pose_id=RUN_SCOPE_POSE_ID,
                category="STATE",
                code=f"RUN_{status}",
                payload={},
            )

    def verify_resume(
        self,
        *,
        config_sha256: str,
        input_manifest_sha256: str,
        retry_failed: bool,
        pose_manifest_sha256: str | None = None,
    ) -> None:
        self.integrity_check()
        with self._transaction() as transaction:
            row = transaction.execute("SELECT * FROM runs").fetchone()
            if row is None:
                raise ResumeValidationError("state database has no run")
            if row["config_sha256"] != config_sha256:
                raise ResumeValidationError("resolved configuration checksum changed")
            if (
                pose_manifest_sha256 is not None
                and row["pose_manifest_sha256"] != pose_manifest_sha256
            ):
                raise ResumeValidationError("resolved pose manifest checksum changed")
            if row["input_manifest_sha256"] != input_manifest_sha256:
                raise ResumeValidationError("input manifest checksum changed")

            run_id = str(row["run_id"])
            recovered_processes: dict[str, str] = {}
            now = utc_now()
            running_tasks = transaction.execute(
                """
                SELECT tasks.task_id, tasks.pose_id, tasks.replica_id,
                       tasks.process_pid, tasks.process_start_token,
                       tasks.process_state, tasks.process_returncode,
                       tasks.process_finished_at, tasks.process_step_id,
                       tasks.stop_requested, tasks.last_error,
                       poses.status AS pose_status
                FROM tasks
                JOIN poses USING (run_id, pose_id)
                WHERE tasks.run_id=? AND tasks.status='RUNNING'
                ORDER BY tasks.task_id
                """,
                (run_id,),
            ).fetchall()
            for task in running_tasks:
                task_id = str(task["task_id"])
                pid_value = task["process_pid"]
                token_value = task["process_start_token"]
                process_state = str(task["process_state"])
                returncode = task["process_returncode"]
                finished_at = task["process_finished_at"]
                process_step_id = task["process_step_id"]
                if process_state not in PROCESS_STATES:
                    raise ResumeValidationError(
                        f"resume refused: task {task_id} has an unknown process "
                        "lifecycle state"
                    )
                if process_state == "EXITED":
                    if (
                        pid_value is None
                        or token_value is None
                        or returncode is None
                        or finished_at is None
                    ):
                        raise ResumeValidationError(
                            f"resume refused: task {task_id} has incomplete durable "
                            "process exit evidence"
                        )
                    durable_returncode = int(returncode)
                    if durable_returncode == 0:
                        recovered_processes[task_id] = "DURABLE_EXITED_SUCCESS"
                        continue
                    cooperative_evidence: str | None = None
                    if bool(task["stop_requested"]):
                        cooperative_evidence = str(
                            task["last_error"] or "persisted task stop intent"
                        )
                    elif str(task["pose_status"]) == "PRUNED":
                        cooperative_evidence = "persisted pose PRUNED state"
                    elif bool(row["stop_requested"]):
                        cooperative_evidence = "persisted run stop intent"
                    step_label = str(process_step_id or "unknown command step")
                    if cooperative_evidence is not None:
                        interruption_reason = (
                            f"{step_label} exited with code {durable_returncode} "
                            "after cooperative cancellation: "
                            f"{cooperative_evidence}"
                        )
                        transaction.execute(
                            """
                            UPDATE tasks SET last_error=? WHERE task_id=?
                            """,
                            (interruption_reason, task_id),
                        )
                        transaction.execute(
                            """
                            UPDATE replicas
                            SET status='INTERRUPTED', gpu_id=NULL,
                                exit_code=?, finished_at=?
                            WHERE run_id=? AND pose_id=? AND replica_id=?
                            """,
                            (
                                durable_returncode,
                                now,
                                run_id,
                                task["pose_id"],
                                task["replica_id"],
                            ),
                        )
                        self._append_event(
                            transaction,
                            run_id=run_id,
                            pose_id=str(task["pose_id"]),
                            category="STATE",
                            code="DURABLE_COMMAND_EXIT_INTERRUPTED",
                            payload={
                                "task_id": task_id,
                                "returncode": durable_returncode,
                                "process_step_id": process_step_id,
                                "cooperative_evidence": cooperative_evidence,
                            },
                        )
                        recovered_processes[task_id] = "DURABLE_EXITED_INTERRUPTED"
                        continue
                    failure_reason = (
                        f"{step_label} exited with code {durable_returncode} "
                        "without durable cooperative cancellation evidence"
                    )
                    transaction.execute(
                        """
                        UPDATE tasks
                        SET status='FAILED', claimed_by=NULL, claim_token=NULL,
                            claimed_at=NULL, heartbeat_at=?, stop_requested=0,
                            last_error=?
                        WHERE task_id=? AND status='RUNNING'
                        """,
                        (now, failure_reason, task_id),
                    )
                    transaction.execute(
                        """
                        UPDATE replicas
                        SET status='FAILED', gpu_id=NULL,
                            exit_code=?, finished_at=?
                        WHERE run_id=? AND pose_id=? AND replica_id=?
                        """,
                        (
                            durable_returncode,
                            now,
                            run_id,
                            task["pose_id"],
                            task["replica_id"],
                        ),
                    )
                    self._append_event(
                        transaction,
                        run_id=run_id,
                        pose_id=str(task["pose_id"]),
                        category="STATE",
                        code="DURABLE_COMMAND_EXIT_FAILED",
                        payload={
                            "task_id": task_id,
                            "returncode": durable_returncode,
                            "process_step_id": process_step_id,
                            "reason": failure_reason,
                        },
                    )
                    recovered_processes[task_id] = "DURABLE_EXITED_FAILED"
                    continue
                if (
                    process_state != "RUNNING"
                    or pid_value is None
                    or token_value is None
                ):
                    raise ResumeValidationError(
                        f"resume refused: task {task_id} has no complete process "
                        "ownership identity or durable exit evidence; run "
                        "gpu-shortmd stop or resolve the process state before "
                        "retrying"
                    )
                if returncode is not None or finished_at is not None:
                    raise ResumeValidationError(
                        f"resume refused: task {task_id} has contradictory process "
                        "lifecycle evidence"
                    )
                identity = inspect_process_identity(
                    pid=int(pid_value),
                    expected_start_token=str(token_value),
                )
                if identity.status is ProcessIdentityStatus.MATCHING_LIVE:
                    raise ResumeValidationError(
                        f"resume refused: task {task_id} still owns a live process; "
                        "run gpu-shortmd stop on this run directory or resolve the "
                        "process state before retrying"
                    )
                if identity.status is ProcessIdentityStatus.UNVERIFIABLE:
                    raise ResumeValidationError(
                        f"resume refused: task {task_id} process ownership cannot "
                        f"be verified ({identity.reason}); run gpu-shortmd stop or "
                        "resolve the process state before retrying"
                    )
                recovered_processes[task_id] = identity.status.value
            stale = transaction.execute(
                """
                UPDATE tasks
                SET status='INTERRUPTED', claimed_by=NULL, claim_token=NULL,
                    claimed_at=NULL, heartbeat_at=NULL, stop_requested=0,
                    process_pid=NULL, process_start_token=NULL,
                    process_state='NONE', process_returncode=NULL,
                    process_finished_at=NULL, process_step_id=NULL
                WHERE run_id=? AND status='RUNNING'
                """,
                (run_id,),
            ).rowcount
            transaction.execute(
                """
                UPDATE replicas
                SET status='INTERRUPTED', gpu_id=NULL, finished_at=?
                WHERE run_id=? AND status='RUNNING'
                """,
                (now, run_id),
            )
            transaction.execute(
                """
                UPDATE tasks
                SET status='PENDING', last_error=NULL, stop_requested=0
                WHERE run_id=? AND status='INTERRUPTED'
                  AND pose_id IN (
                    SELECT pose_id FROM poses
                    WHERE run_id=? AND status != 'PRUNED'
                  )
                """,
                (run_id, run_id),
            )
            transaction.execute(
                """
                UPDATE replicas
                SET status='PENDING', exit_code=NULL, finished_at=NULL
                WHERE run_id=? AND status='INTERRUPTED'
                  AND pose_id IN (
                    SELECT pose_id FROM poses
                    WHERE run_id=? AND status != 'PRUNED'
                  )
                """,
                (run_id, run_id),
            )
            if retry_failed:
                failed_to_retry = transaction.execute(
                    """
                    SELECT tasks.task_id, tasks.pose_id,
                           tasks.process_returncode, tasks.process_step_id,
                           tasks.last_error, replicas.exit_code
                    FROM tasks
                    JOIN replicas USING (run_id, pose_id, replica_id)
                    JOIN poses USING (run_id, pose_id)
                    WHERE tasks.run_id=? AND tasks.status='FAILED'
                      AND poses.status != 'PRUNED'
                    ORDER BY tasks.task_id
                    """,
                    (run_id,),
                ).fetchall()
                for failed_task in failed_to_retry:
                    retained_returncode = failed_task["process_returncode"]
                    if retained_returncode is None:
                        retained_returncode = failed_task["exit_code"]
                    self._append_event(
                        transaction,
                        run_id=run_id,
                        pose_id=str(failed_task["pose_id"]),
                        category="STATE",
                        code="FAILED_TASK_REQUEUED",
                        payload={
                            "task_id": failed_task["task_id"],
                            "returncode": retained_returncode,
                            "process_step_id": failed_task["process_step_id"],
                            "failure_reason": failed_task["last_error"],
                        },
                    )
                transaction.execute(
                    """
                    UPDATE tasks
                    SET status='PENDING', last_error=NULL, stop_requested=0,
                        process_pid=NULL, process_start_token=NULL,
                        process_state='NONE', process_returncode=NULL,
                        process_finished_at=NULL, process_step_id=NULL
                    WHERE run_id=? AND status='FAILED'
                      AND pose_id IN (
                        SELECT pose_id FROM poses
                        WHERE run_id=? AND status != 'PRUNED'
                      )
                    """,
                    (run_id, run_id),
                )
                transaction.execute(
                    """
                    UPDATE replicas
                    SET status='PENDING', exit_code=NULL, finished_at=NULL
                    WHERE run_id=? AND status='FAILED'
                      AND pose_id IN (
                        SELECT pose_id FROM poses
                        WHERE run_id=? AND status != 'PRUNED'
                      )
                    """,
                    (run_id, run_id),
                )
            transaction.execute(
                """
                UPDATE runs
                SET status='RUNNING', stop_requested=0, updated_at=?,
                    resume_count=resume_count+1
                WHERE run_id=?
                """,
                (now, run_id),
            )
            transaction.execute(
                """
                UPDATE poses
                SET status='RUNNING', finished_at=NULL
                WHERE run_id=? AND status IN ('INTERRUPTED', 'INCOMPLETE', 'FAILED')
                  AND EXISTS (
                    SELECT 1 FROM tasks
                    WHERE tasks.run_id=poses.run_id
                      AND tasks.pose_id=poses.pose_id
                      AND tasks.status='PENDING'
                  )
                """,
                (run_id,),
            )
            self._append_event(
                transaction,
                run_id=run_id,
                pose_id=RUN_SCOPE_POSE_ID,
                category="STATE",
                code="RUN_RESUMED",
                payload={
                    "stale_running_recovered": stale,
                    "recovered_process_identities": recovered_processes,
                    "retry_failed": retry_failed,
                },
            )

    def claim_task(
        self,
        *,
        run_id: str,
        worker_id: str,
        gpu_id: int,
        work_stealing: bool,
    ) -> TaskClaim | None:
        with self._transaction() as transaction:
            run = transaction.execute(
                "SELECT stop_requested FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise StateError(f"unknown run_id: {run_id}")
            if run["stop_requested"]:
                return None

            row = transaction.execute(
                """
                SELECT tasks.*, replicas.velocity_seed,
                       poses.pruning_threshold_angstrom
                FROM tasks
                JOIN replicas USING (run_id, pose_id, replica_id)
                JOIN poses USING (run_id, pose_id)
                WHERE tasks.run_id=? AND tasks.status='PENDING'
                  AND tasks.assigned_gpu_id=? AND poses.status='RUNNING'
                ORDER BY tasks.task_id
                LIMIT 1
                """,
                (run_id, gpu_id),
            ).fetchone()
            stolen_from: int | None = None
            if row is None and work_stealing:
                donor = transaction.execute(
                    """
                    SELECT assigned_gpu_id, COUNT(*) AS pending_count
                    FROM tasks
                    JOIN poses USING (run_id, pose_id)
                    WHERE tasks.run_id=? AND tasks.status='PENDING'
                      AND poses.status='RUNNING'
                    GROUP BY assigned_gpu_id
                    ORDER BY pending_count DESC, assigned_gpu_id
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                if donor is not None:
                    stolen_from = int(donor["assigned_gpu_id"])
                    row = transaction.execute(
                        """
                        SELECT tasks.*, replicas.velocity_seed,
                               poses.pruning_threshold_angstrom
                        FROM tasks
                        JOIN replicas USING (run_id, pose_id, replica_id)
                        JOIN poses USING (run_id, pose_id)
                        WHERE tasks.run_id=? AND tasks.status='PENDING'
                          AND tasks.assigned_gpu_id=? AND poses.status='RUNNING'
                        ORDER BY tasks.task_id
                        LIMIT 1
                        """,
                        (run_id, stolen_from),
                    ).fetchone()
            if row is None:
                return None

            token = uuid.uuid4().hex
            now = utc_now()
            changed = transaction.execute(
                """
                UPDATE tasks
                SET status='RUNNING', claimed_by=?, claim_token=?,
                    claimed_at=?, heartbeat_at=?, attempt_count=attempt_count+1,
                    process_pid=NULL, process_start_token=NULL,
                    process_state='NONE', process_returncode=NULL,
                    process_finished_at=NULL, process_step_id=NULL
                WHERE task_id=? AND status='PENDING'
                """,
                (worker_id, token, now, now, row["task_id"]),
            ).rowcount
            if changed != 1:
                raise StateError("transactional task claim invariant failed")
            transaction.execute(
                """
                UPDATE replicas
                SET status='RUNNING', gpu_id=?, started_at=?,
                    finished_at=NULL
                WHERE run_id=? AND pose_id=? AND replica_id=?
                """,
                (
                    gpu_id,
                    now,
                    run_id,
                    row["pose_id"],
                    row["replica_id"],
                ),
            )
            if stolen_from == gpu_id:
                stolen_from = None
            code = "TASK_STOLEN" if stolen_from is not None else "TASK_CLAIMED"
            self._append_event(
                transaction,
                run_id=run_id,
                pose_id=str(row["pose_id"]),
                category="SCHEDULER",
                code=code,
                payload={
                    "task_id": row["task_id"],
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "stolen_from_gpu_id": stolen_from,
                },
            )
            return TaskClaim(
                task_id=str(row["task_id"]),
                run_id=run_id,
                pose_id=str(row["pose_id"]),
                replica_id=str(row["replica_id"]),
                velocity_seed=int(row["velocity_seed"]),
                resolved_ntomp=int(row["resolved_ntomp"]),
                pruning_threshold_angstrom=(
                    float(row["pruning_threshold_angstrom"])
                    if row["pruning_threshold_angstrom"] is not None
                    else None
                ),
                assigned_gpu_id=int(row["assigned_gpu_id"]),
                worker_id=worker_id,
                claim_token=token,
                stolen_from_gpu_id=stolen_from,
                attempt=int(row["attempt_count"]) + 1,
            )

    def heartbeat(self, claim: TaskClaim) -> bool:
        with self._transaction() as transaction:
            row = transaction.execute(
                """
                SELECT stop_requested FROM tasks
                WHERE task_id=? AND status='RUNNING'
                  AND claimed_by=? AND claim_token=?
                """,
                (claim.task_id, claim.worker_id, claim.claim_token),
            ).fetchone()
            if row is None:
                raise StateError("task ownership was lost")
            transaction.execute(
                """
                UPDATE tasks
                SET heartbeat_at=?, heartbeat_write_count=heartbeat_write_count+1
                WHERE task_id=?
                """,
                (utc_now(), claim.task_id),
            )
            return bool(row["stop_requested"])

    def update_progress(
        self,
        claim: TaskClaim,
        *,
        stage_reached: str,
        trajectory_time_completed_ps: float = 0.0,
        max_rmsd_nm: float | None = None,
        max_rmsd_angstrom: float | None = None,
    ) -> None:
        with self._transaction() as transaction:
            task = transaction.execute(
                """
                SELECT 1 FROM tasks
                WHERE task_id=? AND status='RUNNING'
                  AND claimed_by=? AND claim_token=?
                """,
                (claim.task_id, claim.worker_id, claim.claim_token),
            ).fetchone()
            if task is None:
                raise StateError("progress update does not match current owner")
            transaction.execute(
                """
                UPDATE replicas
                SET stage_reached=?, trajectory_time_completed_ps=?,
                    max_rmsd_nm=COALESCE(?, max_rmsd_nm),
                    max_rmsd_angstrom=COALESCE(?, max_rmsd_angstrom)
                WHERE run_id=? AND pose_id=? AND replica_id=?
                """,
                (
                    stage_reached,
                    trajectory_time_completed_ps,
                    max_rmsd_nm,
                    max_rmsd_angstrom,
                    claim.run_id,
                    claim.pose_id,
                    claim.replica_id,
                ),
            )
            transaction.execute(
                """
                UPDATE tasks
                SET heartbeat_at=?, heartbeat_write_count=heartbeat_write_count+1
                WHERE task_id=?
                """,
                (utc_now(), claim.task_id),
            )
            self._append_event(
                transaction,
                run_id=claim.run_id,
                pose_id=claim.pose_id,
                category="STATE",
                code="REPLICA_PROGRESS",
                payload={
                    "task_id": claim.task_id,
                    "stage_reached": stage_reached,
                    "trajectory_time_completed_ps": trajectory_time_completed_ps,
                },
            )

    def record_process(
        self,
        claim: TaskClaim,
        *,
        pid: int,
        start_token: str,
        process_step_id: str | None = None,
    ) -> None:
        if pid <= 1 or not start_token:
            raise StateError("invalid owned process metadata")
        normalized_step_id = (
            process_step_id.strip() if process_step_id is not None else None
        )
        if process_step_id is not None and not normalized_step_id:
            raise StateError("process step ID must not be empty")
        with self._transaction() as transaction:
            changed = transaction.execute(
                """
                UPDATE tasks
                SET process_pid=?, process_start_token=?,
                    process_state='RUNNING', process_returncode=NULL,
                    process_finished_at=NULL, process_step_id=?
                WHERE task_id=? AND status='RUNNING'
                  AND claimed_by=? AND claim_token=?
                """,
                (
                    pid,
                    start_token,
                    normalized_step_id,
                    claim.task_id,
                    claim.worker_id,
                    claim.claim_token,
                ),
            ).rowcount
            if changed != 1:
                raise StateError("process registration does not match current owner")
            self._append_event(
                transaction,
                run_id=claim.run_id,
                pose_id=claim.pose_id,
                category="STATE",
                code="PROCESS_STARTED",
                payload={
                    "task_id": claim.task_id,
                    "pid": pid,
                    "process_step_id": normalized_step_id,
                },
            )

    def record_process_exit(self, claim: TaskClaim, *, returncode: int) -> None:
        """Durably confirm command exit while retaining its ownership identity."""
        finished_at = utc_now()
        with self._transaction() as transaction:
            process = transaction.execute(
                """
                SELECT process_step_id FROM tasks
                WHERE task_id=? AND status='RUNNING'
                  AND claimed_by=? AND claim_token=?
                  AND process_state='RUNNING'
                  AND process_pid IS NOT NULL
                  AND process_start_token IS NOT NULL
                """,
                (claim.task_id, claim.worker_id, claim.claim_token),
            ).fetchone()
            if process is None:
                raise StateError("process exit does not match current owned process")
            changed = transaction.execute(
                """
                UPDATE tasks
                SET process_state='EXITED', process_returncode=?,
                    process_finished_at=?
                WHERE task_id=? AND status='RUNNING'
                  AND claimed_by=? AND claim_token=?
                  AND process_state='RUNNING'
                  AND process_pid IS NOT NULL
                  AND process_start_token IS NOT NULL
                """,
                (
                    returncode,
                    finished_at,
                    claim.task_id,
                    claim.worker_id,
                    claim.claim_token,
                ),
            ).rowcount
            if changed != 1:
                raise StateError("process exit does not match current owned process")
            self._append_event(
                transaction,
                run_id=claim.run_id,
                pose_id=claim.pose_id,
                category="STATE",
                code="PROCESS_EXITED",
                payload={
                    "task_id": claim.task_id,
                    "returncode": returncode,
                    "process_step_id": process["process_step_id"],
                },
            )

    def owned_processes(self, *, run_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT task_id, claimed_by, claim_token,
                           process_pid, process_start_token, process_state
                    FROM tasks
                    WHERE run_id=? AND status='RUNNING'
                      AND process_state='RUNNING'
                      AND process_pid IS NOT NULL
                      AND process_start_token IS NOT NULL
                    ORDER BY task_id
                    """,
                    (run_id,),
                )
            ]
        finally:
            connection.close()

    def replace_artifacts(
        self,
        *,
        run_id: str,
        artifacts: Sequence[dict[str, Any]],
    ) -> None:
        with self._transaction() as transaction:
            transaction.execute(
                "DELETE FROM artifacts WHERE run_id=?",
                (run_id,),
            )
            for artifact in artifacts:
                transaction.execute(
                    """
                    INSERT INTO artifacts (
                        run_id, artifact_path, sha256, size_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(artifact["path"]),
                        str(artifact["sha256"]),
                        int(artifact["size_bytes"]),
                        utc_now(),
                    ),
                )

    def finish_task(
        self,
        claim: TaskClaim,
        *,
        status: str,
        stage_reached: str,
        trajectory_time_completed_ps: float,
        max_rmsd_nm: float | None,
        max_rmsd_angstrom: float | None,
        exit_code: int,
        error: str | None = None,
    ) -> None:
        if status not in TASK_STATUSES - {"PENDING", "RUNNING", "SKIPPED"}:
            raise StateError(f"invalid task completion status: {status}")
        now = utc_now()
        with self._transaction() as transaction:
            row = transaction.execute(
                """
                SELECT run_id, pose_id, replica_id FROM tasks
                WHERE task_id=? AND status='RUNNING'
                  AND claimed_by=? AND claim_token=?
                """,
                (claim.task_id, claim.worker_id, claim.claim_token),
            ).fetchone()
            if row is None:
                raise StateError("task completion does not match current owner")
            transaction.execute(
                """
                UPDATE tasks
                SET status=?, claimed_by=NULL, claim_token=NULL,
                    heartbeat_at=?, stop_requested=0, last_error=?,
                    process_pid=NULL, process_start_token=NULL,
                    process_state='NONE', process_returncode=NULL,
                    process_finished_at=NULL, process_step_id=NULL
                WHERE task_id=?
                """,
                (status, now, error, claim.task_id),
            )
            transaction.execute(
                """
                UPDATE replicas
                SET status=?, stage_reached=?,
                    trajectory_time_completed_ps=?,
                    max_rmsd_nm=?, max_rmsd_angstrom=?,
                    exit_code=?, finished_at=?
                WHERE run_id=? AND pose_id=? AND replica_id=?
                """,
                (
                    status,
                    stage_reached,
                    trajectory_time_completed_ps,
                    max_rmsd_nm,
                    max_rmsd_angstrom,
                    exit_code,
                    now,
                    row["run_id"],
                    row["pose_id"],
                    row["replica_id"],
                ),
            )
            self._append_event(
                transaction,
                run_id=str(row["run_id"]),
                pose_id=str(row["pose_id"]),
                category="STATE",
                code=f"TASK_{status}",
                payload={"task_id": claim.task_id, "exit_code": exit_code},
            )

    def trigger_pruning(
        self,
        claim: TaskClaim,
        *,
        trigger: PruningTrigger,
    ) -> PruningOutcome:
        with self._transaction() as transaction:
            pose = transaction.execute(
                """
                SELECT * FROM poses
                WHERE run_id=? AND pose_id=?
                """,
                (claim.run_id, claim.pose_id),
            ).fetchone()
            if pose is None:
                raise StateError("pruning target pose does not exist")
            if not pose["pruning_enabled"]:
                raise StateError("pruning is disabled for this pose")
            threshold = pose["pruning_threshold_angstrom"]
            if threshold is None or trigger.observed_rmsd_angstrom <= threshold:
                raise StateError("pruning trigger does not exceed the threshold")
            if pose["status"] not in {"RUNNING", "PRUNED"}:
                raise StateError(f"cannot prune pose in {pose['status']} state")
            first_transition = pose["status"] == "RUNNING"

            current: PruningTrigger | None = None
            if pose["trigger_replica_id"] is not None:
                current = PruningTrigger(
                    simulation_time_ps=float(pose["trigger_simulation_time_ps"]),
                    replica_id=str(pose["trigger_replica_id"]),
                    observed_rmsd_angstrom=float(
                        pose["trigger_observed_rmsd_angstrom"]
                    ),
                )
            canonical = canonical_trigger(current, trigger)
            transaction.execute(
                """
                UPDATE poses
                SET status='PRUNED', md_score_angstrom=NULL,
                    observed_max_rmsd_angstrom=MAX(
                        COALESCE(observed_max_rmsd_angstrom, 0), ?
                    ),
                    trigger_replica_id=?,
                    trigger_simulation_time_ps=?,
                    trigger_observed_rmsd_angstrom=?,
                    finished_at=COALESCE(finished_at, ?)
                WHERE run_id=? AND pose_id=?
                """,
                (
                    trigger.observed_rmsd_angstrom,
                    canonical.replica_id,
                    canonical.simulation_time_ps,
                    canonical.observed_rmsd_angstrom,
                    utc_now(),
                    claim.run_id,
                    claim.pose_id,
                ),
            )
            pending: list[sqlite3.Row] = []
            running: list[sqlite3.Row] = []
            if first_transition:
                pending = transaction.execute(
                    """
                    SELECT task_id FROM tasks
                    WHERE run_id=? AND pose_id=? AND status='PENDING'
                    ORDER BY task_id
                    """,
                    (claim.run_id, claim.pose_id),
                ).fetchall()
                transaction.execute(
                    """
                    UPDATE tasks
                    SET status='SKIPPED', last_error='pose pruned'
                    WHERE run_id=? AND pose_id=? AND status='PENDING'
                    """,
                    (claim.run_id, claim.pose_id),
                )
                transaction.execute(
                    """
                    UPDATE replicas
                    SET status='SKIPPED', finished_at=?
                    WHERE run_id=? AND pose_id=? AND status='PENDING'
                    """,
                    (utc_now(), claim.run_id, claim.pose_id),
                )
                running = transaction.execute(
                    """
                    SELECT task_id FROM tasks
                    WHERE run_id=? AND pose_id=? AND status='RUNNING'
                      AND task_id != ?
                    ORDER BY task_id
                    """,
                    (claim.run_id, claim.pose_id, claim.task_id),
                ).fetchall()
                transaction.execute(
                    """
                    UPDATE tasks
                    SET stop_requested=1, last_error='sibling triggered pruning'
                    WHERE run_id=? AND pose_id=? AND status='RUNNING'
                      AND task_id != ?
                    """,
                    (claim.run_id, claim.pose_id, claim.task_id),
                )
            transaction.execute(
                """
                UPDATE replicas
                SET triggered_pruning=1
                WHERE run_id=? AND pose_id=? AND replica_id=?
                """,
                (claim.run_id, claim.pose_id, trigger.replica_id),
            )
            self._append_event(
                transaction,
                run_id=claim.run_id,
                pose_id=claim.pose_id,
                category="PRUNING",
                code=(
                    "POSE_PRUNED" if first_transition else "PRUNING_TRIGGER_OBSERVED"
                ),
                payload={
                    "pose_id": claim.pose_id,
                    "observed_trigger": asdict(trigger),
                    "canonical_trigger": asdict(canonical),
                    "skipped_task_ids": [str(row["task_id"]) for row in pending],
                    "stop_requested_task_ids": [str(row["task_id"]) for row in running],
                },
            )
            return PruningOutcome(
                canonical_trigger=canonical,
                skipped_task_ids=tuple(str(row["task_id"]) for row in pending),
                stop_requested_task_ids=tuple(str(row["task_id"]) for row in running),
            )

    def request_stop(self, *, run_id: str) -> tuple[str, ...]:
        now = utc_now()
        with self._transaction() as transaction:
            run = transaction.execute(
                "SELECT status FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise StateError(f"unknown run_id: {run_id}")
            transaction.execute(
                """
                UPDATE runs
                SET stop_requested=1, status='INTERRUPTED', updated_at=?
                WHERE run_id=?
                """,
                (now, run_id),
            )
            transaction.execute(
                """
                UPDATE tasks
                SET status='INTERRUPTED', last_error='user stop requested'
                WHERE run_id=? AND status='PENDING'
                """,
                (run_id,),
            )
            transaction.execute(
                """
                UPDATE replicas
                SET status='INTERRUPTED', finished_at=?
                WHERE run_id=? AND status='PENDING'
                """,
                (now, run_id),
            )
            running = transaction.execute(
                """
                SELECT task_id FROM tasks
                WHERE run_id=? AND status='RUNNING'
                ORDER BY task_id
                """,
                (run_id,),
            ).fetchall()
            transaction.execute(
                """
                UPDATE tasks
                SET stop_requested=1, last_error='user stop requested'
                WHERE run_id=? AND status='RUNNING'
                """,
                (run_id,),
            )
            transaction.execute(
                """
                UPDATE poses
                SET status='INTERRUPTED', finished_at=?
                WHERE run_id=? AND status IN ('PENDING', 'RUNNING')
                """,
                (now, run_id),
            )
            self._append_event(
                transaction,
                run_id=run_id,
                pose_id=RUN_SCOPE_POSE_ID,
                category="STATE",
                code="STOP_REQUESTED",
                payload={"running_task_ids": [str(row["task_id"]) for row in running]},
            )
            return tuple(str(row["task_id"]) for row in running)

    def finalize_pose(self, *, run_id: str, pose_id: str) -> str:
        now = utc_now()
        with self._transaction() as transaction:
            pose = transaction.execute(
                "SELECT * FROM poses WHERE run_id=? AND pose_id=?",
                (run_id, pose_id),
            ).fetchone()
            if pose is None:
                raise StateError("pose does not exist")
            rows = transaction.execute(
                """
                SELECT status, max_rmsd_angstrom FROM replicas
                WHERE run_id=? AND pose_id=?
                ORDER BY replica_id
                """,
                (run_id, pose_id),
            ).fetchall()
            maxima = [
                float(row["max_rmsd_angstrom"])
                for row in rows
                if row["max_rmsd_angstrom"] is not None
            ]
            observed = max(maxima) if maxima else pose["observed_max_rmsd_angstrom"]
            statuses = [str(row["status"]) for row in rows]
            if pose["status"] == "PRUNED":
                status = "PRUNED"
                score = None
            elif statuses and all(item == "COMPLETED" for item in statuses):
                status = "COMPLETED"
                score = observed
            elif any(item == "FAILED" for item in statuses):
                status = "FAILED"
                score = None
            elif any(item == "INTERRUPTED" for item in statuses):
                status = "INTERRUPTED"
                score = None
            else:
                status = "INCOMPLETE"
                score = None
            transaction.execute(
                """
                UPDATE poses
                SET status=?, observed_max_rmsd_angstrom=?,
                    md_score_angstrom=?, finished_at=?
                WHERE run_id=? AND pose_id=?
                """,
                (status, observed, score, now, run_id, pose_id),
            )
            self._append_event(
                transaction,
                run_id=run_id,
                pose_id=pose_id,
                category="STATE",
                code="POSE_FINALIZED",
                payload={"pose_id": pose_id, "status": status},
            )
            return status

    def finalize_run(self, *, run_id: str) -> str:
        now = utc_now()
        with self._transaction() as transaction:
            rows = transaction.execute(
                """
                SELECT pose_id, status FROM poses
                WHERE run_id=? ORDER BY pose_id
                """,
                (run_id,),
            ).fetchall()
            if not rows:
                raise StateError("run has no poses")
            statuses = [str(row["status"]) for row in rows]
            if len(statuses) == 1:
                status = statuses[0]
            elif any(item == "FAILED" for item in statuses):
                status = "FAILED"
            elif any(item == "INTERRUPTED" for item in statuses):
                status = "INTERRUPTED"
            elif any(item == "INCOMPLETE" for item in statuses):
                status = "INCOMPLETE"
            elif all(item == "PRUNED" for item in statuses):
                status = "PRUNED"
            elif all(item in {"COMPLETED", "PRUNED"} for item in statuses):
                status = "COMPLETED"
            else:
                status = "INCOMPLETE"
            changed = transaction.execute(
                """
                UPDATE runs SET status=?, updated_at=?
                WHERE run_id=?
                """,
                (status, now, run_id),
            ).rowcount
            if changed != 1:
                raise StateError(f"unknown run_id: {run_id}")
            self._append_event(
                transaction,
                run_id=run_id,
                pose_id=RUN_SCOPE_POSE_ID,
                category="STATE",
                code="RUN_FINALIZED",
                payload={
                    "status": status,
                    "poses": {str(row["pose_id"]): str(row["status"]) for row in rows},
                },
            )
            return status

    def rows(self, table: str) -> list[dict[str, Any]]:
        if table not in {"runs", "poses", "replicas", "tasks", "events", "artifacts"}:
            raise ValueError("unsupported state table")
        connection = self._connect()
        try:
            return [
                dict(row)
                for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ]
        finally:
            connection.close()

    def event_codes(self) -> list[str]:
        return [str(row["code"]) for row in self.rows("events")]
