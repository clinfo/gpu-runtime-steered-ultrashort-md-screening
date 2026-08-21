"""Export authoritative SQLite state to the stable CSV contracts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gpu_shortmd.runtime.state import RuntimeState

POSE_FIELDS = [
    "run_id",
    "pose_id",
    "status",
    "n_replicas_requested",
    "n_replicas_completed",
    "md_score_angstrom",
    "observed_max_rmsd_angstrom",
    "pruning_enabled",
    "pruning_threshold_angstrom",
    "trigger_replica_id",
    "trigger_simulation_time_ps",
    "docking_score_kcal_mol",
    "started_at",
    "finished_at",
]

REPLICA_FIELDS = [
    "run_id",
    "pose_id",
    "replica_id",
    "status",
    "gpu_id",
    "velocity_seed",
    "resolved_ntomp",
    "stage_reached",
    "trajectory_time_completed_ps",
    "max_rmsd_nm",
    "max_rmsd_angstrom",
    "triggered_pruning",
    "exit_code",
    "started_at",
    "finished_at",
]


def _write_rows(
    path: Path,
    *,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def export_state_summaries(state: RuntimeState, *, run_dir: Path) -> None:
    replicas = state.rows("replicas")
    completed_by_pose: dict[tuple[str, str], int] = {}
    for replica in replicas:
        if replica["status"] == "COMPLETED":
            key = (str(replica["run_id"]), str(replica["pose_id"]))
            completed_by_pose[key] = completed_by_pose.get(key, 0) + 1
    poses: list[dict[str, Any]] = []
    for row in state.rows("poses"):
        exported = dict(row)
        exported["n_replicas_completed"] = completed_by_pose.get(
            (str(row["run_id"]), str(row["pose_id"])),
            0,
        )
        exported["docking_score_kcal_mol"] = None
        exported["pruning_enabled"] = bool(row["pruning_enabled"])
        poses.append(exported)
    _write_rows(
        run_dir / "pose_summary.csv",
        fields=POSE_FIELDS,
        rows=poses,
    )
    _write_rows(
        run_dir / "replica_summary.csv",
        fields=REPLICA_FIELDS,
        rows=replicas,
    )
