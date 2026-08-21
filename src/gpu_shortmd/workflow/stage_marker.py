"""Fail-closed atomic completion markers for resumable GROMACS stages."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_shortmd.util.checksums import sha256_file
from gpu_shortmd.util.files import write_json

STAGE_COMPLETION_MARKER = ".gpu-shortmd-stage-complete.json"


class StageMarkerError(ValueError):
    """Raised when an existing stage marker is corrupt or mismatched."""


def stage_marker_identity(
    *,
    stage: str,
    pose_id: str,
    replica_id: str,
    velocity_seed: int,
    resolved_mdp: Path,
    resolved_config: Path,
) -> dict[str, object]:
    if stage not in {"nvt", "npt", "production"}:
        raise ValueError(f"unsupported stage marker identity: {stage}")
    if not pose_id or not replica_id or velocity_seed < 1:
        raise ValueError("stage marker identity is incomplete")
    return {
        "schema_version": 1,
        "stage": stage,
        "pose_id": pose_id,
        "replica_id": replica_id,
        "velocity_seed": velocity_seed,
        "resolved_mdp_sha256": sha256_file(resolved_mdp),
        "resolved_config_sha256": sha256_file(resolved_config),
    }


def stage_outputs_complete(base: Path, *, production: bool) -> bool:
    suffixes = [".tpr", ".gro", ".cpt", ".log"]
    if production:
        suffixes.append(".xtc")
    return all(
        base.with_suffix(suffix).is_file()
        and base.with_suffix(suffix).stat().st_size > 0
        for suffix in suffixes
    )


def stage_is_reusable(
    base: Path,
    *,
    production: bool,
    expected_identity: dict[str, object],
) -> bool:
    marker = base.parent / STAGE_COMPLETION_MARKER
    if not marker.is_file():
        return False
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageMarkerError(
            f"{base.parent.name} stage completion marker is corrupt"
        ) from exc
    if actual != expected_identity:
        raise StageMarkerError(
            f"{base.parent.name} stage completion marker identity mismatch"
        )
    return stage_outputs_complete(base, production=production)


def invalidate_stage_marker(stage_dir: Path) -> None:
    (stage_dir / STAGE_COMPLETION_MARKER).unlink(missing_ok=True)


def write_stage_completion_marker(
    stage_dir: Path,
    *,
    identity: dict[str, object],
) -> Path:
    marker = stage_dir / STAGE_COMPLETION_MARKER
    write_json(marker, identity)
    return marker
