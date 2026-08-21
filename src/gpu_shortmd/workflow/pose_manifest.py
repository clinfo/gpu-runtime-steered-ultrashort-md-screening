"""Strict YAML multi-pose manifest loading and per-pose configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from gpu_shortmd.config.models import (
    AppConfig,
    PoseManifest,
    PoseManifestEntry,
)
from gpu_shortmd.workflow.prepared_input import (
    PreparedInputError,
    resolve_prepared_system,
    validate_required_paths,
)


class PoseManifestError(ValueError):
    """Raised when a multi-pose manifest is invalid or incomplete."""


@dataclass(frozen=True)
class ResolvedPoseConfig:
    pose_id: str
    config: AppConfig


def load_pose_manifest(path: str | Path) -> PoseManifest:
    manifest_path = Path(path)
    try:
        raw: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PoseManifestError(f"cannot load multi-pose manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise PoseManifestError("multi-pose manifest root must be a mapping")
    try:
        return PoseManifest.model_validate(raw)
    except ValidationError as exc:
        raise PoseManifestError(str(exc)) from exc


def _apply_entry(
    base: AppConfig,
    *,
    entry: PoseManifestEntry,
    manifest_path: Path,
) -> AppConfig:
    payload = base.model_dump(mode="python")
    configured_root = Path(entry.prepared_system_dir)
    prepared_root = (
        configured_root
        if configured_root.is_absolute()
        else manifest_path.parent / configured_root
    ).resolve()
    payload["run"]["name"] = entry.pose_id
    payload["input"].update(
        {
            "prepared_system_dir": str(prepared_root),
            "start_structure": entry.start_structure,
            "topology": entry.topology,
            "index": entry.index,
            "ligand_resname": entry.ligand_resname,
        }
    )
    overrides = entry.overrides
    if overrides is not None and overrides.trajectory is not None:
        trajectory = overrides.trajectory
        fields = trajectory.model_fields_set
        if "replicas" in fields:
            payload["trajectory"]["replicas"] = trajectory.replicas
        if "base_seed" in fields:
            payload["trajectory"]["base_seed"] = trajectory.base_seed
            if trajectory.base_seed is not None:
                payload["trajectory"]["seeds"] = None
        if "seeds" in fields:
            payload["trajectory"]["seeds"] = trajectory.seeds
            if trajectory.seeds is not None:
                payload["trajectory"]["base_seed"] = None
                if "replicas" not in fields:
                    payload["trajectory"]["replicas"] = len(trajectory.seeds)
    if overrides is not None and overrides.pruning is not None:
        pruning = overrides.pruning
        for field in pruning.model_fields_set:
            payload["pruning"][field] = getattr(pruning, field)
    try:
        return AppConfig.model_validate(payload)
    except ValidationError as exc:
        raise PoseManifestError(
            f"invalid overrides for pose_id {entry.pose_id!r}: {exc}"
        ) from exc


def _single_pose_manifest(base: AppConfig, *, config_path: Path) -> PoseManifest:
    configured_root = Path(base.input.prepared_system_dir)
    prepared_root = (
        configured_root
        if configured_root.is_absolute()
        else config_path.parent / configured_root
    ).resolve()
    pose_id = base.run.name or prepared_root.name
    return PoseManifest(
        schema_version=1,
        poses=[
            PoseManifestEntry(
                pose_id=pose_id,
                prepared_system_dir=str(prepared_root),
                start_structure=base.input.start_structure,
                topology=base.input.topology,
                index=base.input.index,
                ligand_resname=base.input.ligand_resname,
            )
        ],
    )


def resolve_pose_configs(
    base: AppConfig,
    *,
    config_path: Path,
    manifest_path: Path | None,
    validate_files: bool = True,
) -> tuple[ResolvedPoseConfig, ...]:
    source_path = (manifest_path or config_path).resolve()
    manifest = (
        load_pose_manifest(source_path)
        if manifest_path is not None
        else _single_pose_manifest(base, config_path=config_path)
    )
    resolved: list[ResolvedPoseConfig] = []
    for entry in manifest.poses:
        pose_config = _apply_entry(base, entry=entry, manifest_path=source_path)
        if validate_files:
            try:
                validate_required_paths(
                    resolve_prepared_system(pose_config, config_path=source_path)
                )
            except PreparedInputError as exc:
                raise PoseManifestError(f"pose_id {entry.pose_id!r}: {exc}") from exc
        resolved.append(ResolvedPoseConfig(entry.pose_id, pose_config))
    return tuple(resolved)


def resolved_manifest_payload(
    poses: tuple[ResolvedPoseConfig, ...],
    *,
    frozen_roots: dict[str, Path],
    run_dir: Path,
    resolved_seeds: dict[str, tuple[int, ...]],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for pose in poses:
        config = pose.config
        frozen_root = frozen_roots[pose.pose_id]
        entries.append(
            {
                "pose_id": pose.pose_id,
                "prepared_system_dir": frozen_root.relative_to(run_dir).as_posix(),
                "start_structure": config.input.start_structure,
                "topology": config.input.topology,
                "index": config.input.index,
                "ligand_resname": config.input.ligand_resname,
                "overrides": {
                    "trajectory": {
                        "replicas": config.trajectory.replicas,
                        "base_seed": None,
                        "seeds": list(resolved_seeds[pose.pose_id]),
                    },
                    "pruning": config.pruning.model_dump(mode="json"),
                },
            }
        )
    return {"schema_version": 1, "poses": entries}
