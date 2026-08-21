from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.workflow.pose_manifest import (
    PoseManifestError,
    load_pose_manifest,
    resolve_pose_configs,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"


def pose(pose_id: str, prepared_system_dir: str) -> dict[str, Any]:
    return {
        "pose_id": pose_id,
        "prepared_system_dir": prepared_system_dir,
        "start_structure": "p2_em.gro",
        "topology": "p2_topol.top",
        "index": "p2_index.ndx",
        "ligand_resname": "LIG",
    }


def write_manifest(path: Path, poses: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "poses": poses}, sort_keys=False),
        encoding="utf-8",
    )


def test_duplicate_pose_id_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "poses.yaml"
    write_manifest(
        manifest,
        [
            pose("duplicate", str(EXAMPLE / "prepared_input")),
            pose("duplicate", str(EXAMPLE / "prepared_input")),
        ],
    )
    with pytest.raises(PoseManifestError, match="pose_id values must be unique"):
        load_pose_manifest(manifest)


def test_missing_prepared_files_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "poses.yaml"
    write_manifest(manifest, [pose("missing", "not-present")])
    config_path = REPOSITORY_ROOT / "configs" / "default.yaml"
    with pytest.raises(PoseManifestError, match="missing prepared input files"):
        resolve_pose_configs(
            load_config(config_path),
            config_path=config_path,
            manifest_path=manifest,
        )


def test_unknown_or_inconsistent_override_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "poses.yaml"
    invalid = pose("bad-override", str(EXAMPLE / "prepared_input"))
    invalid["overrides"] = {"gromacs": {"ntomp": 2}}
    write_manifest(manifest, [invalid])
    with pytest.raises(PoseManifestError, match="Extra inputs are not permitted"):
        load_pose_manifest(manifest)

    inconsistent = pose("bad-seeds", str(EXAMPLE / "prepared_input"))
    inconsistent["overrides"] = {"trajectory": {"replicas": 2, "seeds": [101]}}
    write_manifest(manifest, [inconsistent])
    with pytest.raises(PoseManifestError, match="seeds length must equal replicas"):
        load_pose_manifest(manifest)


def test_tasks_per_gpu_other_than_one_is_schema_error() -> None:
    raw = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    value = deepcopy(raw)
    value["scheduler"]["tasks_per_gpu"] = 2
    with pytest.raises(ValidationError, match="Input should be 1"):
        AppConfig.model_validate(value)


def test_single_pose_config_normalizes_to_one_pose_manifest() -> None:
    config_path = EXAMPLE / "config.single_replica.yaml"
    poses = resolve_pose_configs(
        load_config(config_path),
        config_path=config_path,
        manifest_path=None,
    )
    assert len(poses) == 1
    assert poses[0].pose_id == "P00533_EGFR|p2|model4"
    assert poses[0].config.input.ligand_resname == "LIG"
