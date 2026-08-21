from __future__ import annotations

from pathlib import Path

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.workflow.pose_manifest import (
    load_pose_manifest,
    resolve_pose_configs,
)

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "templates" / "screening"
POSE_IDS = (
    "TARGET1_CMPD0001_pose01",
    "TARGET1_CMPD0001_pose02",
    "TARGET1_CMPD0002_pose01",
    "TARGET2_CMPD0042_pose01",
)


def test_screening_run_template_is_complete_and_schema_valid() -> None:
    config_path = TEMPLATE / "run.yaml"
    config = load_config(config_path)

    assert config.run.output_dir == "outputs"
    assert config.input.prepared_system_dir == f"prepared/{POSE_IDS[0]}"
    assert config.trajectory.replicas == 5
    assert config.trajectory.base_seed is None
    assert config.trajectory.seeds is None
    assert config.trajectory.production_time_ns == 1.0
    assert config.trajectory.output_interval_ps == 10.0
    assert config.gromacs.ntmpi == 1
    assert config.gromacs.ntomp == "auto"
    assert config.pruning.enabled is False
    assert config.scheduler.gpu_ids == "auto"


def test_screening_pose_manifest_is_schema_valid_and_resolves_locally() -> None:
    config_path = TEMPLATE / "run.yaml"
    manifest_path = TEMPLATE / "poses.yaml"
    manifest = load_pose_manifest(manifest_path)

    assert tuple(entry.pose_id for entry in manifest.poses) == POSE_IDS
    assert len(manifest.poses) == 4
    assert all(entry.overrides is None for entry in manifest.poses)
    assert all(
        not Path(entry.prepared_system_dir).is_absolute() for entry in manifest.poses
    )

    resolved = resolve_pose_configs(
        load_config(config_path),
        config_path=config_path,
        manifest_path=manifest_path,
        validate_files=False,
    )
    assert tuple(pose.pose_id for pose in resolved) == POSE_IDS
    for pose in resolved:
        assert (
            Path(pose.config.input.prepared_system_dir)
            == (TEMPLATE / "prepared" / pose.pose_id).resolve()
        )
        assert pose.config.input.start_structure == "start.gro"
        assert pose.config.input.topology == "topol.top"
        assert pose.config.input.index == "index.ndx"
