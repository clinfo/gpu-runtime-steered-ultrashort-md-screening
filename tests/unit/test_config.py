from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from gpu_shortmd.config.loader import ConfigLoadError, load_config
from gpu_shortmd.config.models import AppConfig, PoseManifest

REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.fixture
def config_data() -> dict[str, Any]:
    value = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_default_config_is_strict_and_pruning_is_off() -> None:
    config = load_config(REPOSITORY_ROOT / "configs" / "default.yaml")
    assert config.schema_version == 1
    assert config.pruning.enabled is False
    assert config.pruning.threshold_angstrom is None
    assert config.gromacs.maxwarn == 0


def test_five_replica_example_has_fixed_unique_seeds() -> None:
    config = load_config(
        REPOSITORY_ROOT
        / "examples"
        / "egfr_p00533_1xkk_fmm_p2_model4"
        / "config.five_replicas.yaml"
    )
    assert config.trajectory.replicas == 5
    assert config.trajectory.seeds == [
        2026073001,
        2026073002,
        2026073003,
        2026073004,
        2026073005,
    ]


def test_unknown_key_is_rejected(config_data: dict[str, Any]) -> None:
    value = deepcopy(config_data)
    value["trajectory"]["rmsd_check_interval"] = 10
    with pytest.raises(ValidationError):
        AppConfig.model_validate(value)


def test_pruning_requires_explicit_angstrom_threshold(
    config_data: dict[str, Any],
) -> None:
    value = deepcopy(config_data)
    value["pruning"]["enabled"] = True
    with pytest.raises(ValidationError):
        AppConfig.model_validate(value)


def test_disabled_pruning_rejects_hidden_threshold(
    config_data: dict[str, Any],
) -> None:
    value = deepcopy(config_data)
    value["pruning"]["threshold_angstrom"] = 5.0
    with pytest.raises(ValidationError):
        AppConfig.model_validate(value)


def test_seed_count_and_uniqueness(config_data: dict[str, Any]) -> None:
    mismatch = deepcopy(config_data)
    mismatch["trajectory"]["seeds"] = [1, 2]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(mismatch)

    duplicate = deepcopy(config_data)
    duplicate["trajectory"]["seeds"] = [1, 2, 3, 4, 4]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(duplicate)


def test_base_seed_range_cannot_overflow_replica_derivation(
    config_data: dict[str, Any],
) -> None:
    value = deepcopy(config_data)
    value["trajectory"]["replicas"] = 2
    value["trajectory"]["base_seed"] = 2_147_483_647
    with pytest.raises(ValidationError, match="seed limit"):
        AppConfig.model_validate(value)


def test_negative_maxwarn_is_rejected(config_data: dict[str, Any]) -> None:
    value = deepcopy(config_data)
    value["gromacs"]["maxwarn"] = -1
    with pytest.raises(ValidationError):
        AppConfig.model_validate(value)


def test_ligand_resname_is_strict_functional_selection_input(
    config_data: dict[str, Any],
) -> None:
    value = deepcopy(config_data)
    value["input"]["ligand_resname"] = 'LIG" or group "Protein'
    with pytest.raises(ValidationError, match="ligand_resname"):
        AppConfig.model_validate(value)
    schema = AppConfig.model_json_schema()
    ligand_resname = schema["$defs"]["InputConfig"]["properties"]["ligand_resname"]
    assert ligand_resname["pattern"] == r"^[A-Za-z0-9_+-]{1,16}$"


def test_scalar_type_coercion_is_rejected(config_data: dict[str, Any]) -> None:
    value = deepcopy(config_data)
    value["gromacs"]["maxwarn"] = "0"
    with pytest.raises(ValidationError):
        AppConfig.model_validate(value)


def test_schema_exposes_ntomp_minimum(config_data: dict[str, Any]) -> None:
    schema = AppConfig.model_json_schema()
    integer_variant = schema["$defs"]["GromacsConfig"]["properties"]["ntomp"]["anyOf"][
        0
    ]
    assert integer_variant["minimum"] == 1


def test_schema_fixes_tasks_per_gpu_to_one() -> None:
    schema = AppConfig.model_json_schema()
    tasks_per_gpu = schema["$defs"]["SchedulerConfig"]["properties"]["tasks_per_gpu"]
    assert tasks_per_gpu["const"] == 1


def test_pose_manifest_schema_requires_pose_identity_and_inputs() -> None:
    schema = PoseManifest.model_json_schema()
    pose_entry = schema["$defs"]["PoseManifestEntry"]
    assert set(pose_entry["required"]) >= {
        "pose_id",
        "prepared_system_dir",
        "start_structure",
        "topology",
        "index",
        "ligand_resname",
    }


def test_non_mapping_yaml_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError):
        load_config(path)


def test_generated_schema_declares_contract_draft_and_id() -> None:
    schema = json.loads(
        (REPOSITORY_ROOT / "configs" / "config.schema.json").read_text(encoding="utf-8")
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://example.org/gpu-shortmd/config.schema.json"
    assert schema["additionalProperties"] is False
