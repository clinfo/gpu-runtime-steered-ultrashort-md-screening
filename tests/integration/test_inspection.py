from __future__ import annotations

import json
from pathlib import Path

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.workflow.inspection import (
    inspect_configuration,
    write_inspection_outputs,
)


def test_inspection_passes_with_explicit_fake_environment(
    single_replica_config: tuple[Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    config_path, environment = single_replica_config
    config = load_config(config_path)
    report = inspect_configuration(
        config,
        config_path=config_path,
        env=environment,
    )
    assert report.overall_status == "PASS"
    assert report.exit_code == 0
    assert any(
        check.code in {"SQLITE_WAL_VALIDATED", "SQLITE_DELETE_VALIDATED"}
        for check in report.checks
    )
    assert report.gromacs_version is not None
    assert report.gromacs_version.version == "2025.4"
    assert report.gromacs_version.raw_version == "2025.4"
    assert any(
        check.code == "SEED_CONFIGURATION_VALID"
        and check.details
        == {
            "mode": "explicit",
            "count": 1,
            "seeds": [2026073001],
        }
        for check in report.checks
    )
    assert any(check.code == "PRUNING_DISABLED" for check in report.checks)
    output = tmp_path / "inspect"
    write_inspection_outputs(report, config=config, output_dir=output)
    payload = json.loads((output / "inspect_report.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] == "PASS"
    assert str(config.input.prepared_system_dir) not in json.dumps(payload)
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["gromacs"]["cuda_compiler"] == "nvcc (release 12.2)"
    assert payload["gromacs"]["cuda_driver"] == "12.2"
    assert (output / "inspect_report.txt").is_file()
    assert (output / "resolved_config.yaml").is_file()


def test_inspection_fails_without_visible_gpu(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    report = inspect_configuration(
        load_config(config_path),
        config_path=config_path,
        env=environment,
    )
    assert report.overall_status == "FAIL"
    assert report.exit_code == 4
    assert any(check.code == "GPU_NOT_VISIBLE" for check in report.checks)


def test_inspection_preserves_physical_cuda_visible_device_ids(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    environment["CUDA_VISIBLE_DEVICES"] = "2,3"
    report = inspect_configuration(
        load_config(config_path),
        config_path=config_path,
        env=environment,
    )

    assert report.exit_code == 0
    assert report.visible_gpu_ids == (2, 3)


def test_inspection_rejects_prepared_input_path_escape(
    single_replica_config: tuple[Path, dict[str, str]],
) -> None:
    config_path, environment = single_replica_config
    config = load_config(config_path)
    escaped_input = config.input.model_copy(
        update={"start_structure": "../outside.gro"}
    )
    config = config.model_copy(update={"input": escaped_input})

    report = inspect_configuration(
        config,
        config_path=config_path,
        env=environment,
    )

    assert report.exit_code == 3
    assert any(
        check.code == "INPUT_FILES_MISSING"
        and "must stay inside prepared_system_dir" in check.message
        for check in report.checks
    )


def test_inspection_text_redacts_explicit_executable_path(
    single_replica_config: tuple[Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    config_path, environment = single_replica_config
    config = load_config(config_path)
    missing_executable = tmp_path / "private-location" / "gmx"
    config = config.model_copy(
        update={
            "gromacs": config.gromacs.model_copy(
                update={"executable": str(missing_executable)}
            )
        }
    )
    report = inspect_configuration(
        config,
        config_path=config_path,
        env=environment,
    )
    output = tmp_path / "inspection"
    write_inspection_outputs(report, config=config, output_dir=output)

    assert report.exit_code != 0
    assert str(missing_executable) not in (output / "inspect_report.txt").read_text(
        encoding="utf-8"
    )
