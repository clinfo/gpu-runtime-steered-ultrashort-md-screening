from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

try:
    from click.utils import strip_ansi
except ModuleNotFoundError:  # Typer 0.27 vendors Click as typer._click.
    from typer._click.utils import strip_ansi

from gpu_shortmd.cli import run as run_cli
from gpu_shortmd.cli import score as score_cli
from gpu_shortmd.cli.app import app

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"
runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_help_labels_analysis_commands_stable() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "score" in result.stdout
    assert "validate-example" in result.stdout
    assert "stop" in result.stdout
    assert "stable" in result.stdout.lower()
    assert "pipeline" not in result.stdout.lower()


def test_run_help_exposes_multi_pose_manifest() -> None:
    result = runner.invoke(
        app,
        ["run", "--help"],
        color=False,
        env={"COLUMNS": "160"},
        terminal_width=160,
    )
    assert result.exit_code == 0

    output = strip_ansi(result.output)
    assert "--manifest" in output
    assert "prepared poses" in output


def test_run_requires_exactly_config_or_resume() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "exactly one" in result.output


def test_stop_rejects_directory_without_state(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stop", str(tmp_path)])
    assert result.exit_code == 7
    assert "state.sqlite3" in result.output


def test_score_rejects_implicit_or_different_units() -> None:
    xvg = EXAMPLE / "reference" / "p2_model4_rmsd_replica_01_nm.xvg"
    result = runner.invoke(
        app,
        [
            "score",
            "--xvg",
            str(xvg),
            "--input-unit",
            "angstrom",
            "--output-unit",
            "angstrom",
        ],
    )
    assert result.exit_code == 2


def test_validate_example_command() -> None:
    result = runner.invoke(app, ["validate-example", str(EXAMPLE)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["md_score_angstrom"] == 4.872631
    assert payload["validation_type"] == "deterministic_xvg_reference"


def test_run_rejects_empty_gpu_override_as_configuration_error() -> None:
    config = EXAMPLE / "config.single_replica.yaml"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--gpu-ids", ""],
    )

    assert result.exit_code == 2
    assert "invalid --gpu-ids value" in result.output


def test_score_output_failure_returns_exit_5(tmp_path: Path) -> None:
    xvg = EXAMPLE / "reference" / "p2_model4_rmsd_replica_01_nm.xvg"
    result = runner.invoke(
        app,
        [
            "score",
            "--xvg",
            str(xvg),
            "--json-output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 5
    assert "could not write score artifacts" in result.output


def test_score_internal_failure_returns_exit_9(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xvg = EXAMPLE / "reference" / "p2_model4_rmsd_replica_01_nm.xvg"

    def fail_internal(*_: object, **__: object) -> None:
        raise AssertionError("injected invariant failure")

    monkeypatch.setattr(score_cli, "calculate_md_score", fail_internal)
    result = runner.invoke(app, ["score", "--xvg", str(xvg)])

    assert result.exit_code == 9
    assert "internal software error: AssertionError" in result.output


def test_run_internal_failure_returns_exit_9(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EXAMPLE / "config.single_replica.yaml"

    def fail_internal(*_: object, **__: object) -> None:
        raise AssertionError("injected invariant failure")

    monkeypatch.setattr(run_cli, "run_fresh", fail_internal)
    result = runner.invoke(app, ["run", "--config", str(config)])

    assert result.exit_code == 9
    assert "internal software error: AssertionError" in result.output
