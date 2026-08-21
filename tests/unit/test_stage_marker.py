from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.workflow.stage_marker import (
    STAGE_COMPLETION_MARKER,
    StageMarkerError,
    stage_is_reusable,
    stage_marker_identity,
    write_stage_completion_marker,
)


def identity(tmp_path: Path, *, seed: int = 101) -> dict[str, object]:
    resolved_mdp = tmp_path / "resolved.mdp"
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_mdp.write_text("integrator = md\n", encoding="utf-8")
    resolved_config.write_text("schema_version: 1\n", encoding="utf-8")
    return stage_marker_identity(
        stage="nvt",
        pose_id="pose/a",
        replica_id="replica_01",
        velocity_seed=seed,
        resolved_mdp=resolved_mdp,
        resolved_config=resolved_config,
    )


def write_nominal_outputs(base: Path) -> None:
    base.parent.mkdir(parents=True)
    for suffix in (".tpr", ".gro", ".cpt", ".log"):
        base.with_suffix(suffix).write_text("complete\n", encoding="utf-8")


def test_outputs_without_completion_marker_are_not_reusable(tmp_path: Path) -> None:
    base = tmp_path / "nvt" / "nvt"
    write_nominal_outputs(base)
    assert not stage_is_reusable(
        base,
        production=False,
        expected_identity=identity(tmp_path),
    )


def test_atomic_completion_marker_enables_exact_stage_reuse(tmp_path: Path) -> None:
    base = tmp_path / "nvt" / "nvt"
    write_nominal_outputs(base)
    expected = identity(tmp_path)
    marker = write_stage_completion_marker(base.parent, identity=expected)
    assert marker.name == STAGE_COMPLETION_MARKER
    assert stage_is_reusable(
        base,
        production=False,
        expected_identity=expected,
    )


@pytest.mark.parametrize("corrupt", [True, False])
def test_corrupt_or_mismatched_completion_marker_is_rejected(
    tmp_path: Path,
    corrupt: bool,
) -> None:
    base = tmp_path / "nvt" / "nvt"
    write_nominal_outputs(base)
    expected = identity(tmp_path)
    marker = base.parent / STAGE_COMPLETION_MARKER
    if corrupt:
        marker.write_text("{not-json\n", encoding="utf-8")
    else:
        write_stage_completion_marker(
            base.parent,
            identity={**expected, "velocity_seed": 999},
        )
    with pytest.raises(StageMarkerError):
        stage_is_reusable(
            base,
            production=False,
            expected_identity=expected,
        )
