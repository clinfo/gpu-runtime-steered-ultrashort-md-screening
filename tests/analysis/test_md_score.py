from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.analysis.md_score import calculate_md_score
from gpu_shortmd.analysis.units import UnitError


def _write(path: Path, values: list[tuple[float, float]]) -> None:
    path.write_text(
        "\n".join(f"{time} {rmsd}" for time, rmsd in values) + "\n",
        encoding="utf-8",
    )


def test_score_is_maximum_over_time_then_replicas(tmp_path: Path) -> None:
    first = tmp_path / "first.xvg"
    second = tmp_path / "second.xvg"
    _write(first, [(0, 0.1), (10, 0.4), (20, 0.2)])
    _write(second, [(0, 0.3), (10, 0.2), (20, 0.45)])
    result = calculate_md_score(
        [first, second], input_unit="nm", output_unit="angstrom"
    )
    assert [item.max_rmsd_nm for item in result.replica_maxima] == [0.4, 0.45]
    assert result.md_score_angstrom == 4.5
    assert result.observed_max_rmsd_angstrom == 4.5


def test_incomplete_pose_has_null_completed_score(tmp_path: Path) -> None:
    path = tmp_path / "one.xvg"
    _write(path, [(0, 0.2)])
    result = calculate_md_score(
        [path],
        input_unit="nm",
        output_unit="angstrom",
        requested_replicas=2,
    )
    assert result.status == "INCOMPLETE"
    assert result.md_score_angstrom is None
    assert result.observed_max_rmsd_angstrom == 2.0


def test_pruned_pose_has_null_completed_score(tmp_path: Path) -> None:
    path = tmp_path / "one.xvg"
    _write(path, [(0, 0.2)])
    result = calculate_md_score(
        [path],
        input_unit="nm",
        output_unit="angstrom",
        requested_replicas=1,
        pruned=True,
    )
    assert result.status == "PRUNED"
    assert result.md_score_angstrom is None
    assert result.observed_max_rmsd_angstrom == 2.0


def test_score_core_rejects_non_contract_units(tmp_path: Path) -> None:
    path = tmp_path / "one.xvg"
    _write(path, [(0, 0.2)])

    with pytest.raises(UnitError, match="explicit nm input"):
        calculate_md_score(
            [path],
            input_unit="angstrom",
            output_unit="angstrom",
        )
