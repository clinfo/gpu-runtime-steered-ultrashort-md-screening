from __future__ import annotations

import pytest

from gpu_shortmd.analysis.units import UnitError, convert_rmsd


def test_nm_to_angstrom_is_explicit_factor_ten() -> None:
    assert convert_rmsd(0.4872631, input_unit="nm", output_unit="angstrom") == (
        4.872631
    )


def test_angstrom_to_nm() -> None:
    assert convert_rmsd(4.872631, input_unit="angstrom", output_unit="nm") == (
        0.4872631
    )


def test_units_are_not_inferred() -> None:
    with pytest.raises(UnitError):
        convert_rmsd(0.2, input_unit="unknown", output_unit="angstrom")  # type: ignore[arg-type]
