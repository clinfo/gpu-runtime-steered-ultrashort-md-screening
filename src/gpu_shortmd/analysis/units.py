"""Explicit RMSD unit conversion utilities."""

from __future__ import annotations

from typing import Literal

RmsdUnit = Literal["nm", "angstrom"]
NM_TO_ANGSTROM = 10.0


class UnitError(ValueError):
    """Raised when a caller supplies an unsupported or implicit unit."""


def convert_rmsd(value: float, *, input_unit: RmsdUnit, output_unit: RmsdUnit) -> float:
    """Convert an RMSD value without inferring units from its magnitude."""
    supported = {"nm", "angstrom"}
    if input_unit not in supported:
        raise UnitError(f"unsupported input unit: {input_unit!r}")
    if output_unit not in supported:
        raise UnitError(f"unsupported output unit: {output_unit!r}")
    if input_unit == output_unit:
        return value
    if input_unit == "nm":
        return value * NM_TO_ANGSTROM
    return value / NM_TO_ANGSTROM
