"""Replica maxima and pose-level MD-score aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpu_shortmd.analysis.units import RmsdUnit, UnitError, convert_rmsd
from gpu_shortmd.analysis.xvg import parse_xvg


@dataclass(frozen=True)
class ReplicaMaximum:
    replica_id: str
    source: str
    time_ps: float
    max_rmsd_nm: float
    max_rmsd_angstrom: float


@dataclass(frozen=True)
class ScoreResult:
    status: str
    input_unit: RmsdUnit
    output_unit: RmsdUnit
    n_replicas_requested: int
    n_replicas_completed: int
    replica_maxima: tuple[ReplicaMaximum, ...]
    md_score_angstrom: float | None
    observed_max_rmsd_angstrom: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "input_unit": self.input_unit,
            "output_unit": self.output_unit,
            "n_replicas_requested": self.n_replicas_requested,
            "n_replicas_completed": self.n_replicas_completed,
            "replica_maxima": [asdict(item) for item in self.replica_maxima],
            "md_score_angstrom": self.md_score_angstrom,
            "observed_max_rmsd_angstrom": self.observed_max_rmsd_angstrom,
        }


def calculate_md_score(
    paths: list[str | Path],
    *,
    input_unit: RmsdUnit,
    output_unit: RmsdUnit,
    requested_replicas: int | None = None,
    pruned: bool = False,
) -> ScoreResult:
    """Calculate max-over-time then max-over-replicas with explicit units."""
    if input_unit != "nm" or output_unit != "angstrom":
        raise UnitError(
            "stable MD-score requires explicit nm input and angstrom output"
        )
    if not paths:
        raise ValueError("at least one XVG file is required")
    if requested_replicas is None:
        requested_replicas = len(paths)
    if requested_replicas < len(paths):
        raise ValueError("completed replica count exceeds requested replica count")

    maxima: list[ReplicaMaximum] = []
    for index, path_value in enumerate(paths, start=1):
        series = parse_xvg(path_value)
        maximum = series.maximum
        maximum_nm = convert_rmsd(
            maximum.rmsd,
            input_unit=input_unit,
            output_unit="nm",
        )
        maximum_angstrom = convert_rmsd(
            maximum.rmsd,
            input_unit=input_unit,
            output_unit="angstrom",
        )
        maxima.append(
            ReplicaMaximum(
                replica_id=f"replica_{index:02d}",
                source=series.path.name,
                time_ps=maximum.time_ps,
                max_rmsd_nm=maximum_nm,
                max_rmsd_angstrom=maximum_angstrom,
            )
        )

    observed_max = max(item.max_rmsd_angstrom for item in maxima)
    complete = len(maxima) == requested_replicas
    status = "PRUNED" if pruned else ("COMPLETED" if complete else "INCOMPLETE")
    completed_score = observed_max if complete and not pruned else None
    return ScoreResult(
        status=status,
        input_unit=input_unit,
        output_unit=output_unit,
        n_replicas_requested=requested_replicas,
        n_replicas_completed=len(maxima),
        replica_maxima=tuple(maxima),
        md_score_angstrom=completed_score,
        observed_max_rmsd_angstrom=observed_max,
    )
