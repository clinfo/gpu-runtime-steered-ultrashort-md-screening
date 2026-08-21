"""Deterministic pose-level pruning decisions in public Å units."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class PruningTrigger:
    """Canonical trigger ordering: time, replica ID, then observed RMSD."""

    simulation_time_ps: float
    replica_id: str
    observed_rmsd_angstrom: float


def evaluate_pruning(
    *,
    replica_id: str,
    simulation_time_ps: float,
    observed_rmsd_angstrom: float,
    threshold_angstrom: float,
) -> PruningTrigger | None:
    """Return a trigger only for the strict contract condition RMSD > threshold."""
    if threshold_angstrom <= 0:
        raise ValueError("pruning threshold must be positive angstrom")
    if simulation_time_ps < 0:
        raise ValueError("simulation time cannot be negative")
    if observed_rmsd_angstrom < 0:
        raise ValueError("RMSD cannot be negative")
    if observed_rmsd_angstrom <= threshold_angstrom:
        return None
    return PruningTrigger(
        simulation_time_ps=simulation_time_ps,
        replica_id=replica_id,
        observed_rmsd_angstrom=observed_rmsd_angstrom,
    )


def canonical_trigger(
    current: PruningTrigger | None,
    candidate: PruningTrigger,
) -> PruningTrigger:
    """Choose the deterministic earliest trigger, breaking ties by replica ID."""
    if current is None:
        return candidate
    current_key = (current.simulation_time_ps, current.replica_id)
    candidate_key = (candidate.simulation_time_ps, candidate.replica_id)
    return candidate if candidate_key < current_key else current
