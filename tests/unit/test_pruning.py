from __future__ import annotations

from gpu_shortmd.runtime.pruning import (
    PruningTrigger,
    canonical_trigger,
    evaluate_pruning,
)


def test_pruning_uses_strict_greater_than_angstrom() -> None:
    assert (
        evaluate_pruning(
            replica_id="replica_01",
            simulation_time_ps=10.0,
            observed_rmsd_angstrom=4.0,
            threshold_angstrom=4.0,
        )
        is None
    )
    trigger = evaluate_pruning(
        replica_id="replica_01",
        simulation_time_ps=20.0,
        observed_rmsd_angstrom=4.000001,
        threshold_angstrom=4.0,
    )
    assert trigger is not None
    assert trigger.observed_rmsd_angstrom == 4.000001


def test_canonical_trigger_is_earliest_then_replica_id() -> None:
    later = PruningTrigger(20.0, "replica_01", 9.0)
    earlier = PruningTrigger(10.0, "replica_05", 4.1)
    tie = PruningTrigger(10.0, "replica_02", 4.2)
    assert canonical_trigger(later, earlier) == earlier
    assert canonical_trigger(earlier, tie) == tie
