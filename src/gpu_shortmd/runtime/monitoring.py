"""Bounded polling for online RMSD snapshots produced during GROMACS MD."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gpu_shortmd.analysis.units import convert_rmsd
from gpu_shortmd.analysis.xvg import XvgSample, XvgSeries
from gpu_shortmd.runtime.scheduler import TaskControl, TaskInterrupted


class MonitoringError(RuntimeError):
    """Raised after persistent online snapshot generation/parsing failures."""


@dataclass(frozen=True)
class MonitoringOutcome:
    samples_observed: int
    latest_time_ps: float
    maximum_rmsd_nm: float | None
    maximum_rmsd_angstrom: float | None
    triggered_pruning: bool
    transient_failures: int


def observe_first_pruning_crossing(
    samples: Sequence[XvgSample],
    *,
    control: TaskControl,
) -> bool:
    """Scan samples in memory and persist only the first strict crossing."""
    threshold = control.pruning_threshold_angstrom
    if threshold is None or control.triggered_pruning:
        return False
    for sample in samples:
        rmsd_angstrom = convert_rmsd(
            sample.rmsd,
            input_unit="nm",
            output_unit="angstrom",
        )
        if rmsd_angstrom > threshold:
            return control.observe_rmsd(
                simulation_time_ps=sample.time_ps,
                rmsd_angstrom=rmsd_angstrom,
            )
    return False


def monitor_rmsd_snapshots(
    *,
    finished: threading.Event,
    poll_interval_seconds: float,
    snapshot_supplier: Callable[[], XvgSeries | None],
    control: TaskControl,
    max_consecutive_failures: int = 3,
) -> MonitoringOutcome:
    """Poll complete snapshots, process each simulation time once, and prune."""
    if poll_interval_seconds <= 0:
        raise ValueError("monitoring poll interval must be positive")
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be positive")
    seen_times: set[float] = set()
    latest_time = 0.0
    maximum_nm: float | None = None
    consecutive_failures = 0
    transient_failures = 0
    while True:
        finished.wait(timeout=poll_interval_seconds)
        control.checkpoint()
        try:
            series = snapshot_supplier()
            consecutive_failures = 0
        except TaskInterrupted:
            raise
        except Exception as exc:
            consecutive_failures += 1
            transient_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise MonitoringError(
                    "online RMSD snapshot failed persistently"
                ) from exc
            if finished.is_set():
                break
            continue
        if series is not None:
            new_samples: list[XvgSample] = []
            for sample in series.samples:
                if sample.time_ps in seen_times:
                    continue
                seen_times.add(sample.time_ps)
                new_samples.append(sample)
                latest_time = max(latest_time, sample.time_ps)
                maximum_nm = (
                    sample.rmsd if maximum_nm is None else max(maximum_nm, sample.rmsd)
                )
            if observe_first_pruning_crossing(new_samples, control=control):
                return MonitoringOutcome(
                    samples_observed=len(seen_times),
                    latest_time_ps=latest_time,
                    maximum_rmsd_nm=maximum_nm,
                    maximum_rmsd_angstrom=(
                        convert_rmsd(
                            maximum_nm,
                            input_unit="nm",
                            output_unit="angstrom",
                        )
                        if maximum_nm is not None
                        else None
                    ),
                    triggered_pruning=True,
                    transient_failures=transient_failures,
                )
        if finished.is_set():
            break
    return MonitoringOutcome(
        samples_observed=len(seen_times),
        latest_time_ps=latest_time,
        maximum_rmsd_nm=maximum_nm,
        maximum_rmsd_angstrom=(
            convert_rmsd(
                maximum_nm,
                input_unit="nm",
                output_unit="angstrom",
            )
            if maximum_nm is not None
            else None
        ),
        triggered_pruning=False,
        transient_failures=transient_failures,
    )
