from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from gpu_shortmd.analysis.xvg import XvgParseError, XvgSample, XvgSeries
from gpu_shortmd.gromacs.rmsd import PbcRmsdWorkspace, write_clustered_rmsd_index
from gpu_shortmd.runtime.monitoring import (
    MonitoringError,
    MonitoringOutcome,
    monitor_rmsd_snapshots,
    observe_first_pruning_crossing,
)
from gpu_shortmd.runtime.scheduler import TaskInterrupted
from gpu_shortmd.util.subprocess import CommandResult
from gpu_shortmd.workflow import replica as replica_module


class FakeControl:
    def __init__(self, threshold_angstrom: float | None = 4.0) -> None:
        self.claim = type(
            "Claim",
            (),
            {"replica_id": "replica_01", "pose_id": "pose-a"},
        )()
        self.pruning_threshold_angstrom = threshold_angstrom
        self.triggered_pruning = False
        self.observations: list[tuple[float, float]] = []
        self.checkpoints = 0

    def checkpoint(self) -> None:
        self.checkpoints += 1

    def stop_requested(self) -> bool:
        return False

    def observe_rmsd(
        self,
        *,
        simulation_time_ps: float,
        rmsd_angstrom: float,
    ) -> bool:
        self.observations.append((simulation_time_ps, rmsd_angstrom))
        triggered = (
            self.pruning_threshold_angstrom is not None
            and rmsd_angstrom > self.pruning_threshold_angstrom
        )
        self.triggered_pruning = triggered
        return triggered


def series(*values: tuple[float, float]) -> XvgSeries:
    return XvgSeries(
        path=Path("online.xvg"),
        samples=tuple(
            XvgSample(time_ps=time_ps, rmsd=rmsd) for time_ps, rmsd in values
        ),
    )


def test_online_monitor_deduplicates_snapshots_and_triggers_strict_pruning() -> None:
    finished = threading.Event()
    snapshots = iter(
        [
            series((0.0, 0.1), (10.0, 0.3)),
            series((0.0, 0.1), (10.0, 0.3), (20.0, 0.41)),
        ]
    )
    control = FakeControl()

    outcome = monitor_rmsd_snapshots(
        finished=finished,
        poll_interval_seconds=0.001,
        snapshot_supplier=lambda: next(snapshots),
        control=control,  # type: ignore[arg-type]
    )
    assert outcome.triggered_pruning is True
    assert outcome.samples_observed == 3
    assert outcome.maximum_rmsd_angstrom == 4.1
    assert control.checkpoints == 2
    assert control.observations == [(20.0, 4.1)]


def test_online_monitor_escalates_persistent_snapshot_failure() -> None:
    finished = threading.Event()

    def fail() -> XvgSeries:
        raise ValueError("partial snapshot")

    with pytest.raises(MonitoringError, match="persistently"):
        monitor_rmsd_snapshots(
            finished=finished,
            poll_interval_seconds=0.001,
            snapshot_supplier=fail,
            control=FakeControl(),  # type: ignore[arg-type]
            max_consecutive_failures=2,
        )


def test_online_monitor_preserves_supplier_task_interruption() -> None:
    finished = threading.Event()
    interruption = TaskInterrupted("cooperative online analysis stop")
    supplier_calls = 0

    def interrupt() -> XvgSeries:
        nonlocal supplier_calls
        supplier_calls += 1
        raise interruption

    with pytest.raises(TaskInterrupted) as caught:
        monitor_rmsd_snapshots(
            finished=finished,
            poll_interval_seconds=0.001,
            snapshot_supplier=interrupt,
            control=FakeControl(),  # type: ignore[arg-type]
        )

    assert caught.value is interruption
    assert supplier_calls == 1
    assert not isinstance(caught.value, MonitoringError)


def test_monitor_target_classifies_supplier_task_interruption() -> None:
    finished = threading.Event()
    cancel = threading.Event()
    interruption = TaskInterrupted("cooperative online analysis stop")
    supplier_calls = 0
    outcomes: list[MonitoringOutcome] = []
    interruptions: list[TaskInterrupted] = []
    errors: list[BaseException] = []

    def interrupt() -> XvgSeries:
        nonlocal supplier_calls
        supplier_calls += 1
        raise interruption

    replica_module._monitor_target(
        finished=finished,
        poll_interval_seconds=0.001,
        snapshot_supplier=interrupt,
        control=FakeControl(),  # type: ignore[arg-type]
        outcomes=outcomes,
        interruptions=interruptions,
        errors=errors,
        cancel=cancel,
    )

    assert supplier_calls == 1
    assert interruptions == [interruption]
    assert errors == []
    assert outcomes == []
    assert cancel.is_set()


def test_monitoring_state_calls_do_not_scale_with_sample_count() -> None:
    samples = tuple((float(index), 0.1) for index in range(1000))
    disabled_finished = threading.Event()
    disabled_finished.set()
    disabled = FakeControl(threshold_angstrom=None)
    disabled_outcome = monitor_rmsd_snapshots(
        finished=disabled_finished,
        poll_interval_seconds=0.001,
        snapshot_supplier=lambda: series(*samples),
        control=disabled,  # type: ignore[arg-type]
    )
    assert disabled_outcome.samples_observed == 1000
    assert disabled.checkpoints == 1
    assert disabled.observations == []

    enabled_samples = tuple(
        XvgSample(time_ps=float(index), rmsd=0.41 if index >= 500 else 0.1)
        for index in range(1000)
    )
    enabled = FakeControl()
    assert observe_first_pruning_crossing(
        enabled_samples,
        control=enabled,  # type: ignore[arg-type]
    )
    assert enabled.observations == [(500.0, 4.1)]


class _RedactingLogger:
    @staticmethod
    def redact(value: str) -> str:
        return value

    @staticmethod
    def event(**_kwargs: object) -> None:
        return None


def test_online_snapshot_atomic_replace_avoids_backups_after_111_polls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = tmp_path / "production.xtc"
    trajectory.write_text("changing trajectory", encoding="utf-8")
    stable_snapshot = tmp_path / "online_rmsd_snapshot_nm.xvg"
    command_outputs: list[Path] = []
    emit_invalid_snapshot = False

    source_index = tmp_path / "source.ndx"
    source_index.write_text(
        "[ C-alpha ]\n1 2\n[ LIG ]\n3 4 5\n"
        "[ Protein_LIG ]\n1 2 3 4 5\n[ Water_and_ions ]\n6\n",
        encoding="utf-8",
    )
    generated_index = tmp_path / "rmsd_groups.ndx"
    generated_index.write_text(
        "[ C-alpha ]\n1 2\n[ LIG_HEAVY ]\n3 4 5\n",
        encoding="utf-8",
    )
    workspace = PbcRmsdWorkspace.under(tmp_path)
    workspace.reset()
    write_clustered_rmsd_index(
        source_index=source_index,
        generated_index=generated_index,
        output_index=workspace.rmsd_index,
    )

    def fake_run_command(
        args: Sequence[str],
        **_kwargs: object,
    ) -> CommandResult:
        output = Path(args[args.index("-o") + 1])
        command_outputs.append(output)
        if output.exists():
            (output.parent / f"#{output.name}.1#").write_text(
                "backup",
                encoding="utf-8",
            )
        if args[1] == "rms":
            output.write_text(
                "0.0 invalid\n" if emit_invalid_snapshot else "0.0 0.10\n10.0 0.20\n",
                encoding="utf-8",
            )
        else:
            output.write_text("clustered", encoding="utf-8")
        return CommandResult(tuple(args), 0, "", "", 1)

    monkeypatch.setattr(
        replica_module,
        "run_cancellable_command",
        fake_run_command,
    )
    for _ in range(111):
        snapshot = replica_module._online_snapshot(
            executable=tmp_path / "gmx",
            reference_topology=tmp_path / "production.tpr",
            trajectory=trajectory,
            source_index=source_index,
            workspace=workspace,
            begin_time_ps=0.0,
            output_xvg=stable_snapshot,
            cwd=tmp_path,
            environment={},
            logger=_RedactingLogger(),  # type: ignore[arg-type]
            control=FakeControl(),  # type: ignore[arg-type]
            timeout_seconds=1.0,
        )
        assert snapshot is not None
        assert snapshot.path == stable_snapshot

    rms_outputs = [path for path in command_outputs if path.suffix == ".xvg"]
    assert len(rms_outputs) == 111
    assert len(set(command_outputs)) == len(command_outputs)
    assert {path.suffix for path in command_outputs} <= {".gro", ".xtc", ".xvg"}
    assert stable_snapshot.is_file()
    last_valid_snapshot = stable_snapshot.read_text(encoding="utf-8")
    emit_invalid_snapshot = True
    with pytest.raises(XvgParseError):
        replica_module._online_snapshot(
            executable=tmp_path / "gmx",
            reference_topology=tmp_path / "production.tpr",
            trajectory=trajectory,
            source_index=source_index,
            workspace=workspace,
            begin_time_ps=0.0,
            output_xvg=stable_snapshot,
            cwd=tmp_path,
            environment={},
            logger=_RedactingLogger(),  # type: ignore[arg-type]
            control=FakeControl(),  # type: ignore[arg-type]
            timeout_seconds=1.0,
        )
    assert stable_snapshot.read_text(encoding="utf-8") == last_valid_snapshot
    assert not list(tmp_path.glob("#online_rmsd_snapshot*#"))
    assert not list(tmp_path.glob(".online_rmsd_snapshot_nm.*.xvg"))
    assert not list(workspace.root.glob("#*#"))
    workspace.cleanup()
    assert not workspace.root.exists()
