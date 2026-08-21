from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from gpu_shortmd.analysis.xvg import XvgSample, XvgSeries, parse_xvg
from gpu_shortmd.gromacs.rmsd import (
    PbcRmsdWorkspace,
    build_rmsd_command,
    reconstruct_clustered_complex,
    write_clustered_rmsd_index,
)
from gpu_shortmd.runtime.scheduler import TaskInterrupted
from gpu_shortmd.util.subprocess import CommandResult, run_command
from gpu_shortmd.workflow import replica as replica_module


class _Control:
    triggered_pruning = False
    claim = type(
        "Claim",
        (),
        {"replica_id": "replica_01", "pose_id": "artificial-pbc"},
    )()

    @staticmethod
    def stop_requested() -> bool:
        return False


class _Logger:
    @staticmethod
    def redact(value: str) -> str:
        return value

    @staticmethod
    def event(**_kwargs: object) -> None:
        return None


def _artificial_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_index = tmp_path / "prepared.ndx"
    source_index.write_text(
        "[ C-alpha ]\n1 2\n[ LIG ]\n3 4 5\n"
        "[ Protein_LIG ]\n1 2 3 4 5\n[ Water_and_ions ]\n6\n",
        encoding="utf-8",
    )
    generated_index = tmp_path / "full_system_rmsd.ndx"
    generated_index.write_text(
        "[ C-alpha ]\n1 2\n[ LIG_HEAVY ]\n3 4 5\n",
        encoding="utf-8",
    )
    trajectory = tmp_path / "artificial_pbc_trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "fake_pbc_trajectory": True,
                "box_nm": 10.0,
                "frames": [
                    {
                        "time_ps": 0.0,
                        "coordinates_nm": [
                            [1.0, 1.0, 1.0],
                            [2.0, 1.0, 1.0],
                            [3.0, 1.0, 1.0],
                            [4.0, 1.0, 1.0],
                            [3.0, 2.0, 1.0],
                        ],
                    },
                    {
                        "time_ps": 10.0,
                        "coordinates_nm": [
                            [1.1, 1.0, 1.0],
                            [2.1, 1.0, 1.0],
                            [13.1, 1.0, 1.0],
                            [14.1, 1.0, 1.0],
                            [13.1, 2.0, 1.0],
                        ],
                    },
                    {
                        "time_ps": 20.0,
                        "coordinates_nm": [
                            [1.2, 1.0, 1.0],
                            [2.2, 1.0, 1.0],
                            [3.2, 1.0, 1.0],
                            [4.2, 1.0, 1.0],
                            [3.2, 2.0, 1.0],
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return source_index, generated_index, trajectory


def _invoke(
    *,
    cwd: Path,
    environment: dict[str, str],
):
    def run(args: list[str], stdin_text: str) -> None:
        result = run_command(
            args,
            cwd=cwd,
            env=environment,
            stdin_text=stdin_text,
        )
        assert result.returncode == 0, result.stderr

    return run


def test_artificial_periodic_image_jump_is_reconstructed_not_clipped(
    tmp_path: Path,
    fake_gromacs: tuple[Path, Path],
) -> None:
    executable, _ = fake_gromacs
    environment = dict(os.environ)
    source_index, generated_index, trajectory = _artificial_inputs(tmp_path)
    raw_xvg = tmp_path / "raw.xvg"
    raw_result = run_command(
        build_rmsd_command(
            executable=executable,
            reference_structure=trajectory,
            trajectory=trajectory,
            generated_index=generated_index,
            output_xvg=raw_xvg,
        ),
        cwd=tmp_path,
        env=environment,
        stdin_text="C-alpha\nLIG_HEAVY\n",
    )
    assert raw_result.returncode == 0, raw_result.stderr
    assert parse_xvg(raw_xvg).maximum.rmsd == pytest.approx(10.0)

    workspace = PbcRmsdWorkspace.under(tmp_path)
    workspace.reset()
    write_clustered_rmsd_index(
        source_index=source_index,
        generated_index=generated_index,
        output_index=workspace.rmsd_index,
    )
    invoke = _invoke(cwd=tmp_path, environment=environment)
    reconstruct_clustered_complex(
        executable=executable,
        reference_topology=trajectory,
        trajectory=trajectory,
        source_index=source_index,
        output=workspace.reference,
        begin_time_ps=0.0,
        end_time_ps=0.0,
        invoke=invoke,
    )
    reconstruct_clustered_complex(
        executable=executable,
        reference_topology=trajectory,
        trajectory=trajectory,
        source_index=source_index,
        output=workspace.final_trajectory,
        begin_time_ps=0.0,
        end_time_ps=None,
        invoke=invoke,
    )
    clustered_xvg = tmp_path / "clustered.xvg"
    clustered_result = run_command(
        build_rmsd_command(
            executable=executable,
            reference_structure=workspace.reference,
            trajectory=workspace.final_trajectory,
            generated_index=workspace.rmsd_index,
            output_xvg=clustered_xvg,
        ),
        cwd=tmp_path,
        env=environment,
        stdin_text="C-alpha\nLIG_HEAVY\n",
    )
    assert clustered_result.returncode == 0, clustered_result.stderr
    clustered = parse_xvg(clustered_xvg)
    assert [sample.rmsd for sample in clustered.samples] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-8
    )
    workspace.cleanup()


def test_online_and_final_paths_have_identical_clustered_rmsd(
    tmp_path: Path,
    fake_gromacs: tuple[Path, Path],
) -> None:
    executable, _ = fake_gromacs
    environment = dict(os.environ)
    source_index, generated_index, trajectory = _artificial_inputs(tmp_path)
    workspace = PbcRmsdWorkspace.under(tmp_path)
    workspace.reset()
    write_clustered_rmsd_index(
        source_index=source_index,
        generated_index=generated_index,
        output_index=workspace.rmsd_index,
    )
    online = replica_module._online_snapshot(
        executable=executable,
        reference_topology=trajectory,
        trajectory=trajectory,
        source_index=source_index,
        workspace=workspace,
        begin_time_ps=0.0,
        output_xvg=tmp_path / "online.xvg",
        cwd=tmp_path,
        environment=environment,
        logger=_Logger(),  # type: ignore[arg-type]
        control=_Control(),  # type: ignore[arg-type]
        timeout_seconds=5.0,
    )
    assert online is not None
    invoke = _invoke(cwd=tmp_path, environment=environment)
    reconstruct_clustered_complex(
        executable=executable,
        reference_topology=trajectory,
        trajectory=trajectory,
        source_index=source_index,
        output=workspace.final_trajectory,
        begin_time_ps=0.0,
        end_time_ps=None,
        invoke=invoke,
    )
    final_xvg = tmp_path / "final.xvg"
    result = run_command(
        build_rmsd_command(
            executable=executable,
            reference_structure=workspace.reference,
            trajectory=workspace.final_trajectory,
            generated_index=workspace.rmsd_index,
            output_xvg=final_xvg,
        ),
        cwd=tmp_path,
        env=environment,
        stdin_text="C-alpha\nLIG_HEAVY\n",
    )
    assert result.returncode == 0, result.stderr
    final = parse_xvg(final_xvg)
    assert [(s.time_ps, s.rmsd) for s in online.samples] == pytest.approx(
        [(s.time_ps, s.rmsd) for s in final.samples]
    )
    workspace.cleanup()


def test_online_supplier_advances_cluster_begin_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    begin_times: list[float] = []

    def fake_snapshot(**kwargs: object) -> XvgSeries:
        begin = float(kwargs["begin_time_ps"])
        begin_times.append(begin)
        latest = begin + 10.0
        return XvgSeries(
            path=tmp_path / "online.xvg",
            samples=(XvgSample(time_ps=begin, rmsd=0.1), XvgSample(latest, 0.2)),
        )

    monkeypatch.setattr(replica_module, "_online_snapshot", fake_snapshot)
    supplier = replica_module._make_online_snapshot_supplier(
        executable=tmp_path / "gmx",
        reference_topology=tmp_path / "production.tpr",
        trajectory=tmp_path / "production.xtc",
        source_index=tmp_path / "prepared.ndx",
        workspace=PbcRmsdWorkspace.under(tmp_path),
        output_xvg=tmp_path / "online.xvg",
        cwd=tmp_path,
        environment={},
        logger=_Logger(),  # type: ignore[arg-type]
        control=_Control(),  # type: ignore[arg-type]
        timeout_seconds=1.0,
    )

    supplier()
    supplier()
    supplier()
    assert begin_times == [0.0, 10.0, 20.0]


def test_interrupted_cluster_removes_unique_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_index, _, trajectory = _artificial_inputs(tmp_path)
    workspace = PbcRmsdWorkspace.under(tmp_path)
    workspace.reset()

    def interrupted(
        args: Sequence[str],
        **_kwargs: object,
    ) -> CommandResult:
        return CommandResult(tuple(args), -15, "", "", 1, interrupted=True)

    monkeypatch.setattr(replica_module, "run_cancellable_command", interrupted)
    with pytest.raises(TaskInterrupted):
        replica_module._online_snapshot(
            executable=tmp_path / "gmx",
            reference_topology=trajectory,
            trajectory=trajectory,
            source_index=source_index,
            workspace=workspace,
            begin_time_ps=0.0,
            output_xvg=tmp_path / "online.xvg",
            cwd=tmp_path,
            environment={},
            logger=_Logger(),  # type: ignore[arg-type]
            control=_Control(),  # type: ignore[arg-type]
            timeout_seconds=1.0,
        )
    assert list(workspace.root.iterdir()) == []
    workspace.cleanup()
    assert not workspace.root.exists()
