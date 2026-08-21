from __future__ import annotations

import sys
from pathlib import Path

from gpu_shortmd.runtime.processes import process_start_token
from gpu_shortmd.runtime.scheduler import TaskControl
from gpu_shortmd.runtime.state import RuntimeState
from gpu_shortmd.util.subprocess import run_cancellable_command
from gpu_shortmd.workflow.stage_marker import (
    STAGE_COMPLETION_MARKER,
    stage_is_reusable,
    stage_marker_identity,
)

RUN_ID = "RUN-20260801-000000-boundary"
POSE_ID = "P00533_EGFR|p2|model4"


def claimed_control(tmp_path: Path) -> tuple[RuntimeState, TaskControl]:
    state = RuntimeState.create(
        tmp_path / "state.sqlite3",
        run_id=RUN_ID,
        pose_id=POSE_ID,
        seeds=(2026080101,),
        gpu_ids=(0,),
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        pruning_enabled=False,
        pruning_threshold_angstrom=None,
    )
    claim = state.claim_task(
        run_id=RUN_ID,
        worker_id="gpu-0",
        gpu_id=0,
        work_stealing=False,
    )
    assert claim is not None
    return state, TaskControl(
        state=state,
        claim=claim,
        pruning_threshold_angstrom=None,
    )


def run_boundary_command(
    *,
    control: TaskControl,
    cwd: Path,
    script: str,
    process_step_id: str,
) -> None:
    result = run_cancellable_command(
        [sys.executable, "-c", script],
        cwd=cwd,
        env={},
        stop_requested=control.stop_requested,
        on_start=lambda pid: control.record_process(
            pid=pid,
            start_token=process_start_token(pid),
            process_step_id=process_step_id,
        ),
        on_exit=control.record_process_exit,
    )
    assert result.returncode == 0
    assert result.interrupted is False


def test_resume_after_grompp_exit_before_next_mdrun(tmp_path: Path) -> None:
    state, control = claimed_control(tmp_path)
    stage = tmp_path / "nvt"
    stage.mkdir()
    run_boundary_command(
        control=control,
        cwd=stage,
        script=(
            "from pathlib import Path; "
            "Path('nvt.tpr').write_text('tpr'); "
            "Path('grompp_processed.mdp').write_text('processed')"
        ),
        process_step_id="replica_01.nvt.grompp",
    )

    before = state.rows("tasks")[0]
    assert before["status"] == "RUNNING"
    assert before["process_state"] == "EXITED"
    assert before["process_returncode"] == 0
    assert before["process_finished_at"]
    assert before["process_step_id"] == "replica_01.nvt.grompp"

    reopened = RuntimeState(state.path)
    reopened.verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=False,
    )
    after = reopened.rows("tasks")[0]
    assert after["status"] == "PENDING"
    assert after["process_state"] == "NONE"
    assert after["process_pid"] is None
    assert after["process_start_token"] is None
    assert after["process_step_id"] is None
    assert (stage / "nvt.tpr").is_file()
    assert not (stage / "nvt.cpt").exists()


def test_resume_after_production_exit_before_marker_or_finish_task(
    tmp_path: Path,
) -> None:
    state, control = claimed_control(tmp_path)
    stage = tmp_path / "production"
    stage.mkdir()
    run_boundary_command(
        control=control,
        cwd=stage,
        script=(
            "from pathlib import Path; "
            "[(Path('production.' + suffix)).write_text(suffix) "
            "for suffix in ('tpr', 'gro', 'cpt', 'log', 'xtc')]"
        ),
        process_step_id="replica_01.production.mdrun",
    )
    resolved_mdp = tmp_path / "production.mdp"
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_mdp.write_text("integrator = md\n", encoding="utf-8")
    resolved_config.write_text("schema_version: 1\n", encoding="utf-8")
    identity = stage_marker_identity(
        stage="production",
        pose_id=POSE_ID,
        replica_id=control.claim.replica_id,
        velocity_seed=control.claim.velocity_seed,
        resolved_mdp=resolved_mdp,
        resolved_config=resolved_config,
    )

    assert not (stage / STAGE_COMPLETION_MARKER).exists()
    assert not stage_is_reusable(
        stage / "production",
        production=True,
        expected_identity=identity,
    )
    assert state.rows("tasks")[0]["process_state"] == "EXITED"
    assert state.rows("tasks")[0]["process_step_id"] == "replica_01.production.mdrun"

    RuntimeState(state.path).verify_resume(
        config_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        retry_failed=False,
    )
    after = state.rows("tasks")[0]
    assert after["status"] == "PENDING"
    assert after["process_state"] == "NONE"
    assert (stage / "production.cpt").is_file()
    assert not (stage / STAGE_COMPLETION_MARKER).exists()
