"""Persist stop intent and signal only verified run-owned process groups."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gpu_shortmd.config.loader import ConfigLoadError, load_config
from gpu_shortmd.runtime.processes import terminate_owned_process_group
from gpu_shortmd.runtime.state import RuntimeState, StateError


def stop_command(
    run_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
) -> None:
    resolved = run_dir.resolve()
    state_path = resolved / "state.sqlite3"
    if not state_path.is_file():
        typer.echo("state error: run directory has no state.sqlite3", err=True)
        raise typer.Exit(code=7)
    try:
        state = RuntimeState(state_path)
        state.integrity_check()
        run_id = str(state.run_row()["run_id"])
        config = load_config(resolved / "resolved_config.yaml")
        state.request_stop(run_id=run_id)
        results = [
            terminate_owned_process_group(
                pid=int(process["process_pid"]),
                expected_start_token=str(process["process_start_token"]),
                grace_seconds=config.pruning.grace_period_seconds,
            )
            for process in state.owned_processes(run_id=run_id)
        ]
    except (ConfigLoadError, OSError, StateError) as exc:
        typer.echo(f"state error: {exc}", err=True)
        raise typer.Exit(code=7) from exc
    failed = [result for result in results if not result.terminated]
    if failed:
        typer.echo(
            "stop intent persisted, but one or more owned processes could not "
            "be verified; inspect state before resume",
            err=True,
        )
        raise typer.Exit(code=8)
    typer.echo(f"STOP_REQUESTED: {run_id}")
