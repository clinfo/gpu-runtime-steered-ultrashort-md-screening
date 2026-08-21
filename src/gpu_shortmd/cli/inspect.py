"""CLI adapter for prepared-input inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gpu_shortmd.config.loader import ConfigLoadError, load_config
from gpu_shortmd.workflow.inspection import (
    inspect_configuration,
    inspection_json,
    write_inspection_outputs,
)


def inspect_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    json_stdout: Annotated[
        bool,
        typer.Option("--json", help="Print the machine-readable report."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Directory for persistent inspect reports."),
    ] = None,
) -> None:
    try:
        config = load_config(config_path)
    except ConfigLoadError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    try:
        report = inspect_configuration(config, config_path=config_path.resolve())
        output_dir = output or config_path.parent / "inspect_report"
        write_inspection_outputs(report, config=config, output_dir=output_dir)
    except OSError as exc:
        typer.echo("inspection output error", err=True)
        raise typer.Exit(code=5) from exc
    except Exception as exc:
        typer.echo(
            f"internal software error: {type(exc).__name__}",
            err=True,
        )
        raise typer.Exit(code=9) from exc
    if json_stdout:
        typer.echo(inspection_json(report), nl=False)
    else:
        typer.echo(
            f"{report.overall_status}: {len(report.checks)} checks; "
            f"reports written to {output_dir}"
        )
    if report.exit_code:
        raise typer.Exit(code=report.exit_code)
