"""Unified gpu-shortmd command."""

from __future__ import annotations

from typing import Annotated

import typer

from gpu_shortmd import __version__
from gpu_shortmd.cli.inspect import inspect_command
from gpu_shortmd.cli.run import run_command
from gpu_shortmd.cli.score import score_command
from gpu_shortmd.cli.stop import stop_command
from gpu_shortmd.cli.validate_example import validate_example_command

app = typer.Typer(
    name="gpu-shortmd",
    help=("Stable multi-pose prepared-system short-MD workflow and analysis."),
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print the package version and exit.",
        ),
    ] = None,
) -> None:
    """GPU ultrashort-MD screening with explicit scientific contracts."""


app.command("score", help="Score complete existing XVG replicas (stable).")(
    score_command
)
app.command(
    "inspect",
    help="Inspect a prepared system and environment without starting MD (stable).",
)(inspect_command)
app.command(
    "run",
    help="Run the prepared-system NVT, NPT, and production workflow (stable).",
)(run_command)
app.command(
    "validate-example",
    help="Validate the bundled deterministic EGFR XVG example (stable).",
)(validate_example_command)
app.command(
    "stop",
    help="Persist stop intent and stop verified owned processes (stable).",
)(stop_command)


def main() -> None:
    app()
