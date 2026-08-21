"""CLI adapter for deterministic reference validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from gpu_shortmd.analysis.validation import (
    ReferenceValidationError,
    validate_reference_example,
)
from gpu_shortmd.analysis.xvg import XvgParseError


def validate_example_command(
    example_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the validation JSON to a file."),
    ] = None,
) -> None:
    try:
        report = validate_reference_example(example_dir)
    except XvgParseError as exc:
        typer.echo(f"analysis error: {exc}", err=True)
        raise typer.Exit(code=6) from exc
    except ReferenceValidationError as exc:
        typer.echo(f"example validation error: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except Exception as exc:
        typer.echo(
            f"internal software error: {type(exc).__name__}",
            err=True,
        )
        raise typer.Exit(code=9) from exc
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    typer.echo(serialized, nl=False)
    if output is not None:
        try:
            output.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            typer.echo(
                "output error: could not write validation artifact",
                err=True,
            )
            raise typer.Exit(code=5) from exc
