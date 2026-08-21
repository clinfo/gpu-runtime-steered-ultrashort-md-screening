"""CLI adapter for offline XVG scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from gpu_shortmd.analysis.md_score import calculate_md_score
from gpu_shortmd.analysis.summaries import write_replica_csv
from gpu_shortmd.analysis.xvg import XvgParseError


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = (
            json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata must contain a mapping")
    return value


def score_command(
    xvg: Annotated[
        list[Path] | None,
        typer.Option(
            "--xvg",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="XVG file; repeat --xvg for each completed replica.",
        ),
    ] = None,
    input_unit: Annotated[
        str,
        typer.Option("--input-unit", help="Required XVG RMSD unit."),
    ] = "nm",
    output_unit: Annotated[
        str,
        typer.Option("--output-unit", help="Required public score unit."),
    ] = "angstrom",
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Write the structured result to a file."),
    ] = None,
    csv_output: Annotated[
        Path | None,
        typer.Option("--csv-output", help="Write per-replica maxima as CSV."),
    ] = None,
) -> None:
    if not xvg:
        typer.echo("error: at least one --xvg file is required", err=True)
        raise typer.Exit(code=2)
    if input_unit != "nm" or output_unit != "angstrom":
        typer.echo(
            "error: the stable contract requires --input-unit nm "
            "--output-unit angstrom",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        result = calculate_md_score(
            list(xvg),
            input_unit="nm",
            output_unit="angstrom",
        )
        payload = result.to_dict()
        if metadata is not None:
            payload["metadata"] = _load_metadata(metadata)
    except XvgParseError as exc:
        typer.echo(f"analysis error: {exc}", err=True)
        raise typer.Exit(code=6) from exc
    except ValueError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        typer.echo(
            f"internal software error: {type(exc).__name__}",
            err=True,
        )
        raise typer.Exit(code=9) from exc

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    typer.echo(serialized, nl=False)
    try:
        if json_output is not None:
            json_output.write_text(serialized, encoding="utf-8")
        if csv_output is not None:
            write_replica_csv(result, csv_output)
    except OSError as exc:
        typer.echo("output error: could not write score artifacts", err=True)
        raise typer.Exit(code=5) from exc
