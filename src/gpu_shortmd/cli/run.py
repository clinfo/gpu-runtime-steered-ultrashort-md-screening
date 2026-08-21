"""CLI adapter for the stable run workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from gpu_shortmd.config.loader import ConfigLoadError, load_config
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.workflow.ensemble import resume_run, run_fresh
from gpu_shortmd.workflow.runner import RunConfigurationError, RunExecutionError


def _with_gpu_ids(config: AppConfig, value: str | None) -> AppConfig:
    if value is None:
        return config
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise RunConfigurationError(
            "--gpu-ids must be a comma-separated integer list"
        ) from exc
    payload: dict[str, Any] = config.model_dump(mode="python")
    payload["scheduler"]["gpu_ids"] = gpu_ids
    try:
        return AppConfig.model_validate(payload)
    except ValidationError as exc:
        raise RunConfigurationError(f"invalid --gpu-ids value: {exc}") from exc


def run_command(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="YAML manifest of prepared poses for one shared scheduler run.",
        ),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Resume an existing transactional run directory.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Override the configured output root."),
    ] = None,
    gpu_ids: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated local GPU IDs."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Write the resolved transactional execution plan without MD.",
        ),
    ] = False,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="On resume, explicitly requeue failed replicas.",
        ),
    ] = False,
) -> None:
    try:
        if (config_path is None) == (resume is None):
            raise RunConfigurationError(
                "provide exactly one of --config CONFIG or --resume RUN_DIR"
            )
        if resume is not None:
            if (
                output_dir is not None
                or gpu_ids is not None
                or dry_run
                or manifest is not None
            ):
                raise RunConfigurationError(
                    "--manifest, --output-dir, --gpu-ids, and --dry-run "
                    "are invalid with --resume"
                )
            result = resume_run(
                resume,
                retry_failed=retry_failed,
            )
        else:
            assert config_path is not None
            if retry_failed:
                raise RunConfigurationError("--retry-failed requires --resume RUN_DIR")
            if not config_path.is_file():
                raise RunConfigurationError("configuration file does not exist")
            config = _with_gpu_ids(load_config(config_path), gpu_ids)
            result = run_fresh(
                config,
                config_path=config_path.resolve(),
                pose_manifest_path=(
                    manifest.resolve() if manifest is not None else None
                ),
                output_dir_override=output_dir,
                dry_run=dry_run,
            )
    except ConfigLoadError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except RunConfigurationError as exc:
        typer.echo(f"run configuration error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc
    except RunExecutionError as exc:
        typer.echo(
            f"run failed: {exc}; diagnostics retained in {exc.run_dir}",
            err=True,
        )
        raise typer.Exit(code=exc.exit_code) from exc
    except OSError as exc:
        typer.echo("run I/O error before diagnostics completed", err=True)
        raise typer.Exit(code=5) from exc
    except Exception as exc:
        typer.echo(
            f"internal software error: {type(exc).__name__}",
            err=True,
        )
        raise typer.Exit(code=9) from exc
    typer.echo(f"{result.status}: {result.run_dir}")
    for pose in result.pose_results:
        typer.echo(
            f"Pose {pose['pose_id']}: status={pose['status']}, MD-score="
            + (
                f"{pose['md_score_angstrom']} angstrom"
                if pose["md_score_angstrom"] is not None
                else "null"
            )
        )
    if result.status in {"FAILED"}:
        raise typer.Exit(code=5)
    if result.status in {"INTERRUPTED", "INCOMPLETE"}:
        raise typer.Exit(code=8)
