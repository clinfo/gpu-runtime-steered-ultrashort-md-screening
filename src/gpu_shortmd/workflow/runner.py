"""Single-replica NVT → NPT → production runner for Chunk 2."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_shortmd import __version__
from gpu_shortmd.analysis.md_score import ScoreResult, calculate_md_score
from gpu_shortmd.analysis.units import convert_rmsd
from gpu_shortmd.analysis.xvg import XvgParseError, parse_xvg
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.gromacs.grompp import build_grompp_command
from gpu_shortmd.gromacs.index import IndexValidationError
from gpu_shortmd.gromacs.mdp import resolve_stage_mdps
from gpu_shortmd.gromacs.mdrun import build_mdrun_command
from gpu_shortmd.gromacs.rmsd import (
    PbcRmsdWorkspace,
    build_heavy_atom_index_command,
    build_rmsd_command,
    reconstruct_clustered_complex,
    validate_rmsd_groups,
    write_clustered_rmsd_index,
)
from gpu_shortmd.util.checksums import sha256_file, write_checksum_file
from gpu_shortmd.util.files import ensure_new_directory, write_json, write_yaml
from gpu_shortmd.util.logging import RunLogger, utc_now
from gpu_shortmd.util.subprocess import (
    CommandResult,
    CommandTimeoutError,
    run_command,
)
from gpu_shortmd.workflow.filesystem_identity import pose_filesystem_key
from gpu_shortmd.workflow.inspection import InspectionReport, inspect_configuration
from gpu_shortmd.workflow.prepared_input import snapshot_prepared_system


class RunConfigurationError(ValueError):
    """Raised when Chunk 2 receives a later-phase configuration."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class RunExecutionError(RuntimeError):
    """Raised after a failed run has persisted diagnostic output."""

    def __init__(self, message: str, *, exit_code: int, run_dir: Path) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.run_dir = run_dir


class StageExecutionError(RuntimeError):
    """Raised when a GROMACS stage fails or omits a required output."""

    def __init__(
        self,
        message: str,
        *,
        stage_reached: str = "NONE",
        step_id: str = "run",
    ) -> None:
        super().__init__(message)
        self.stage_reached = stage_reached
        self.step_id = step_id


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    status: str
    score: ScoreResult


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "run"


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"RUN-{timestamp}-{uuid.uuid4().hex[:8]}"


def _output_root(
    config: AppConfig,
    *,
    config_path: Path,
    override: Path | None,
) -> Path:
    configured = override if override is not None else Path(config.run.output_dir)
    return (
        configured if configured.is_absolute() else config_path.parent / configured
    ).resolve()


def _selected_gpu(
    config: AppConfig,
    *,
    report: InspectionReport,
    env: Mapping[str, str],
) -> tuple[int, str]:
    requested = config.scheduler.gpu_ids
    if requested == "auto":
        if not report.visible_gpu_ids:
            raise RunConfigurationError("no GPU is available")
        visible_environment = env.get("CUDA_VISIBLE_DEVICES")
        if visible_environment:
            physical = visible_environment.split(",", 1)[0].strip()
            return report.visible_gpu_ids[0], physical
        return report.visible_gpu_ids[0], str(report.visible_gpu_ids[0])
    if len(requested) != 1:
        raise RunConfigurationError(
            "Chunk 2 single-replica runner requires exactly one GPU ID"
        )
    return requested[0], str(requested[0])


def _validate_chunk2_scope(config: AppConfig) -> int:
    if config.run.resume:
        raise RunConfigurationError("resume is not enabled before Chunk 3")
    if config.restart.retry_failed:
        raise RunConfigurationError(
            "retry_failed is not enabled before transactional state in Chunk 3"
        )
    if config.trajectory.replicas != 1:
        raise RunConfigurationError(
            "single-replica validation requires trajectory.replicas = 1"
        )
    if config.trajectory.seeds is None:
        raise RunConfigurationError(
            "single-replica validation requires one explicit persisted seed"
        )
    if config.pruning.enabled:
        raise RunConfigurationError("pruning is not enabled before Chunk 3")
    if config.scheduler.backend != "local":
        raise RunConfigurationError("stable runner requires scheduler.backend = local")
    if config.scheduler.work_stealing:
        raise RunConfigurationError("work stealing is not enabled before Chunk 4")
    if config.scheduler.tasks_per_gpu != 1:
        raise RunConfigurationError(
            "Chunk 2 runner requires scheduler.tasks_per_gpu = 1"
        )
    return config.trajectory.seeds[0]


def _validate_output(
    path: Path,
    *,
    step_id: str,
    stage_reached: str = "NONE",
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise StageExecutionError(
            f"{step_id}: required output is missing or empty",
            stage_reached=stage_reached,
            step_id=step_id,
        )


def _run_logged_command(
    *,
    command: list[str],
    step_id: str,
    cwd: Path,
    env: Mapping[str, str],
    logger: RunLogger,
    stdin_text: str | None = None,
    stage_reached: str = "NONE",
) -> CommandResult:
    logger.event(
        step_id=step_id,
        level="INFO",
        category="EXECUTION",
        status="STARTED",
        code="EXTERNAL_COMMAND_STARTED",
        message=f"Started {step_id}.",
        params={"args": command},
    )
    try:
        result = run_command(
            command,
            cwd=cwd,
            env=env,
            stdin_text=stdin_text,
        )
    except CommandTimeoutError as exc:
        logger.event(
            step_id=step_id,
            level="ERROR",
            category="TIMEOUT",
            status="FAILED",
            code="EXTERNAL_COMMAND_TIMED_OUT",
            message=str(exc),
            params={"args": command},
            exception_type=type(exc).__name__,
            exception_excerpt=str(exc),
        )
        raise StageExecutionError(
            str(exc),
            stage_reached=stage_reached,
            step_id=step_id,
        ) from exc
    except OSError as exc:
        logger.event(
            step_id=step_id,
            level="ERROR",
            category="EXECUTION",
            status="FAILED",
            code="EXTERNAL_COMMAND_COULD_NOT_EXECUTE",
            message=str(exc),
            params={"args": command},
            exception_type=type(exc).__name__,
            exception_excerpt=str(exc),
        )
        raise StageExecutionError(
            str(exc),
            stage_reached=stage_reached,
            step_id=step_id,
        ) from exc
    logger.command_output(
        step_id=step_id,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    logger.event(
        step_id=step_id,
        level="INFO" if result.returncode == 0 else "ERROR",
        category="EXECUTION",
        status="SUCCEEDED" if result.returncode == 0 else "FAILED",
        code=(
            "EXTERNAL_COMMAND_SUCCEEDED"
            if result.returncode == 0
            else "EXTERNAL_COMMAND_FAILED"
        ),
        message=f"{step_id} exited with code {result.returncode}.",
        params={"args": command, "exit_code": result.returncode},
        duration_ms=result.duration_ms,
    )
    if result.returncode != 0:
        raise StageExecutionError(
            f"{step_id} failed with exit code {result.returncode}",
            stage_reached=stage_reached,
            step_id=step_id,
        )
    return result


def _stage_inputs(
    stage: str,
    *,
    frozen_start: Path,
    replica_dir: Path,
) -> tuple[Path, Path, Path | None]:
    if stage == "nvt":
        return frozen_start, frozen_start, None
    predecessor = "nvt" if stage == "npt" else "npt"
    predecessor_base = replica_dir / predecessor / predecessor
    coordinates = predecessor_base.with_suffix(".gro")
    return coordinates, coordinates, predecessor_base.with_suffix(".cpt")


def _execute_stages(
    *,
    config: AppConfig,
    report: InspectionReport,
    frozen_start: Path,
    frozen_topology: Path,
    frozen_index: Path,
    resolved_mdps: dict[str, Path],
    replica_dir: Path,
    process_env: Mapping[str, str],
    logger: RunLogger,
) -> None:
    if report.gromacs_executable is None:
        raise StageExecutionError(
            "preflight did not resolve GROMACS",
            step_id="preflight",
        )
    for stage in ("nvt", "npt", "production"):
        try:
            stage_dir = replica_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=False)
            base = stage_dir / stage
            coordinates, reference, checkpoint = _stage_inputs(
                stage,
                frozen_start=frozen_start,
                replica_dir=replica_dir,
            )
            command = build_grompp_command(
                executable=report.gromacs_executable,
                mdp=resolved_mdps[stage],
                coordinates=coordinates,
                reference_coordinates=reference,
                checkpoint=checkpoint,
                topology=frozen_topology,
                index=frozen_index,
                output_tpr=base.with_suffix(".tpr"),
                processed_mdp=stage_dir / "grompp_processed.mdp",
                maxwarn=config.gromacs.maxwarn,
            )
            _run_logged_command(
                command=command,
                step_id=f"{stage}.grompp",
                cwd=stage_dir,
                env=process_env,
                logger=logger,
            )
            _validate_output(base.with_suffix(".tpr"), step_id=f"{stage}.grompp")
            processed_mdp = stage_dir / "grompp_processed.mdp"
            _validate_output(processed_mdp, step_id=f"{stage}.grompp")
            processed_mdp.write_text(
                logger.redact(processed_mdp.read_text(encoding="utf-8")),
                encoding="utf-8",
            )

            _run_logged_command(
                command=build_mdrun_command(
                    executable=report.gromacs_executable,
                    deffnm=base,
                    config=config.gromacs,
                ),
                step_id=f"{stage}.mdrun",
                cwd=stage_dir,
                env=process_env,
                logger=logger,
            )
            for suffix in (".gro", ".cpt", ".log"):
                _validate_output(base.with_suffix(suffix), step_id=f"{stage}.mdrun")
            gromacs_log = base.with_suffix(".log")
            gromacs_log.write_text(
                logger.redact(
                    gromacs_log.read_text(encoding="utf-8", errors="replace")
                ),
                encoding="utf-8",
            )
            if stage == "production":
                _validate_output(
                    base.with_suffix(".xtc"),
                    step_id="production.mdrun",
                )
        except StageExecutionError as exc:
            raise StageExecutionError(
                str(exc),
                stage_reached=stage.upper(),
                step_id=exc.step_id,
            ) from exc


def _write_rmsd_angstrom_csv(xvg_path: Path, output: Path) -> None:
    series = parse_xvg(xvg_path)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simulation_time_ps", "rmsd_angstrom"])
        for sample in series.samples:
            writer.writerow(
                [
                    sample.time_ps,
                    convert_rmsd(
                        sample.rmsd,
                        input_unit="nm",
                        output_unit="angstrom",
                    ),
                ]
            )


def _write_summaries(
    *,
    run_dir: Path,
    run_id: str,
    pose_id: str,
    seed: int,
    gpu_id: int,
    started_at: str,
    finished_at: str,
    score: ScoreResult | None,
    status: str,
    stage_reached: str,
    exit_code: int,
    production_time_ps: float,
) -> None:
    replica_maximum = score.replica_maxima[0] if score is not None else None
    pose_fields = [
        "run_id",
        "pose_id",
        "status",
        "n_replicas_requested",
        "n_replicas_completed",
        "md_score_angstrom",
        "observed_max_rmsd_angstrom",
        "pruning_enabled",
        "pruning_threshold_angstrom",
        "trigger_replica_id",
        "trigger_simulation_time_ps",
        "docking_score_kcal_mol",
        "started_at",
        "finished_at",
    ]
    pose_row: dict[str, Any] = {
        "run_id": run_id,
        "pose_id": pose_id,
        "status": status,
        "n_replicas_requested": 1,
        "n_replicas_completed": 1 if score is not None else 0,
        "md_score_angstrom": score.md_score_angstrom if score else None,
        "observed_max_rmsd_angstrom": (
            score.observed_max_rmsd_angstrom if score else None
        ),
        "pruning_enabled": False,
        "pruning_threshold_angstrom": None,
        "trigger_replica_id": None,
        "trigger_simulation_time_ps": None,
        "docking_score_kcal_mol": None,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    with (run_dir / "pose_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pose_fields)
        writer.writeheader()
        writer.writerow(pose_row)

    replica_fields = [
        "run_id",
        "pose_id",
        "replica_id",
        "status",
        "gpu_id",
        "velocity_seed",
        "stage_reached",
        "trajectory_time_completed_ps",
        "max_rmsd_nm",
        "max_rmsd_angstrom",
        "triggered_pruning",
        "exit_code",
        "started_at",
        "finished_at",
    ]
    replica_row: dict[str, Any] = {
        "run_id": run_id,
        "pose_id": pose_id,
        "replica_id": "replica_01",
        "status": status,
        "gpu_id": gpu_id,
        "velocity_seed": seed,
        "stage_reached": stage_reached,
        "trajectory_time_completed_ps": production_time_ps if score else 0.0,
        "max_rmsd_nm": replica_maximum.max_rmsd_nm if replica_maximum else None,
        "max_rmsd_angstrom": (
            replica_maximum.max_rmsd_angstrom if replica_maximum else None
        ),
        "triggered_pruning": False,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    with (run_dir / "replica_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=replica_fields)
        writer.writeheader()
        writer.writerow(replica_row)


def _write_artifact_manifest(run_dir: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_manifest.csv", "checksums.sha256"}
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name in excluded
            or path.name.endswith("_artifacts.json")
        ):
            continue
        artifacts.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (run_dir / "artifact_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(artifacts)
    return artifacts


def run_single_replica(
    config: AppConfig,
    *,
    config_path: Path,
    output_dir_override: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RunResult:
    seed = _validate_chunk2_scope(config)
    environment = dict(os.environ if env is None else env)
    report = inspect_configuration(
        config,
        config_path=config_path,
        env=environment,
        output_dir_override=output_dir_override,
    )
    if report.exit_code != 0:
        raise RunConfigurationError(
            f"preflight failed with exit code {report.exit_code}",
            exit_code=report.exit_code,
        )
    if report.topology_resolution is None:
        raise RunConfigurationError("preflight did not resolve topology")
    if report.gromacs_executable is None:
        raise RunConfigurationError("preflight did not resolve GROMACS")

    gpu_id, gpu_environment_id = _selected_gpu(
        config,
        report=report,
        env=environment,
    )
    run_id = _new_run_id()
    pose_id = config.run.name or report.prepared_system.root.name
    run_dir = (
        _output_root(
            config,
            config_path=config_path,
            override=output_dir_override,
        )
        / f"{pose_filesystem_key(pose_id)}_{run_id}"
    )
    ensure_new_directory(run_dir)
    started_at = utc_now()
    logger = RunLogger(
        run_dir=run_dir,
        run_id=run_id,
        redacted_values=[
            str(run_dir),
            str(report.prepared_system.root),
            str(config_path.parent.resolve()),
            str(Path.home()),
            str(report.gromacs_executable.parent),
            (
                str(report.gromacs_version.data_prefix)
                if report.gromacs_version is not None
                and report.gromacs_version.data_prefix is not None
                else ""
            ),
        ],
        redacted_tokens=[platform.node()],
    )
    with (run_dir / "logs" / "run_index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "ts": started_at,
                    "run_id": run_id,
                    "status": "RUNNING",
                    "manifest": f"{run_id}_manifest.json",
                },
                sort_keys=True,
            )
            + "\n"
        )
    logger.event(
        step_id="run",
        level="INFO",
        category="ORCHESTRATION",
        status="STARTED",
        code="RUN_CREATED",
        message="Created an immutable single-replica run.",
    )
    write_json(run_dir / "audit" / "preflight.json", report.public_dict())

    frozen, input_manifest = snapshot_prepared_system(
        report.prepared_system,
        topology_resolution=report.topology_resolution,
        destination=run_dir / "inputs",
    )
    write_json(
        run_dir / "input_manifest.json",
        {"schema_version": 1, "files": input_manifest},
    )
    resolved_config = config.model_dump(mode="json")
    resolved_config["run"]["output_dir"] = "."
    resolved_config["input"]["prepared_system_dir"] = "inputs"
    resolved_config["gromacs"]["executable"] = report.gromacs_executable.name
    resolved_config["scheduler"]["gpu_ids"] = [gpu_id]
    for stage, mdp_path in frozen.mdps.items():
        resolved_config["stages"][stage]["mdp"] = mdp_path.relative_to(
            frozen.root
        ).as_posix()
    write_yaml(run_dir / "resolved_config.yaml", resolved_config)
    write_json(
        run_dir / "environment.json",
        {
            "schema_version": 1,
            "package_version": __version__,
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "gromacs": (
                report.gromacs_version.public_dict()
                if report.gromacs_version is not None
                else None
            ),
            "gpu_id": gpu_id,
        },
    )
    logger.write_manifest(
        {
            "schema_version": 1,
            "run_id": run_id,
            "pose_id": pose_id,
            "status": "RUNNING",
            "started_at": started_at,
            "input_manifest": "input_manifest.json",
            "resolved_config": "resolved_config.yaml",
        }
    )

    replica_dir = run_dir / "poses" / pose_filesystem_key(pose_id) / "replica_01"
    replica_dir.mkdir(parents=True, exist_ok=False)
    resolved_mdps = resolve_stage_mdps(
        config,
        prepared_root=frozen.root,
        destination=replica_dir / "resolved_mdp",
        velocity_seed=seed,
    )
    process_env = dict(environment)
    process_env["CUDA_VISIBLE_DEVICES"] = gpu_environment_id
    score: ScoreResult | None = None
    status = "FAILED"
    stage_reached = "NONE"
    failure: Exception | None = None
    failure_exit = 5
    failed_step_id = "run"
    pbc_workspace = PbcRmsdWorkspace.under(replica_dir)
    pbc_workspace.reset()
    try:
        _execute_stages(
            config=config,
            report=report,
            frozen_start=frozen.start_structure,
            frozen_topology=frozen.topology,
            frozen_index=frozen.index,
            resolved_mdps=resolved_mdps,
            replica_dir=replica_dir,
            process_env=process_env,
            logger=logger,
        )
        stage_reached = "PRODUCTION"
        rmsd_xvg = replica_dir / "rmsd_time_series_nm.xvg"
        if report.gromacs_executable is None:
            raise StageExecutionError(
                "GROMACS executable disappeared after preflight",
                stage_reached=stage_reached,
                step_id="analysis.rmsd",
            )
        production_tpr = replica_dir / "production" / "production.tpr"
        generated_index = replica_dir / "rmsd_groups.ndx"
        _run_logged_command(
            command=build_heavy_atom_index_command(
                executable=report.gromacs_executable,
                reference_topology=production_tpr,
                source_index=frozen.index,
                generated_index=generated_index,
                ligand_resname=config.input.ligand_resname,
            ),
            step_id="analysis.rmsd_groups",
            cwd=replica_dir,
            env=process_env,
            logger=logger,
            stage_reached=stage_reached,
        )
        _validate_output(
            generated_index,
            step_id="analysis.rmsd_groups",
            stage_reached=stage_reached,
        )
        alpha_count, heavy_count = validate_rmsd_groups(generated_index)
        write_clustered_rmsd_index(
            source_index=frozen.index,
            generated_index=generated_index,
            output_index=pbc_workspace.rmsd_index,
        )

        def invoke_cluster(command: list[str], stdin_text: str) -> None:
            _run_logged_command(
                command=command,
                step_id="analysis.pbc_cluster",
                cwd=replica_dir,
                env=process_env,
                logger=logger,
                stdin_text=stdin_text,
                stage_reached=stage_reached,
            )

        reconstruct_clustered_complex(
            executable=report.gromacs_executable,
            reference_topology=production_tpr,
            trajectory=replica_dir / "production" / "production.xtc",
            source_index=frozen.index,
            output=pbc_workspace.reference,
            begin_time_ps=0.0,
            end_time_ps=0.0,
            invoke=invoke_cluster,
        )
        reconstruct_clustered_complex(
            executable=report.gromacs_executable,
            reference_topology=production_tpr,
            trajectory=replica_dir / "production" / "production.xtc",
            source_index=frozen.index,
            output=pbc_workspace.final_trajectory,
            begin_time_ps=0.0,
            end_time_ps=None,
            invoke=invoke_cluster,
        )
        rmsd_command = build_rmsd_command(
            executable=report.gromacs_executable,
            reference_structure=pbc_workspace.reference,
            trajectory=pbc_workspace.final_trajectory,
            generated_index=pbc_workspace.rmsd_index,
            output_xvg=rmsd_xvg,
        )
        logger.event(
            step_id="analysis.rmsd",
            level="INFO",
            category="RMSD",
            status="STARTED",
            code="RMSD_COMMAND_STARTED",
            message="Started unweighted ligand-heavy RMSD after protein C-alpha fit.",
            params={
                "args": rmsd_command,
                "fit_group": "C-alpha",
                "measurement_group": "LIG_HEAVY",
                "pbc_reconstruction_group": "Protein_LIG",
                "pbc_mode": "cluster",
                "mass_weighted": False,
                "heavy_atom_selection": "TPR mass > 2.5 Da",
            },
        )
        _run_logged_command(
            command=rmsd_command,
            step_id="analysis.rmsd",
            cwd=replica_dir,
            env=process_env,
            logger=logger,
            stdin_text="C-alpha\nLIG_HEAVY\n",
            stage_reached=stage_reached,
        )
        _validate_output(
            rmsd_xvg,
            step_id="analysis.rmsd",
            stage_reached=stage_reached,
        )
        rmsd_xvg.write_text(
            logger.redact(rmsd_xvg.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
        score = calculate_md_score(
            [rmsd_xvg],
            input_unit="nm",
            output_unit="angstrom",
            requested_replicas=1,
        )
        _write_rmsd_angstrom_csv(
            rmsd_xvg,
            replica_dir / "rmsd_time_series_angstrom.csv",
        )
        logger.event(
            step_id="analysis.rmsd",
            level="INFO",
            category="RMSD",
            status="SUCCEEDED",
            code="RMSD_CALCULATED",
            message="Calculated ligand-heavy RMSD after protein C-alpha fit.",
            metrics={
                "fit_atom_count": alpha_count,
                "ligand_heavy_atom_count": heavy_count,
                "max_rmsd_angstrom": score.md_score_angstrom,
            },
        )
        status = "COMPLETED"
    except XvgParseError as exc:
        failure = exc
        failure_exit = 6
        failed_step_id = "analysis.score"
    except IndexValidationError as exc:
        failure = exc
        failure_exit = 6
        failed_step_id = "analysis.rmsd"
    except StageExecutionError as exc:
        failure = exc
        failure_exit = 5
        stage_reached = exc.stage_reached
        failed_step_id = exc.step_id
    except (CommandTimeoutError, OSError) as exc:
        failure = exc
        failure_exit = 5
    except Exception as exc:
        failure = exc
        failure_exit = 9
    finally:
        pbc_workspace.cleanup()

    finished_at = utc_now()
    _write_summaries(
        run_dir=run_dir,
        run_id=run_id,
        pose_id=pose_id,
        seed=seed,
        gpu_id=gpu_id,
        started_at=started_at,
        finished_at=finished_at,
        score=score,
        status=status,
        stage_reached=stage_reached,
        exit_code=0 if failure is None else failure_exit,
        production_time_ps=config.trajectory.production_time_ns * 1000,
    )
    if failure is not None:
        internal_failure = failure_exit == 9
        logger.issue(
            step_id=failed_step_id,
            severity="CRITICAL" if internal_failure else "HIGH",
            code=(
                "INTERNAL_SOFTWARE_ERROR"
                if internal_failure
                else "RUN_EXECUTION_FAILED"
            ),
            message=str(failure),
            evidence=["Inspect stage logs and expected output files."],
            suggested_action=(
                "File a bug report with the sanitized run bundle."
                if internal_failure
                else "Correct the environment/input and start a new run."
            ),
        )
        logger.event(
            step_id=failed_step_id,
            level="ERROR",
            category="EXECUTION",
            status="FAILED",
            code="INTERNAL_SOFTWARE_ERROR" if internal_failure else "RUN_FAILED",
            message=str(failure),
            exception_type=type(failure).__name__,
            exception_excerpt=str(failure),
            suggested_action=(
                "File a bug report with the sanitized run bundle."
                if internal_failure
                else "Inspect the structured issue and stage logs."
            ),
        )
    else:
        logger.event(
            step_id="run",
            level="INFO",
            category="ORCHESTRATION",
            status="SUCCEEDED",
            code="RUN_COMPLETED",
            message="Single-replica run completed.",
        )

    (run_dir / "run_report.md").write_text(
        "\n".join(
            [
                f"# Run report: {run_id}",
                "",
                f"- Status: {status}",
                f"- Pose: `{pose_id}`",
                "- Replicas requested: 1",
                (
                    "- MD-score (angstrom): "
                    f"{score.md_score_angstrom if score else 'null'}"
                ),
                "- Validation surface: single-replica trajectory integration",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.write_summary(
        {
            "overall_status": status,
            "top_issues": (
                []
                if failure is None
                else [
                    (
                        "INTERNAL_SOFTWARE_ERROR"
                        if failure_exit == 9
                        else "RUN_EXECUTION_FAILED"
                    )
                ]
            ),
            "likely_root_causes": [] if failure is None else [type(failure).__name__],
            "first_checks": [
                "Review the preflight report.",
                "Review stage stderr and required outputs.",
            ],
            "failed_steps": [] if failure is None else [failed_step_id],
            "partial_outputs": [] if failure is None else ["poses/"],
            "next_actions": (
                ["Archive the integration report."]
                if failure is None
                else ["Correct the recorded cause and start a new run."]
            ),
        }
    )
    logger.write_manifest(
        {
            "schema_version": 1,
            "run_id": run_id,
            "pose_id": pose_id,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "input_manifest": "input_manifest.json",
            "resolved_config": "resolved_config.yaml",
        }
    )
    with (run_dir / "logs" / "run_index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "ts": finished_at,
                    "run_id": run_id,
                    "status": status,
                    "manifest": f"{run_id}_manifest.json",
                },
                sort_keys=True,
            )
            + "\n"
        )
    artifacts = _write_artifact_manifest(run_dir)
    logger.write_artifacts({"schema_version": 1, "artifacts": artifacts})
    write_checksum_file(
        (path for path in run_dir.rglob("*") if path.is_file()),
        root=run_dir,
        output=run_dir / "checksums.sha256",
    )
    if failure is not None:
        raise RunExecutionError(
            str(failure),
            exit_code=failure_exit,
            run_dir=run_dir,
        ) from failure
    assert score is not None
    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        score=score,
    )
