"""One GROMACS replica executed under transactional scheduler control."""

from __future__ import annotations

import csv
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from gpu_shortmd.analysis.units import convert_rmsd
from gpu_shortmd.analysis.xvg import XvgSeries, parse_xvg
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.gromacs.grompp import build_grompp_command
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
from gpu_shortmd.runtime.monitoring import (
    MonitoringError,
    MonitoringOutcome,
    monitor_rmsd_snapshots,
    observe_first_pruning_crossing,
)
from gpu_shortmd.runtime.processes import process_start_token
from gpu_shortmd.runtime.scheduler import (
    ExecutionResult,
    TaskControl,
    TaskInterrupted,
)
from gpu_shortmd.runtime.state import TaskClaim
from gpu_shortmd.util.logging import RunLogger
from gpu_shortmd.util.subprocess import (
    CommandResult,
    CommandTimeoutError,
    run_cancellable_command,
)
from gpu_shortmd.workflow.filesystem_identity import pose_filesystem_key
from gpu_shortmd.workflow.inspection import InspectionReport
from gpu_shortmd.workflow.prepared_input import PreparedSystem
from gpu_shortmd.workflow.runner import StageExecutionError
from gpu_shortmd.workflow.stage_marker import (
    StageMarkerError,
    invalidate_stage_marker,
    stage_is_reusable,
    stage_marker_identity,
    write_stage_completion_marker,
)


def _required_output(path: Path, *, step_id: str, stage: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise StageExecutionError(
            f"{step_id}: required output is missing or empty",
            stage_reached=stage,
            step_id=step_id,
        )


def _logged_command(
    *,
    command: list[str],
    step_id: str,
    cwd: Path,
    env: Mapping[str, str],
    logger: RunLogger,
    control: TaskControl,
    stdin_text: str | None = None,
    local_stop_requested: Callable[[], bool] | None = None,
    termination_grace_seconds: float = 5.0,
) -> CommandResult:
    logger.event(
        step_id=step_id,
        level="INFO",
        category="EXECUTION",
        status="STARTED",
        code="EXTERNAL_COMMAND_STARTED",
        message=f"Started {step_id}.",
        params={"args": command, "task_id": control.claim.task_id},
    )

    def register(pid: int) -> None:
        control.record_process(
            pid=pid,
            start_token=process_start_token(pid),
            process_step_id=step_id,
        )

    try:
        result = run_cancellable_command(
            command,
            cwd=cwd,
            env=env,
            stdin_text=stdin_text,
            stop_requested=(
                lambda: (
                    control.triggered_pruning
                    or control.stop_requested()
                    or (
                        local_stop_requested()
                        if local_stop_requested is not None
                        else False
                    )
                )
            ),
            on_start=register,
            on_exit=control.record_process_exit,
            termination_grace_seconds=termination_grace_seconds,
        )
    except CommandTimeoutError as exc:
        raise StageExecutionError(
            str(exc),
            step_id=step_id,
        ) from exc
    logger.command_output(
        step_id=step_id,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if result.interrupted:
        raise TaskInterrupted(f"{step_id} stopped by runtime steering")
    if result.returncode != 0:
        raise StageExecutionError(
            f"{step_id} failed with exit code {result.returncode}",
            step_id=step_id,
        )
    logger.event(
        step_id=step_id,
        level="INFO",
        category="EXECUTION",
        status="SUCCEEDED",
        code="EXTERNAL_COMMAND_SUCCEEDED",
        message=f"{step_id} exited with code 0.",
        params={"args": command, "exit_code": 0},
        duration_ms=result.duration_ms,
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


def _write_angstrom_series(xvg_path: Path, output: Path) -> None:
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


def _online_snapshot(
    *,
    executable: Path,
    reference_topology: Path,
    trajectory: Path,
    source_index: Path,
    workspace: PbcRmsdWorkspace,
    begin_time_ps: float,
    output_xvg: Path,
    cwd: Path,
    environment: Mapping[str, str],
    logger: RunLogger,
    control: TaskControl,
    timeout_seconds: float,
) -> XvgSeries | None:
    if not trajectory.is_file() or trajectory.stat().st_size == 0:
        return None

    def invoke(command: list[str], stdin_text: str) -> None:
        result = run_cancellable_command(
            command,
            cwd=cwd,
            env=environment,
            stdin_text=stdin_text,
            stop_requested=lambda: (
                control.triggered_pruning or control.stop_requested()
            ),
            on_start=lambda _pid: None,
            on_exit=lambda _returncode: None,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=1.0,
        )
        if result.interrupted:
            raise TaskInterrupted("online RMSD reconstruction was interrupted")
        if result.returncode != 0:
            raise RuntimeError("online GROMACS PBC/RMSD snapshot returned nonzero")
        params = {"command": command[1]}
        if "-b" in command:
            params["begin_time_ps"] = command[command.index("-b") + 1]
        logger.event(
            step_id=f"{control.claim.replica_id}.analysis.online_snapshot",
            level="INFO",
            category="RMSD",
            status="SUCCEEDED",
            code="ONLINE_ANALYSIS_COMMAND_SUCCEEDED",
            message="Completed one online PBC/RMSD analysis command.",
            pose_id=control.claim.pose_id,
            params=params,
            duration_ms=result.duration_ms,
        )

    if not workspace.reference.is_file():
        reconstruct_clustered_complex(
            executable=executable,
            reference_topology=reference_topology,
            trajectory=trajectory,
            source_index=source_index,
            output=workspace.reference,
            begin_time_ps=0.0,
            end_time_ps=0.0,
            invoke=invoke,
        )
    reconstruct_clustered_complex(
        executable=executable,
        reference_topology=reference_topology,
        trajectory=trajectory,
        source_index=source_index,
        output=workspace.online_trajectory,
        begin_time_ps=begin_time_ps,
        end_time_ps=None,
        invoke=invoke,
    )
    temporary_xvg = output_xvg.with_name(
        f".{output_xvg.stem}.{uuid.uuid4().hex}{output_xvg.suffix}"
    )
    snapshot_command = build_rmsd_command(
        executable=executable,
        reference_structure=workspace.reference,
        trajectory=workspace.online_trajectory,
        generated_index=workspace.rmsd_index,
        output_xvg=temporary_xvg,
    )
    try:
        invoke(snapshot_command, "C-alpha\nLIG_HEAVY\n")
        series = parse_xvg(temporary_xvg)
        temporary_xvg.write_text(
            logger.redact(
                temporary_xvg.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            ),
            encoding="utf-8",
        )
        temporary_xvg.replace(output_xvg)
        return XvgSeries(path=output_xvg, samples=series.samples)
    finally:
        if temporary_xvg.exists():
            temporary_xvg.unlink()


def _make_online_snapshot_supplier(
    *,
    executable: Path,
    reference_topology: Path,
    trajectory: Path,
    source_index: Path,
    workspace: PbcRmsdWorkspace,
    output_xvg: Path,
    cwd: Path,
    environment: Mapping[str, str],
    logger: RunLogger,
    control: TaskControl,
    timeout_seconds: float,
) -> Callable[[], XvgSeries | None]:
    begin_time_ps = 0.0

    def supply() -> XvgSeries | None:
        nonlocal begin_time_ps
        snapshot = _online_snapshot(
            executable=executable,
            reference_topology=reference_topology,
            trajectory=trajectory,
            source_index=source_index,
            workspace=workspace,
            begin_time_ps=begin_time_ps,
            output_xvg=output_xvg,
            cwd=cwd,
            environment=environment,
            logger=logger,
            control=control,
            timeout_seconds=timeout_seconds,
        )
        if snapshot is not None and snapshot.samples:
            begin_time_ps = snapshot.samples[-1].time_ps
        return snapshot

    return supply


def _monitor_target(
    *,
    finished: threading.Event,
    poll_interval_seconds: float,
    snapshot_supplier: Callable[[], XvgSeries | None],
    control: TaskControl,
    outcomes: list[MonitoringOutcome],
    interruptions: list[TaskInterrupted],
    errors: list[BaseException],
    cancel: threading.Event,
) -> None:
    try:
        outcomes.append(
            monitor_rmsd_snapshots(
                finished=finished,
                poll_interval_seconds=poll_interval_seconds,
                snapshot_supplier=snapshot_supplier,
                control=control,
            )
        )
    except TaskInterrupted as exc:
        interruptions.append(exc)
        cancel.set()
    except BaseException as exc:
        errors.append(exc)
        cancel.set()


class GromacsReplicaExecutor:
    """Execute a replica without owning global scheduler or pose state."""

    def __init__(
        self,
        *,
        config: AppConfig,
        inspection: InspectionReport,
        frozen: PreparedSystem,
        run_dir: Path,
        environment: Mapping[str, str],
        logger: RunLogger,
    ) -> None:
        self.config = config
        self.inspection = inspection
        self.frozen = frozen
        self.run_dir = run_dir
        self.environment = dict(environment)
        self.logger = logger

    def __call__(
        self,
        claim: TaskClaim,
        control: TaskControl,
    ) -> ExecutionResult:
        replica_dir = (
            self.run_dir
            / "poses"
            / pose_filesystem_key(claim.pose_id)
            / claim.replica_id
        )
        workspace = PbcRmsdWorkspace.under(replica_dir)
        workspace.reset()
        try:
            return self._execute(claim, control, workspace=workspace)
        finally:
            workspace.cleanup()

    def _execute(
        self,
        claim: TaskClaim,
        control: TaskControl,
        *,
        workspace: PbcRmsdWorkspace,
    ) -> ExecutionResult:
        self.logger.set_pose_context(claim.pose_id)
        if (
            not isinstance(self.config.gromacs.ntomp, int)
            or self.config.gromacs.ntomp != claim.resolved_ntomp
        ):
            raise StageExecutionError(
                "resolved task ntomp does not match the immutable runtime config",
                step_id="preflight",
            )
        executable = self.inspection.gromacs_executable
        if executable is None:
            raise StageExecutionError(
                "preflight did not resolve GROMACS",
                step_id="preflight",
            )
        replica_dir = (
            self.run_dir
            / "poses"
            / pose_filesystem_key(claim.pose_id)
            / claim.replica_id
        )
        replica_dir.mkdir(parents=True, exist_ok=True)
        resolved_mdps = resolve_stage_mdps(
            self.config,
            prepared_root=self.frozen.root,
            destination=replica_dir / "resolved_mdp",
            velocity_seed=claim.velocity_seed,
        )
        resolved_config = self.run_dir / "resolved_config.yaml"
        _required_output(
            resolved_config,
            step_id="preflight.resolved_config",
            stage="NONE",
        )
        process_env = dict(self.environment)
        process_env["CUDA_VISIBLE_DEVICES"] = str(
            int(claim.worker_id.removeprefix("gpu-"))
        )
        generated_index = replica_dir / "rmsd_groups.ndx"
        online_outcome: MonitoringOutcome | None = None

        for stage in ("nvt", "npt", "production"):
            control.checkpoint()
            stage_upper = stage.upper()
            stage_dir = replica_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            base = stage_dir / stage
            marker_identity = stage_marker_identity(
                stage=stage,
                pose_id=claim.pose_id,
                replica_id=claim.replica_id,
                velocity_seed=claim.velocity_seed,
                resolved_mdp=resolved_mdps[stage],
                resolved_config=resolved_config,
            )
            try:
                reusable = (
                    self.config.restart.validate_existing_outputs
                    and stage_is_reusable(
                        base,
                        production=stage == "production",
                        expected_identity=marker_identity,
                    )
                )
            except StageMarkerError as exc:
                raise StageExecutionError(
                    str(exc),
                    stage_reached=stage_upper,
                    step_id=f"{claim.replica_id}.{stage}.completion_marker",
                ) from exc
            if reusable:
                self.logger.event(
                    step_id=f"{claim.replica_id}.{stage}",
                    level="INFO",
                    category="STATE",
                    status="SKIPPED",
                    code="VALIDATED_STAGE_REUSED",
                    message=f"Reused validated {stage} outputs during resume.",
                )
                control.update_progress(stage_reached=stage_upper)
                continue
            invalidate_stage_marker(stage_dir)

            coordinates, reference, checkpoint = _stage_inputs(
                stage,
                frozen_start=self.frozen.start_structure,
                replica_dir=replica_dir,
            )
            tpr = base.with_suffix(".tpr")
            processed_mdp = stage_dir / "grompp_processed.mdp"
            if not (
                tpr.is_file()
                and tpr.stat().st_size > 0
                and processed_mdp.is_file()
                and processed_mdp.stat().st_size > 0
            ):
                _logged_command(
                    command=build_grompp_command(
                        executable=executable,
                        mdp=resolved_mdps[stage],
                        coordinates=coordinates,
                        reference_coordinates=reference,
                        checkpoint=checkpoint,
                        topology=self.frozen.topology,
                        index=self.frozen.index,
                        output_tpr=tpr,
                        processed_mdp=processed_mdp,
                        maxwarn=self.config.gromacs.maxwarn,
                    ),
                    step_id=f"{claim.replica_id}.{stage}.grompp",
                    cwd=stage_dir,
                    env=process_env,
                    logger=self.logger,
                    control=control,
                )
                _required_output(tpr, step_id=f"{stage}.grompp", stage=stage_upper)
                _required_output(
                    processed_mdp,
                    step_id=f"{stage}.grompp",
                    stage=stage_upper,
                )
                processed_mdp.write_text(
                    self.logger.redact(processed_mdp.read_text(encoding="utf-8")),
                    encoding="utf-8",
                )

            resume_checkpoint = base.with_suffix(".cpt")
            use_checkpoint = (
                resume_checkpoint
                if resume_checkpoint.is_file() and resume_checkpoint.stat().st_size > 0
                else None
            )
            monitor_finished = threading.Event()
            monitor_cancel = threading.Event()
            monitor_outcomes: list[MonitoringOutcome] = []
            monitor_interruptions: list[TaskInterrupted] = []
            monitor_errors: list[BaseException] = []
            monitor_thread: threading.Thread | None = None
            monitor_join_timeout_seconds = max(
                self.config.monitoring.poll_interval_seconds * 2,
                2.0,
            )
            if stage == "production" and self.config.pruning.enabled:
                if not generated_index.is_file():
                    _logged_command(
                        command=build_heavy_atom_index_command(
                            executable=executable,
                            reference_topology=tpr,
                            source_index=self.frozen.index,
                            generated_index=generated_index,
                            ligand_resname=self.config.input.ligand_resname,
                        ),
                        step_id=f"{claim.replica_id}.analysis.rmsd_groups",
                        cwd=replica_dir,
                        env=process_env,
                        logger=self.logger,
                        control=control,
                    )
                _required_output(
                    generated_index,
                    step_id="analysis.rmsd_groups",
                    stage="PRODUCTION",
                )
                validate_rmsd_groups(generated_index)
                write_clustered_rmsd_index(
                    source_index=self.frozen.index,
                    generated_index=generated_index,
                    output_index=workspace.rmsd_index,
                )
                online_xvg = replica_dir / "online_rmsd_snapshot_nm.xvg"
                online_command_timeout_seconds = max(
                    self.config.monitoring.poll_interval_seconds * 5,
                    60.0,
                )
                monitor_join_timeout_seconds = online_command_timeout_seconds * 3 + 5.0
                snapshot_supplier = _make_online_snapshot_supplier(
                    executable=executable,
                    reference_topology=tpr,
                    trajectory=base.with_suffix(".xtc"),
                    source_index=self.frozen.index,
                    workspace=workspace,
                    output_xvg=online_xvg,
                    cwd=replica_dir,
                    environment=process_env,
                    logger=self.logger,
                    control=control,
                    timeout_seconds=online_command_timeout_seconds,
                )

                monitor_thread = threading.Thread(
                    target=_monitor_target,
                    kwargs={
                        "finished": monitor_finished,
                        "poll_interval_seconds": (
                            self.config.monitoring.poll_interval_seconds
                        ),
                        "snapshot_supplier": snapshot_supplier,
                        "control": control,
                        "outcomes": monitor_outcomes,
                        "interruptions": monitor_interruptions,
                        "errors": monitor_errors,
                        "cancel": monitor_cancel,
                    },
                    name=f"gpu-shortmd-monitor-{claim.replica_id}",
                )
                monitor_thread.start()
            interrupted: TaskInterrupted | None = None
            try:
                _logged_command(
                    command=build_mdrun_command(
                        executable=executable,
                        deffnm=base,
                        config=self.config.gromacs,
                        checkpoint=use_checkpoint,
                        append=use_checkpoint is not None,
                    ),
                    step_id=f"{claim.replica_id}.{stage}.mdrun",
                    cwd=stage_dir,
                    env=process_env,
                    logger=self.logger,
                    control=control,
                    local_stop_requested=monitor_cancel.is_set,
                    termination_grace_seconds=(
                        self.config.pruning.grace_period_seconds
                        if self.config.pruning.enabled
                        else 5.0
                    ),
                )
            except TaskInterrupted as exc:
                interrupted = exc
            finally:
                monitor_finished.set()
                if monitor_thread is not None:
                    monitor_thread.join(timeout=monitor_join_timeout_seconds)
                    if monitor_thread.is_alive():
                        monitor_errors.append(
                            MonitoringError("online RMSD monitor did not stop")
                        )
            if monitor_errors:
                raise StageExecutionError(
                    "online RMSD monitoring failed persistently",
                    stage_reached=stage_upper,
                    step_id=f"{claim.replica_id}.monitoring",
                ) from monitor_errors[0]
            if monitor_interruptions and not control.triggered_pruning:
                raise monitor_interruptions[0]
            if monitor_outcomes:
                online_outcome = monitor_outcomes[0]
            if control.triggered_pruning:
                return ExecutionResult(
                    status="PRUNED",
                    stage_reached="PRODUCTION",
                    trajectory_time_completed_ps=(
                        online_outcome.latest_time_ps if online_outcome else 0.0
                    ),
                    max_rmsd_nm=(
                        online_outcome.maximum_rmsd_nm if online_outcome else None
                    ),
                    max_rmsd_angstrom=(
                        online_outcome.maximum_rmsd_angstrom if online_outcome else None
                    ),
                    exit_code=0,
                )
            if interrupted is not None:
                raise interrupted
            for suffix in (".gro", ".cpt", ".log"):
                _required_output(
                    base.with_suffix(suffix),
                    step_id=f"{stage}.mdrun",
                    stage=stage_upper,
                )
            if stage == "production":
                _required_output(
                    base.with_suffix(".xtc"),
                    step_id="production.mdrun",
                    stage=stage_upper,
                )
            log_path = base.with_suffix(".log")
            log_path.write_text(
                self.logger.redact(
                    log_path.read_text(encoding="utf-8", errors="replace")
                ),
                encoding="utf-8",
            )
            write_stage_completion_marker(
                stage_dir,
                identity=marker_identity,
            )
            control.update_progress(
                stage_reached=stage_upper,
                trajectory_time_completed_ps=(
                    self.config.trajectory.production_time_ns * 1000
                    if stage == "production"
                    else 0.0
                ),
            )

        production_tpr = replica_dir / "production" / "production.tpr"
        if not generated_index.is_file():
            _logged_command(
                command=build_heavy_atom_index_command(
                    executable=executable,
                    reference_topology=production_tpr,
                    source_index=self.frozen.index,
                    generated_index=generated_index,
                    ligand_resname=self.config.input.ligand_resname,
                ),
                step_id=f"{claim.replica_id}.analysis.rmsd_groups",
                cwd=replica_dir,
                env=process_env,
                logger=self.logger,
                control=control,
            )
        _required_output(
            generated_index,
            step_id="analysis.rmsd_groups",
            stage="PRODUCTION",
        )
        alpha_count, heavy_count = validate_rmsd_groups(generated_index)
        write_clustered_rmsd_index(
            source_index=self.frozen.index,
            generated_index=generated_index,
            output_index=workspace.rmsd_index,
        )

        def invoke_reference_cluster(
            cluster_command: list[str], stdin_text: str
        ) -> None:
            _logged_command(
                command=cluster_command,
                step_id=f"{claim.replica_id}.analysis.pbc_reference",
                cwd=replica_dir,
                env=process_env,
                logger=self.logger,
                control=control,
                stdin_text=stdin_text,
            )

        def invoke_trajectory_cluster(
            cluster_command: list[str], stdin_text: str
        ) -> None:
            _logged_command(
                command=cluster_command,
                step_id=f"{claim.replica_id}.analysis.pbc_trajectory",
                cwd=replica_dir,
                env=process_env,
                logger=self.logger,
                control=control,
                stdin_text=stdin_text,
            )

        if not workspace.reference.is_file():
            reconstruct_clustered_complex(
                executable=executable,
                reference_topology=production_tpr,
                trajectory=replica_dir / "production" / "production.xtc",
                source_index=self.frozen.index,
                output=workspace.reference,
                begin_time_ps=0.0,
                end_time_ps=0.0,
                invoke=invoke_reference_cluster,
            )
        reconstruct_clustered_complex(
            executable=executable,
            reference_topology=production_tpr,
            trajectory=replica_dir / "production" / "production.xtc",
            source_index=self.frozen.index,
            output=workspace.final_trajectory,
            begin_time_ps=0.0,
            end_time_ps=None,
            invoke=invoke_trajectory_cluster,
        )
        rmsd_xvg = replica_dir / "rmsd_time_series_nm.xvg"
        temporary_rmsd_xvg = rmsd_xvg.with_name(
            f".{rmsd_xvg.stem}.{uuid.uuid4().hex}{rmsd_xvg.suffix}"
        )
        command = build_rmsd_command(
            executable=executable,
            reference_structure=workspace.reference,
            trajectory=workspace.final_trajectory,
            generated_index=workspace.rmsd_index,
            output_xvg=temporary_rmsd_xvg,
        )
        self.logger.event(
            step_id=f"{claim.replica_id}.analysis.rmsd",
            level="INFO",
            category="RMSD",
            status="STARTED",
            code="RMSD_COMMAND_STARTED",
            message="Started unweighted ligand-heavy RMSD after protein C-alpha fit.",
            params={
                "args": command,
                "fit_group": "C-alpha",
                "measurement_group": "LIG_HEAVY",
                "pbc_reconstruction_group": "Protein_LIG",
                "pbc_mode": "cluster",
                "mass_weighted": False,
                "heavy_atom_selection": "TPR mass > 2.5 Da",
            },
        )
        try:
            _logged_command(
                command=command,
                step_id=f"{claim.replica_id}.analysis.rmsd",
                cwd=replica_dir,
                env=process_env,
                logger=self.logger,
                control=control,
                stdin_text="C-alpha\nLIG_HEAVY\n",
            )
            _required_output(
                temporary_rmsd_xvg,
                step_id="analysis.rmsd",
                stage="PRODUCTION",
            )
            parse_xvg(temporary_rmsd_xvg)
            temporary_rmsd_xvg.write_text(
                self.logger.redact(
                    temporary_rmsd_xvg.read_text(encoding="utf-8", errors="replace")
                ),
                encoding="utf-8",
            )
            temporary_rmsd_xvg.replace(rmsd_xvg)
        finally:
            if temporary_rmsd_xvg.exists():
                temporary_rmsd_xvg.unlink()
        _required_output(
            rmsd_xvg,
            step_id="analysis.rmsd",
            stage="PRODUCTION",
        )
        series = parse_xvg(rmsd_xvg)
        maximum = series.maximum
        maximum_angstrom = convert_rmsd(
            maximum.rmsd,
            input_unit="nm",
            output_unit="angstrom",
        )
        _write_angstrom_series(
            rmsd_xvg,
            replica_dir / "rmsd_time_series_angstrom.csv",
        )
        trajectory_time_completed_ps = self.config.trajectory.production_time_ns * 1000
        control.checkpoint()
        control.update_progress(
            stage_reached="PRODUCTION",
            trajectory_time_completed_ps=trajectory_time_completed_ps,
            max_rmsd_nm=maximum.rmsd,
            max_rmsd_angstrom=maximum_angstrom,
        )
        final_unobserved_samples = tuple(
            sample
            for sample in series.samples
            if online_outcome is None or sample.time_ps > online_outcome.latest_time_ps
        )
        observe_first_pruning_crossing(
            final_unobserved_samples,
            control=control,
        )
        self.logger.event(
            step_id=f"{claim.replica_id}.analysis.rmsd",
            level="INFO",
            category="RMSD",
            status="SUCCEEDED",
            code="RMSD_CALCULATED",
            message="Calculated ligand-heavy RMSD after protein C-alpha fit.",
            metrics={
                "fit_atom_count": alpha_count,
                "ligand_heavy_atom_count": heavy_count,
                "max_rmsd_angstrom": maximum_angstrom,
                "max_rmsd_time_ps": maximum.time_ps,
            },
        )
        return ExecutionResult(
            status="COMPLETED",
            stage_reached="PRODUCTION",
            trajectory_time_completed_ps=trajectory_time_completed_ps,
            max_rmsd_nm=maximum.rmsd,
            max_rmsd_angstrom=maximum_angstrom,
            exit_code=0,
        )
