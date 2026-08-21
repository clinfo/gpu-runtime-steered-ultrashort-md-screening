"""Fresh and resumed local ensemble runs backed by authoritative SQLite state."""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_shortmd import __version__
from gpu_shortmd.config.loader import load_config
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.runtime.export import export_state_summaries
from gpu_shortmd.runtime.scheduler import (
    DryRunPoseSpec,
    ExecutionResult,
    ReplicaExecutor,
    TaskControl,
    available_cpu_count,
    build_multi_pose_dry_run_plan,
    resolve_ntomp,
    run_local_scheduler,
)
from gpu_shortmd.runtime.seeds import resolve_seeds
from gpu_shortmd.runtime.state import (
    PoseTaskSpec,
    RuntimeState,
    StateError,
    TaskClaim,
)
from gpu_shortmd.util.checksums import sha256_file, write_checksum_file
from gpu_shortmd.util.files import ensure_new_directory, write_json, write_yaml
from gpu_shortmd.util.logging import RunLogger, utc_now
from gpu_shortmd.util.provenance import detect_source_revision
from gpu_shortmd.workflow.filesystem_identity import (
    PoseFilesystemIdentityError,
    derive_pose_filesystem_keys,
)
from gpu_shortmd.workflow.inspection import InspectionReport, inspect_configuration
from gpu_shortmd.workflow.pose_manifest import (
    PoseManifestError,
    ResolvedPoseConfig,
    resolve_pose_configs,
    resolved_manifest_payload,
)
from gpu_shortmd.workflow.prepared_input import (
    PreparedSystem,
    external_topology_identifier,
    snapshot_prepared_system,
)
from gpu_shortmd.workflow.replica import GromacsReplicaExecutor
from gpu_shortmd.workflow.runner import (
    RunConfigurationError,
    _new_run_id,
    _output_root,
    _safe_name,
    _write_artifact_manifest,
)

ExecutorFactory = Callable[
    [AppConfig, InspectionReport, PreparedSystem, Path, Mapping[str, str], RunLogger],
    ReplicaExecutor,
]

EXTERNAL_VALIDATION_REFERENCE = "SEE_RELEASE_DOCUMENTATION"
EXTERNAL_VALIDATION_NOTICE = (
    "External validation is documented at the software-release level. This run "
    "records environment and source provenance but does not independently certify "
    "the current checkout. See the repository validation documentation for the "
    "validated commit and scope."
)


@dataclass(frozen=True)
class EnsembleRunResult:
    run_id: str
    run_dir: Path
    status: str
    md_score_angstrom: float | None
    pose_results: tuple[dict[str, object], ...]


def _selected_gpus(
    config: AppConfig,
    *,
    report: InspectionReport,
) -> tuple[int, ...]:
    if config.scheduler.backend != "local":
        raise RunConfigurationError(
            "stable execution requires scheduler.backend = local"
        )
    if config.scheduler.tasks_per_gpu != 1:
        raise RunConfigurationError(
            "stable worker ownership requires scheduler.tasks_per_gpu = 1"
        )
    if config.scheduler.gpu_ids == "auto":
        gpu_ids = report.visible_gpu_ids
    else:
        gpu_ids = tuple(config.scheduler.gpu_ids)
        unavailable = sorted(set(gpu_ids) - set(report.visible_gpu_ids))
        if unavailable:
            raise RunConfigurationError(
                "configured GPU IDs are not visible: "
                + ", ".join(str(item) for item in unavailable),
                exit_code=4,
            )
    if not gpu_ids:
        raise RunConfigurationError("no GPU is available", exit_code=4)
    return gpu_ids


def _redacted_logger(
    *,
    run_dir: Path,
    run_id: str,
    config_path: Path,
    reports: Sequence[InspectionReport],
    resume: bool,
) -> RunLogger:
    report = reports[0]
    executable_parent = (
        str(report.gromacs_executable.parent)
        if report.gromacs_executable is not None
        else ""
    )
    data_prefix = (
        str(report.gromacs_version.data_prefix)
        if report.gromacs_version is not None
        and report.gromacs_version.data_prefix is not None
        else ""
    )
    return RunLogger(
        run_dir=run_dir,
        run_id=run_id,
        redacted_values=[
            str(run_dir),
            *(str(item.prepared_system.root) for item in reports),
            str(config_path.parent.resolve()),
            str(Path.home()),
            executable_parent,
            data_prefix,
        ],
        redacted_tokens=[platform.node()],
        resume=resume,
    )


def _default_executor_factory(
    config: AppConfig,
    report: InspectionReport,
    frozen: PreparedSystem,
    run_dir: Path,
    environment: Mapping[str, str],
    logger: RunLogger,
) -> ReplicaExecutor:
    return GromacsReplicaExecutor(
        config=config,
        inspection=report,
        frozen=frozen,
        run_dir=run_dir,
        environment=environment,
        logger=logger,
    )


class _PoseExecutor:
    def __init__(self, executors: Mapping[str, ReplicaExecutor]) -> None:
        self._executors = dict(executors)

    def __call__(
        self,
        claim: TaskClaim,
        control: TaskControl,
    ) -> ExecutionResult:
        try:
            executor = self._executors[claim.pose_id]
        except KeyError as exc:
            raise StateError(
                f"task references an unknown pose executor: {claim.pose_id}"
            ) from exc
        return executor(claim, control)


def _input_checksum_map(
    manifest: list[dict[str, Any]],
    *,
    pose_id: str,
) -> dict[str, str]:
    return {
        str(item["destination"]): str(item["sha256"])
        for item in manifest
        if item.get("destination") != "<EXTERNAL_GROMACS_DATA>"
        and item.get("pose_id") == pose_id
    }


def _load_and_verify_frozen_inputs(
    *,
    run_dir: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Strictly verify every immutable local input recorded at run creation."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot read input manifest: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or not isinstance(payload.get("files"), list)
    ):
        raise StateError("input manifest has an unsupported schema")

    entries: list[dict[str, Any]] = []
    destinations: set[str] = set()
    for index, raw_entry in enumerate(payload["files"]):
        if not isinstance(raw_entry, dict):
            raise StateError(f"input manifest entry {index} is not an object")
        entry = dict(raw_entry)
        if not all(
            isinstance(entry.get(key), str)
            for key in ("pose_id", "source", "destination", "kind", "sha256")
        ):
            raise StateError(f"input manifest entry {index} has invalid fields")
        expected = str(entry["sha256"])
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise StateError(f"input manifest entry {index} has invalid SHA-256")
        destination = str(entry["destination"])
        if destination == "<EXTERNAL_GROMACS_DATA>":
            if entry["kind"] != "external_topology_dependency":
                raise StateError(
                    f"input manifest entry {index} has an invalid external kind"
                )
            entries.append(entry)
            continue
        if destination in destinations:
            raise StateError(f"input manifest has duplicate destination: {destination}")
        destinations.add(destination)
        relative = Path(destination)
        if relative.is_absolute() or ".." in relative.parts:
            raise StateError(f"input manifest has unsafe destination: {destination}")
        target = (run_dir / relative).resolve()
        if run_dir != target and run_dir not in target.parents:
            raise StateError(
                f"input manifest destination escapes run directory: {destination}"
            )
        if not target.is_file():
            raise StateError(f"frozen input is missing: {destination}")
        if sha256_file(target) != expected:
            raise StateError(f"frozen input checksum changed: {destination}")
        entries.append(entry)
    return entries


def _verify_external_topology_dependencies(
    entries: Sequence[Mapping[str, Any]],
    *,
    reports: Mapping[str, InspectionReport],
) -> None:
    for pose_id, report in reports.items():
        if report.topology_resolution is None:
            raise StateError(
                f"resume preflight did not resolve topology for pose_id {pose_id}"
            )
        expected = {
            str(entry["source"]): str(entry["sha256"])
            for entry in entries
            if entry["kind"] == "external_topology_dependency"
            and entry["pose_id"] == pose_id
        }
        actual: dict[str, str] = {}
        for path in report.topology_resolution.external_files:
            identifier = external_topology_identifier(path)
            if identifier in actual:
                raise StateError(
                    "duplicate external topology identifier for "
                    f"{pose_id}: {identifier}"
                )
            actual[identifier] = sha256_file(path)
        if actual.keys() != expected.keys():
            missing = sorted(expected.keys() - actual.keys())
            unexpected = sorted(actual.keys() - expected.keys())
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            raise StateError(
                f"external topology dependency set changed for pose_id {pose_id}"
                + (": " + "; ".join(details) if details else "")
            )
        changed = sorted(
            identifier
            for identifier, checksum in actual.items()
            if checksum != expected[identifier]
        )
        if changed:
            raise StateError(
                f"external topology dependency checksum changed for pose_id {pose_id}: "
                + ", ".join(changed)
            )


def _verify_registered_artifacts(state: RuntimeState, *, run_dir: Path) -> None:
    """Verify persisted artifacts without treating mutable SQLite as an artifact."""
    seen: set[str] = set()
    for row in state.rows("artifacts"):
        relative_value = str(row["artifact_path"])
        if relative_value in seen:
            raise StateError(f"duplicate registered artifact: {relative_value}")
        seen.add(relative_value)
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise StateError(f"registered artifact has unsafe path: {relative_value}")
        target = (run_dir / relative).resolve()
        if run_dir != target and run_dir not in target.parents:
            raise StateError(
                f"registered artifact escapes run directory: {relative_value}"
            )
        if not target.is_file():
            raise StateError(f"registered artifact is missing: {relative_value}")
        if target.stat().st_size != int(row["size_bytes"]):
            raise StateError(f"registered artifact size changed: {relative_value}")
        if sha256_file(target) != str(row["sha256"]):
            raise StateError(f"registered artifact checksum changed: {relative_value}")


def _write_environment(
    *,
    run_dir: Path,
    report: InspectionReport,
    gpu_ids: Sequence[int],
    pose_ids: Sequence[str],
    resolved_ntomp: int,
    available_cpus: int,
) -> None:
    source_revision = detect_source_revision()
    write_json(
        run_dir / "environment.json",
        {
            "schema_version": 1,
            "package_version": __version__,
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "source_revision": source_revision,
            "gromacs": (
                report.gromacs_version.public_dict()
                if report.gromacs_version is not None
                else None
            ),
            "gpu_ids": list(gpu_ids),
            "pose_ids": list(pose_ids),
            "tasks_per_gpu": 1,
            "resolved_ntomp_per_task": resolved_ntomp,
            "available_cpu_count": available_cpus,
            # Retain the existing field name for artifact-consumer compatibility.
            "external_gpu_validation": EXTERNAL_VALIDATION_REFERENCE,
            "external_validation_notice": EXTERNAL_VALIDATION_NOTICE,
        },
    )


def _export_state_events(state: RuntimeState, logger: RunLogger) -> None:
    exported_sequences: set[int] = set()
    if logger.events_path.is_file():
        for line in logger.events_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = payload.get("params", {})
            if isinstance(params, dict) and isinstance(
                params.get("state_sequence"),
                int,
            ):
                exported_sequences.add(int(params["state_sequence"]))
    for row in state.rows("events"):
        sequence = int(row["sequence"])
        if sequence in exported_sequences:
            continue
        state_payload = json.loads(str(row["payload_json"]))
        logger.event(
            step_id="runtime.state",
            level="INFO",
            category=str(row["category"]),
            status="OBSERVED",
            code=str(row["code"]),
            message=f"Observed authoritative state transition {row['code']}.",
            pose_id=str(row["pose_id"]),
            params={
                "state_sequence": sequence,
                "state_payload": state_payload,
            },
        )


def _finalize_bundle(
    *,
    state: RuntimeState,
    run_dir: Path,
    run_id: str,
    pose_ids: Sequence[str],
    logger: RunLogger,
    status: str,
    started_at: str,
    dry_run: bool,
) -> EnsembleRunResult:
    logger.event(
        step_id="run",
        level="ERROR" if status == "FAILED" else "INFO",
        category="ORCHESTRATION",
        status=(
            "FAILED"
            if status == "FAILED"
            else "PARTIAL"
            if status in {"INTERRUPTED", "INCOMPLETE"}
            else "SUCCEEDED"
        ),
        code=f"RUN_{status}",
        message=f"Local run reached terminal status {status}.",
    )
    export_state_summaries(state, run_dir=run_dir)
    _export_state_events(state, logger)
    poses = state.rows("poses")
    pose_by_id = {str(row["pose_id"]): row for row in poses}
    finished_at = utc_now()
    score = poses[0]["md_score_angstrom"] if len(poses) == 1 else None
    pose_lines = [
        (
            f"- Pose `{pose_id}`: status={pose_by_id[pose_id]['status']}, "
            f"replicas={pose_by_id[pose_id]['n_replicas_requested']}, "
            "MD-score (angstrom)="
            + (
                str(pose_by_id[pose_id]["md_score_angstrom"])
                if pose_by_id[pose_id]["md_score_angstrom"] is not None
                else "null"
            )
        )
        for pose_id in pose_ids
    ]
    (run_dir / "run_report.md").write_text(
        "\n".join(
            [
                f"# Run report: {run_id}",
                "",
                f"- Status: {status}",
                f"- Pose count: {len(poses)}",
                *pose_lines,
                *(
                    []
                    if dry_run
                    else ["- Validation surface: local prepared-system execution"]
                ),
                f"- External validation: {EXTERNAL_VALIDATION_NOTICE}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    failed = [
        f"{row['pose_id']}:{row['replica_id']}"
        for row in state.rows("replicas")
        if row["status"] == "FAILED"
    ]
    if failed:
        logger.issue(
            step_id="scheduler",
            severity="HIGH",
            code="REPLICA_EXECUTION_FAILED",
            message="One or more replica tasks failed.",
            evidence=[str(item) for item in failed],
            suggested_action="Inspect structured events and retry explicitly.",
        )
    logger.write_summary(
        {
            "overall_status": status,
            "top_issues": ["REPLICA_EXECUTION_FAILED"] if failed else [],
            "likely_root_causes": ["external execution failure"] if failed else [],
            "first_checks": [
                "Review preflight and scheduler events.",
                "Review replica stage logs and expected outputs.",
            ],
            "failed_steps": [str(item) for item in failed],
            "partial_outputs": ["poses/"] if failed else [],
            "next_actions": (
                ["Correct the recorded cause and use --retry-failed."]
                if failed
                else ["Retain the run bundle with its checksums."]
            ),
        }
    )
    logger.write_manifest(
        {
            "schema_version": 2,
            "run_id": run_id,
            "pose_ids": list(pose_ids),
            "poses": [
                {
                    "pose_id": str(row["pose_id"]),
                    "status": str(row["status"]),
                    "n_replicas_requested": int(row["n_replicas_requested"]),
                    "md_score_angstrom": row["md_score_angstrom"],
                }
                for row in poses
            ],
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "input_manifest": "input_manifest.json",
            "pose_manifest": "resolved_pose_manifest.yaml",
            "resolved_config": "resolved_config.yaml",
            "state": "state.sqlite3",
        }
    )
    with (run_dir / "logs" / "run_index.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
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
    state.replace_artifacts(
        run_id=run_id,
        artifacts=[
            artifact
            for artifact in artifacts
            if artifact["path"]
            not in {
                "state.sqlite3",
                "artifact_manifest.csv",
                "checksums.sha256",
            }
            and not str(artifact["path"]).endswith("_artifacts.json")
        ],
    )
    artifacts = _write_artifact_manifest(run_dir)
    logger.write_artifacts({"schema_version": 1, "artifacts": artifacts})
    write_checksum_file(
        (path for path in run_dir.rglob("*") if path.is_file()),
        root=run_dir,
        output=run_dir / "checksums.sha256",
    )
    return EnsembleRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        md_score_angstrom=float(score) if score is not None else None,
        pose_results=tuple(
            {
                "pose_id": str(row["pose_id"]),
                "status": str(row["status"]),
                "md_score_angstrom": (
                    float(row["md_score_angstrom"])
                    if row["md_score_angstrom"] is not None
                    else None
                ),
            }
            for row in poses
        ),
    )


def _config_with_resolved_runtime(
    config: AppConfig,
    *,
    resolved_ntomp: int,
    gpu_ids: Sequence[int],
    executable_name: str,
) -> AppConfig:
    payload = config.model_dump(mode="python")
    payload["gromacs"]["ntomp"] = resolved_ntomp
    payload["gromacs"]["executable"] = executable_name
    payload["scheduler"]["gpu_ids"] = list(gpu_ids)
    return AppConfig.model_validate(payload)


def _resolve_pose_seeds(
    poses: Sequence[ResolvedPoseConfig],
) -> dict[str, tuple[int, ...]]:
    resolved: dict[str, tuple[int, ...]] = {}
    all_seeds: list[int] = []
    for pose in poses:
        seeds = resolve_seeds(
            replicas=pose.config.trajectory.replicas,
            explicit=pose.config.trajectory.seeds,
            base_seed=pose.config.trajectory.base_seed,
        )
        resolved[pose.pose_id] = seeds
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise RunConfigurationError(
            "velocity seeds must be unique across all poses in one run"
        )
    return resolved


def run_fresh(
    config: AppConfig,
    *,
    config_path: Path,
    pose_manifest_path: Path | None = None,
    output_dir_override: Path | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    executor_factory: ExecutorFactory = _default_executor_factory,
) -> EnsembleRunResult:
    if config.run.resume:
        raise RunConfigurationError(
            "run.resume in configuration is unsupported; use --resume RUN_DIR"
        )
    environment = dict(os.environ if env is None else env)
    try:
        poses = resolve_pose_configs(
            config,
            config_path=config_path,
            manifest_path=pose_manifest_path,
        )
        pose_filesystem_keys = derive_pose_filesystem_keys(
            pose.pose_id for pose in poses
        )
    except (PoseManifestError, PoseFilesystemIdentityError) as exc:
        raise RunConfigurationError(str(exc), exit_code=3) from exc
    inspection_source = (pose_manifest_path or config_path).resolve()
    reports: dict[str, InspectionReport] = {}
    for pose in poses:
        report = inspect_configuration(
            pose.config,
            config_path=inspection_source,
            env=environment,
            output_dir_override=output_dir_override,
        )
        reports[pose.pose_id] = report
        if report.exit_code != 0:
            raise RunConfigurationError(
                f"preflight failed for pose_id {pose.pose_id!r} "
                f"with exit code {report.exit_code}",
                exit_code=report.exit_code,
            )
        if report.topology_resolution is None:
            raise RunConfigurationError(
                f"preflight did not resolve topology for pose_id {pose.pose_id!r}",
                exit_code=3,
            )
        if report.gromacs_executable is None:
            raise RunConfigurationError(
                f"preflight did not resolve GROMACS for pose_id {pose.pose_id!r}",
                exit_code=4,
            )
    first_report = reports[poses[0].pose_id]
    first_executable = first_report.gromacs_executable
    if first_executable is None:
        raise RunConfigurationError(
            "preflight did not resolve GROMACS for the first pose",
            exit_code=4,
        )
    gpu_ids = _selected_gpus(config, report=first_report)
    cpu_count = available_cpu_count()
    try:
        resolved_ntomp = resolve_ntomp(
            config.gromacs.ntomp,
            gpu_ids=gpu_ids,
            ntmpi=config.gromacs.ntmpi,
            cpu_count=cpu_count,
        )
    except ValueError as exc:
        raise RunConfigurationError(str(exc), exit_code=4) from exc
    seeds_by_pose = _resolve_pose_seeds(poses)
    runtime_poses = tuple(
        ResolvedPoseConfig(
            pose.pose_id,
            _config_with_resolved_runtime(
                pose.config,
                resolved_ntomp=resolved_ntomp,
                gpu_ids=gpu_ids,
                executable_name=first_executable.name,
            ),
        )
        for pose in poses
    )
    run_id = _new_run_id()
    run_label = config.run.name or (
        runtime_poses[0].pose_id if len(runtime_poses) == 1 else "multi_pose"
    )
    run_directory_label = (
        pose_filesystem_keys[runtime_poses[0].pose_id]
        if len(runtime_poses) == 1
        else (config.run.name or "multi_pose")
    )
    run_dir = (
        _output_root(
            config,
            config_path=config_path,
            override=output_dir_override,
        )
        / f"{_safe_name(run_directory_label)}_{run_id}"
    )
    ensure_new_directory(run_dir)
    started_at = utc_now()
    logger = _redacted_logger(
        run_dir=run_dir,
        run_id=run_id,
        config_path=inspection_source,
        reports=tuple(reports.values()),
        resume=False,
    )
    with (run_dir / "logs" / "run_index.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "ts": started_at,
                    "run_id": run_id,
                    "pose_ids": [pose.pose_id for pose in runtime_poses],
                    "status": "DRY_RUN" if dry_run else "RUNNING",
                    "manifest": f"{run_id}_manifest.json",
                },
                sort_keys=True,
            )
            + "\n"
        )
    write_json(
        run_dir / "audit" / "preflight.json",
        {
            "schema_version": 2,
            "pose_ids": [pose.pose_id for pose in runtime_poses],
            "pose_filesystem_keys": pose_filesystem_keys,
            "poses": {
                pose_id: report.public_dict() for pose_id, report in reports.items()
            },
        },
    )
    frozen_by_pose: dict[str, PreparedSystem] = {}
    input_manifest: list[dict[str, Any]] = []
    for pose in runtime_poses:
        report = reports[pose.pose_id]
        assert report.topology_resolution is not None
        destination = (
            run_dir / "inputs"
            if len(runtime_poses) == 1
            else run_dir / "inputs" / pose_filesystem_keys[pose.pose_id]
        )
        frozen, entries = snapshot_prepared_system(
            report.prepared_system,
            topology_resolution=report.topology_resolution,
            destination=destination,
            manifest_root=run_dir,
        )
        frozen_by_pose[pose.pose_id] = frozen
        for entry in entries:
            input_manifest.append({"pose_id": pose.pose_id, **entry})
    write_json(
        run_dir / "input_manifest.json",
        {
            "schema_version": 2,
            "pose_ids": [pose.pose_id for pose in runtime_poses],
            "files": input_manifest,
        },
    )
    pose_manifest_payload = resolved_manifest_payload(
        runtime_poses,
        frozen_roots={
            pose_id: frozen.root for pose_id, frozen in frozen_by_pose.items()
        },
        run_dir=run_dir,
        resolved_seeds=seeds_by_pose,
    )
    write_yaml(run_dir / "resolved_pose_manifest.yaml", pose_manifest_payload)

    first_pose = runtime_poses[0]
    first_frozen = frozen_by_pose[first_pose.pose_id]
    resolved = first_pose.config.model_dump(mode="json")
    resolved["run"]["name"] = run_label
    resolved["run"]["output_dir"] = "."
    resolved["run"]["resume"] = False
    resolved["input"]["prepared_system_dir"] = first_frozen.root.relative_to(
        run_dir
    ).as_posix()
    resolved["input"]["start_structure"] = first_frozen.start_structure.relative_to(
        first_frozen.root
    ).as_posix()
    resolved["input"]["topology"] = first_frozen.topology.relative_to(
        first_frozen.root
    ).as_posix()
    resolved["input"]["index"] = first_frozen.index.relative_to(
        first_frozen.root
    ).as_posix()
    resolved["trajectory"]["base_seed"] = None
    resolved["trajectory"]["seeds"] = list(seeds_by_pose[first_pose.pose_id])
    for stage, mdp_path in first_frozen.mdps.items():
        resolved["stages"][stage]["mdp"] = mdp_path.relative_to(
            first_frozen.root
        ).as_posix()
    write_yaml(run_dir / "resolved_config.yaml", resolved)
    _write_environment(
        run_dir=run_dir,
        report=first_report,
        gpu_ids=gpu_ids,
        pose_ids=[pose.pose_id for pose in runtime_poses],
        resolved_ntomp=resolved_ntomp,
        available_cpus=cpu_count,
    )
    state = RuntimeState.create_batch(
        run_dir / "state.sqlite3",
        run_id=run_id,
        poses=tuple(
            PoseTaskSpec(
                pose_id=pose.pose_id,
                seeds=seeds_by_pose[pose.pose_id],
                pruning_enabled=pose.config.pruning.enabled,
                pruning_threshold_angstrom=(pose.config.pruning.threshold_angstrom),
                resolved_ntomp=resolved_ntomp,
            )
            for pose in runtime_poses
        ),
        gpu_ids=gpu_ids,
        config_sha256=sha256_file(run_dir / "resolved_config.yaml"),
        pose_manifest_sha256=sha256_file(run_dir / "resolved_pose_manifest.yaml"),
        input_manifest_sha256=sha256_file(run_dir / "input_manifest.json"),
    )
    logger.event(
        step_id="run",
        level="INFO",
        category="ORCHESTRATION",
        status="STARTED",
        code="RUN_CREATED",
        message="Created a transactional local multi-pose ensemble run.",
        metrics={
            "poses": len(runtime_poses),
            "replicas": sum(len(seeds) for seeds in seeds_by_pose.values()),
            "workers": len(gpu_ids),
            "resolved_ntomp": resolved_ntomp,
        },
    )
    plan = build_multi_pose_dry_run_plan(
        run_id=run_id,
        poses=tuple(
            DryRunPoseSpec(
                pose_id=pose.pose_id,
                seeds=seeds_by_pose[pose.pose_id],
                pruning_enabled=pose.config.pruning.enabled,
                pruning_threshold_angstrom=(pose.config.pruning.threshold_angstrom),
                input_checksums=_input_checksum_map(
                    input_manifest,
                    pose_id=pose.pose_id,
                ),
            )
            for pose in runtime_poses
        ),
        gpu_ids=gpu_ids,
        stages=("nvt", "npt", "production"),
        work_stealing=config.scheduler.work_stealing,
        resolved_ntomp=resolved_ntomp,
        available_cpus=cpu_count,
    )
    write_json(run_dir / "execution_plan.json", plan)
    pose_ids = tuple(pose.pose_id for pose in runtime_poses)
    if dry_run:
        state.set_run_status(run_id=run_id, status="DRY_RUN")
        return _finalize_bundle(
            state=state,
            run_dir=run_dir,
            run_id=run_id,
            pose_ids=pose_ids,
            logger=logger,
            status="DRY_RUN",
            started_at=started_at,
            dry_run=True,
        )

    executors = {
        pose.pose_id: executor_factory(
            pose.config,
            reports[pose.pose_id],
            frozen_by_pose[pose.pose_id],
            run_dir,
            environment,
            logger,
        )
        for pose in runtime_poses
    }
    run_local_scheduler(
        state=state,
        run_id=run_id,
        gpu_ids=gpu_ids,
        work_stealing=config.scheduler.work_stealing,
        pruning_threshold_angstrom=None,
        executor=_PoseExecutor(executors),
    )
    for pose_id in pose_ids:
        state.finalize_pose(run_id=run_id, pose_id=pose_id)
    status = state.finalize_run(run_id=run_id)
    return _finalize_bundle(
        state=state,
        run_dir=run_dir,
        run_id=run_id,
        pose_ids=pose_ids,
        logger=logger,
        status=status,
        started_at=started_at,
        dry_run=False,
    )


def resume_run(
    run_dir: Path,
    *,
    retry_failed: bool,
    env: Mapping[str, str] | None = None,
    executor_factory: ExecutorFactory = _default_executor_factory,
) -> EnsembleRunResult:
    resolved_run_dir = run_dir.resolve()
    config_path = resolved_run_dir / "resolved_config.yaml"
    pose_manifest_path = resolved_run_dir / "resolved_pose_manifest.yaml"
    state_path = resolved_run_dir / "state.sqlite3"
    input_manifest = resolved_run_dir / "input_manifest.json"
    if not all(
        path.is_file()
        for path in (
            config_path,
            pose_manifest_path,
            state_path,
            input_manifest,
        )
    ):
        raise RunConfigurationError(
            "resume directory is missing resolved config, pose manifest, "
            "input manifest, or state",
            exit_code=7,
        )
    try:
        config = load_config(config_path)
        poses = resolve_pose_configs(
            config,
            config_path=config_path,
            manifest_path=pose_manifest_path,
        )
        derive_pose_filesystem_keys(pose.pose_id for pose in poses)
        state = RuntimeState(state_path)
        manifest_entries = _load_and_verify_frozen_inputs(
            run_dir=resolved_run_dir,
            manifest_path=input_manifest,
        )
        _verify_registered_artifacts(state, run_dir=resolved_run_dir)
        state.verify_resume(
            config_sha256=sha256_file(config_path),
            input_manifest_sha256=sha256_file(input_manifest),
            retry_failed=retry_failed,
            pose_manifest_sha256=sha256_file(pose_manifest_path),
        )
        run_row = state.run_row()
        state_pose_ids = tuple(str(row["pose_id"]) for row in state.rows("poses"))
        manifest_pose_ids = tuple(pose.pose_id for pose in poses)
        if set(state_pose_ids) != set(manifest_pose_ids):
            raise StateError("resolved pose manifest does not match SQLite pose state")
    except (
        OSError,
        PoseManifestError,
        PoseFilesystemIdentityError,
        StateError,
    ) as exc:
        raise RunConfigurationError(str(exc), exit_code=7) from exc
    run_id = str(run_row["run_id"])
    environment = dict(os.environ if env is None else env)
    reports: dict[str, InspectionReport] = {}
    for pose in poses:
        report = inspect_configuration(
            pose.config,
            config_path=pose_manifest_path,
            env=environment,
            output_dir_override=resolved_run_dir.parent,
        )
        reports[pose.pose_id] = report
        if report.exit_code != 0:
            raise RunConfigurationError(
                f"resume preflight failed for pose_id {pose.pose_id!r} "
                f"with exit code {report.exit_code}",
                exit_code=report.exit_code,
            )
    try:
        _verify_external_topology_dependencies(
            manifest_entries,
            reports=reports,
        )
    except (OSError, StateError) as exc:
        raise RunConfigurationError(str(exc), exit_code=7) from exc
    first_report = reports[poses[0].pose_id]
    gpu_ids = _selected_gpus(config, report=first_report)
    logger = _redacted_logger(
        run_dir=resolved_run_dir,
        run_id=run_id,
        config_path=config_path,
        reports=tuple(reports.values()),
        resume=True,
    )
    started_at = str(run_row["created_at"])
    logger.event(
        step_id="resume",
        level="INFO",
        category="STATE",
        status="STARTED",
        code="RUN_RESUME_STARTED",
        message="Validated immutable run inputs and resumed pending work.",
    )
    executors = {
        pose.pose_id: executor_factory(
            pose.config,
            reports[pose.pose_id],
            reports[pose.pose_id].prepared_system,
            resolved_run_dir,
            environment,
            logger,
        )
        for pose in poses
    }
    run_local_scheduler(
        state=state,
        run_id=run_id,
        gpu_ids=gpu_ids,
        work_stealing=config.scheduler.work_stealing,
        pruning_threshold_angstrom=None,
        executor=_PoseExecutor(executors),
    )
    pose_ids = tuple(pose.pose_id for pose in poses)
    for pose_id in pose_ids:
        state.finalize_pose(run_id=run_id, pose_id=pose_id)
    status = state.finalize_run(run_id=run_id)
    return _finalize_bundle(
        state=state,
        run_dir=resolved_run_dir,
        run_id=run_id,
        pose_ids=pose_ids,
        logger=logger,
        status=status,
        started_at=started_at,
        dry_run=False,
    )
