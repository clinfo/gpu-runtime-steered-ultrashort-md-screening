"""Fail-fast prepared-input and environment inspection."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpu_shortmd import __version__
from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.gromacs.executable import (
    ExecutableNotFoundError,
    resolve_executable,
)
from gpu_shortmd.gromacs.index import (
    IndexValidationError,
    parse_index,
    validate_required_groups,
)
from gpu_shortmd.gromacs.mdp import MdpValidationError, validate_stage_mdps
from gpu_shortmd.gromacs.topology import (
    TopologyResolution,
    TopologyValidationError,
    resolve_topology,
)
from gpu_shortmd.gromacs.version import (
    GromacsVersion,
    GromacsVersionError,
    gromacs_topology_search_dirs,
    query_gromacs_version,
)
from gpu_shortmd.runtime.state import (
    StateCompatibilityError,
    validate_sqlite_filesystem,
)
from gpu_shortmd.util.files import write_json, write_yaml
from gpu_shortmd.util.subprocess import CommandTimeoutError, run_command
from gpu_shortmd.workflow.prepared_input import (
    PreparedInputError,
    PreparedSystem,
    resolve_prepared_system,
    validate_required_paths,
)


@dataclass(frozen=True)
class InspectionCheck:
    code: str
    severity: str
    status: str
    message: str
    remediation: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class InspectionReport:
    overall_status: str
    checks: tuple[InspectionCheck, ...]
    package_version: str
    prepared_system: PreparedSystem
    gromacs_executable: Path | None
    gromacs_version: GromacsVersion | None
    topology_resolution: TopologyResolution | None
    visible_gpu_ids: tuple[int, ...]
    estimated_disk_bytes: int
    exit_code: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "package_version": self.package_version,
            "checks": [_redact_absolute_paths(asdict(check)) for check in self.checks],
            "resolved_paths": {
                "prepared_system": "<PREPARED_SYSTEM>",
                "start_structure": self.prepared_system.start_structure.name,
                "topology": self.prepared_system.topology.name,
                "index": self.prepared_system.index.name,
            },
            "gromacs": (
                self.gromacs_version.public_dict()
                if self.gromacs_version is not None
                else None
            ),
            "visible_gpu_ids": list(self.visible_gpu_ids),
            "estimated_disk_bytes": self.estimated_disk_bytes,
            "exit_code": self.exit_code,
        }


def _redact_absolute_paths(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"(?<![A-Za-z0-9_.-])/(?:[^/\s,:;]+/)*[^/\s,:;]+",
            "<REDACTED_PATH>",
            value,
        )
    if isinstance(value, list):
        return [_redact_absolute_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_absolute_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_absolute_paths(item) for key, item in value.items()}
    return value


def _check(
    checks: list[InspectionCheck],
    *,
    code: str,
    severity: str,
    status: str,
    message: str,
    remediation: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        InspectionCheck(
            code=code,
            severity=severity,
            status=status,
            message=message,
            remediation=remediation,
            details=details,
        )
    )


def _gro_atom_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as handle:
            next(handle)
            return int(next(handle).strip())
    except (OSError, UnicodeError, StopIteration, ValueError) as exc:
        raise PreparedInputError(
            "start structure has an invalid GRO atom count"
        ) from exc


def _estimate_disk_bytes(config: AppConfig, system: PreparedSystem) -> int:
    atom_count = _gro_atom_count(system.start_structure)
    production_ps = config.trajectory.production_time_ns * 1000
    frames = math.ceil(production_ps / config.trajectory.output_interval_ps) + 1
    trajectory_estimate = atom_count * frames * 16
    stage_overhead = atom_count * 256 + 20 * 1024 * 1024
    return (trajectory_estimate + stage_overhead) * config.trajectory.replicas


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _visible_gpus(
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> tuple[int, ...]:
    configured = env.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        values = [item.strip() for item in configured.split(",") if item.strip()]
        if not values or values == ["-1"]:
            return ()
        try:
            gpu_ids = tuple(int(item) for item in values)
        except ValueError as exc:
            raise ValueError("CUDA_VISIBLE_DEVICES must contain integer IDs") from exc
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain nonnegative IDs")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain unique IDs")
        return gpu_ids

    try:
        executable = resolve_executable("nvidia-smi", env=env)
    except ExecutableNotFoundError:
        return ()
    try:
        result = run_command(
            [str(executable), "-L"],
            cwd=cwd,
            env=env,
            timeout_seconds=30,
        )
    except CommandTimeoutError as exc:
        raise ValueError("nvidia-smi GPU query timed out") from exc
    except OSError as exc:
        raise ValueError("nvidia-smi GPU query could not execute") from exc
    if result.returncode != 0:
        return ()
    lines = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
    return tuple(range(len(lines)))


def inspect_configuration(
    config: AppConfig,
    *,
    config_path: Path,
    env: Mapping[str, str] | None = None,
    output_dir_override: Path | None = None,
) -> InspectionReport:
    environment = dict(os.environ if env is None else env)
    checks: list[InspectionCheck] = []
    system = resolve_prepared_system(config, config_path=config_path)
    gromacs_executable: Path | None = None
    version: GromacsVersion | None = None
    topology_resolution: TopologyResolution | None = None
    visible_gpu_ids: tuple[int, ...] = ()
    estimated_disk_bytes = 0
    input_failed = False
    environment_failed = False

    if config.trajectory.seeds is not None:
        seed_details: dict[str, Any] = {
            "mode": "explicit",
            "count": len(config.trajectory.seeds),
            "seeds": config.trajectory.seeds,
        }
    elif config.trajectory.base_seed is not None:
        seed_details = {
            "mode": "base_seed",
            "count": config.trajectory.replicas,
            "base_seed": config.trajectory.base_seed,
        }
    else:
        seed_details = {
            "mode": "resolve_and_persist_at_run_creation",
            "count": config.trajectory.replicas,
        }
    _check(
        checks,
        code="SEED_CONFIGURATION_VALID",
        severity="INFO",
        status="PASS",
        message="Replica seed configuration is internally consistent.",
        details=seed_details,
    )
    _check(
        checks,
        code="PRUNING_ENABLED" if config.pruning.enabled else "PRUNING_DISABLED",
        severity="INFO",
        status="PASS",
        message=(
            "Pruning uses an explicit angstrom threshold."
            if config.pruning.enabled
            else "Pruning is disabled."
        ),
        details={
            "enabled": config.pruning.enabled,
            "threshold_angstrom": config.pruning.threshold_angstrom,
        },
    )

    try:
        validate_required_paths(system)
        _check(
            checks,
            code="INPUT_FILES_OK",
            severity="INFO",
            status="PASS",
            message="Required prepared-system files are present.",
        )
    except PreparedInputError as exc:
        input_failed = True
        _check(
            checks,
            code="INPUT_FILES_MISSING",
            severity="ERROR",
            status="FAIL",
            message=str(exc),
            remediation="Provide every required prepared-system file.",
        )

    try:
        gromacs_executable = resolve_executable(
            config.gromacs.executable, env=environment
        )
        version = query_gromacs_version(
            gromacs_executable,
            env=environment,
            cwd=config_path.parent,
        )
        _check(
            checks,
            code=(
                "GROMACS_TESTED_VERSION"
                if version.is_tested_version
                else "GROMACS_UNTESTED_VERSION"
            ),
            severity="INFO" if version.is_tested_version else "WARNING",
            status="PASS" if version.is_tested_version else "WARN",
            message=(
                f"GROMACS {version.version} detected."
                if version.is_tested_version
                else (
                    f"GROMACS {version.version} detected; only 2025.4 is a tested "
                    "compatibility claim."
                )
            ),
            remediation=(
                None
                if version.is_tested_version
                else "Use GROMACS 2025.4 CUDA for release validation."
            ),
            details=version.public_dict(),
        )
        if not version.has_gpu_support:
            environment_failed = True
            _check(
                checks,
                code="GROMACS_GPU_DISABLED",
                severity="ERROR",
                status="FAIL",
                message="The configured GROMACS build has no GPU support.",
                remediation="Use a CUDA-enabled GROMACS build.",
            )
        elif not version.has_cuda_support:
            environment_failed = True
            _check(
                checks,
                code="GROMACS_CUDA_REQUIRED",
                severity="ERROR",
                status="FAIL",
                message="The configured GROMACS GPU backend is not CUDA.",
                remediation="Use the tested GROMACS 2025.4 CUDA build.",
            )
    except (ExecutableNotFoundError, GromacsVersionError) as exc:
        environment_failed = True
        _check(
            checks,
            code="GROMACS_UNAVAILABLE",
            severity="ERROR",
            status="FAIL",
            message=str(exc),
            remediation="Configure a parseable GROMACS executable.",
        )

    if not input_failed:
        try:
            topology_resolution = resolve_topology(
                system.topology,
                prepared_root=system.root,
                external_search_dirs=(
                    gromacs_topology_search_dirs(version) if version else []
                ),
            )
            _check(
                checks,
                code="TOPOLOGY_INCLUDES_OK",
                severity="INFO",
                status="PASS",
                message="All topology includes resolve.",
                details={
                    "local_include_files": len(topology_resolution.local_files),
                    "external_include_files": len(topology_resolution.external_files),
                },
            )
        except TopologyValidationError as exc:
            input_failed = True
            _check(
                checks,
                code="TOPOLOGY_INCLUDE_ERROR",
                severity="ERROR",
                status="FAIL",
                message=str(exc),
                remediation="Provide local includes or a valid GROMACS data prefix.",
            )

        try:
            required_groups = validate_required_groups(parse_index(system.index))
            _check(
                checks,
                code="INDEX_GROUPS_OK",
                severity="INFO",
                status="PASS",
                message="Required index groups exist exactly once.",
                details={
                    name: len(group.atoms) for name, group in required_groups.items()
                },
            )
        except IndexValidationError as exc:
            input_failed = True
            _check(
                checks,
                code="INDEX_GROUP_ERROR",
                severity="ERROR",
                status="FAIL",
                message=str(exc),
                remediation="Create exact unique required index groups.",
            )

        try:
            parsed_mdps = validate_stage_mdps(config, system.root)
            _check(
                checks,
                code="MDP_SEMANTICS_OK",
                severity="INFO",
                status="PASS",
                message="NVT, NPT, and production MDP semantics are valid.",
                details={
                    stage: {
                        "dt_ps": mdp.float_value("dt"),
                        "source_nsteps": mdp.int_value("nsteps"),
                    }
                    for stage, mdp in parsed_mdps.items()
                },
            )
        except MdpValidationError as exc:
            input_failed = True
            _check(
                checks,
                code="MDP_SEMANTICS_ERROR",
                severity="ERROR",
                status="FAIL",
                message=str(exc),
                remediation="Correct the stage MDP settings without mutating inputs.",
            )

        try:
            estimated_disk_bytes = _estimate_disk_bytes(config, system)
            configured_output = output_dir_override or Path(config.run.output_dir)
            output_path = (
                configured_output
                if configured_output.is_absolute()
                else config_path.parent / configured_output
            )
            disk = shutil.disk_usage(_nearest_existing_parent(output_path))
            disk_ok = disk.free > estimated_disk_bytes
            if not disk_ok:
                input_failed = True
            _check(
                checks,
                code="DISK_ESTIMATE_OK" if disk_ok else "DISK_SPACE_LOW",
                severity="INFO" if disk_ok else "ERROR",
                status="PASS" if disk_ok else "FAIL",
                message=(
                    "Estimated run artifacts fit in available space."
                    if disk_ok
                    else "Estimated run artifacts exceed available space."
                ),
                remediation=None
                if disk_ok
                else "Select an output filesystem with space.",
                details={
                    "estimated_bytes": estimated_disk_bytes,
                    "available_bytes": disk.free,
                },
            )
            capability = validate_sqlite_filesystem(
                _nearest_existing_parent(output_path)
            )
            _check(
                checks,
                code=(
                    "SQLITE_WAL_VALIDATED"
                    if capability.journal_mode == "WAL"
                    else "SQLITE_DELETE_VALIDATED"
                ),
                severity="WARNING" if capability.warning else "INFO",
                status="WARN" if capability.warning else "PASS",
                message=(
                    capability.warning
                    or "SQLite creation, transaction, locking, and WAL are valid."
                ),
                details={
                    "journal_mode": capability.journal_mode,
                    "locking_validated": capability.locking_validated,
                },
            )
        except (OSError, PreparedInputError, StateCompatibilityError) as exc:
            input_failed = True
            _check(
                checks,
                code="DISK_ESTIMATE_ERROR",
                severity="ERROR",
                status="FAIL",
                message=str(exc),
                remediation="Correct the start structure or output filesystem.",
            )

    try:
        visible_gpu_ids = _visible_gpus(env=environment, cwd=config_path.parent)
        if not visible_gpu_ids:
            environment_failed = True
            _check(
                checks,
                code="GPU_NOT_VISIBLE",
                severity="ERROR",
                status="FAIL",
                message="No NVIDIA GPU is visible.",
                remediation="Expose at least one GPU for the stable run command.",
            )
        else:
            requested = config.scheduler.gpu_ids
            unavailable = (
                []
                if requested == "auto"
                else [gpu_id for gpu_id in requested if gpu_id not in visible_gpu_ids]
            )
            if unavailable:
                environment_failed = True
                _check(
                    checks,
                    code="GPU_ID_UNAVAILABLE",
                    severity="ERROR",
                    status="FAIL",
                    message="Configured GPU IDs are not visible.",
                    remediation="Choose IDs from the visible local GPU set.",
                    details={"unavailable_gpu_ids": unavailable},
                )
            else:
                _check(
                    checks,
                    code="GPU_VISIBLE",
                    severity="INFO",
                    status="PASS",
                    message="Configured GPU selection is visible.",
                    details={"visible_gpu_ids": list(visible_gpu_ids)},
                )
    except ValueError as exc:
        environment_failed = True
        _check(
            checks,
            code="GPU_VISIBILITY_INVALID",
            severity="ERROR",
            status="FAIL",
            message=str(exc),
            remediation="Correct GPU visibility configuration.",
        )

    if input_failed:
        exit_code = 3
    elif environment_failed:
        exit_code = 4
    else:
        exit_code = 0
    return InspectionReport(
        overall_status="PASS" if exit_code == 0 else "FAIL",
        checks=tuple(checks),
        package_version=__version__,
        prepared_system=system,
        gromacs_executable=gromacs_executable,
        gromacs_version=version,
        topology_resolution=topology_resolution,
        visible_gpu_ids=visible_gpu_ids,
        estimated_disk_bytes=estimated_disk_bytes,
        exit_code=exit_code,
    )


def write_inspection_outputs(
    report: InspectionReport,
    *,
    config: AppConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.public_dict()
    write_json(output_dir / "inspect_report.json", payload)
    lines = [
        f"overall_status: {report.overall_status}",
        f"exit_code: {report.exit_code}",
        f"estimated_disk_bytes: {report.estimated_disk_bytes}",
        "checks:",
    ]
    lines.extend(
        f"- [{check['status']}] {check['code']}: {check['message']}"
        for check in payload["checks"]
    )
    (output_dir / "inspect_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    resolved = config.model_dump(mode="json")
    resolved["run"]["output_dir"] = "<OUTPUT_DIR>"
    resolved["input"]["prepared_system_dir"] = "<PREPARED_SYSTEM>"
    for key in ("start_structure", "topology", "index"):
        resolved["input"][key] = Path(resolved["input"][key]).name
    for stage in ("nvt", "npt", "production"):
        resolved["stages"][stage]["mdp"] = Path(resolved["stages"][stage]["mdp"]).name
    resolved["gromacs"]["executable"] = Path(resolved["gromacs"]["executable"]).name
    write_yaml(output_dir / "resolved_config.yaml", resolved)


def inspection_json(report: InspectionReport) -> str:
    return json.dumps(report.public_dict(), indent=2, sort_keys=True) + "\n"
