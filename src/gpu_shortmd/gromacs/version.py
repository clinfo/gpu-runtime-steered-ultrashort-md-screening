"""Parse a minimal, non-sensitive GROMACS capability report."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from gpu_shortmd.util.subprocess import CommandTimeoutError, run_command

TESTED_GROMACS_VERSION = "2025.4"


class GromacsVersionError(RuntimeError):
    """Raised when version or build capabilities cannot be determined."""


@dataclass(frozen=True)
class GromacsVersion:
    version: str
    raw_version: str
    gpu_support: str
    precision: str
    cuda_compiler: str | None
    cuda_driver: str | None
    data_prefix: Path | None

    @property
    def is_tested_version(self) -> bool:
        return self.version == TESTED_GROMACS_VERSION

    @property
    def has_gpu_support(self) -> bool:
        return self.gpu_support.lower() not in {"disabled", "none", "unknown"}

    @property
    def has_cuda_support(self) -> bool:
        return "cuda" in self.gpu_support.lower()

    def public_dict(self) -> dict[str, str | bool | None]:
        return {
            "version": self.version,
            "raw_version": self.raw_version,
            "tested_version": TESTED_GROMACS_VERSION,
            "is_tested_version": self.is_tested_version,
            "gpu_support": self.gpu_support,
            "has_gpu_support": self.has_gpu_support,
            "has_cuda_support": self.has_cuda_support,
            "precision": self.precision,
            "cuda_compiler": self.cuda_compiler,
            "cuda_driver": self.cuda_driver,
        }


def _field(output: str, label: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1) if match else None


def _public_build_value(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(
        r"(?:/[A-Za-z0-9._+~-]+)+",
        lambda match: Path(match.group(0)).name,
        value,
    )


def query_gromacs_version(
    executable: Path,
    *,
    env: Mapping[str, str],
    cwd: Path,
) -> GromacsVersion:
    try:
        result = run_command(
            [str(executable), "--version"],
            cwd=cwd,
            env=env,
            timeout_seconds=30,
        )
    except CommandTimeoutError as exc:
        raise GromacsVersionError("GROMACS version command timed out") from exc
    except OSError as exc:
        raise GromacsVersionError("GROMACS version command could not execute") from exc
    if result.returncode != 0:
        raise GromacsVersionError(
            f"GROMACS version command failed with exit {result.returncode}"
        )
    version = _field(result.stdout, "GROMACS version")
    if version is None:
        raise GromacsVersionError("GROMACS version output is not parseable")
    normalized_version_match = re.match(r"(\d+\.\d+)", version)
    if normalized_version_match is None:
        raise GromacsVersionError(f"unsupported GROMACS version format: {version}")
    gpu_support = _field(result.stdout, "GPU support") or "unknown"
    precision = _field(result.stdout, "Precision") or "unknown"
    cuda_compiler = _public_build_value(_field(result.stdout, "CUDA compiler"))
    cuda_driver = _public_build_value(_field(result.stdout, "CUDA driver"))
    data_prefix_value = _field(result.stdout, "Data prefix")
    data_prefix = None
    if data_prefix_value:
        configured_prefix = Path(data_prefix_value)
        data_prefix = (
            configured_prefix
            if configured_prefix.is_absolute()
            else cwd / configured_prefix
        ).resolve()
    return GromacsVersion(
        version=normalized_version_match.group(1),
        raw_version=version,
        gpu_support=gpu_support,
        precision=precision,
        cuda_compiler=cuda_compiler,
        cuda_driver=cuda_driver,
        data_prefix=data_prefix,
    )


def gromacs_topology_search_dirs(version: GromacsVersion) -> list[Path]:
    if version.data_prefix is None:
        return []
    candidates = [
        version.data_prefix / "share" / "gromacs" / "top",
        version.data_prefix / "share" / "top",
    ]
    return [candidate for candidate in candidates if candidate.is_dir()]
