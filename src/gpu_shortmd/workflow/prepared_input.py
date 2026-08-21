"""Resolve and freeze the prepared-system input contract."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from gpu_shortmd.config.models import AppConfig
from gpu_shortmd.gromacs.mdp import stage_mdp_paths
from gpu_shortmd.gromacs.topology import TopologyResolution
from gpu_shortmd.util.checksums import sha256_file


class PreparedInputError(ValueError):
    """Raised when a required prepared input cannot be resolved."""


@dataclass(frozen=True)
class PreparedSystem:
    root: Path
    start_structure: Path
    topology: Path
    index: Path
    mdps: dict[str, Path]


def resolve_prepared_system(config: AppConfig, *, config_path: Path) -> PreparedSystem:
    configured_root = Path(config.input.prepared_system_dir)
    root = (
        configured_root
        if configured_root.is_absolute()
        else config_path.parent / configured_root
    ).resolve()
    return PreparedSystem(
        root=root,
        start_structure=(root / config.input.start_structure).resolve(),
        topology=(root / config.input.topology).resolve(),
        index=(root / config.input.index).resolve(),
        mdps={
            name: path.resolve() for name, path in stage_mdp_paths(config, root).items()
        },
    )


def required_paths(system: PreparedSystem) -> tuple[Path, ...]:
    return (
        system.start_structure,
        system.topology,
        system.index,
        *system.mdps.values(),
    )


def external_topology_identifier(path: Path) -> str:
    """Return the stable public identifier used for an external topology file."""
    forcefield_index = next(
        (index for index, part in enumerate(path.parts) if part.endswith(".ff")),
        None,
    )
    return (
        Path(*path.parts[forcefield_index:]).as_posix()
        if forcefield_index is not None
        else path.name
    )


def validate_required_paths(system: PreparedSystem) -> None:
    escaped = [
        path.name
        for path in required_paths(system)
        if path != system.root and system.root not in path.parents
    ]
    if escaped:
        raise PreparedInputError(
            "prepared input paths must stay inside prepared_system_dir: "
            + ", ".join(sorted(escaped))
        )
    missing = [
        path.relative_to(system.root).as_posix()
        if system.root in path.parents
        else path.name
        for path in required_paths(system)
        if not path.is_file()
    ]
    if missing:
        raise PreparedInputError(
            "missing prepared input files: " + ", ".join(sorted(missing))
        )


def snapshot_prepared_system(
    system: PreparedSystem,
    *,
    topology_resolution: TopologyResolution,
    destination: Path,
    manifest_root: Path | None = None,
) -> tuple[PreparedSystem, list[dict[str, str]]]:
    destination.mkdir(parents=True, exist_ok=False)
    relative_root = manifest_root or destination.parent
    source_files = set(required_paths(system))
    source_files.update(topology_resolution.local_files)
    manifest: list[dict[str, str]] = []
    copied: dict[Path, Path] = {}
    for source in sorted(source_files):
        try:
            relative = source.relative_to(system.root)
        except ValueError as exc:
            raise PreparedInputError(
                f"local prepared input escaped its root: {source.name}"
            ) from exc
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[source] = target
        manifest.append(
            {
                "source": relative.as_posix(),
                "destination": target.relative_to(relative_root).as_posix(),
                "kind": "frozen_local_input",
                "sha256": sha256_file(target),
            }
        )
    for source in sorted(topology_resolution.external_files):
        manifest.append(
            {
                "source": external_topology_identifier(source),
                "destination": "<EXTERNAL_GROMACS_DATA>",
                "kind": "external_topology_dependency",
                "sha256": sha256_file(source),
            }
        )
    frozen = PreparedSystem(
        root=destination,
        start_structure=copied[system.start_structure],
        topology=copied[system.topology],
        index=copied[system.index],
        mdps={name: copied[path] for name, path in system.mdps.items()},
    )
    return frozen, manifest
