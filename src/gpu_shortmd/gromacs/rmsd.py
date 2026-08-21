"""GROMACS PBC reconstruction and ligand-heavy RMSD commands."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from gpu_shortmd.config.models import LIGAND_RESNAME_PATTERN
from gpu_shortmd.gromacs.index import (
    IndexValidationError,
    parse_index,
    validate_required_groups,
)

HEAVY_ATOM_MASS_THRESHOLD_DA = 2.5
PBC_CLUSTER_GROUP = "Protein_LIG"
PBC_WORKSPACE_NAME = ".pbc-rmsd-work"

CommandInvoker = Callable[[list[str], str], None]


@dataclass(frozen=True)
class PbcRmsdWorkspace:
    """Replica-local transient files shared by online and final RMSD paths."""

    root: Path

    @classmethod
    def under(cls, replica_dir: Path) -> PbcRmsdWorkspace:
        return cls(root=replica_dir / PBC_WORKSPACE_NAME)

    @property
    def reference(self) -> Path:
        return self.root / "clustered_reference.gro"

    @property
    def online_trajectory(self) -> Path:
        return self.root / "online_clustered.xtc"

    @property
    def final_trajectory(self) -> Path:
        return self.root / "final_clustered.xtc"

    @property
    def rmsd_index(self) -> Path:
        return self.root / "clustered_rmsd_groups.ndx"

    def reset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup(remove_root=False)

    def cleanup(self, *, remove_root: bool = True) -> None:
        if not self.root.exists():
            return
        for path in self.root.iterdir():
            if path.is_file():
                path.unlink()
        if remove_root:
            with suppress(OSError):
                self.root.rmdir()


def build_heavy_atom_index_command(
    *,
    executable: Path,
    reference_topology: Path,
    source_index: Path,
    generated_index: Path,
    ligand_resname: str,
) -> list[str]:
    """Use TPR masses so ligand-heavy selection is independent of atom names."""
    if re.fullmatch(LIGAND_RESNAME_PATTERN, ligand_resname) is None:
        raise ValueError("ligand_resname is unsafe for GROMACS selection syntax")
    selection = (
        '"C-alpha" group "C-alpha"; '
        f'"LIG_HEAVY" group "LIG" and resname "{ligand_resname}" and '
        f"mass > {HEAVY_ATOM_MASS_THRESHOLD_DA}"
    )
    return [
        str(executable),
        "select",
        "-s",
        str(reference_topology),
        "-n",
        str(source_index),
        "-select",
        selection,
        "-on",
        str(generated_index),
    ]


def validate_rmsd_groups(generated_index: Path) -> tuple[int, int]:
    groups = parse_index(generated_index)
    by_name = {group.name: group for group in groups}
    if len(by_name) != len(groups):
        raise IndexValidationError("generated RMSD index contains duplicate groups")
    if set(by_name) != {"C-alpha", "LIG_HEAVY"}:
        raise IndexValidationError(
            "generated RMSD index must contain C-alpha and LIG_HEAVY"
        )
    alpha_count = len(by_name["C-alpha"].atoms)
    ligand_heavy_count = len(by_name["LIG_HEAVY"].atoms)
    if alpha_count == 0 or ligand_heavy_count == 0:
        raise IndexValidationError("generated RMSD groups must be non-empty")
    if any(len(set(group.atoms)) != len(group.atoms) for group in groups):
        raise IndexValidationError(
            "generated RMSD groups contain duplicate atom indices"
        )
    return alpha_count, ligand_heavy_count


def write_clustered_rmsd_index(
    *,
    source_index: Path,
    generated_index: Path,
    output_index: Path,
) -> tuple[int, int]:
    """Remap full-system RMSD groups to Protein_LIG extraction order."""
    source = validate_required_groups(parse_index(source_index))
    alpha_count, heavy_count = validate_rmsd_groups(generated_index)
    generated = {group.name: group for group in parse_index(generated_index)}
    if generated["C-alpha"].atoms != source["C-alpha"].atoms:
        raise IndexValidationError(
            "generated C-alpha group does not match the prepared index"
        )
    ligand_atoms = set(source["LIG"].atoms)
    if not set(generated["LIG_HEAVY"].atoms).issubset(ligand_atoms):
        raise IndexValidationError("LIG_HEAVY must be a non-empty subset of LIG")
    positions = {
        atom: position
        for position, atom in enumerate(source["Protein_LIG"].atoms, start=1)
    }
    temporary = output_index.with_name(f".{output_index.name}.{uuid.uuid4().hex}.tmp")
    output_index.parent.mkdir(parents=True, exist_ok=True)
    try:
        lines: list[str] = []
        for name in ("C-alpha", "LIG_HEAVY"):
            try:
                remapped = [positions[atom] for atom in generated[name].atoms]
            except KeyError as exc:
                raise IndexValidationError(
                    f"{name} contains an atom outside Protein_LIG"
                ) from exc
            lines.append(f"[ {name} ]\n")
            for offset in range(0, len(remapped), 15):
                chunk = remapped[offset : offset + 15]
                lines.append(" ".join(str(value) for value in chunk))
                lines.append("\n")
            lines.append("\n")
        temporary.write_text("".join(lines), encoding="utf-8")
        validate_rmsd_groups(temporary)
        temporary.replace(output_index)
    finally:
        if temporary.exists():
            temporary.unlink()
    return alpha_count, heavy_count


def build_pbc_cluster_command(
    *,
    executable: Path,
    reference_topology: Path,
    trajectory: Path,
    source_index: Path,
    output: Path,
    begin_time_ps: float,
    end_time_ps: float | None = None,
) -> list[str]:
    """Build the Fugaku-compatible Protein_LIG cluster reconstruction."""
    command = [
        str(executable),
        "trjconv",
        "-s",
        str(reference_topology),
        "-f",
        str(trajectory),
        "-n",
        str(source_index),
        "-o",
        str(output),
        "-b",
        f"{begin_time_ps:g}",
    ]
    if end_time_ps is not None:
        command.extend(["-e", f"{end_time_ps:g}"])
    command.extend(["-pbc", "cluster"])
    return command


def reconstruct_clustered_complex(
    *,
    executable: Path,
    reference_topology: Path,
    trajectory: Path,
    source_index: Path,
    output: Path,
    begin_time_ps: float,
    end_time_ps: float | None,
    invoke: CommandInvoker,
) -> None:
    """Run cluster reconstruction into a unique file and publish atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}{output.suffix}")
    command = build_pbc_cluster_command(
        executable=executable,
        reference_topology=reference_topology,
        trajectory=trajectory,
        source_index=source_index,
        output=temporary,
        begin_time_ps=begin_time_ps,
        end_time_ps=end_time_ps,
    )
    try:
        invoke(command, f"{PBC_CLUSTER_GROUP}\n{PBC_CLUSTER_GROUP}\n")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("PBC cluster reconstruction produced no output")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_rmsd_command(
    *,
    executable: Path,
    reference_structure: Path,
    trajectory: Path,
    generated_index: Path,
    output_xvg: Path,
) -> list[str]:
    """Build the fidelity-critical unweighted RMSD command."""
    return [
        str(executable),
        "rms",
        "-s",
        str(reference_structure),
        "-f",
        str(trajectory),
        "-n",
        str(generated_index),
        "-o",
        str(output_xvg),
        "-tu",
        "ps",
        "-fit",
        "rot+trans",
        "-what",
        "rmsd",
        "-nomw",
    ]
