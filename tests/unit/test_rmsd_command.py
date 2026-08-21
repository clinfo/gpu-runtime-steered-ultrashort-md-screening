from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.gromacs.index import IndexValidationError
from gpu_shortmd.gromacs.rmsd import (
    build_heavy_atom_index_command,
    build_pbc_cluster_command,
    build_rmsd_command,
    validate_rmsd_groups,
    write_clustered_rmsd_index,
)


def test_rmsd_command_is_explicitly_unweighted_and_protein_fitted() -> None:
    command = build_rmsd_command(
        executable=Path("/opt/gromacs/bin/gmx"),
        reference_structure=Path("production_start.gro"),
        trajectory=Path("production.xtc"),
        generated_index=Path("rmsd_groups.ndx"),
        output_xvg=Path("rmsd_nm.xvg"),
    )

    assert command[1] == "rms"
    assert command[command.index("-fit") + 1] == "rot+trans"
    assert command[command.index("-what") + 1] == "rmsd"
    assert "-pbc" not in command
    assert "-nomw" in command
    assert "-mw" not in command


def test_pbc_cluster_command_uses_incremental_protein_lig_reconstruction() -> None:
    command = build_pbc_cluster_command(
        executable=Path("gmx"),
        reference_topology=Path("production.tpr"),
        trajectory=Path("production.xtc"),
        source_index=Path("prepared.ndx"),
        output=Path("clustered.xtc"),
        begin_time_ps=350.0,
    )

    assert command[1] == "trjconv"
    assert command[command.index("-pbc") + 1] == "cluster"
    assert command[command.index("-b") + 1] == "350"
    assert "-e" not in command
    assert not {"mol", "nojump", "center", "compact"}.intersection(command)


def test_reference_cluster_command_selects_first_frame() -> None:
    command = build_pbc_cluster_command(
        executable=Path("gmx"),
        reference_topology=Path("production.tpr"),
        trajectory=Path("production.xtc"),
        source_index=Path("prepared.ndx"),
        output=Path("reference.gro"),
        begin_time_ps=0.0,
        end_time_ps=0.0,
    )

    assert command[command.index("-b") + 1] == "0"
    assert command[command.index("-e") + 1] == "0"


def test_heavy_atom_index_uses_tpr_mass_not_atom_name() -> None:
    command = build_heavy_atom_index_command(
        executable=Path("/opt/gromacs/bin/gmx"),
        reference_topology=Path("production.tpr"),
        source_index=Path("source.ndx"),
        generated_index=Path("rmsd_groups.ndx"),
        ligand_resname="DRG",
    )

    assert command[1] == "select"
    selection = command[command.index("-select") + 1]
    assert 'group "C-alpha"' in selection
    assert 'group "LIG"' in selection
    assert 'resname "DRG"' in selection
    assert "mass > 2.5" in selection
    assert "name" not in selection.split()


def test_ligand_resname_rejects_selection_syntax() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_heavy_atom_index_command(
            executable=Path("gmx"),
            reference_topology=Path("production.tpr"),
            source_index=Path("source.ndx"),
            generated_index=Path("rmsd_groups.ndx"),
            ligand_resname='LIG" or group "Protein',
        )


def test_generated_rmsd_groups_must_be_exact_and_nonempty(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ndx"
    valid.write_text(
        "[ C-alpha ]\n1 2\n\n[ LIG_HEAVY ]\n3 4 5\n",
        encoding="utf-8",
    )
    assert validate_rmsd_groups(valid) == (2, 3)

    invalid = tmp_path / "invalid.ndx"
    invalid.write_text("[ C-alpha ]\n1\n", encoding="utf-8")
    with pytest.raises(IndexValidationError, match="must contain"):
        validate_rmsd_groups(invalid)


def test_clustered_index_remaps_full_system_atom_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.ndx"
    source.write_text(
        "[ C-alpha ]\n10 30\n[ LIG ]\n70 90\n"
        "[ Protein_LIG ]\n10 30 70 90\n[ Water_and_ions ]\n100\n",
        encoding="utf-8",
    )
    generated = tmp_path / "generated.ndx"
    generated.write_text(
        "[ C-alpha ]\n10 30\n[ LIG_HEAVY ]\n70 90\n",
        encoding="utf-8",
    )
    output = tmp_path / "clustered.ndx"

    assert write_clustered_rmsd_index(
        source_index=source,
        generated_index=generated,
        output_index=output,
    ) == (2, 2)
    assert "1 2" in output.read_text(encoding="utf-8")
    assert "3 4" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "generated_text",
    [
        "[ C-alpha ]\n10 30\n[ LIG_HEAVY ]\n\n",
        "[ C-alpha ]\n10 30\n[ LIG_HEAVY ]\n100\n",
        "[ C-alpha ]\n10\n[ LIG_HEAVY ]\n70\n",
    ],
)
def test_clustered_index_rejects_invalid_ligand_or_alpha_group(
    tmp_path: Path,
    generated_text: str,
) -> None:
    source = tmp_path / "source.ndx"
    source.write_text(
        "[ C-alpha ]\n10 30\n[ LIG ]\n70 90\n"
        "[ Protein_LIG ]\n10 30 70 90\n[ Water_and_ions ]\n100\n",
        encoding="utf-8",
    )
    generated = tmp_path / "generated.ndx"
    generated.write_text(generated_text, encoding="utf-8")

    with pytest.raises(IndexValidationError):
        write_clustered_rmsd_index(
            source_index=source,
            generated_index=generated,
            output_index=tmp_path / "clustered.ndx",
        )
