"""GROMACS preprocessing command construction."""

from __future__ import annotations

from pathlib import Path


def build_grompp_command(
    *,
    executable: Path,
    mdp: Path,
    coordinates: Path,
    reference_coordinates: Path,
    checkpoint: Path | None,
    topology: Path,
    index: Path,
    output_tpr: Path,
    processed_mdp: Path,
    maxwarn: int,
) -> list[str]:
    if maxwarn < 0:
        raise ValueError("negative maxwarn is prohibited")
    command = [
        str(executable),
        "grompp",
        "-f",
        str(mdp),
        "-c",
        str(coordinates),
        "-r",
        str(reference_coordinates),
    ]
    if checkpoint is not None:
        command.extend(["-t", str(checkpoint)])
    command.extend(
        [
            "-p",
            str(topology),
            "-n",
            str(index),
            "-o",
            str(output_tpr),
            "-po",
            str(processed_mdp),
            "-maxwarn",
            str(maxwarn),
        ]
    )
    return command
