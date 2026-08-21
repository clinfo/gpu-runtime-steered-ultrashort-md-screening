"""GROMACS MD execution command construction."""

from __future__ import annotations

from pathlib import Path

from gpu_shortmd.config.models import GromacsConfig


def build_mdrun_command(
    *,
    executable: Path,
    deffnm: Path,
    config: GromacsConfig,
    checkpoint: Path | None = None,
    append: bool = False,
) -> list[str]:
    command = [
        str(executable),
        "mdrun",
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        str(config.ntmpi),
        "-pin",
        "on" if config.pin else "off",
        "-nb",
        config.offload.nonbonded,
        "-pme",
        config.offload.pme,
        "-bonded",
        config.offload.bonded,
        "-update",
        config.offload.update,
    ]
    if isinstance(config.ntomp, int):
        command.extend(["-ntomp", str(config.ntomp)])
    if checkpoint is not None:
        command.extend(["-cpi", str(checkpoint)])
        command.append("-append" if append else "-noappend")
    return command
