#!/usr/bin/env python3
"""Small deterministic GROMACS test double; never real-MD evidence."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("GROMACS version:     2025.4")
    print("Precision:           mixed")
    print("GPU support:         CUDA")
    print(
        "CUDA compiler:       "
        + os.environ.get(
            "FAKE_GMX_CUDA_COMPILER",
            "/opt/cuda/bin/nvcc (release 12.2)",
        )
    )
    print("CUDA driver:         12.2")
    print("Data prefix:         " + os.environ["FAKE_GMX_DATA_PREFIX"])
    raise SystemExit(0)

command = args[0] if args else ""
if os.environ.get("FAKE_GMX_FAIL_STAGE") == Path.cwd().name + ":" + command:
    print("injected failure", file=sys.stderr)
    raise SystemExit(17)


def option(name: str) -> str:
    return args[args.index(name) + 1]


def write_unless_omitted(path: str, contents: str) -> None:
    key = Path.cwd().name + ":" + Path(path).suffix
    if os.environ.get("FAKE_GMX_OMIT_OUTPUT") != key:
        Path(path).write_text(contents, encoding="utf-8")


def parse_index(path: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    name: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            groups[name] = []
        elif name is not None:
            groups[name].extend(int(value) for value in line.split())
    return groups


def pbc_payload(path: str) -> dict[str, object] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    if isinstance(value, dict) and value.get("fake_pbc_trajectory") is True:
        return value
    return None


def centroid(coordinates: list[list[float]]) -> list[float]:
    return [
        sum(coordinate[axis] for coordinate in coordinates) / len(coordinates)
        for axis in range(3)
    ]


if command == "grompp":
    write_unless_omitted(option("-o"), "fake tpr\n")
    write_unless_omitted(option("-po"), "integrator = md\n")
elif command == "mdrun":
    base = Path(option("-deffnm"))
    shutil.copyfile(os.environ["FAKE_GMX_START"], str(base) + ".gro")
    write_unless_omitted(str(base) + ".cpt", "fake checkpoint\n")
    write_unless_omitted(
        str(base) + ".log",
        f"Working dir: {Path.cwd()}\n",
    )
    write_unless_omitted(str(base) + ".xtc", "fake trajectory\n")
elif command == "select":
    source_groups = parse_index(option("-n"))
    alpha = source_groups["C-alpha"]
    ligand = source_groups["LIG"]
    Path(option("-on")).write_text(
        "[ C-alpha ]\n"
        + " ".join(str(atom) for atom in alpha)
        + "\n\n[ LIG_HEAVY ]\n"
        + " ".join(str(atom) for atom in ligand[: min(3, len(ligand))])
        + "\n",
        encoding="utf-8",
    )
elif command == "trjconv":
    selections = [line.strip() for line in sys.stdin if line.strip()]
    if selections != ["Protein_LIG", "Protein_LIG"]:
        print("trjconv requires Protein_LIG twice", file=sys.stderr)
        raise SystemExit(20)
    trajectory_payload = pbc_payload(option("-f"))
    begin = float(option("-b"))
    end = float(option("-e")) if "-e" in args else None
    if trajectory_payload is None:
        Path(option("-o")).write_text(
            f"fake clustered trajectory\nbegin_ps={begin:g}\n",
            encoding="utf-8",
        )
    else:
        source_groups = parse_index(option("-n"))
        complex_atoms = source_groups["Protein_LIG"]
        ligand_atoms = set(source_groups["LIG"])
        box = float(trajectory_payload["box_nm"])
        frames: list[dict[str, object]] = []
        for frame in trajectory_payload["frames"]:  # type: ignore[union-attr]
            time_ps = float(frame["time_ps"])
            if time_ps < begin or (end is not None and time_ps > end):
                continue
            coordinates = frame["coordinates_nm"]
            protein = [
                coordinates[atom - 1]
                for atom in complex_atoms
                if atom not in ligand_atoms
            ]
            ligand = [
                coordinates[atom - 1] for atom in complex_atoms if atom in ligand_atoms
            ]
            protein_center = centroid(protein)
            ligand_center = centroid(ligand)
            shift = [
                round((protein_center[axis] - ligand_center[axis]) / box) * box
                for axis in range(3)
            ]
            selected = []
            for atom in complex_atoms:
                coordinate = list(coordinates[atom - 1])
                if atom in ligand_atoms:
                    coordinate = [coordinate[axis] + shift[axis] for axis in range(3)]
                selected.append(coordinate)
            frames.append({"time_ps": time_ps, "coordinates_nm": selected})
        Path(option("-o")).write_text(
            json.dumps(
                {
                    "fake_pbc_trajectory": True,
                    "box_nm": box,
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
elif command == "rms":
    selections = [line.strip() for line in sys.stdin if line.strip()]
    if selections != ["C-alpha", "LIG_HEAVY"]:
        print("rms requires C-alpha then LIG_HEAVY", file=sys.stderr)
        raise SystemExit(21)
    reference_payload = pbc_payload(option("-s"))
    trajectory_payload = pbc_payload(option("-f"))
    if os.environ.get("FAKE_GMX_INVALID_XVG"):
        Path(option("-o")).write_text(
            "0.0 not-a-number\n",
            encoding="utf-8",
        )
    elif reference_payload is not None and trajectory_payload is not None:
        groups = parse_index(option("-n"))
        reference_frame = reference_payload["frames"][0]  # type: ignore[index]
        reference_coordinates = reference_frame["coordinates_nm"]
        alpha_indices = [atom - 1 for atom in groups["C-alpha"]]
        ligand_indices = [atom - 1 for atom in groups["LIG_HEAVY"]]
        reference_alpha = centroid(
            [reference_coordinates[index] for index in alpha_indices]
        )
        output_lines = ['@ yaxis label "RMSD (nm)"']
        for frame in trajectory_payload["frames"]:  # type: ignore[union-attr]
            coordinates = frame["coordinates_nm"]
            alpha_center = centroid([coordinates[index] for index in alpha_indices])
            translation = [
                reference_alpha[axis] - alpha_center[axis] for axis in range(3)
            ]
            squared_distances = []
            for index in ligand_indices:
                fitted = [
                    coordinates[index][axis] + translation[axis] for axis in range(3)
                ]
                squared_distances.append(
                    sum(
                        (fitted[axis] - reference_coordinates[index][axis]) ** 2
                        for axis in range(3)
                    )
                )
            rmsd = math.sqrt(sum(squared_distances) / len(squared_distances))
            output_lines.append(f"{float(frame['time_ps']):g} {rmsd:.9f}")
        Path(option("-o")).write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    else:
        peak_time = os.environ.get("FAKE_GMX_RMSD_PEAK_TIME_PS", "10")
        profiles = {
            "0": "0.0 0.42\n10.0 0.25\n20.0 0.10\n",
            "10": "0.0 0.10\n10.0 0.42\n20.0 0.25\n",
        }
        if peak_time not in profiles:
            print("unsupported fake RMSD peak time", file=sys.stderr)
            raise SystemExit(19)
        begin = 0.0
        trajectory_text = Path(option("-f")).read_text(
            encoding="utf-8", errors="replace"
        )
        for line in trajectory_text.splitlines():
            if line.startswith("begin_ps="):
                begin = float(line.split("=", 1)[1])
        profile = "".join(
            line + "\n"
            for line in profiles[peak_time].splitlines()
            if float(line.split()[0]) >= begin
        )
        Path(option("-o")).write_text(
            f"# Working dir: {Path.cwd()}\n"
            '@ title "RMSD"\n@ yaxis label "RMSD (nm)"\n' + profile,
            encoding="utf-8",
        )
else:
    print("unsupported fake command", file=sys.stderr)
    raise SystemExit(18)
