"""Deterministic validation for the bundled EGFR reference example."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from gpu_shortmd.analysis.md_score import calculate_md_score
from gpu_shortmd.analysis.xvg import numeric_data_sha256

EXPECTED_POSE_ID = "P00533_EGFR|p2|model4"
EXPECTED_PUBLIC_INDEX = "00753"
EXPECTED_SCORE_ANGSTROM = 4.872631
REFERENCE_GLOB = "p2_model4_rmsd_replica_*_nm.xvg"
FORBIDDEN_EXTENSIONS = {
    ".cpt",
    ".dcd",
    ".edr",
    ".gz",
    ".nc",
    ".tar",
    ".tpr",
    ".trr",
    ".tgz",
    ".xtc",
    ".zip",
}
MAX_EXAMPLE_FILE_BYTES = 20 * 1024 * 1024
REQUIRED_PREPARED_FILES = {
    "p2_em.gro",
    "p2_topol.top",
    "p2_ligand.itp",
    "p2_posre_protein.itp",
    "p2_posre_ligand.itp",
    "p2_index.ndx",
    "mdp/nvt.mdp",
    "mdp/npt.mdp",
    "mdp/production_5ns.mdp",
}
REQUIRED_INDEX_GROUPS = {"C-alpha", "LIG", "Protein_LIG", "Water_and_ions"}
INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
INDEX_GROUP_PATTERN = re.compile(r"^\s*\[\s*([^\]]+?)\s*\]\s*$", re.MULTILINE)


class ReferenceValidationError(ValueError):
    """Raised when the bundled example violates its scientific contract."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ReferenceValidationError(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceValidationError(f"{path.name} must contain a mapping")
    return value


def _validate_public_file_policy(example_dir: Path) -> list[str]:
    checked: list[str] = []
    for path in sorted(example_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(example_dir).as_posix()
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            raise ReferenceValidationError(f"forbidden example file: {relative}")
        if path.stat().st_size > MAX_EXAMPLE_FILE_BYTES:
            raise ReferenceValidationError(f"oversized example file: {relative}")
        checked.append(relative)
    return checked


def _validate_transformations(example_dir: Path, reference_files: list[Path]) -> None:
    path = example_dir / "provenance" / "transformations.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceValidationError(
            f"cannot parse transformation provenance: {exc}"
        ) from exc
    transformations = manifest.get("transformations")
    if not isinstance(transformations, list):
        raise ReferenceValidationError("transformation provenance has no entries")
    by_destination = {
        item.get("destination"): item
        for item in transformations
        if isinstance(item, dict)
    }
    for reference_file in reference_files:
        relative = reference_file.relative_to(example_dir).as_posix()
        entry = by_destination.get(relative)
        if entry is None:
            raise ReferenceValidationError(
                f"missing transformation entry for {relative}"
            )
        current_numeric_hash = numeric_data_sha256(reference_file)
        if entry.get("original_numeric_sha256") != current_numeric_hash:
            raise ReferenceValidationError(
                f"numeric provenance mismatch for {relative}"
            )
        if entry.get("cleaned_numeric_sha256") != current_numeric_hash:
            raise ReferenceValidationError(
                f"cleaned numeric checksum mismatch for {relative}"
            )


def _validate_prepared_input_links(prepared: Path) -> None:
    topology = (prepared / "p2_topol.top").read_text(encoding="utf-8")
    missing_local_includes = []
    for include in INCLUDE_PATTERN.findall(topology):
        if ".ff/" in include:
            continue
        if not (prepared / include).is_file():
            missing_local_includes.append(include)
    if missing_local_includes:
        raise ReferenceValidationError(
            "prepared topology has missing local includes: "
            + ", ".join(sorted(missing_local_includes))
        )

    index_text = (prepared / "p2_index.ndx").read_text(encoding="utf-8")
    groups = [match.strip() for match in INDEX_GROUP_PATTERN.findall(index_text)]
    invalid = sorted(name for name in REQUIRED_INDEX_GROUPS if groups.count(name) != 1)
    if invalid:
        raise ReferenceValidationError(
            "prepared index groups must exist exactly once: " + ", ".join(invalid)
        )


def validate_reference_example(example: str | Path) -> dict[str, Any]:
    example_dir = Path(example)
    if not example_dir.is_dir():
        raise ReferenceValidationError(f"example directory not found: {example_dir}")

    metadata = _load_yaml(example_dir / "metadata.yaml")
    mapping = _load_yaml(example_dir / "zenodo_mapping.yaml")
    expected = _load_yaml(example_dir / "reference" / "expected_md_score.yaml")

    pose_id = metadata.get("docking_pose", {}).get("pose_id")
    if pose_id != EXPECTED_POSE_ID:
        raise ReferenceValidationError("metadata pose ID does not match the contract")
    mapping_pose = mapping.get("pose", {})
    if mapping_pose.get("pose_id") != EXPECTED_POSE_ID:
        raise ReferenceValidationError("Zenodo pose ID does not match the contract")
    if str(mapping_pose.get("public_pose_index")) != EXPECTED_PUBLIC_INDEX:
        raise ReferenceValidationError("Zenodo public pose index must be 00753")
    if mapping.get("rmsd_unit") != "nm" or expected.get("input_unit") != "nm":
        raise ReferenceValidationError("reference XVG input unit must be explicit nm")

    reference_files = sorted((example_dir / "reference").glob(REFERENCE_GLOB))
    if len(reference_files) != 5:
        raise ReferenceValidationError("exactly five reference XVG files are required")

    prepared = example_dir / "prepared_input"
    missing = sorted(
        relative
        for relative in REQUIRED_PREPARED_FILES
        if not (prepared / relative).is_file()
    )
    if missing:
        raise ReferenceValidationError(
            f"prepared example files are missing: {', '.join(missing)}"
        )
    _validate_prepared_input_links(prepared)

    result = calculate_md_score(
        list(reference_files),
        input_unit="nm",
        output_unit="angstrom",
        requested_replicas=5,
    )
    expected_score = float(expected.get("expected_md_score_angstrom", math.nan))
    if not math.isfinite(expected_score):
        raise ReferenceValidationError("expected score is missing or non-finite")
    if not math.isclose(
        expected_score,
        EXPECTED_SCORE_ANGSTROM,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReferenceValidationError("expected score metadata is inconsistent")
    if result.md_score_angstrom is None or not math.isclose(
        result.md_score_angstrom,
        EXPECTED_SCORE_ANGSTROM,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ReferenceValidationError(
            "five-XVG MD-score does not equal 4.872631 angstrom"
        )

    _validate_transformations(example_dir, reference_files)
    checked_files = _validate_public_file_policy(example_dir)
    report = result.to_dict()
    report.update(
        {
            "example_id": metadata.get("example_id"),
            "pose_id": pose_id,
            "public_pose_index": EXPECTED_PUBLIC_INDEX,
            "expected_md_score_angstrom": EXPECTED_SCORE_ANGSTROM,
            "tolerance_angstrom": 1e-6,
            "prepared_input_complete": True,
            "numeric_provenance_valid": True,
            "public_file_policy_valid": True,
            "n_files_checked": len(checked_files),
            "validation_type": "deterministic_xvg_reference",
        }
    )
    return report
