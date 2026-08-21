from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"


def _mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_mapping_keys(item) for item in value.values()),
        )
    if isinstance(value, list):
        return set().union(*(_mapping_keys(item) for item in value))
    return set()


def _restricted_patterns() -> tuple[re.Pattern[str], ...]:
    database_names = ("PDB" + "bind", "Ch" + "EMBL", "Pub" + "Chem")
    activity_measures = (
        "K" + "i",
        "K" + "d",
        "IC" + "50",
        "p" + "Ch" + "EMBL",
    )
    active_label = "act" + "ive"
    inactive_label = "in" + "active"
    return (
        re.compile(r"(?i)\b(?:" + "|".join(map(re.escape, database_names)) + r")\b"),
        re.compile(r"(?i)\b(?:" + "|".join(map(re.escape, activity_measures)) + r")\b"),
        re.compile(
            r"(?i)\blabel\s*[:=]\s*(?:"
            + "|".join(map(re.escape, (active_label, inactive_label)))
            + r")\b"
        ),
        re.compile(r"(?i)\b(?:1|10)[- ]?(?:u|µ|μ)m\b"),
    )


def test_public_example_omits_binding_activity_annotations() -> None:
    removed_keys = {
        "_".join(("activity", field)) for field in ("type", "value", "unit")
    }
    for path in sorted(EXAMPLE.rglob("*")):
        if path.suffix.lower() not in {".json", ".md", ".yaml", ".yml"}:
            continue
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            assert removed_keys.isdisjoint(_mapping_keys(yaml.safe_load(content)))
        for pattern in _restricted_patterns():
            assert pattern.search(content) is None, path.relative_to(ROOT)


def test_structural_and_computational_example_metadata_is_preserved() -> None:
    metadata = yaml.safe_load((EXAMPLE / "metadata.yaml").read_text(encoding="utf-8"))
    assert metadata["target"] == {
        "name": "EGFR",
        "uniprot_id": "P00533",
        "reference_pdb_id": "1XKK",
    }
    assert metadata["ligand"] == {
        "pdb_ligand_code": "FMM",
        "source_identifier": "1XKK_FMM_A_91",
    }
    assert metadata["docking_pose"] == {
        "pose_set": "p2",
        "model": 4,
        "pose_id": "P00533_EGFR|p2|model4",
        "pose_label": "correct",
        "docking_score_kcal_mol": -10.319,
    }
    assert metadata["shortmd"]["expected_md_score_angstrom"] == 4.872631


def test_public_mapping_adds_version_2_record_and_preserves_crosswalk() -> None:
    mapping = yaml.safe_load(
        (EXAMPLE / "zenodo_mapping.yaml").read_text(encoding="utf-8")
    )
    dataset_doi = "10.5281/" + "zenodo.21835249"
    assert mapping["record"] == {
        "title": (
            "Six-target docking poses and MD-scores for runtime-steered ultrashort "
            "molecular dynamics"
        ),
        "doi": dataset_doi,
        "url": "https://doi.org/" + dataset_doi,
        "version": 2,
        "publication_date": "2026-08-18",
        "creators": [
            "Natsuki Kanazawa",
            "Junta Asano",
            "Shigeyuki Matsumoto",
        ],
        "license": "CC BY 4.0",
        "resource_type": "Dataset",
    }
    assert mapping["pose"] == {
        "pose_id": "P00533_EGFR|p2|model4",
        "public_pose_index": "00753",
        "directory": "data/P00533_EGFR/pose_00753",
    }
    assert mapping["rmsd_files"] == [f"rmsd_eq{index}.xvg" for index in range(1, 6)]
    assert mapping["rmsd_unit"] == "nm"
    assert mapping["expected_md_score_angstrom"] == 4.872631

    dataset_citation = (
        "Kanazawa, N., Asano, J., & Matsumoto, S. (2026). Six-target docking "
        "poses and MD-scores for runtime-steered ultrashort molecular dynamics "
        "(Version 2) [Dataset]. Zenodo. https://doi.org/" + dataset_doi
    )
    assert dataset_citation in (ROOT / "README.md").read_text(encoding="utf-8")
