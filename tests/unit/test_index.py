from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.gromacs.index import (
    IndexValidationError,
    parse_index,
    validate_required_groups,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
PREPARED = (
    REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4" / "prepared_input"
)


def test_example_required_groups_are_unique() -> None:
    groups = validate_required_groups(parse_index(PREPARED / "p2_index.ndx"))
    assert len(groups["C-alpha"].atoms) == 317
    assert len(groups["LIG"].atoms) == 66


def test_duplicate_required_group_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.ndx"
    path.write_text(
        "[ C-alpha ]\n1\n[ LIG ]\n2\n[ LIG ]\n2\n"
        "[ Protein_LIG ]\n1 2\n[ Water_and_ions ]\n3\n",
        encoding="utf-8",
    )
    with pytest.raises(IndexValidationError):
        validate_required_groups(parse_index(path))


@pytest.mark.parametrize(
    ("group", "atoms"),
    [
        ("Protein_LIG", ""),
        ("C-alpha", ""),
        ("LIG", ""),
    ],
)
def test_empty_rmsd_required_group_is_rejected(
    tmp_path: Path,
    group: str,
    atoms: str,
) -> None:
    values = {
        "C-alpha": "1",
        "LIG": "2",
        "Protein_LIG": "1 2",
        "Water_and_ions": "3",
    }
    values[group] = atoms
    path = tmp_path / "empty.ndx"
    path.write_text(
        "".join(f"[ {name} ]\n{value}\n" for name, value in values.items()),
        encoding="utf-8",
    )

    with pytest.raises(IndexValidationError, match="non-empty"):
        validate_required_groups(parse_index(path))


@pytest.mark.parametrize("missing", ["Protein_LIG", "C-alpha"])
def test_missing_pbc_or_fit_group_is_rejected(
    tmp_path: Path,
    missing: str,
) -> None:
    values = {
        "C-alpha": "1",
        "LIG": "2",
        "Protein_LIG": "1 2",
        "Water_and_ions": "3",
    }
    del values[missing]
    path = tmp_path / "missing.ndx"
    path.write_text(
        "".join(f"[ {name} ]\n{value}\n" for name, value in values.items()),
        encoding="utf-8",
    )

    with pytest.raises(IndexValidationError, match="exactly once"):
        validate_required_groups(parse_index(path))
