from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.analysis.validation import validate_reference_example

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"


def test_reference_example_is_exact_and_provenance_linked() -> None:
    report = validate_reference_example(EXAMPLE)
    assert report["validation_type"] == "deterministic_xvg_reference"
    assert report["status"] == "COMPLETED"
    assert report["n_replicas_completed"] == 5
    assert report["md_score_angstrom"] == 4.872631
    assert report["pose_id"] == "P00533_EGFR|p2|model4"
    assert report["public_pose_index"] == "00753"
    assert report["numeric_provenance_valid"] is True


def test_reference_replica_maxima_are_deterministic() -> None:
    report = validate_reference_example(EXAMPLE)
    assert [
        item["max_rmsd_angstrom"] for item in report["replica_maxima"]
    ] == pytest.approx(
        [4.872631, 2.629156, 2.638972, 2.343307, 2.827962],
        abs=1e-12,
    )
