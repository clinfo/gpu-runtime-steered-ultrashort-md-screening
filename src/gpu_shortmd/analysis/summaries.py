"""Structured score summary writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gpu_shortmd.analysis.md_score import ScoreResult


def write_score_json(result: ScoreResult, output: Path) -> None:
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_replica_csv(result: ScoreResult, output: Path) -> None:
    fieldnames = [
        "replica_id",
        "source",
        "time_ps",
        "max_rmsd_nm",
        "max_rmsd_angstrom",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for maximum in result.replica_maxima:
            row: dict[str, Any] = {
                field: getattr(maximum, field) for field in fieldnames
            }
            writer.writerow(row)
