from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.gromacs.topology import (
    TopologyValidationError,
    resolve_topology,
)


def test_recursive_local_and_external_includes(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    external = tmp_path / "external"
    prepared.mkdir()
    (external / "ff").mkdir(parents=True)
    (prepared / "ligand.itp").write_text("; ligand\n", encoding="utf-8")
    (external / "ff" / "forcefield.itp").write_text(
        '#include "ffnonbonded.itp"\n',
        encoding="utf-8",
    )
    (external / "ff" / "ffnonbonded.itp").write_text("; ff\n", encoding="utf-8")
    topology = prepared / "topol.top"
    topology.write_text(
        '#include "ff/forcefield.itp"\n#include "ligand.itp"\n',
        encoding="utf-8",
    )
    result = resolve_topology(
        topology,
        prepared_root=prepared,
        external_search_dirs=[external],
    )
    assert len(result.local_files) == 2
    assert len(result.external_files) == 2


def test_missing_include_is_not_silently_ignored(tmp_path: Path) -> None:
    topology = tmp_path / "topol.top"
    topology.write_text('#include "missing.itp"\n', encoding="utf-8")
    with pytest.raises(TopologyValidationError, match="unresolved"):
        resolve_topology(topology, prepared_root=tmp_path)


def test_local_include_cannot_escape_prepared_root(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    outside = tmp_path / "outside.itp"
    outside.write_text("; must not be read\n", encoding="utf-8")
    topology = prepared / "topol.top"
    topology.write_text('#include "../outside.itp"\n', encoding="utf-8")

    with pytest.raises(TopologyValidationError, match="unresolved"):
        resolve_topology(topology, prepared_root=prepared)
