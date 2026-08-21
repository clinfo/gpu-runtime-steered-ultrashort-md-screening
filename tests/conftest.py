from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLE = REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"


@pytest.fixture
def fake_gromacs(tmp_path: Path) -> tuple[Path, Path]:
    data_prefix = tmp_path / "gromacs-prefix"
    forcefield = data_prefix / "share" / "gromacs" / "top" / "amber99sb-ildn.ff"
    forcefield.mkdir(parents=True)
    for name in ("forcefield.itp", "tip3p.itp", "ions.itp"):
        (forcefield / name).write_text(f"; fake {name}\n", encoding="utf-8")

    executable = tmp_path / "fake-gmx"
    shutil.copyfile(
        REPOSITORY_ROOT / "tests" / "fixtures" / "fake_gmx.py",
        executable,
    )
    executable.chmod(0o755)
    return executable, data_prefix


@pytest.fixture
def single_replica_config(
    tmp_path: Path,
    fake_gromacs: tuple[Path, Path],
) -> tuple[Path, dict[str, str]]:
    executable, _ = fake_gromacs
    value: dict[str, Any] = yaml.safe_load(
        (EXAMPLE / "config.single_replica.yaml").read_text(encoding="utf-8")
    )
    value["run"]["output_dir"] = str(tmp_path / "outputs")
    value["input"]["prepared_system_dir"] = str(EXAMPLE / "prepared_input")
    value["gromacs"]["executable"] = str(executable)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    environment = dict(os.environ)
    environment["PATH"] = str(executable.parent) + os.pathsep + environment["PATH"]
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["FAKE_GMX_START"] = str(EXAMPLE / "prepared_input" / "p2_em.gro")
    environment["FAKE_GMX_DATA_PREFIX"] = str(fake_gromacs[1])
    environment["FAKE_GMX_CUDA_COMPILER"] = (
        str(tmp_path / "cuda" / "bin" / "nvcc") + " (release 12.2)"
    )
    return config_path, environment
