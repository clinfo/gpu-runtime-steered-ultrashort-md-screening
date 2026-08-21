from __future__ import annotations

from pathlib import Path

from gpu_shortmd.config.loader import load_config
from gpu_shortmd.gromacs.mdp import parse_mdp, resolve_stage_mdps

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "egfr_p00533_1xkk_fmm_p2_model4"


def test_resolve_mdp_does_not_mutate_source(tmp_path: Path) -> None:
    config = load_config(EXAMPLE / "config.single_replica.yaml")
    prepared = EXAMPLE / "prepared_input"
    source = prepared / "mdp" / "nvt.mdp"
    before = source.read_bytes()
    outputs = resolve_stage_mdps(
        config,
        prepared_root=prepared,
        destination=tmp_path,
        velocity_seed=12345,
    )
    assert source.read_bytes() == before
    nvt = parse_mdp(outputs["nvt"])
    assert nvt.values["gen-vel"] == "yes"
    assert nvt.values["gen-seed"] == "12345"
    production = parse_mdp(outputs["production"])
    assert production.int_value("nsteps") == 10_000
    assert production.int_value("nstxout-compressed") == 5_000
