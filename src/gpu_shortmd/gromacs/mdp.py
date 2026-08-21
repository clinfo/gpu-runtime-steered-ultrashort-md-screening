"""MDP parsing, validation, and non-mutating resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from gpu_shortmd.config.models import AppConfig


class MdpValidationError(ValueError):
    """Raised when stage settings violate stable workflow semantics."""


@dataclass(frozen=True)
class ParsedMdp:
    path: Path
    values: dict[str, str]

    def float_value(self, key: str) -> float:
        try:
            return float(self.values[key])
        except (KeyError, ValueError) as exc:
            raise MdpValidationError(
                f"{self.path.name}: {key} must be a numeric value"
            ) from exc

    def int_value(self, key: str) -> int:
        try:
            return int(self.values[key])
        except (KeyError, ValueError) as exc:
            raise MdpValidationError(
                f"{self.path.name}: {key} must be an integer value"
            ) from exc


def normalize_key(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def parse_mdp(path: str | Path) -> ParsedMdp:
    resolved = Path(path)
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise MdpValidationError(f"cannot read MDP {resolved.name}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        content = line.split(";", 1)[0].strip()
        if not content:
            continue
        if "=" not in content:
            raise MdpValidationError(f"{resolved}:{line_number}: expected key = value")
        key, raw_value = (part.strip() for part in content.split("=", 1))
        normalized = normalize_key(key)
        if normalized in values:
            raise MdpValidationError(f"{resolved}: duplicate MDP key {normalized}")
        if not raw_value:
            raise MdpValidationError(f"{resolved}: empty MDP value for {normalized}")
        values[normalized] = raw_value
    if not values:
        raise MdpValidationError(f"{resolved}: MDP has no settings")
    return ParsedMdp(path=resolved, values=values)


def stage_mdp_paths(config: AppConfig, prepared_root: Path) -> dict[str, Path]:
    return {
        "nvt": prepared_root / config.stages.nvt.mdp,
        "npt": prepared_root / config.stages.npt.mdp,
        "production": prepared_root / config.stages.production.mdp,
    }


def validate_stage_mdps(config: AppConfig, prepared_root: Path) -> dict[str, ParsedMdp]:
    if not all(
        (
            config.stages.nvt.enabled,
            config.stages.npt.enabled,
            config.stages.production.enabled,
        )
    ):
        raise MdpValidationError("stable workflow requires NVT, NPT, and production")
    parsed = {
        stage: parse_mdp(path)
        for stage, path in stage_mdp_paths(config, prepared_root).items()
    }
    for stage, mdp in parsed.items():
        dt = mdp.float_value("dt")
        nsteps = mdp.int_value("nsteps")
        if dt <= 0 or nsteps <= 0:
            raise MdpValidationError(f"{mdp.path.name}: dt/nsteps must be positive")
        integrator = mdp.values.get("integrator", "").lower()
        if integrator != "md":
            raise MdpValidationError(
                f"{mdp.path.name}: integrator must be md for {stage}"
            )

    if parsed["nvt"].values.get("gen-vel", "").lower() != "yes":
        raise MdpValidationError("NVT MDP must set gen_vel = yes")
    for stage in ("npt", "production"):
        if parsed[stage].values.get("gen-vel", "").lower() != "no":
            raise MdpValidationError(f"{stage} MDP must set gen_vel = no")

    production = parsed["production"]
    dt = production.float_value("dt")
    interval_steps = config.trajectory.output_interval_ps / dt
    if not math.isclose(interval_steps, round(interval_steps), abs_tol=1e-9):
        raise MdpValidationError(
            "trajectory.output_interval_ps must be an integer multiple of MDP dt"
        )
    configured_steps = config.trajectory.production_time_ns * 1000.0 / dt
    if not math.isclose(configured_steps, round(configured_steps), abs_tol=1e-9):
        raise MdpValidationError(
            "trajectory.production_time_ns must resolve to an integer step count"
        )
    source_interval = production.int_value("nstxout-compressed") * dt
    if not math.isclose(
        source_interval,
        config.trajectory.output_interval_ps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise MdpValidationError(
            "production MDP compressed-output interval does not match configuration"
        )
    return parsed


def write_resolved_mdp(
    parsed: ParsedMdp,
    *,
    destination: Path,
    overrides: dict[str, str | int],
) -> None:
    values = dict(parsed.values)
    for key, value in overrides.items():
        values[normalize_key(key)] = str(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "; Resolved by gpu-shortmd; source file was not mutated.\n"
        + "\n".join(f"{key} = {value}" for key, value in values.items())
        + "\n",
        encoding="utf-8",
    )


def resolve_stage_mdps(
    config: AppConfig,
    *,
    prepared_root: Path,
    destination: Path,
    velocity_seed: int,
) -> dict[str, Path]:
    parsed = validate_stage_mdps(config, prepared_root)
    production_dt = parsed["production"].float_value("dt")
    production_steps = round(
        config.trajectory.production_time_ns * 1000 / production_dt
    )
    interval_steps = round(config.trajectory.output_interval_ps / production_dt)
    outputs = {
        stage: destination / f"{stage}.mdp" for stage in ("nvt", "npt", "production")
    }
    write_resolved_mdp(
        parsed["nvt"],
        destination=outputs["nvt"],
        overrides={"gen-vel": "yes", "gen-seed": velocity_seed},
    )
    write_resolved_mdp(
        parsed["npt"],
        destination=outputs["npt"],
        overrides={"gen-vel": "no"},
    )
    write_resolved_mdp(
        parsed["production"],
        destination=outputs["production"],
        overrides={
            "gen-vel": "no",
            "nsteps": production_steps,
            "nstxout-compressed": interval_steps,
        },
    )
    return outputs
