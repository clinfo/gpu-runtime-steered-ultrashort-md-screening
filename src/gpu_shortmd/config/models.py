"""Pydantic source of truth for stable-core configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GROMACS_SEED_MAX = 2_147_483_647
LIGAND_RESNAME_PATTERN = r"^[A-Za-z0-9_+-]{1,16}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunConfig(StrictModel):
    name: str | None = None
    output_dir: str = Field(min_length=1)
    resume: bool = False
    overwrite: Literal[False] = False


class InputConfig(StrictModel):
    prepared_system_dir: str = Field(min_length=1)
    start_structure: str = Field(min_length=1)
    topology: str = Field(min_length=1)
    index: str = Field(min_length=1)
    ligand_resname: str = Field(pattern=LIGAND_RESNAME_PATTERN)
    fit_group: Literal["C-alpha"]
    ligand_group: Literal["LIG"]


class TrajectoryConfig(StrictModel):
    production_time_ns: float = Field(gt=0)
    output_interval_ps: float = Field(gt=0)
    replicas: int = Field(ge=1)
    base_seed: int | None = None
    seeds: list[int] | None = None

    @model_validator(mode="after")
    def validate_seeds(self) -> TrajectoryConfig:
        if self.base_seed is not None and self.seeds is not None:
            raise ValueError("base_seed and seeds are mutually exclusive")
        if self.base_seed is not None:
            if self.base_seed < 1 or self.base_seed > GROMACS_SEED_MAX:
                raise ValueError("base_seed must be a positive 32-bit integer")
            if self.base_seed + self.replicas - 1 > GROMACS_SEED_MAX:
                raise ValueError(
                    "base_seed range exceeds the GROMACS 32-bit seed limit"
                )
        if self.seeds is None:
            return self
        if len(self.seeds) != self.replicas:
            raise ValueError("seeds length must equal trajectory.replicas")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("explicit seeds must be unique")
        invalid = [seed for seed in self.seeds if seed < 1 or seed > GROMACS_SEED_MAX]
        if invalid:
            raise ValueError("explicit seeds must be positive 32-bit integers")
        return self


class StageConfig(StrictModel):
    enabled: bool = True
    mdp: str = Field(min_length=1)


class StagesConfig(StrictModel):
    nvt: StageConfig
    npt: StageConfig
    production: StageConfig


class OffloadConfig(StrictModel):
    nonbonded: Literal["gpu", "cpu", "auto"]
    pme: Literal["gpu", "cpu", "auto"]
    bonded: Literal["gpu", "cpu", "auto"]
    update: Literal["gpu", "cpu", "auto"]


class GromacsConfig(StrictModel):
    executable: str = Field(min_length=1)
    ntmpi: int = Field(ge=1)
    ntomp: Annotated[int, Field(ge=1)] | Literal["auto"]
    pin: bool
    offload: OffloadConfig
    maxwarn: int = Field(ge=0)


class RmsdConfig(StrictModel):
    backend: Literal["gromacs"]
    input_unit: Literal["nm"]
    output_unit: Literal["angstrom"]
    fit_group: Literal["C-alpha"]
    ligand_group: Literal["LIG"]
    heavy_atoms_only: Literal[True]


class MonitoringConfig(StrictModel):
    poll_interval_seconds: float = Field(gt=0)


class PruningConfig(StrictModel):
    enabled: bool = False
    threshold_angstrom: float | None = Field(default=None, gt=0)
    grace_period_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_threshold(self) -> PruningConfig:
        if self.enabled and self.threshold_angstrom is None:
            raise ValueError(
                "pruning.threshold_angstrom is required when pruning is enabled"
            )
        if not self.enabled and self.threshold_angstrom is not None:
            raise ValueError(
                "pruning.threshold_angstrom must be null when pruning is disabled"
            )
        return self


class SchedulerConfig(StrictModel):
    backend: Literal["local"]
    gpu_ids: Literal["auto"] | list[int]
    work_stealing: bool
    tasks_per_gpu: Literal[1]

    @model_validator(mode="after")
    def validate_gpu_ids(self) -> SchedulerConfig:
        if isinstance(self.gpu_ids, list):
            if any(gpu_id < 0 for gpu_id in self.gpu_ids):
                raise ValueError("scheduler.gpu_ids cannot contain negative IDs")
            if len(set(self.gpu_ids)) != len(self.gpu_ids):
                raise ValueError("scheduler.gpu_ids must be unique")
            if not self.gpu_ids:
                raise ValueError("scheduler.gpu_ids list cannot be empty")
        return self


class PoseTrajectoryOverrides(StrictModel):
    replicas: int | None = Field(default=None, ge=1)
    base_seed: int | None = Field(default=None, ge=1, le=GROMACS_SEED_MAX)
    seeds: list[int] | None = None

    @model_validator(mode="after")
    def validate_seed_override(self) -> PoseTrajectoryOverrides:
        if self.base_seed is not None and self.seeds is not None:
            raise ValueError("per-pose base_seed and seeds are mutually exclusive")
        if self.seeds is None:
            return self
        if not self.seeds:
            raise ValueError("per-pose seeds cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("per-pose explicit seeds must be unique")
        if any(seed < 1 or seed > GROMACS_SEED_MAX for seed in self.seeds):
            raise ValueError("per-pose seeds must be positive 32-bit integers")
        if self.replicas is not None and len(self.seeds) != self.replicas:
            raise ValueError("per-pose seeds length must equal replicas")
        return self


class PosePruningOverrides(StrictModel):
    enabled: bool | None = None
    threshold_angstrom: float | None = Field(default=None, gt=0)
    grace_period_seconds: float | None = Field(default=None, ge=0)


class PoseOverrides(StrictModel):
    trajectory: PoseTrajectoryOverrides | None = None
    pruning: PosePruningOverrides | None = None


class PoseManifestEntry(StrictModel):
    pose_id: str = Field(min_length=1)
    prepared_system_dir: str = Field(min_length=1)
    start_structure: str = Field(min_length=1)
    topology: str = Field(min_length=1)
    index: str = Field(min_length=1)
    ligand_resname: str = Field(pattern=LIGAND_RESNAME_PATTERN)
    overrides: PoseOverrides | None = None

    @model_validator(mode="after")
    def validate_pose_id(self) -> PoseManifestEntry:
        if self.pose_id != self.pose_id.strip():
            raise ValueError("pose_id cannot have leading or trailing whitespace")
        if any(ord(character) < 32 for character in self.pose_id):
            raise ValueError("pose_id cannot contain control characters")
        return self


class PoseManifest(StrictModel):
    schema_version: Literal[1]
    poses: Annotated[list[PoseManifestEntry], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_pose_ids(self) -> PoseManifest:
        pose_ids = [pose.pose_id for pose in self.poses]
        if len(set(pose_ids)) != len(pose_ids):
            raise ValueError("multi-pose manifest pose_id values must be unique")
        return self


class RestartConfig(StrictModel):
    retry_failed: bool
    validate_existing_outputs: bool


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    event_log: bool
    console: bool


class AppConfig(StrictModel):
    schema_version: Literal[1]
    run: RunConfig
    input: InputConfig
    trajectory: TrajectoryConfig
    stages: StagesConfig
    gromacs: GromacsConfig
    rmsd: RmsdConfig
    monitoring: MonitoringConfig
    pruning: PruningConfig
    scheduler: SchedulerConfig
    restart: RestartConfig
    logging: LoggingConfig
