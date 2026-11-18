"""Immutable scientific specification (ExperimentSpec) and ExecutionProfile.

ExperimentSpec holds everything that changes the scientific meaning of a run
(rules, world size, seed, numeric policy, stopping rules, observations).
ExecutionProfile holds compute placement, preview and resource budgets only.
The two are hashed independently; changing the profile never changes the
scientific hash, and the profile must never alter the dynamics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alife_lab.core.hashing import sha256_json

MAX_WORLD_SIDE = 1024
MAX_STEPS = 10_000_000

_RULE_RE = re.compile(r"^B([0-8]*)/S([0-8]*)$")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# ExperimentSpec
# --------------------------------------------------------------------------- #


class WorldSpec(_Frozen):
    width: Annotated[int, Field(ge=8, le=MAX_WORLD_SIDE)]
    height: Annotated[int, Field(ge=8, le=MAX_WORLD_SIDE)]
    boundary: Literal["periodic", "dead"] = "periodic"


PatternPreset = Literal["glider", "blinker", "block", "r_pentomino", "random", "empty"]


class InitializationSpec(_Frozen):
    preset: PatternPreset = "glider"
    # Density is only meaningful for preset="random".
    density: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    # Top-left offset of a pattern preset; None centers the pattern.
    offset_x: Annotated[int, Field(ge=0, lt=MAX_WORLD_SIDE)] | None = None
    offset_y: Annotated[int, Field(ge=0, lt=MAX_WORLD_SIDE)] | None = None


class DynamicsSpec(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["exact_integer"] = "exact_integer"
    rule: str = "B3/S23"

    @field_validator("rule")
    @classmethod
    def _check_rule(cls, value: str) -> str:
        match = _RULE_RE.match(value)
        if not match:
            raise ValueError("rule must use B<digits>/S<digits> notation, e.g. B3/S23")
        for group in match.groups():
            if len(set(group)) != len(group):
                raise ValueError("rule digits must not repeat")
        return value


class RunnerSpec(_Frozen):
    max_steps: Annotated[int, Field(ge=1, le=MAX_STEPS)] = 1000
    checkpoint_interval: Annotated[int, Field(ge=1, le=MAX_STEPS)] = 100


KnownMetric = Literal["alive_cell_count", "glider_displacement", "centroid_displacement"]


class ObservationSpec(_Frozen):
    metrics: tuple[KnownMetric, ...] = ("alive_cell_count",)


class ExperimentBody(_Frozen):
    name: Annotated[str, Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.\- ]+$")]
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["conway"]
    seed: Annotated[int, Field(ge=0, lt=2**63)]
    world: WorldSpec
    initialization: InitializationSpec = InitializationSpec()
    dynamics: DynamicsSpec = DynamicsSpec()
    runner: RunnerSpec = RunnerSpec()
    observations: ObservationSpec = ObservationSpec()

    @model_validator(mode="after")
    def _check_reference_rule(self) -> ExperimentBody:
        if self.mode == "reference" and self.dynamics.rule != "B3/S23":
            raise ValueError("reference mode for substrate 'conway' requires rule B3/S23")
        return self


class ExperimentSpec(_Frozen):
    schema_version: Literal[1] = 1
    experiment: ExperimentBody

    def digest(self) -> str:
        """SHA-256 of the canonical JSON of the fully-resolved scientific spec."""
        return sha256_json(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# ExecutionProfile
# --------------------------------------------------------------------------- #

ExecutionRole = Literal[
    "cpu_dev", "local_interactive", "remote_compute", "multi_gpu", "cluster_batch"
]


class TargetSelector(_Frozen):
    role: ExecutionRole
    backend: str = "auto"
    gpu_count: Annotated[int, Field(ge=0, le=64)] = 0
    scheduler: str | None = None
    require_explicit_target: bool = False


class MemoryPolicy(_Frozen):
    policy: Literal["probe_available"] = "probe_available"
    max_fraction_of_available: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    reserve_fraction: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.15


class SchedulingPolicy(_Frozen):
    mode: Literal["interactive", "batch"] = "interactive"
    max_active_jobs: Annotated[int, Field(ge=1, le=64)] | None = None
    max_active_gpu_jobs: Annotated[int, Field(ge=0, le=64)] | None = None
    adaptive_microbatch: bool = False
    checkpoint_and_exit: str | None = None
    job_arrays: str | None = None


class PreviewPolicy(_Frozen):
    max_fps: Annotated[float, Field(gt=0.0, le=60.0)] = 5.0
    adaptive_quality: bool = False


class TransportPolicy(_Frozen):
    mode: str
    artifact_transfer: Literal["checksum_verified"] = "checksum_verified"


class ResourceLimits(_Frozen):
    max_walltime_seconds: Annotated[int, Field(ge=1, le=30 * 24 * 3600)] = 3600
    max_disk_mb: Annotated[int, Field(ge=1, le=10_000_000)] = 1024


class SitePolicy(_Frozen):
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    notes: str | None = None


class ExecutionBody(_Frozen):
    id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-]+$")]
    target_selector: TargetSelector
    memory: MemoryPolicy = MemoryPolicy()
    scheduling: SchedulingPolicy = SchedulingPolicy()
    preview: PreviewPolicy = PreviewPolicy()
    transport: TransportPolicy | None = None
    limits: ResourceLimits = ResourceLimits()
    site_policy: SitePolicy | None = None


class ExecutionProfile(_Frozen):
    schema_version: Literal[1] = 1
    execution: ExecutionBody

    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_spec(path: Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(_load_yaml(path))


def load_execution_profile(path: Path) -> ExecutionProfile:
    return ExecutionProfile.model_validate(_load_yaml(path))
