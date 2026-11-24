"""SweepSpec: a set of runs = conditions (factor levels) × seeds, plus a pre-declared endpoint.

The sweep is itself a scientific specification (hashed). Each member run still has its own
validated, hashed ExperimentSpec. Factors may only touch scientific sub-sections of the
experiment; the seed is set per run from ``seeds``.
"""

from __future__ import annotations

import itertools
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alife_lab.core.hashing import sha256_json

MAX_SWEEP_RUNS = 200
_PATH = (
    r"^((dynamics|world|initialization|runner|observations|soup)(\.[a-z_]{1,64}){1,2}"
    r"|wiring|silenced_ports|silenced_cell_types)$"
)
Scalar = float | int | str | bool
# Structured levels (e.g. Particle Life matrices or connectome silenced cell-type lists)
# are allowed when labelled.
Level = Scalar | list[str] | list[list[float]]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SweepFactor(_Frozen):
    path: Annotated[str, Field(pattern=_PATH)]
    values: Annotated[list[Level], Field(min_length=1, max_length=20)]
    labels: list[Annotated[str, Field(min_length=1, max_length=40)]] | None = None

    @model_validator(mode="after")
    def _check(self) -> SweepFactor:
        if self.labels is not None and len(self.labels) != len(self.values):
            raise ValueError("labels must match values")
        if len({repr(v) for v in self.values}) != len(self.values):
            raise ValueError("factor values must be distinct")
        if self.labels is None and any(isinstance(v, list) for v in self.values):
            raise ValueError("structured factor values (e.g. matrices) require labels")
        return self

    def label(self, i: int) -> str:
        if self.labels:
            return self.labels[i]
        return f"{self.path.split('.')[-1]}={self.values[i]}"


class SweepEndpoint(_Frozen):
    metric: Annotated[str, Field(pattern=r"^[a-z0-9_]{1,64}$")]
    # "final" = the last tick of each run; an int = that tick (must be <= max_steps).
    at: Literal["final"] | Annotated[int, Field(ge=0)] = "final"


class SweepBody(_Frozen):
    name: Annotated[str, Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.\- ]+$")]
    base_preset: str | None = None
    base_spec: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = None
    seeds: Annotated[
        list[Annotated[int, Field(ge=0, lt=2**63)]], Field(min_length=1, max_length=100)
    ]
    factors: Annotated[list[SweepFactor], Field(max_length=2)] = []
    reference_condition: Annotated[int, Field(ge=0)] = 0
    endpoint: SweepEndpoint
    design: Literal["exploratory", "confirmatory"] = "exploratory"
    hypothesis: Annotated[str, Field(max_length=2000)] | None = None
    profile_id: str = "cpu_dev"

    @field_validator("seeds")
    @classmethod
    def _unique(cls, seeds: list[int]) -> list[int]:
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        return seeds

    @model_validator(mode="after")
    def _check(self) -> SweepBody:
        if (self.base_preset is None) == (self.base_spec is None):
            raise ValueError("provide exactly one of base_preset or base_spec")
        if len({f.path for f in self.factors}) != len(self.factors):
            raise ValueError("factor paths must be distinct")
        if self.reference_condition >= len(self.conditions()):
            raise ValueError("reference_condition out of range")
        if len(self.conditions()) * len(self.seeds) > MAX_SWEEP_RUNS:
            raise ValueError(f"a sweep may create at most {MAX_SWEEP_RUNS} runs")
        if self.design == "confirmatory" and not self.hypothesis:
            raise ValueError("confirmatory sweeps require a pre-declared hypothesis")
        return self

    def conditions(self) -> list[dict[str, Any]]:
        """Cartesian product of factor levels: [{label, assignments{path: value}}]."""
        if not self.factors:
            return [{"label": "baseline", "assignments": {}}]
        out = []
        for combo in itertools.product(*(range(len(f.values)) for f in self.factors)):
            out.append(
                {
                    "label": " & ".join(
                        f.label(i) for f, i in zip(self.factors, combo, strict=True)
                    ),
                    "assignments": {
                        f.path: f.values[i] for f, i in zip(self.factors, combo, strict=True)
                    },
                }
            )
        return out


class SweepSpec(_Frozen):
    schema_version: Literal[1] = 1
    sweep: SweepBody

    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))
