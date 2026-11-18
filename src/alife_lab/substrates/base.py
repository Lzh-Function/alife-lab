"""Minimal substrate adapter contract (spec section 5.1 / 5.3).

The contract is intentionally small: substrates own their state type and
serialization. Metrics and rendering are passive (they must never mutate
state); interventions are applied only by the worker at tick boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from alife_lab.core.spec import ExperimentSpec


@dataclass(frozen=True)
class Capabilities:
    state_kind: Literal["grid", "particles", "rigid_body", "program_soup"]
    clock_unit: Literal["tick", "epoch", "physics_step"]
    supports_pause: bool
    supports_interventions: bool
    supports_inspection: bool
    supports_lineage: bool
    supports_differentiation: bool
    supports_checkpoint: bool
    supports_deterministic_cpu: bool
    intervention_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginManifest:
    plugin_id: str
    plugin_version: str
    api_contract_version: int
    kind: Literal["substrate"]
    slug: str
    display_name_key: str
    scientific_modes: tuple[str, ...]
    capabilities: Capabilities
    backends: tuple[str, ...]
    gpu_required: bool
    serialize_version: int
    metrics: tuple[MetricDefinition, ...]
    reference: dict[str, Any] = field(default_factory=dict)
    licence: str = "project"


@dataclass(frozen=True)
class MetricDefinition:
    id: str
    display_name_key: str
    unit: str
    clock: str
    definition: str
    aggregation: str
    sampling_interval: str
    implementation_version: int
    null_policy: str


class SubstrateAdapter(Protocol):
    """Per-substrate pure functions; state is an opaque substrate-owned object."""

    @property
    def manifest(self) -> PluginManifest: ...

    def initialize(self, spec: ExperimentSpec) -> Any: ...

    def step(self, state: Any, spec: ExperimentSpec) -> Any: ...

    def metrics(self, state: Any, spec: ExperimentSpec) -> dict[str, float]: ...

    def render(self, state: Any) -> tuple[dict[str, Any], bytes]:
        """Display payload: header (``codec`` + codec-specific dimensions) and bytes."""
        ...

    def inspect(self, state: Any, spec: ExperimentSpec, query: dict[str, Any]) -> dict[str, Any]:
        """Read-only description of the state around a query point (must not mutate)."""
        ...

    def state_distance(self, a: Any, b: Any) -> dict[str, Any]:
        """Exact comparison of two states of the same spec.

        Returns ``{"metric": str, "value": float | int | None, "exact_equal": bool, ...}``.
        """
        ...

    def apply_intervention(
        self, state: Any, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]: ...

    def serialize_state(self, state: Any) -> dict[str, Any]: ...

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> Any: ...
