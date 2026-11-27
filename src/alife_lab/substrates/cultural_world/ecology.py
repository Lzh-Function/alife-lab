"""Ecological resource fields, regime schedules, and exact per-resource mass ledger."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import CulturalWorldExperiment

F64 = npt.NDArray[np.float64]
RESOURCES = ("food", "material_a", "material_b")


@dataclass
class ResourceLedger:
    """Cumulative accounting per resource (initial + inflow == field + inv + consumed + decayed)."""

    initial: dict[str, float] = field(
        default_factory=lambda: {"food": 0.0, "material_a": 0.0, "material_b": 0.0}
    )
    inflow: dict[str, float] = field(
        default_factory=lambda: {"food": 0.0, "material_a": 0.0, "material_b": 0.0}
    )
    consumed: dict[str, float] = field(
        default_factory=lambda: {"food": 0.0, "material_a": 0.0, "material_b": 0.0}
    )
    decayed: dict[str, float] = field(
        default_factory=lambda: {"food": 0.0, "material_a": 0.0, "material_b": 0.0}
    )

    def to_json(self) -> dict[str, dict[str, float]]:
        return {
            "initial": dict(self.initial),
            "inflow": dict(self.inflow),
            "consumed": dict(self.consumed),
            "decayed": dict(self.decayed),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ResourceLedger:
        return cls(
            initial={k: float(v) for k, v in data["initial"].items()},
            inflow={k: float(v) for k, v in data["inflow"].items()},
            consumed={k: float(v) for k, v in data["consumed"].items()},
            decayed={k: float(v) for k, v in data["decayed"].items()},
        )


@dataclass
class ResourceField:
    food: F64
    material_a: F64
    material_b: F64
    ledger: ResourceLedger

    def grid(self, kind: str) -> F64:
        if kind == "food":
            return self.food
        if kind == "material_a":
            return self.material_a
        if kind == "material_b":
            return self.material_b
        raise ValueError(f"unknown resource kind {kind!r}")


def _biome_masks(r: int) -> tuple[F64, F64]:
    """Segregated west (material_a, x≈0.25) and east (material_b, x≈0.75) biome masks."""
    _, xx = np.meshgrid(
        np.linspace(0.0, 1.0, r, endpoint=False),
        np.linspace(0.0, 1.0, r, endpoint=False),
        indexing="ij",
    )
    west = 0.03 + 0.97 * np.maximum(0.0, np.cos(2.0 * math.pi * (xx - 0.25)))
    east = 0.03 + 0.97 * np.maximum(0.0, np.cos(2.0 * math.pi * (xx - 0.75)))
    return west, east


def init_ecology(exp: CulturalWorldExperiment, rng_env: np.random.Generator) -> ResourceField:
    r = exp.world.resolution
    res = exp.dynamics.resources
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, r, endpoint=False),
        np.linspace(0.0, 1.0, r, endpoint=False),
        indexing="ij",
    )
    noise_f = 0.85 + 0.3 * rng_env.random((r, r))
    if exp.dynamics.social.interdependent_resources:
        patch_a, patch_b = _biome_masks(r)
        scale_m = 0.80
    else:
        patch_a = 0.5 + 0.5 * np.cos(2.0 * math.pi * (xx - 0.25)) * np.cos(
            2.0 * math.pi * (yy - 0.25)
        )
        patch_b = 0.5 + 0.5 * np.cos(2.0 * math.pi * (xx - 0.75)) * np.cos(
            2.0 * math.pi * (yy - 0.75)
        )
        scale_m = 0.65
    food = np.clip(0.55 * res.food_capacity * noise_f, 0.0, res.food_capacity)
    mat_a = np.clip(scale_m * res.material_a_capacity * patch_a, 0.0, res.material_a_capacity)
    mat_b = np.clip(scale_m * res.material_b_capacity * patch_b, 0.0, res.material_b_capacity)
    ledger = ResourceLedger(
        initial={
            "food": float(food.sum()),
            "material_a": float(mat_a.sum()),
            "material_b": float(mat_b.sum()),
        }
    )
    return ResourceField(food=food, material_a=mat_a, material_b=mat_b, ledger=ledger)


def regime_phase(tick: int, exp: CulturalWorldExperiment, phase_offset: float = 0.0) -> float:
    """Optimal skill tuning phase in [0, 1) driven by environmental regime."""
    if exp.world.environment_regime == "stationary":
        return float(phase_offset % 1.0)
    period = max(1, int(exp.world.regime_period))
    return float(((tick / period) + phase_offset) % 1.0)


def step_ecology(
    eco: ResourceField,
    exp: CulturalWorldExperiment,
    tick: int,
    rng_env: np.random.Generator,
    phase_offset: float = 0.0,
) -> None:
    """Advance resource fields with strict non-negative inflow and decay bookkeeping."""
    res = exp.dynamics.resources
    r = exp.world.resolution
    phase = regime_phase(tick, exp, phase_offset)
    interdep = exp.dynamics.social.interdependent_resources
    if exp.world.environment_regime == "stationary":
        mult = 1.0
        spatial_a, spatial_b = (
            _biome_masks(r)
            if interdep
            else (np.ones((r, r), dtype=np.float64), np.ones((r, r), dtype=np.float64))
        )
    elif exp.world.environment_regime == "seasonal":
        # Scarcity seasons make material-processing skills critical for survival.
        mult = 0.55 + 0.45 * math.sin(2.0 * math.pi * phase)
        spatial_a, spatial_b = (
            _biome_masks(r)
            if interdep
            else (np.ones((r, r), dtype=np.float64), np.ones((r, r), dtype=np.float64))
        )
    else:
        # shifting_patches: food & material centers rotate over the torus.
        mult = 0.65 + 0.35 * math.cos(2.0 * math.pi * phase)
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, r, endpoint=False),
            np.linspace(0.0, 1.0, r, endpoint=False),
            indexing="ij",
        )
        if interdep:
            west, east = _biome_masks(r)
            y_a = 0.65 + 0.35 * np.maximum(0.0, np.cos(2.0 * math.pi * (yy - phase)))
            y_b = 0.65 + 0.35 * np.maximum(0.0, np.cos(2.0 * math.pi * (yy - (1.0 - phase))))
            spatial_a = west * y_a
            spatial_b = east * y_b
        else:
            spatial_a = 0.3 + 0.7 * np.maximum(0.0, np.cos(2.0 * math.pi * (xx - phase)))
            spatial_b = 0.3 + 0.7 * np.maximum(0.0, np.cos(2.0 * math.pi * (yy - (1.0 - phase))))

    # Subtle spatial jitter draws from rng_env so env stream stays active & shared across arms.
    _ = float(rng_env.random())

    specs = (
        ("food", eco.food, res.food_capacity * mult, res.food_regrowth, res.food_decay, 1.0),
        (
            "material_a",
            eco.material_a,
            res.material_a_capacity,
            res.material_a_regrowth,
            res.material_decay,
            spatial_a,
        ),
        (
            "material_b",
            eco.material_b,
            res.material_b_capacity,
            res.material_b_regrowth,
            res.material_decay,
            spatial_b,
        ),
    )
    for name, grid, cap_base, rate, decay_rate, mask in specs:
        decay = grid * decay_rate
        grid -= decay
        eco.ledger.decayed[name] += float(decay.sum())
        cap = cap_base * mask
        add = np.maximum(0.0, cap - grid) * rate
        grid += add
        eco.ledger.inflow[name] += float(add.sum())


def ledger_residuals(eco: ResourceField, inv_a_sum: float, inv_b_sum: float) -> dict[str, float]:
    """Return signed (observed - expected) residual for every resource."""
    lg = eco.ledger
    inv_map = {"food": 0.0, "material_a": inv_a_sum, "material_b": inv_b_sum}
    out: dict[str, float] = {}
    for name in RESOURCES:
        observed = float(eco.grid(name).sum()) + inv_map[name]
        expected = lg.initial[name] + lg.inflow[name] - lg.consumed[name] - lg.decayed[name]
        out[name] = observed - expected
    return out


def max_ledger_residual(eco: ResourceField, inv_a_sum: float, inv_b_sum: float) -> float:
    res = ledger_residuals(eco, inv_a_sum, inv_b_sum)
    return max(abs(v) for v in res.values())
