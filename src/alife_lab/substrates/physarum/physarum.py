"""Physarum (Jones 2010) adapter: agents + trail field. See ``model.py`` for the update order."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, PhysarumExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.physarum import model
from alife_lab.substrates.physarum.model import F64

SERIALIZE_VERSION = 1
COVERAGE_THRESHOLD = 0.5
MAX_INTERVENTION_AMOUNT = 1_000.0


def _ph(spec: ExperimentSpec) -> PhysarumExperiment:
    exp = spec.experiment
    if not isinstance(exp, PhysarumExperiment):
        raise TypeError("physarum adapter received a non-physarum spec")
    return exp


@dataclass
class PhysarumState:
    x: F64
    y: F64
    h: F64
    trail: F64
    tick: int
    rng: np.random.Generator = field(repr=False)
    occ: Any = field(repr=False, default=None)  # bool (H, W), derived from positions
    last_blocked: int = 0

    def rebuild_occupancy(self) -> None:
        hgt, wid = self.trail.shape
        occ = np.zeros((hgt, wid), dtype=np.bool_)
        occ[np.floor(self.y).astype(np.int64) % hgt, np.floor(self.x).astype(np.int64) % wid] = True
        self.occ = occ


_M = MetricDefinition
_METRICS = (
    _M("trail_mean", "metric.trail_mean", "trail units", "tick", "Mean trail value.", "none",
       "every tick", 1, "never null"),
    _M("trail_coverage", "metric.trail_coverage", "fraction", "tick",
       f"Fraction of cells with trail > {COVERAGE_THRESHOLD}.", "none", "every tick", 1,
       "never null"),
    _M("trail_contrast", "metric.trail_contrast", "ratio", "tick",
       "Coefficient of variation of the trail (std / mean); network formation raises it.",
       "none", "every tick", 1, "null when the trail is empty"),
    _M("heading_order", "metric.heading_order", "ratio", "tick",
       "|mean(exp(i * heading))| over agents (1 = all aligned).", "none", "every tick", 1,
       "never null"),
    _M("blocked_fraction", "metric.blocked_fraction", "fraction", "tick",
       "Fraction of agents whose move was blocked by exclusion in the last motor stage.",
       "none", "every tick", 1, "never null"),
)  # fmt: skip

PHYSARUM_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.physarum",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="physarum",
    display_name_key="substrate.physarum",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="grid",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("attract", "erase"),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Jones (2010) Physarum transport network approximation, Artificial Life "
        "16(2):127-153. A phenomenological model, not a biological simulation of slime moulds.",
        "paper_urls": ["https://doi.org/10.1162/artl.2010.16.2.16202"],
        "implementation": "this repository",
    },
)


class PhysarumAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return PHYSARUM_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> PhysarumState:
        exp = _ph(spec)
        hgt, wid = exp.world.height, exp.world.width
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        n = max(1, round(hgt * wid * exp.initialization.population_fraction))
        cells = rng.choice(hgt * wid, size=n, replace=False)  # one agent per cell
        y = (cells // wid).astype(np.float64) + rng.random(n)
        x = (cells % wid).astype(np.float64) + rng.random(n)
        h = rng.random(n) * model.TWO_PI
        state = PhysarumState(
            x=x, y=y, h=h, trail=np.zeros((hgt, wid), dtype=np.float64), tick=0, rng=rng
        )
        state.rebuild_occupancy()
        return state

    def step(self, state: PhysarumState, spec: ExperimentSpec) -> PhysarumState:
        d = _ph(spec).dynamics
        r = model.draw_randoms(state.rng, state.x.size)
        state.last_blocked = model.motor_stage(
            state.x, state.y, state.h, state.occ, state.trail, r, step_size=d.step_size,
            deposit=d.deposit, exclusion=d.collisions == "exclusion",
        )  # fmt: skip
        model.sensory_stage(
            state.x, state.y, state.h, state.trail, r,
            sensor_angle=math.radians(d.sensor_angle_deg), sensor_offset=d.sensor_offset,
            rotation_angle=math.radians(d.rotation_angle_deg), p_random_turn=d.p_random_turn,
            sensing=d.sensing,
        )  # fmt: skip
        hgt, wid = state.trail.shape
        for f in d.food:
            state.trail[model.disc_mask(hgt, wid, f.x, f.y, f.radius)] += f.amount
        state.trail = model.diffuse_decay(state.trail, d.decay)
        state.tick += 1
        return state

    def metrics(self, state: PhysarumState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_ph(spec).observations.metrics)
        mean = float(state.trail.mean())
        out: dict[str, float] = {}
        if "trail_mean" in wanted:
            out["trail_mean"] = mean
        if "trail_coverage" in wanted:
            out["trail_coverage"] = float((state.trail > COVERAGE_THRESHOLD).mean())
        if "trail_contrast" in wanted:
            out["trail_contrast"] = float(state.trail.std() / mean) if mean > 0 else math.nan
        if "heading_order" in wanted:
            out["heading_order"] = float(abs(np.exp(1j * state.h).mean())) if state.h.size else 0.0
        if "blocked_fraction" in wanted:
            out["blocked_fraction"] = state.last_blocked / max(1, state.x.size)
        return out

    def render(self, state: PhysarumState) -> tuple[dict[str, Any], bytes]:
        hgt, wid = state.trail.shape
        # sqrt scaling: steady trails reach ~deposit/decay; 50 = 5/0.1 for Jones' defaults.
        scaled = np.sqrt(np.clip(state.trail / 50.0, 0.0, 1.0)) * 255
        header = {
            "codec": "field_u8",
            "width": wid,
            "height": hgt,
            "channels": 1,
            "scale": "sqrt(trail / 50), clipped",
        }
        return header, np.rint(scaled).astype(np.uint8).tobytes()

    def apply_intervention(
        self, state: PhysarumState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[PhysarumState, dict[str, Any]]:
        hgt, wid = state.trail.shape
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 5))
            amount = float(payload.get("amount", 50.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs integer x, y (optional radius, amount)") from exc
        if not (0 <= x < wid and 0 <= y < hgt):
            raise ValueError("(x, y) is outside the world")
        if not 0 <= radius <= 128:
            raise ValueError("radius must be within 0..128")
        if not 0 <= amount <= MAX_INTERVENTION_AMOUNT:
            raise ValueError(f"amount must be within 0..{MAX_INTERVENTION_AMOUNT}")
        mask = model.disc_mask(hgt, wid, x, y, radius)
        before = float(state.trail[mask].sum())
        if kind == "attract":
            state.trail[mask] += amount
        elif kind == "erase":
            state.trail[mask] = 0.0
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        return state, {
            "cells": int(mask.sum()),
            "trail_before": before,
            "trail_after": float(state.trail[mask].sum()),
        }

    def inspect(
        self, state: PhysarumState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        d = _ph(spec).dynamics
        hgt, wid = state.trail.shape
        x, y = float(query["x"]), float(query["y"])
        if not (0 <= x < wid and 0 <= y < hgt):
            raise ValueError("(x, y) is outside the world")
        radius = max(1.0, min(float(query.get("radius", 4)), 64.0))
        ddx = (state.x - x + wid / 2) % wid - wid / 2
        ddy = (state.y - y + hgt / 2) % hgt - hgt / 2
        dist = np.hypot(ddx, ddy)
        inside = np.flatnonzero(dist <= radius)
        nearest: dict[str, Any] | None = None
        if state.x.size:
            a = int(np.argmin(dist))
            f, left, right = model.sensor_values(
                state.x[a : a + 1], state.y[a : a + 1], state.h[a : a + 1], state.trail,
                math.radians(d.sensor_angle_deg), d.sensor_offset,
            )  # fmt: skip
            fv, lv, rv = float(f[0]), float(left[0]), float(right[0])
            if fv > lv and fv > rv:
                decision = "keep"
            elif fv < lv and fv < rv:
                decision = "random_left_or_right"
            elif lv > rv:
                decision = "turn_left"
            elif rv > lv:
                decision = "turn_right"
            else:
                decision = "keep"
            nearest = {
                "id": a,
                "x": float(state.x[a]),
                "y": float(state.y[a]),
                "heading_deg": math.degrees(float(state.h[a])),
                "distance": float(dist[a]),
                "sensors": {"front": fv, "left": lv, "right": rv},
                "decision": decision if d.sensing else "sensing_disabled",
            }
        return {
            "substrate": "physarum",
            "tick": state.tick,
            "query": {"x": x, "y": y, "radius": radius},
            "trail": float(state.trail[int(y) % hgt, int(x) % wid]),
            "agents_in_radius": int(inside.size),
            "nearest_agent": nearest,
        }

    def state_distance(self, a: PhysarumState, b: PhysarumState) -> dict[str, Any]:
        if a.trail.shape != b.trail.shape or a.x.shape != b.x.shape:
            return {"metric": "trail_max_abs_difference", "value": None, "exact_equal": False}
        exact = all(
            np.array_equal(p, q)
            for p, q in ((a.x, b.x), (a.y, b.y), (a.h, b.h), (a.trail, b.trail))
        )
        return {
            "metric": "trail_max_abs_difference",
            "value": float(np.abs(a.trail - b.trail).max()),
            "exact_equal": exact,
            "agent_position_max_difference": float(
                max(np.abs(a.x - b.x).max(initial=0.0), np.abs(a.y - b.y).max(initial=0.0))
            ),
        }

    def serialize_state(self, state: PhysarumState) -> dict[str, Any]:
        return {
            "arrays": {"x": state.x, "y": state.y, "h": state.h, "trail": state.trail},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "rng_state": state.rng.bit_generator.state,
                "last_blocked": state.last_blocked,
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> PhysarumState:
        exp = _ph(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible physarum checkpoint serialize_version")
        arr = payload["arrays"]
        x, y, h = (np.asarray(arr[k], dtype=np.float64).copy() for k in ("x", "y", "h"))
        trail = np.asarray(arr["trail"], dtype=np.float64).copy()
        if trail.shape != (exp.world.height, exp.world.width):
            raise ValueError("physarum checkpoint trail shape does not match the spec")
        if not (x.shape == y.shape == h.shape and x.ndim == 1):
            raise ValueError("physarum checkpoint agent arrays are inconsistent")
        if not all(np.isfinite(a).all() for a in (x, y, h, trail)) or trail.min() < 0:
            raise ValueError("physarum checkpoint values must be finite (trail >= 0)")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        state = PhysarumState(
            x=x, y=y, h=h, trail=trail, tick=int(meta["tick"]), rng=rng,
            last_blocked=int(meta.get("last_blocked", 0)),
        )  # fmt: skip
        state.rebuild_occupancy()
        return state
