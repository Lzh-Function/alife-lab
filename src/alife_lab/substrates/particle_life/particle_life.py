"""Particle Life (CPU / NumPy float64), exploratory only.

Community model popularised by J. Ventrella ("Clusters") and T. Mohr ("particle-life");
there is no canonical numerical reference, so this formulation is documented in
``docs/reference_fidelity/particle_life.md`` and runs only in exploratory mode.

Per tick, for each particle *i* and neighbour *j* within ``r_max`` (minimal image),
with r = |d_ij| / r_max, unit vector u_ij = d_ij / |d_ij| and a = matrix[type_i][type_j]:

    F(r, a) = r / beta − 1                              for r < beta   (universal repulsion)
            = a · (1 − |2r − 1 − beta| / (1 − beta))    for beta ≤ r < 1
    acc_i   = force_factor · r_max · Σ_j F(r_ij, a_ij) · u_ij
    v_i ← v_i · 0.5^(dt / friction_half_life) + dt · acc_i ;  x_i ← x_i + dt · v_i

The interaction matrix need not be symmetric, so momentum is not conserved (by design of
the model). Metrics are passive; ``segregation_index`` is the observable for type sorting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import MAX_BOIDS, ExperimentSpec, ParticleLifeExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.particles.neighbors import (
    Pairs,
    _wrap_delta,
    neighbor_pairs,
    rms_id_distance,
)

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
SERIALIZE_VERSION = 1
MAX_SPAWN = 500


def _pl(spec: ExperimentSpec) -> ParticleLifeExperiment:
    exp = spec.experiment
    if not isinstance(exp, ParticleLifeExperiment):
        raise TypeError("particle_life adapter received a different spec")
    return exp


def force_profile(r: F64, a: F64, beta: float) -> F64:
    """F(r, a) for normalised distance r in [0, 1)."""
    repulse = r / beta - 1.0
    attract = a * (1.0 - np.abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
    return np.asarray(np.where(r < beta, repulse, attract), dtype=np.float64)


def accelerations(
    pos: F64, types: I64, matrix: F64, exp: ParticleLifeExperiment, pairs: Pairs
) -> F64:
    n = pos.shape[0]
    dyn = exp.dynamics
    dist = np.sqrt(pairs.dist2)
    ok = dist > 0.0
    i, j = pairs.i[ok], pairs.j[ok]
    dx, dy, d = pairs.dx[ok], pairs.dy[ok], dist[ok]
    f = force_profile(d / dyn.r_max, matrix[types[i], types[j]], dyn.beta)
    scale = dyn.force_factor * dyn.r_max
    acc = np.zeros((n, 2), dtype=np.float64)
    acc[:, 0] = np.bincount(i, weights=f * dx / d, minlength=n) * scale
    acc[:, 1] = np.bincount(i, weights=f * dy / d, minlength=n) * scale
    return acc


def accelerations_oracle(pos: F64, types: I64, matrix: F64, exp: ParticleLifeExperiment) -> F64:
    """Scalar O(N²) reference (tests only)."""
    dyn = exp.dynamics
    w, h = exp.world.width, exp.world.height
    periodic = exp.world.boundary == "periodic"
    n = pos.shape[0]
    out = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            if periodic:
                dx = (dx + w / 2) % w - w / 2
                dy = (dy + h / 2) % h - h / 2
            d = math.hypot(dx, dy)
            if d == 0.0 or d >= dyn.r_max:
                continue
            r = d / dyn.r_max
            a = matrix[types[i], types[j]]
            f = (
                r / dyn.beta - 1
                if r < dyn.beta
                else a * (1 - abs(2 * r - 1 - dyn.beta) / (1 - dyn.beta))
            )
            out[i, 0] += f * dx / d * dyn.force_factor * dyn.r_max
            out[i, 1] += f * dy / d * dyn.force_factor * dyn.r_max
    return out


@dataclass
class ParticleLifeState:
    pos: F64
    vel: F64
    types: I64
    ids: I64
    matrix: F64
    next_id: int
    tick: int
    rng: np.random.Generator = field(repr=False)
    world_width: float = 0.0
    world_height: float = 0.0
    periodic: bool = True


def _boundary(pos: F64, vel: F64, w: float, h: float, periodic: bool) -> None:
    if periodic:
        pos[:, 0] %= w
        pos[:, 1] %= h
        return
    for axis, size in ((0, w), (1, h)):
        low = pos[:, axis] < 0.0
        pos[low, axis] = -pos[low, axis]
        vel[low, axis] = -vel[low, axis]
        high = pos[:, axis] > size
        pos[high, axis] = 2.0 * size - pos[high, axis]
        vel[high, axis] = -vel[high, axis]
        np.clip(pos[:, axis], 0.0, size, out=pos[:, axis])


def segregation_index(types: I64, pairs: Pairs) -> float:
    """Fraction of neighbour pairs (within r_max) that share a type; NaN without pairs."""
    if pairs.i.size == 0:
        return math.nan
    return float((types[pairs.i] == types[pairs.j]).mean())


_METRICS = (
    MetricDefinition(
        "population",
        "metric.population",
        "particles",
        "tick",
        "Number of particles.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    MetricDefinition(
        "mean_speed",
        "metric.mean_speed",
        "world units / time",
        "tick",
        "Mean |v_i|.",
        "none",
        "every tick",
        1,
        "null when N = 0",
    ),
    MetricDefinition(
        "mean_neighbor_count",
        "metric.mean_neighbor_count",
        "particles",
        "tick",
        "Mean number of other particles within r_max (minimal image).",
        "none",
        "every tick",
        1,
        "null when N = 0",
    ),
    MetricDefinition(
        "segregation_index",
        "metric.segregation_index",
        "fraction [0, 1]",
        "tick",
        "Fraction of directed neighbour pairs within r_max whose types are equal. For "
        "well-mixed particles its expectation is sum_k p_k^2 (p_k = type fractions).",
        "none",
        "every tick",
        1,
        "null when there are no neighbour pairs",
    ),
)

PARTICLE_LIFE_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.particle_life",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="particle_life",
    display_name_key="substrate.particle_life",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="particles",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("spawn", "remove"),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Ventrella 'Clusters'; T. Mohr 'particle-life' (community model). "
        "No canonical numerical reference.",
        "paper_urls": [],
        "implementation_commit": None,
    },
)


class ParticleLifeAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return PARTICLE_LIFE_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> ParticleLifeState:
        exp = _pl(spec)
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        k = exp.initialization.types
        if exp.dynamics.matrix is None:
            matrix = rng.uniform(-1.0, 1.0, size=(k, k))
        else:
            matrix = np.asarray(exp.dynamics.matrix, dtype=np.float64)
        n = exp.initialization.count
        pos = rng.random((n, 2)) * np.array([exp.world.width, exp.world.height])
        types = rng.integers(0, k, size=n).astype(np.int64)
        return ParticleLifeState(
            pos=pos,
            vel=np.zeros((n, 2)),
            types=types,
            ids=np.arange(n, dtype=np.int64),
            matrix=matrix,
            next_id=n,
            tick=0,
            rng=rng,
            world_width=exp.world.width,
            world_height=exp.world.height,
            periodic=exp.world.boundary == "periodic",
        )

    def _pairs(self, state: ParticleLifeState, exp: ParticleLifeExperiment) -> Pairs:
        return neighbor_pairs(
            state.pos, exp.dynamics.r_max, exp.world.width, exp.world.height, state.periodic
        )

    def step(self, state: ParticleLifeState, spec: ExperimentSpec) -> ParticleLifeState:
        exp = _pl(spec)
        dyn = exp.dynamics
        acc = accelerations(state.pos, state.types, state.matrix, exp, self._pairs(state, exp))
        friction = 0.5 ** (dyn.dt / dyn.friction_half_life)
        vel = state.vel * friction + dyn.dt * acc
        pos = state.pos + dyn.dt * vel
        _boundary(pos, vel, exp.world.width, exp.world.height, state.periodic)
        state.pos, state.vel = pos, vel
        state.tick += 1
        return state

    def metrics(self, state: ParticleLifeState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _pl(spec)
        n = state.pos.shape[0]
        out: dict[str, float] = {"population": float(n)}
        wanted = set(exp.observations.metrics)
        if "mean_speed" in wanted:
            out["mean_speed"] = float(np.hypot(*state.vel.T).mean()) if n else math.nan
        if {"mean_neighbor_count", "segregation_index"} & wanted:
            pairs = self._pairs(state, exp)
            if "mean_neighbor_count" in wanted:
                out["mean_neighbor_count"] = pairs.i.size / n if n else math.nan
            if "segregation_index" in wanted:
                out["segregation_index"] = segregation_index(state.types, pairs)
        return out

    def render(self, state: ParticleLifeState) -> tuple[dict[str, Any], bytes]:
        data = np.concatenate(
            [state.pos, state.vel, state.types[:, None].astype(np.float64)], axis=1
        ).astype("<f4")
        return {
            "codec": "particles_typed_f32",
            "count": int(state.pos.shape[0]),
            "types": int(state.matrix.shape[0]),
            "world_width": state.world_width,
            "world_height": state.world_height,
        }, data.tobytes()

    def apply_intervention(
        self,
        state: ParticleLifeState,
        spec: ExperimentSpec,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[ParticleLifeState, dict[str, Any]]:
        exp = _pl(spec)
        w, h = exp.world.width, exp.world.height
        try:
            x, y = float(payload["x"]), float(payload["y"])
            radius = float(payload.get("radius", 5.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs numeric x, y (and optional radius)") from exc
        if not (0.0 <= x <= w and 0.0 <= y <= h):
            raise ValueError("(x, y) is outside the world")
        if not 0.0 <= radius <= max(w, h):
            raise ValueError("radius out of range")
        before = int(state.pos.shape[0])
        k = state.matrix.shape[0]
        if kind == "spawn":
            count = int(payload.get("count", 10))
            if not 1 <= count <= MAX_SPAWN:
                raise ValueError(f"count must be within 1..{MAX_SPAWN}")
            if before + count > MAX_BOIDS:
                raise ValueError(f"population would exceed {MAX_BOIDS}")
            ptype = payload.get("type")
            if ptype is not None and not (isinstance(ptype, int) and 0 <= ptype < k):
                raise ValueError(f"type must be an integer within 0..{k - 1}")
            r = radius * np.sqrt(state.rng.random(count))
            theta = state.rng.random(count) * (2.0 * math.pi)
            new_pos = np.stack([x + r * np.cos(theta), y + r * np.sin(theta)], axis=1)
            new_vel = np.zeros((count, 2))
            new_types = (
                np.full(count, ptype, dtype=np.int64)
                if ptype is not None
                else state.rng.integers(0, k, size=count).astype(np.int64)
            )
            _boundary(new_pos, new_vel, w, h, state.periodic)
            state.pos = np.concatenate([state.pos, new_pos])
            state.vel = np.concatenate([state.vel, new_vel])
            state.types = np.concatenate([state.types, new_types])
            state.ids = np.concatenate(
                [state.ids, np.arange(state.next_id, state.next_id + count, dtype=np.int64)]
            )
            state.next_id += count
        elif kind == "remove":
            dx = state.pos[:, 0] - x
            dy = state.pos[:, 1] - y
            if state.periodic:
                dx, dy = _wrap_delta(dx, w), _wrap_delta(dy, h)
            keep = dx * dx + dy * dy > radius * radius
            state.pos, state.vel = state.pos[keep], state.vel[keep]
            state.types, state.ids = state.types[keep], state.ids[keep]
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        return state, {"population_before": before, "population_after": int(state.pos.shape[0])}

    def inspect(
        self, state: ParticleLifeState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _pl(spec)
        x, y = float(query["x"]), float(query["y"])
        radius = min(float(query.get("radius", 5.0)), max(exp.world.width, exp.world.height))
        dx = state.pos[:, 0] - x
        dy = state.pos[:, 1] - y
        if state.periodic:
            dx, dy = _wrap_delta(dx, exp.world.width), _wrap_delta(dy, exp.world.height)
        dist = np.hypot(dx, dy)
        inside = np.flatnonzero(dist <= radius)
        inside = inside[np.argsort(dist[inside], kind="stable")][:20]
        k = state.matrix.shape[0]
        result: dict[str, Any] = {
            "substrate": "particle_life",
            "tick": state.tick,
            "query": {"x": x, "y": y, "radius": radius},
            "count_in_radius": int((dist <= radius).sum()),
            "type_counts_in_radius": np.bincount(state.types[dist <= radius], minlength=k).tolist(),
            "matrix": state.matrix.tolist(),
            "particles": [
                {
                    "id": int(state.ids[i]),
                    "type": int(state.types[i]),
                    "x": float(state.pos[i, 0]),
                    "y": float(state.pos[i, 1]),
                    "speed": float(math.hypot(*state.vel[i])),
                    "distance": float(dist[i]),
                }
                for i in inside.tolist()
            ],
            "nearest_particle": None,
        }
        if inside.size:
            i = int(inside[0])
            pairs = self._pairs(state, exp)
            mine = pairs.i == i
            result["nearest_particle"] = {
                **result["particles"][0],
                "neighbors_by_type": np.bincount(state.types[pairs.j[mine]], minlength=k).tolist(),
            }
        return result

    def state_distance(self, a: ParticleLifeState, b: ParticleLifeState) -> dict[str, Any]:
        exact = (
            np.array_equal(a.ids, b.ids)
            and np.array_equal(a.pos, b.pos)
            and np.array_equal(a.vel, b.vel)
            and np.array_equal(a.types, b.types)
            and np.array_equal(a.matrix, b.matrix)
        )
        value, common, only_a, only_b = rms_id_distance(
            a.pos, a.ids, b.pos, b.ids, a.world_width, a.world_height, a.periodic
        )
        return {
            "metric": "rms_position_distance",
            "value": 0.0 if exact else value,
            "exact_equal": bool(exact),
            "common_ids": common,
            "only_in_first": only_a,
            "only_in_second": only_b,
        }

    def serialize_state(self, state: ParticleLifeState) -> dict[str, Any]:
        return {
            "arrays": {
                "pos": state.pos,
                "vel": state.vel,
                "types": state.types,
                "ids": state.ids,
                "matrix": state.matrix,
            },
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "next_id": state.next_id,
                "rng_state": state.rng.bit_generator.state,
                "count": int(state.pos.shape[0]),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> ParticleLifeState:
        exp = _pl(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible particle_life checkpoint serialize_version")
        arr = payload["arrays"]
        pos = np.asarray(arr["pos"], dtype=np.float64).copy()
        vel = np.asarray(arr["vel"], dtype=np.float64).copy()
        types = np.asarray(arr["types"], dtype=np.int64).copy()
        ids = np.asarray(arr["ids"], dtype=np.int64).copy()
        matrix = np.asarray(arr["matrix"], dtype=np.float64).copy()
        n, k = ids.shape[0], exp.initialization.types
        if pos.shape != (n, 2) or vel.shape != (n, 2) or types.shape != (n,) or n > MAX_BOIDS:
            raise ValueError("inconsistent particle_life checkpoint shapes")
        if matrix.shape != (k, k) or not np.isfinite(matrix).all():
            raise ValueError("invalid interaction matrix in checkpoint")
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            raise ValueError("non-finite values in checkpoint")
        if n and (types.min() < 0 or types.max() >= k or np.unique(ids).size != n):
            raise ValueError("invalid types or ids in checkpoint")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        return ParticleLifeState(
            pos=pos,
            vel=vel,
            types=types,
            ids=ids,
            matrix=matrix,
            next_id=int(meta["next_id"]),
            tick=int(meta["tick"]),
            rng=rng,
            world_width=exp.world.width,
            world_height=exp.world.height,
            periodic=exp.world.boundary == "periodic",
        )
