"""Boids (CPU / NumPy float64).

Scientific contract (exploratory mode only — there is no canonical numerical reference
implementation of Reynolds' 1987 boids; this formulation is documented in
``docs/reference_fidelity/boids.md``):

* State: positions ``(N, 2)`` and velocities ``(N, 2)`` in float64, stable integer ids,
  PCG64 RNG state (used for initialization, optional velocity noise and spawning).
* Per tick, for each boid *i* with neighbours *j* within ``perception_radius``:
  - separation  s_i = Σ_{|d_ij| < r_sep} -d_ij / |d_ij|²       (d_ij = x_j - x_i, minimal image)
  - alignment   a_i = mean_j v_j - v_i
  - cohesion    c_i = mean_j d_ij
  - v_i ← v_i + dt · (w_s s_i + w_a a_i + w_c c_i) [+ N(0, noise²)]
  - |v_i| clamped to [min_speed, max_speed];  x_i ← x_i + dt · v_i;  boundary applied.
* The rules are a *scripted local behaviour*; flock-level order (polarization) is the
  emergent observable. Metrics are passive.

Neighbour search uses a uniform cell list (no all-pairs O(N²)); ``neighbor_pairs_bruteforce``
is the O(N²) oracle used only by tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import MAX_BOIDS, BoidsExperiment, ExperimentSpec
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.particles.neighbors import (
    Pairs,
    _wrap_delta,
    neighbor_pairs,
    neighbor_pairs_bruteforce,
)

__all__ = ["BoidsAdapter", "neighbor_pairs", "neighbor_pairs_bruteforce", "steering"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

SERIALIZE_VERSION = 1
MAX_SPAWN = 500


def _boids(spec: ExperimentSpec) -> BoidsExperiment:
    exp = spec.experiment
    if not isinstance(exp, BoidsExperiment):
        raise TypeError("boids adapter received a non-boids spec")
    return exp


# --------------------------------------------------------------------------- #
# Neighbour search
# --------------------------------------------------------------------------- #


def steering(
    pos: F64, vel: F64, exp: BoidsExperiment, pairs: Pairs
) -> tuple[F64, npt.NDArray[np.int64]]:
    """Acceleration from separation / alignment / cohesion and the neighbour counts."""
    n = pos.shape[0]
    dyn = exp.dynamics
    counts = np.bincount(pairs.i, minlength=n).astype(np.int64)
    has = counts > 0
    denom = np.maximum(counts, 1).astype(np.float64)
    mean_vx = np.bincount(pairs.i, weights=vel[pairs.j, 0], minlength=n) / denom
    mean_vy = np.bincount(pairs.i, weights=vel[pairs.j, 1], minlength=n) / denom
    mean_dx = np.bincount(pairs.i, weights=pairs.dx, minlength=n) / denom
    mean_dy = np.bincount(pairs.i, weights=pairs.dy, minlength=n) / denom
    close = (pairs.dist2 < dyn.separation_radius**2) & (pairs.dist2 > 0.0)
    ci = pairs.i[close]
    inv = 1.0 / pairs.dist2[close]
    sep_x = np.bincount(ci, weights=-pairs.dx[close] * inv, minlength=n)
    sep_y = np.bincount(ci, weights=-pairs.dy[close] * inv, minlength=n)
    acc = np.zeros((n, 2), dtype=np.float64)
    acc[:, 0] = dyn.separation_weight * sep_x
    acc[:, 1] = dyn.separation_weight * sep_y
    acc[has, 0] += dyn.alignment_weight * (mean_vx[has] - vel[has, 0])
    acc[has, 1] += dyn.alignment_weight * (mean_vy[has] - vel[has, 1])
    acc[has, 0] += dyn.cohesion_weight * mean_dx[has]
    acc[has, 1] += dyn.cohesion_weight * mean_dy[has]
    return acc, counts


def steering_components(
    pos: F64, vel: F64, exp: BoidsExperiment, pairs: Pairs, index: int
) -> dict[str, Any]:
    """Weighted rule contributions for one boid (inspection only; never used for dynamics)."""
    dyn = exp.dynamics
    mine = pairs.i == index
    n_nb = int(mine.sum())
    dx, dy, d2 = pairs.dx[mine], pairs.dy[mine], pairs.dist2[mine]
    close = (d2 < dyn.separation_radius**2) & (d2 > 0.0)
    sep = np.array([(-dx[close] / d2[close]).sum(), (-dy[close] / d2[close]).sum()])
    if n_nb:
        ali = vel[pairs.j[mine]].mean(axis=0) - vel[index]
        coh = np.array([dx.mean(), dy.mean()])
    else:
        ali = np.zeros(2)
        coh = np.zeros(2)

    def vec(v: F64, w: float) -> dict[str, float]:
        x, y = float(w * v[0]), float(w * v[1])
        return {"x": x, "y": y, "magnitude": math.hypot(x, y)}

    return {
        "neighbors": n_nb,
        "too_close": int(close.sum()),
        "separation": vec(sep, dyn.separation_weight),
        "alignment": vec(ali, dyn.alignment_weight),
        "cohesion": vec(coh, dyn.cohesion_weight),
    }


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass
class BoidsState:
    pos: F64
    vel: F64
    ids: I64
    next_id: int
    tick: int
    rng: np.random.Generator = field(repr=False)
    # Copied from the spec (not serialized): needed for display and minimal-image distances.
    world_width: float = 0.0
    world_height: float = 0.0


def _random_velocities(rng: np.random.Generator, n: int, speed: float) -> F64:
    angles = rng.random(n) * (2.0 * math.pi)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * speed


def _apply_boundary(pos: F64, vel: F64, width: float, height: float, periodic: bool) -> None:
    if periodic:
        pos[:, 0] %= width
        pos[:, 1] %= height
        return
    for axis, size in ((0, width), (1, height)):
        low = pos[:, axis] < 0.0
        pos[low, axis] = -pos[low, axis]
        vel[low, axis] = -vel[low, axis]
        high = pos[:, axis] > size
        pos[high, axis] = 2.0 * size - pos[high, axis]
        vel[high, axis] = -vel[high, axis]
        np.clip(pos[:, axis], 0.0, size, out=pos[:, axis])


_METRICS = (
    MetricDefinition(
        id="population",
        display_name_key="metric.population",
        unit="boids",
        clock="tick",
        definition="Number of boids after the update (and interventions) of the tick.",
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="never null",
    ),
    MetricDefinition(
        id="polarization",
        display_name_key="metric.polarization",
        unit="dimensionless [0, 1]",
        clock="tick",
        definition=(
            "Norm of the mean unit heading |(1/N) Σ v_i/|v_i||. 1 = all aligned; for N "
            "independent uniformly random headings its expectation is about sqrt(pi/(4N))."
        ),
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="null when N = 0",
    ),
    MetricDefinition(
        id="mean_speed",
        display_name_key="metric.mean_speed",
        unit="world units / tick",
        clock="tick",
        definition="Mean |v_i| after speed clamping.",
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="null when N = 0",
    ),
    MetricDefinition(
        id="mean_neighbor_count",
        display_name_key="metric.mean_neighbor_count",
        unit="boids",
        clock="tick",
        definition="Mean number of other boids within perception_radius (minimal image).",
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="null when N = 0",
    ),
)

BOIDS_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.boids",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="boids",
    display_name_key="substrate.boids",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="particles",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=False,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        # Bitwise deterministic for the same NumPy build/platform; replay verification
        # checks this empirically for each run.
        supports_deterministic_cpu=True,
        intervention_types=("spawn", "remove"),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": (
            "Reynolds (1987) 'Flocks, herds and schools: a distributed behavioral model'. "
            "Conceptual reference only; numerical formulation is project-specific."
        ),
        "paper_urls": ["https://doi.org/10.1145/37402.37406"],
        "implementation_commit": None,
    },
)


class BoidsAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return BOIDS_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> BoidsState:
        exp = _boids(spec)
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        n = exp.initialization.count
        size = np.array([exp.world.width, exp.world.height], dtype=np.float64)
        pos = rng.random((n, 2)) * size
        speed = 0.5 * (exp.dynamics.min_speed + exp.dynamics.max_speed)
        vel = _random_velocities(rng, n, speed)
        return BoidsState(
            pos=pos,
            vel=vel,
            ids=np.arange(n, dtype=np.int64),
            next_id=n,
            tick=0,
            rng=rng,
            world_width=exp.world.width,
            world_height=exp.world.height,
        )

    def step(self, state: BoidsState, spec: ExperimentSpec) -> BoidsState:
        exp = _boids(spec)
        dyn = exp.dynamics
        periodic = exp.world.boundary == "periodic"
        pairs = neighbor_pairs(
            state.pos, dyn.perception_radius, exp.world.width, exp.world.height, periodic
        )
        acc, _ = steering(state.pos, state.vel, exp, pairs)
        vel = state.vel + dyn.dt * acc
        if dyn.noise > 0.0 and vel.shape[0]:
            vel = vel + state.rng.normal(0.0, dyn.noise, size=vel.shape)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        stalled = speed < 1e-12
        if stalled.any():
            vel[stalled] = np.array([dyn.min_speed, 0.0])
            speed[stalled] = dyn.min_speed
        clamped = np.clip(speed, dyn.min_speed, dyn.max_speed)
        scale = np.divide(clamped, speed, out=np.ones_like(speed), where=speed > 0)
        vel = vel * scale[:, None]
        pos = state.pos + dyn.dt * vel
        _apply_boundary(pos, vel, exp.world.width, exp.world.height, periodic)
        state.pos, state.vel = pos, vel
        state.tick += 1
        return state

    def metrics(self, state: BoidsState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _boids(spec)
        n = state.pos.shape[0]
        out: dict[str, float] = {"population": float(n)}
        wanted = set(exp.observations.metrics)
        speed = np.hypot(state.vel[:, 0], state.vel[:, 1])
        if "polarization" in wanted:
            if n:
                unit = state.vel / np.maximum(speed, 1e-300)[:, None]
                out["polarization"] = float(np.hypot(*unit.mean(axis=0)))
            else:
                out["polarization"] = math.nan
        if "mean_speed" in wanted:
            out["mean_speed"] = float(speed.mean()) if n else math.nan
        if "mean_neighbor_count" in wanted:
            if n:
                pairs = neighbor_pairs(
                    state.pos,
                    exp.dynamics.perception_radius,
                    exp.world.width,
                    exp.world.height,
                    exp.world.boundary == "periodic",
                )
                out["mean_neighbor_count"] = pairs.i.size / n
            else:
                out["mean_neighbor_count"] = math.nan
        return out

    def render(self, state: BoidsState) -> tuple[dict[str, Any], bytes]:
        data = np.concatenate([state.pos, state.vel], axis=1).astype("<f4")
        return {
            "codec": "particles_f32",
            "count": int(state.pos.shape[0]),
            "world_width": state.world_width,
            "world_height": state.world_height,
        }, data.tobytes()

    def apply_intervention(
        self, state: BoidsState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[BoidsState, dict[str, Any]]:
        exp = _boids(spec)
        width, height = exp.world.width, exp.world.height
        try:
            x, y = float(payload["x"]), float(payload["y"])
            radius = float(payload.get("radius", 5.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs numeric x, y (and optional radius)") from exc
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            raise ValueError("(x, y) is outside the world")
        if not (0.0 <= radius <= max(width, height)):
            raise ValueError("radius out of range")
        before = int(state.pos.shape[0])
        if kind == "spawn":
            count = int(payload.get("count", 10))
            if not 1 <= count <= MAX_SPAWN:
                raise ValueError(f"count must be within 1..{MAX_SPAWN}")
            if before + count > MAX_BOIDS:
                raise ValueError(f"population would exceed {MAX_BOIDS}")
            r = radius * np.sqrt(state.rng.random(count))
            theta = state.rng.random(count) * (2.0 * math.pi)
            new_pos = np.stack([x + r * np.cos(theta), y + r * np.sin(theta)], axis=1)
            speed = 0.5 * (exp.dynamics.min_speed + exp.dynamics.max_speed)
            new_vel = _random_velocities(state.rng, count, speed)
            _apply_boundary(new_pos, new_vel, width, height, exp.world.boundary == "periodic")
            state.pos = np.concatenate([state.pos, new_pos])
            state.vel = np.concatenate([state.vel, new_vel])
            state.ids = np.concatenate(
                [state.ids, np.arange(state.next_id, state.next_id + count, dtype=np.int64)]
            )
            state.next_id += count
        elif kind == "remove":
            dx = state.pos[:, 0] - x
            dy = state.pos[:, 1] - y
            if exp.world.boundary == "periodic":
                dx = _wrap_delta(dx, width)
                dy = _wrap_delta(dy, height)
            keep = dx * dx + dy * dy > radius * radius
            state.pos, state.vel, state.ids = state.pos[keep], state.vel[keep], state.ids[keep]
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        return state, {"population_before": before, "population_after": int(state.pos.shape[0])}

    def inspect(
        self, state: BoidsState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        """Boids within ``radius`` of (x, y) and rule contributions for the nearest one."""
        exp = _boids(spec)
        x, y = float(query["x"]), float(query["y"])
        radius = min(float(query.get("radius", 5.0)), max(exp.world.width, exp.world.height))
        periodic = exp.world.boundary == "periodic"
        dx = state.pos[:, 0] - x
        dy = state.pos[:, 1] - y
        if periodic:
            dx = _wrap_delta(dx, exp.world.width)
            dy = _wrap_delta(dy, exp.world.height)
        dist = np.hypot(dx, dy)
        inside = np.flatnonzero(dist <= radius)
        inside = inside[np.argsort(dist[inside], kind="stable")][:20]

        def describe(i: int) -> dict[str, Any]:
            vx, vy = float(state.vel[i, 0]), float(state.vel[i, 1])
            return {
                "id": int(state.ids[i]),
                "x": float(state.pos[i, 0]),
                "y": float(state.pos[i, 1]),
                "vx": vx,
                "vy": vy,
                "speed": math.hypot(vx, vy),
                "heading_deg": (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0,
                "distance": float(dist[i]),
            }

        result: dict[str, Any] = {
            "substrate": "boids",
            "tick": state.tick,
            "query": {"x": x, "y": y, "radius": radius},
            "count_in_radius": int((dist <= radius).sum()),
            "boids": [describe(int(i)) for i in inside],
            "nearest": None,
        }
        if inside.size:
            nearest = int(inside[0])
            pairs = neighbor_pairs(
                state.pos,
                exp.dynamics.perception_radius,
                exp.world.width,
                exp.world.height,
                periodic,
            )
            result["nearest"] = {
                **describe(nearest),
                "steering": steering_components(state.pos, state.vel, exp, pairs, nearest),
            }
        return result

    def state_distance(self, a: BoidsState, b: BoidsState) -> dict[str, Any]:
        common, ia, ib = np.intersect1d(a.ids, b.ids, assume_unique=True, return_indices=True)
        exact = (
            a.ids.shape == b.ids.shape
            and bool(np.array_equal(a.ids, b.ids))
            and bool(np.array_equal(a.pos, b.pos))
            and bool(np.array_equal(a.vel, b.vel))
        )
        value: float | None = None
        if common.size:
            diff = a.pos[ia] - b.pos[ib]
            width, height = a.world_width, a.world_height
            if width > 0 and height > 0:
                diff[:, 0] = _wrap_delta(diff[:, 0], width)
                diff[:, 1] = _wrap_delta(diff[:, 1], height)
            value = float(np.sqrt((diff * diff).sum(axis=1).mean()))
        return {
            "metric": "rms_position_distance",
            "value": 0.0 if exact else value,
            "exact_equal": exact,
            "common_ids": int(common.size),
            "only_in_first": int(a.ids.size - common.size),
            "only_in_second": int(b.ids.size - common.size),
        }

    def serialize_state(self, state: BoidsState) -> dict[str, Any]:
        return {
            "arrays": {"pos": state.pos, "vel": state.vel, "ids": state.ids},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "next_id": state.next_id,
                "rng_state": state.rng.bit_generator.state,
                "dtype": "float64",
                "count": int(state.pos.shape[0]),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> BoidsState:
        exp = _boids(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible boids checkpoint serialize_version")
        arrays = payload["arrays"]
        pos = np.asarray(arrays["pos"], dtype=np.float64).copy()
        vel = np.asarray(arrays["vel"], dtype=np.float64).copy()
        ids = np.asarray(arrays["ids"], dtype=np.int64).copy()
        n = ids.shape[0]
        if pos.shape != (n, 2) or vel.shape != (n, 2) or n > MAX_BOIDS:
            raise ValueError("inconsistent boids checkpoint shapes")
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            raise ValueError("non-finite values in boids checkpoint")
        if n and (np.unique(ids).size != n or int(ids.max()) >= int(meta["next_id"])):
            raise ValueError("invalid boid ids in checkpoint")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        return BoidsState(
            pos=pos,
            vel=vel,
            ids=ids,
            next_id=int(meta["next_id"]),
            tick=int(meta["tick"]),
            rng=rng,
            world_width=exp.world.width,
            world_height=exp.world.height,
        )
