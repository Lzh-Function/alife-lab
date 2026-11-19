"""Gray-Scott adapter (float64 explicit Euler; NumPy or CuPy). See ``model.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, GrayScottExperiment
from alife_lab.substrates._xp import to_backend, to_host
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.gray_scott import model

SERIALIZE_VERSION = 1
V_THRESHOLD = 0.1
DISPLAY_V_MAX = 0.4


def _gs(spec: ExperimentSpec) -> GrayScottExperiment:
    exp = spec.experiment
    if not isinstance(exp, GrayScottExperiment):
        raise TypeError("gray_scott adapter received a non-gray_scott spec")
    return exp


def _params(exp: GrayScottExperiment) -> dict[str, Any]:
    d = exp.dynamics
    return {
        "Du": d.Du,
        "Dv": d.Dv,
        "F": d.F,
        "k": d.k,
        "dx": exp.world.length / exp.world.width,
        "reaction": d.reaction,
    }


@dataclass
class GrayScottState:
    u: Any  # float64 (H, W): numpy or cupy
    v: Any
    tick: int


_M = MetricDefinition
_METRICS = (
    _M("mean_u", "metric.mean_u", "concentration", "tick", "Spatial mean of u.", "none",
       "every tick", 1, "never null"),
    _M("mean_v", "metric.mean_v", "concentration", "tick", "Spatial mean of v.", "none",
       "every tick", 1, "never null"),
    _M("v_area_fraction", "metric.v_area_fraction", "fraction", "tick",
       f"Fraction of cells with v > {V_THRESHOLD}.", "none", "every tick", 1, "never null"),
    _M("v_std", "metric.v_std", "concentration", "tick",
       "Spatial standard deviation of v (pattern contrast).", "none", "every tick", 1,
       "never null"),
    _M("total_u_plus_v", "metric.total_u_plus_v", "sum of concentrations", "tick",
       "Sum of u + v over all cells. Conserved exactly (up to rounding) by diffusion alone.",
       "none", "every tick", 1, "never null"),
)  # fmt: skip

GRAY_SCOTT_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.gray_scott",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="gray_scott",
    display_name_key="substrate.gray_scott",
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
        intervention_types=("seed", "erase"),
    ),
    backends=("numpy", "cupy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Pearson (1993) Complex Patterns in a Simple System, Science 261:189.",
        "paper_urls": ["https://doi.org/10.1126/science.261.5118.189"],
        "implementation": "this repository (5-point Laplacian, forward Euler)",
    },
)


class GrayScottAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in GRAY_SCOTT_MANIFEST.backends:
            raise ValueError(f"unsupported gray_scott backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return GRAY_SCOTT_MANIFEST

    def _device(self, state: GrayScottState) -> GrayScottState:
        state.u = to_backend(state.u, self.backend)
        state.v = to_backend(state.v, self.backend)
        return state

    def initialize(self, spec: ExperimentSpec) -> GrayScottState:
        exp = _gs(spec)
        n = exp.world.width
        u = np.ones((n, n), dtype=np.float64)
        v = np.zeros((n, n), dtype=np.float64)
        init = exp.initialization
        if init.preset == "pearson":
            size = min(init.size, n)
            o = (n - size) // 2
            u[o : o + size, o : o + size] = 0.5
            v[o : o + size, o : o + size] = 0.25
            if init.noise > 0:
                rng = np.random.Generator(np.random.PCG64(exp.seed))
                u *= 1.0 + init.noise * (2.0 * rng.random((n, n)) - 1.0)
                v *= 1.0 + init.noise * (2.0 * rng.random((n, n)) - 1.0)
        return self._device(GrayScottState(u=u, v=v, tick=0))

    def step(self, state: GrayScottState, spec: ExperimentSpec) -> GrayScottState:
        exp = _gs(spec)
        state.u, state.v = model.step(state.u, state.v, dt=exp.dynamics.dt, **_params(exp))
        state.tick += 1
        return state

    def metrics(self, state: GrayScottState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_gs(spec).observations.metrics)
        out: dict[str, float] = {}
        if "mean_u" in wanted:
            out["mean_u"] = float(state.u.mean())
        if "mean_v" in wanted:
            out["mean_v"] = float(state.v.mean())
        if "v_area_fraction" in wanted:
            out["v_area_fraction"] = float((state.v > V_THRESHOLD).mean())
        if "v_std" in wanted:
            out["v_std"] = float(state.v.std())
        if "total_u_plus_v" in wanted:
            out["total_u_plus_v"] = float(state.u.sum() + state.v.sum())
        return out

    def render(self, state: GrayScottState) -> tuple[dict[str, Any], bytes]:
        n = state.v.shape[0]
        scaled = (state.v / DISPLAY_V_MAX).clip(0.0, 1.0) * 255
        data = to_host(scaled.round().astype(np.uint8)).tobytes()
        header = {
            "codec": "field_u8",
            "width": n,
            "height": n,
            "channels": 1,
            "scale": f"v / {DISPLAY_V_MAX}, clipped",
        }
        return header, data

    def apply_intervention(
        self, state: GrayScottState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[GrayScottState, dict[str, Any]]:
        n = state.u.shape[0]
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 5))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs integer x, y (and optional radius)") from exc
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError("(x, y) is outside the world")
        if not 0 <= radius <= 128:
            raise ValueError("radius must be within 0..128")
        yy, xx = np.ogrid[0:n, 0:n]
        dyy = (yy - y + n // 2) % n - n // 2
        dxx = (xx - x + n // 2) % n - n // 2
        mask = dxx * dxx + dyy * dyy <= radius * radius
        u, v = to_host(state.u).copy(), to_host(state.v).copy()
        if kind == "seed":  # Pearson's initial perturbation values
            u[mask], v[mask] = 0.5, 0.25
        elif kind == "erase":  # back to the trivial steady state
            u[mask], v[mask] = 1.0, 0.0
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        state.u, state.v = to_backend(u, self.backend), to_backend(v, self.backend)
        return state, {"cells": int(mask.sum())}

    def inspect(
        self, state: GrayScottState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _gs(spec)
        n = state.u.shape[0]
        x, y = int(query["x"]), int(query["y"])
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError("(x, y) is outside the world")
        p = _params(exp)
        du, dv, uvv = model.rates(state.u, state.v, **p)
        lu, lv = model.laplacian(state.u, p["dx"]), model.laplacian(state.v, p["dx"])

        def pick(a: Any) -> float:
            return float(to_host(a[y, x]))

        return {
            "substrate": "gray_scott",
            "tick": state.tick,
            "query": {"x": x, "y": y},
            "u": pick(state.u),
            "v": pick(state.v),
            "laplacian_u": pick(lu),
            "laplacian_v": pick(lv),
            "reaction_uvv": pick(uvv),
            "du_dt": pick(du),
            "dv_dt": pick(dv),
        }

    def state_distance(self, a: GrayScottState, b: GrayScottState) -> dict[str, Any]:
        ua, va, ub, vb = to_host(a.u), to_host(a.v), to_host(b.u), to_host(b.v)
        if ua.shape != ub.shape:
            return {"metric": "max_abs_difference", "value": None, "exact_equal": False}
        d = max(float(np.abs(ua - ub).max()), float(np.abs(va - vb).max()))
        return {
            "metric": "max_abs_difference",
            "value": d,
            "exact_equal": bool(np.array_equal(ua, ub) and np.array_equal(va, vb)),
        }

    def serialize_state(self, state: GrayScottState) -> dict[str, Any]:
        return {
            "arrays": {"u": to_host(state.u), "v": to_host(state.v)},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "dtype": "float64",
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> GrayScottState:
        exp = _gs(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible gray_scott checkpoint serialize_version")
        u = np.asarray(payload["arrays"]["u"], dtype=np.float64).copy()
        v = np.asarray(payload["arrays"]["v"], dtype=np.float64).copy()
        n = exp.world.width
        if u.shape != (n, n) or v.shape != (n, n):
            raise ValueError("gray_scott checkpoint shape does not match the spec")
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise ValueError("gray_scott checkpoint values must be finite")
        return self._device(GrayScottState(u=u, v=v, tick=int(meta["tick"])))
