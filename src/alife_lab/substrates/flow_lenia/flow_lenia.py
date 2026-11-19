"""Flow-Lenia adapter (CPU, NumPy, float64). See ``core.py`` for the update rule.

The rule (kernel parameters) is either explicit in the spec or sampled from the experiment seed;
the realised rule is deterministic, recomputed on load and recorded in checkpoints/inspection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, FlowLeniaExperiment
from alife_lab.substrates._xp import to_backend, to_host
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.flow_lenia import core
from alife_lab.substrates.flow_lenia.core import F64, Rule

SERIALIZE_VERSION = 1
ALIVE_THRESHOLD = 0.01
DEFAULT_KERNELS = 10  # official Config.k for one channel


def _exp(spec: ExperimentSpec) -> FlowLeniaExperiment:
    exp = spec.experiment
    if not isinstance(exp, FlowLeniaExperiment):
        raise TypeError("flow_lenia adapter received a non-flow_lenia spec")
    return exp


def _streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    rule_ss, init_ss = np.random.SeedSequence(seed).spawn(2)
    return np.random.Generator(np.random.PCG64(rule_ss)), np.random.Generator(
        np.random.PCG64(init_ss)
    )


def realise_rule(exp: FlowLeniaExperiment) -> Rule:
    rule = exp.dynamics.rule
    if rule.kernels is not None:
        R = rule.R if rule.R is not None else 0.0
        return core.rule_from_kernels(R, [k.model_dump() for k in rule.kernels])
    connectivity = rule.connectivity or [[DEFAULT_KERNELS]]
    return core.sample_rule(_streams(exp.seed)[0], connectivity, rule.R)


def _theta(exp: FlowLeniaExperiment) -> float:
    d = exp.dynamics
    return float(d.theta_A) if d.theta_A is not None else float(d.channels)


def _centroid(mass: Any, torus: bool) -> tuple[float, float] | None:
    total = float(mass.sum())
    if total <= 1e-12:
        return None
    h, w = mass.shape
    out = []
    for axis, size in ((0, w), (1, h)):  # x from columns, y from rows
        weights = to_host(mass.sum(axis=axis))  # marginals only leave the device
        if torus:
            angles = np.arange(size) * (2 * math.pi / size)
            s = float((weights * np.sin(angles)).sum())
            c = float((weights * np.cos(angles)).sum())
            if math.hypot(s, c) < 1e-12:
                return None
            out.append((math.atan2(s, c) * size / (2 * math.pi)) % size)
        else:
            out.append(float((weights * (np.arange(size) + 0.5)).sum() / total))
    return out[0], out[1]


@dataclass
class FlowLeniaState:
    cells: Any  # float64 (H, W, C): numpy, or cupy for the GPU backend
    tick: int
    rule: Rule = field(repr=False)
    kfft: Any = field(repr=False)
    torus: bool = False
    mass_baseline: float = 0.0
    last_centroid: tuple[float, float] | None = None
    dx: float = 0.0
    dy: float = 0.0

    def track(self) -> None:
        current = _centroid(self.cells.sum(axis=-1), self.torus)
        if current is not None and self.last_centroid is not None:
            h, w = self.cells.shape[:2]
            ddx = current[0] - self.last_centroid[0]
            ddy = current[1] - self.last_centroid[1]
            if self.torus:
                ddx = (ddx + w / 2) % w - w / 2
                ddy = (ddy + h / 2) % h - h / 2
            self.dx += ddx
            self.dy += ddy
        self.last_centroid = current

    def reset_baselines(self) -> None:
        self.mass_baseline = float(self.cells.sum())
        self.last_centroid, self.dx, self.dy = None, 0.0, 0.0
        self.track()


_M = MetricDefinition
_METRICS = (
    _M("mass", "metric.mass", "sum of states", "tick", "Sum over all cells and channels.",
       "none", "every tick", 1, "never null"),
    _M("mass_drift", "metric.mass_drift", "relative", "tick",
       "(mass - baseline) / baseline; baseline = mass at tick 0 or after the last "
       "intervention. Flow-Lenia conserves mass, so this measures numerical drift.",
       "none", "every tick", 1, "null when the baseline is 0"),
    _M("alive_fraction", "metric.alive_fraction", "fraction", "tick",
       f"Fraction of cells whose channel sum exceeds {ALIVE_THRESHOLD}.", "none",
       "every tick", 1, "never null"),
    _M("max_value", "metric.max_value", "state", "tick",
       "Maximum value over cells and channels (not bounded by 1).", "none", "every tick", 1,
       "never null"),
    _M("centroid_displacement", "metric.centroid_displacement", "cells", "tick",
       "Norm of the displacement of the mass centroid (circular mean on the torus) since the "
       "last baseline. Not a creature tracker.", "none", "every tick", 1,
       "null when the world is empty"),
)  # fmt: skip

FLOW_LENIA_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.flow_lenia",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="flow_lenia",
    display_name_key="substrate.flow_lenia",
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
        intervention_types=("paint", "erase"),
    ),
    backends=("numpy", "cupy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Plantec et al. (2023) Flow-Lenia; semantics of the authors' JAX code "
        "(reimplemented, no code copied: the repository has no licence).",
        "paper_urls": ["https://arxiv.org/abs/2212.07906"],
        "implementation": "https://github.com/erwanplantec/FlowLenia",
        "implementation_commit": "dce428c6b0c5079a06e5606fb7b5ac1fe1323bc5",
    },
)


class FlowLeniaAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        """``cupy``: field on the GPU (cuFFT + fused reintegration kernel), float64.
        Not bitwise equal to numpy (FFT library); see docs/reference_fidelity/flow_lenia.md."""
        if backend not in FLOW_LENIA_MANIFEST.backends:
            raise ValueError(f"unsupported flow_lenia backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return FLOW_LENIA_MANIFEST

    def _new_state(self, exp: FlowLeniaExperiment, cells: F64, tick: int) -> FlowLeniaState:
        rule = realise_rule(exp)
        return FlowLeniaState(
            cells=to_backend(cells, self.backend),
            tick=tick,
            rule=rule,
            kfft=to_backend(
                core.kernels_fft(exp.world.height, exp.world.width, rule), self.backend
            ),
            torus=exp.world.boundary == "torus",
        )

    def initialize(self, spec: ExperimentSpec) -> FlowLeniaState:
        exp = _exp(spec)
        h, w, c = exp.world.height, exp.world.width, exp.dynamics.channels
        cells = np.zeros((h, w, c), dtype=np.float64)
        init = exp.initialization
        if init.preset == "random":
            rng = _streams(exp.seed)[1]
            size = min(init.size, h, w)
            block = rng.random((size, size, c))
            if init.density < 1.0:
                block *= rng.random((size, size, 1)) < init.density
            y0, x0 = (h - size) // 2, (w - size) // 2
            cells[y0 : y0 + size, x0 : x0 + size] = block
        state = self._new_state(exp, cells, 0)
        state.reset_baselines()
        return state

    def step(self, state: FlowLeniaState, spec: ExperimentSpec) -> FlowLeniaState:
        exp = _exp(spec)
        d = exp.dynamics
        state.cells = core.step(
            state.cells,
            state.kfft,
            state.rule,
            dt=d.dt,
            dd=d.dd,
            sigma=d.sigma,
            theta_a=_theta(exp),
            n=d.n,
            boundary=exp.world.boundary,
        )
        state.tick += 1
        state.track()
        return state

    def metrics(self, state: FlowLeniaState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_exp(spec).observations.metrics)
        mass = float(state.cells.sum())
        out: dict[str, float] = {}
        if "mass" in wanted:
            out["mass"] = mass
        if "mass_drift" in wanted:
            base = state.mass_baseline
            out["mass_drift"] = (mass - base) / base if base > 0 else math.nan
        if "alive_fraction" in wanted:
            out["alive_fraction"] = float((state.cells.sum(axis=-1) > ALIVE_THRESHOLD).mean())
        if "max_value" in wanted:
            out["max_value"] = float(state.cells.max())
        if "centroid_displacement" in wanted:
            out["centroid_displacement"] = (
                math.hypot(state.dx, state.dy) if state.last_centroid is not None else math.nan
            )
        return out

    def render(self, state: FlowLeniaState) -> tuple[dict[str, Any], bytes]:
        h, w, c = state.cells.shape
        data = to_host((state.cells.clip(0.0, 1.0) * 255).round().astype(np.uint8)).tobytes()
        header = {
            "codec": "field_u8",
            "width": w,
            "height": h,
            "channels": c,
            "scale": "linear 0..1, clipped",
        }
        return header, data

    def apply_intervention(
        self, state: FlowLeniaState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[FlowLeniaState, dict[str, Any]]:
        h, w, c = state.cells.shape
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 5))
            channel = payload.get("channel")
            channel = None if channel is None else int(channel)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs integer x, y (optional radius, channel)") from exc
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        if not 0 <= radius <= 128:
            raise ValueError("radius must be within 0..128")
        if channel is not None and not 0 <= channel < c:
            raise ValueError(f"channel must be within 0..{c - 1}")
        yy, xx = np.ogrid[0:h, 0:w]
        dyy, dxx = yy - y, xx - x
        if state.torus:
            dyy = (dyy + h // 2) % h - h // 2
            dxx = (dxx + w // 2) % w - w // 2
        mask = dxx * dxx + dyy * dyy <= radius * radius
        chans = slice(None) if channel is None else slice(channel, channel + 1)
        before = float(state.cells.sum())
        cells = to_host(state.cells).copy()
        if kind == "paint":
            value = float(payload.get("value", 1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError("value must be within 0..1")
            cells[mask, chans] = value
        elif kind == "erase":
            cells[mask, chans] = 0.0
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        state.cells = to_backend(cells, self.backend)
        state.reset_baselines()
        return state, {
            "cells": int(mask.sum()),
            "mass_before": before,
            "mass_after": float(state.cells.sum()),
        }

    def inspect(
        self, state: FlowLeniaState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _exp(spec)
        h, w, c = state.cells.shape
        x, y = int(query["x"]), int(query["y"])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        u = core.affinity(state.cells, state.kfft, state.rule, c)
        f, alpha = core.flow(state.cells, u, _theta(exp), exp.dynamics.n)
        d = exp.dynamics
        mu = core.target_positions(state.cells.shape, f, d.dt, d.dd, d.sigma, exp.world.boundary)
        u, f, alpha, mu = to_host(u), to_host(f), to_host(alpha), to_host(mu)
        cells = to_host(state.cells)
        r = max(1, min(int(query.get("radius", 3)), 32))
        ys = [(y + k) % h for k in range(-r, r + 1)]
        xs = [(x + k) % w for k in range(-r, r + 1)]
        return {
            "substrate": "flow_lenia",
            "tick": state.tick,
            "query": {"x": x, "y": y, "radius": r},
            "channels": [
                {
                    "value": float(cells[y, x, ch]),
                    "affinity": float(u[y, x, ch]),
                    "alpha": float(alpha[y, x, 0, ch]),
                    "flow": {"x": float(f[y, x, 1, ch]), "y": float(f[y, x, 0, ch])},
                    "displacement": {
                        "x": float(mu[y, x, 1, ch] - (x + 0.5)),
                        "y": float(mu[y, x, 0, ch] - (y + 0.5)),
                    },
                }
                for ch in range(c)
            ],
            "patch_mass": float(cells[np.ix_(ys, xs)].sum()),
            "rule": state.rule.to_json(),
        }

    def state_distance(self, a: FlowLeniaState, b: FlowLeniaState) -> dict[str, Any]:
        if a.cells.shape != b.cells.shape:
            return {"metric": "max_abs_difference", "value": None, "exact_equal": False}
        ca, cb = to_host(a.cells), to_host(b.cells)
        diff = np.abs(ca - cb)
        return {
            "metric": "max_abs_difference",
            "value": float(diff.max()),
            "exact_equal": bool(np.array_equal(ca, cb)),
            "mean_abs_difference": float(diff.mean()),
        }

    def serialize_state(self, state: FlowLeniaState) -> dict[str, Any]:
        return {
            "arrays": {"cells": to_host(state.cells)},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "dtype": "float64",
                "mass_baseline": state.mass_baseline,
                "tracker": {
                    "last": list(state.last_centroid) if state.last_centroid else None,
                    "dx": state.dx,
                    "dy": state.dy,
                },
                "rule": state.rule.to_json(),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> FlowLeniaState:
        exp = _exp(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible flow_lenia checkpoint serialize_version")
        cells = np.asarray(payload["arrays"]["cells"], dtype=np.float64).copy()
        expected = (exp.world.height, exp.world.width, exp.dynamics.channels)
        if cells.shape != expected:
            raise ValueError("flow_lenia checkpoint shape does not match the spec")
        if not np.isfinite(cells).all() or cells.min() < 0:
            raise ValueError("flow_lenia checkpoint values must be finite and non-negative")
        state = self._new_state(exp, cells, int(meta["tick"]))
        if "rule" in meta and meta["rule"] != state.rule.to_json():
            raise ValueError("checkpoint rule differs from the rule realised from the spec")
        tr = meta.get("tracker") or {}
        last = tr.get("last")
        state.mass_baseline = float(meta.get("mass_baseline", cells.sum()))
        state.last_centroid = (float(last[0]), float(last[1])) if last else None
        state.dx, state.dy = float(tr.get("dx", 0.0)), float(tr.get("dy", 0.0))
        return state
