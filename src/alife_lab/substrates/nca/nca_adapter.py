"""``nca``: run a Growing NCA with fixed weights (one tick = one CA step).

The weights are part of the scientific spec (base64 + sha256), so the rule is hashed with the
experiment. Controls: ``untrained`` (fresh initialisation: zero output layer, nothing grows) and
``noise`` (random output layer). Interventions: ``damage`` (erase a disc) and ``seed``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, NCAExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.nca.render import cell_info, rgb_bytes
from alife_lab.substrates.nca.targets import padded_target
from alife_lab.substrates.nca.train_adapter import JAX_AVAILABLE, seed_state

SERIALIZE_VERSION = 1


def _nca(spec: ExperimentSpec) -> NCAExperiment:
    exp = spec.experiment
    if not isinstance(exp, NCAExperiment):
        raise TypeError("nca adapter received a different spec")
    return exp


def _model() -> Any:
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed (uv sync --extra nca)")
    from alife_lab.substrates.nca import model

    return model


def _params(exp: NCAExperiment) -> dict[str, Any]:
    m = _model()
    import jax

    d = exp.dynamics
    a = d.architecture
    if d.weights_kind == "trained":
        assert d.weights is not None
        return dict(m.unpack_params(base64.b64decode(d.weights.data_b64), a.channels, a.hidden))
    key = jax.random.PRNGKey((exp.seed + 1) % (2**32))
    k1, k2 = jax.random.split(key)
    params = dict(m.init_params(k1, a.channels, a.hidden))
    if d.weights_kind == "noise":
        params["w2"] = 0.1 * jax.random.normal(k2, params["w2"].shape)
    return params


@dataclass
class NCAState:
    x: np.ndarray[Any, Any]  # (H, W, C) float32
    tick: int
    key: np.ndarray[Any, Any] = field(repr=False)
    params: dict[str, Any] = field(default_factory=dict, repr=False)
    step_fn: Any = field(default=None, repr=False)
    target: np.ndarray[Any, Any] | None = field(default=None, repr=False)


_M = MetricDefinition
_METRICS = (
    _M("alive_fraction", "metric.alive_fraction", "fraction", "tick",
       "Fraction of cells inside the alive mask (3x3 max alpha > threshold).", "none",
       "every tick", 1, "never null"),
    _M("alpha_mass", "metric.alpha_mass", "sum of alpha", "tick", "Sum of clipped alpha.",
       "none", "every tick", 1, "never null"),
    _M("target_mse", "metric.target_mse", "MSE", "tick",
       "RGBA MSE to the padded target (passive measurement; not used by the dynamics).",
       "none", "every tick", 1, "never null"),
)  # fmt: skip

NCA_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.nca",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="nca",
    display_name_key="substrate.nca",
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
        intervention_types=("damage", "seed"),
    ),
    backends=("jax",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Growing NCA dynamics with fixed weights (Mordvintsev et al. 2020).",
        "paper_urls": ["https://distill.pub/2020/growing-ca/"],
        "implementation": "this repository (JAX, CPU)",
    },
)


class NCAAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return NCA_MANIFEST

    def _prepare(self, exp: NCAExperiment, state: NCAState) -> NCAState:
        m = _model()
        a = exp.dynamics.architecture
        state.params = _params(exp)
        state.step_fn = m.make_ca_step(a.fire_rate, a.alive_threshold)
        state.target = padded_target(exp.target, exp.pad)
        return state

    def initialize(self, spec: ExperimentSpec) -> NCAState:
        exp = _nca(spec)
        _model()
        import jax

        side = exp.target.size + 2 * exp.pad
        x = seed_state(side, side, exp.dynamics.architecture.channels)
        key = np.asarray(jax.random.PRNGKey(exp.seed % (2**32)), dtype=np.uint32)
        return self._prepare(exp, NCAState(x=x, tick=0, key=key))

    def step(self, state: NCAState, spec: ExperimentSpec) -> NCAState:
        import jax
        import jax.numpy as jnp

        key, sub = jax.random.split(jnp.asarray(state.key))
        out = state.step_fn(state.params, jnp.asarray(state.x[None]), sub)
        state.x = np.array(out[0])  # writable copy (JAX buffers are read-only)
        state.key = np.asarray(key, dtype=np.uint32)
        state.tick += 1
        return state

    def metrics(self, state: NCAState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _nca(spec)
        wanted = set(exp.observations.metrics)
        x = state.x
        out: dict[str, float] = {}
        if "alive_fraction" in wanted:
            a = x[..., 3]
            p = np.pad(a, 1, constant_values=-np.inf)
            pooled = np.max(
                np.stack([p[i : i + a.shape[0], j : j + a.shape[1]] for i in range(3)
                          for j in range(3)]), axis=0,
            )  # fmt: skip
            out["alive_fraction"] = float(
                (pooled > exp.dynamics.architecture.alive_threshold).mean()
            )
        if "alpha_mass" in wanted:
            out["alpha_mass"] = float(np.clip(x[..., 3], 0.0, 1.0).sum())
        if "target_mse" in wanted and state.target is not None:
            out["target_mse"] = float(np.mean((x[..., :4] - state.target) ** 2))
        return out

    def render(self, state: NCAState) -> tuple[dict[str, Any], bytes]:
        return rgb_bytes(state.x)

    def apply_intervention(
        self, state: NCAState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[NCAState, dict[str, Any]]:
        h, w = state.x.shape[:2]
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 6))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs integer x, y (optional radius)") from exc
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        if not 0 <= radius <= 64:
            raise ValueError("radius must be within 0..64")
        if kind == "damage":
            yy, xx = np.mgrid[0:h, 0:w]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius * radius
            before = float(np.clip(state.x[..., 3], 0, 1).sum())
            state.x[mask] = 0.0
            after = float(np.clip(state.x[..., 3], 0, 1).sum())
            return state, {
                "cells": int(mask.sum()),
                "alpha_mass_before": before,
                "alpha_mass_after": after,
            }
        if kind == "seed":
            state.x[y, x, 3:] = 1.0
            state.x[y, x, :3] = 0.0
            return state, {"cells": 1}
        raise ValueError(f"unsupported intervention kind {kind!r}")

    def inspect(
        self, state: NCAState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        x, y = int(query["x"]), int(query["y"])
        info = cell_info(state.x, x, y)
        target = state.target[y, x].tolist() if state.target is not None else None
        return {"substrate": "nca", "tick": state.tick, "query": {"x": x, "y": y},
                "target_rgba": target, **info}  # fmt: skip

    def state_distance(self, a: NCAState, b: NCAState) -> dict[str, Any]:
        if a.x.shape != b.x.shape:
            return {"metric": "max_abs_difference", "value": None, "exact_equal": False}
        return {
            "metric": "max_abs_difference",
            "value": float(np.abs(a.x - b.x).max()),
            "exact_equal": bool(np.array_equal(a.x, b.x)),
        }

    def serialize_state(self, state: NCAState) -> dict[str, Any]:
        return {
            "arrays": {"cells": state.x, "key": state.key},
            "meta": {"serialize_version": SERIALIZE_VERSION, "tick": state.tick},
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> NCAState:
        exp = _nca(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible nca checkpoint serialize_version")
        x = np.asarray(payload["arrays"]["cells"], dtype=np.float32).copy()
        side = exp.target.size + 2 * exp.pad
        if x.shape != (side, side, exp.dynamics.architecture.channels):
            raise ValueError("nca checkpoint shape does not match the spec")
        if not np.isfinite(x).all():
            raise ValueError("nca checkpoint values must be finite")
        key = np.asarray(payload["arrays"]["key"], dtype=np.uint32).copy()
        return self._prepare(exp, NCAState(x=x, tick=int(meta["tick"]), key=key))
