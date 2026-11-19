"""Lenia adapter (CPU, NumPy FFT, float64).

Reference mode = the default update path of Chan's LeniaND.py (see ``reference.py``):
``A ← clip(A + G(K * A) / T, 0, 1)`` with a periodic FFT convolution. Metrics are passive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, LeniaExperiment
from alife_lab.substrates._xp import to_backend, to_host
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.lenia import reference as ref

F64 = npt.NDArray[np.float64]
SERIALIZE_VERSION = 1
ALIVE_THRESHOLD = 0.01


def _lenia(spec: ExperimentSpec) -> LeniaExperiment:
    exp = spec.experiment
    if not isinstance(exp, LeniaExperiment):
        raise TypeError("lenia adapter received a non-lenia spec")
    return exp


def _kfft(exp: LeniaExperiment) -> F64:
    k = exp.dynamics.kernel
    return ref.kernel_fft(exp.world.height, exp.world.width, k.R, list(k.b), k.core)


def _circular_centroid(cells: Any) -> tuple[float, float] | None:
    total = float(cells.sum())
    if total <= 1e-12:
        return None
    h, w = cells.shape
    out = []
    for axis, size in ((0, w), (1, h)):  # x from columns, y from rows
        weights = to_host(cells.sum(axis=axis))  # marginals only leave the device
        angles = np.arange(size) * (2 * math.pi / size)
        s = float((weights * np.sin(angles)).sum())
        c = float((weights * np.cos(angles)).sum())
        if math.hypot(s, c) < 1e-12:
            return None
        out.append((math.atan2(s, c) * size / (2 * math.pi)) % size)
    return out[0], out[1]


@dataclass
class LeniaState:
    cells: Any  # float64 (H, W): numpy, or cupy for the GPU backend
    tick: int
    kfft: Any = field(repr=False)
    last_centroid: tuple[float, float] | None = None
    dx: float = 0.0
    dy: float = 0.0

    def track(self) -> None:
        current = _circular_centroid(self.cells)
        if current is not None and self.last_centroid is not None:
            h, w = self.cells.shape
            self.dx += (current[0] - self.last_centroid[0] + w / 2) % w - w / 2
            self.dy += (current[1] - self.last_centroid[1] + h / 2) % h - h / 2
        self.last_centroid = current


_M = MetricDefinition
_METRICS = (
    _M(
        "mass",
        "metric.mass",
        "sum of states",
        "tick",
        "Sum of all cell values.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "alive_fraction",
        "metric.alive_fraction",
        "fraction",
        "tick",
        f"Fraction of cells with value > {ALIVE_THRESHOLD}.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "max_value",
        "metric.max_value",
        "state",
        "tick",
        "Maximum cell value.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "centroid_displacement",
        "metric.centroid_displacement",
        "cells",
        "tick",
        "Norm of the unwrapped displacement of the mass-weighted circular centroid since tick 0 "
        "(reset by interventions). Not a creature tracker.",
        "none",
        "every tick",
        1,
        "null when the world is empty",
    ),
)

LENIA_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.lenia",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="lenia",
    display_name_key="substrate.lenia",
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
        "description": "Chan (2019) Lenia — Biology of Artificial Life; LeniaND.py (MIT).",
        "paper_urls": ["https://arxiv.org/abs/1812.05433"],
        "implementation": "https://github.com/Chakazul/Lenia",
        "implementation_commit": "adfc542939266de7f4bb7ebb552e8499701ee107",
    },
)


class LeniaAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        """``cupy`` keeps the field on the GPU (cuFFT, float64). Not bitwise equal to numpy:
        rounding differences grow with time (see docs/reference_fidelity/lenia.md)."""
        if backend not in LENIA_MANIFEST.backends:
            raise ValueError(f"unsupported lenia backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return LENIA_MANIFEST

    def _device(self, state: LeniaState) -> LeniaState:
        state.cells = to_backend(state.cells, self.backend)
        state.kfft = to_backend(state.kfft, self.backend)
        return state

    def initialize(self, spec: ExperimentSpec) -> LeniaState:
        exp = _lenia(spec)
        h, w = exp.world.height, exp.world.width
        cells = np.zeros((h, w), dtype=np.float64)
        init = exp.initialization
        if init.preset == "pattern":
            ref.place_centered(cells, ref.rle2arr(ref.ANIMALS[init.pattern]["cells"]))
        elif init.preset == "random":
            rng = np.random.Generator(np.random.PCG64(exp.seed))
            size = min(init.size, h, w)
            block = rng.random((size, size)) * (rng.random((size, size)) < init.density)
            ref.place_centered(cells, block)
        state = self._device(LeniaState(cells=cells, tick=0, kfft=_kfft(exp)))
        state.track()
        return state

    def step(self, state: LeniaState, spec: ExperimentSpec) -> LeniaState:
        exp = _lenia(spec)
        d = exp.dynamics
        state.cells = ref.calc_once(
            state.cells, state.kfft, d.T, d.growth.m, d.growth.s, d.growth.function
        )
        state.tick += 1
        state.track()
        return state

    def metrics(self, state: LeniaState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_lenia(spec).observations.metrics)
        out: dict[str, float] = {}
        if "mass" in wanted:
            out["mass"] = float(state.cells.sum())
        if "alive_fraction" in wanted:
            out["alive_fraction"] = float((state.cells > ALIVE_THRESHOLD).mean())
        if "max_value" in wanted:
            out["max_value"] = float(state.cells.max())
        if "centroid_displacement" in wanted:
            out["centroid_displacement"] = (
                math.hypot(state.dx, state.dy) if state.last_centroid is not None else math.nan
            )
        return out

    def render(self, state: LeniaState) -> tuple[dict[str, Any], bytes]:
        h, w = state.cells.shape
        data = to_host((state.cells * 255).round().astype(np.uint8)).tobytes()
        return {"codec": "field_u8", "width": w, "height": h, "scale": "linear 0..1"}, data

    def apply_intervention(
        self, state: LeniaState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[LeniaState, dict[str, Any]]:
        h, w = state.cells.shape
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 5))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs integer x, y (and optional radius)") from exc
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        if not 0 <= radius <= 128:
            raise ValueError("radius must be within 0..128")
        yy, xx = np.ogrid[0:h, 0:w]
        dyy = (yy - y + h // 2) % h - h // 2
        dxx = (xx - x + w // 2) % w - w // 2
        mask = dxx * dxx + dyy * dyy <= radius * radius
        before = float(state.cells.sum())
        cells = to_host(state.cells).copy()
        if kind == "paint":
            value = float(payload.get("value", 1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError("value must be within 0..1")
            cells[mask] = value
        elif kind == "erase":
            cells[mask] = 0.0
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        state.cells = to_backend(cells, self.backend)
        state.last_centroid, state.dx, state.dy = None, 0.0, 0.0
        state.track()
        return state, {
            "cells": int(mask.sum()),
            "mass_before": before,
            "mass_after": float(state.cells.sum()),
        }

    def inspect(
        self, state: LeniaState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _lenia(spec)
        h, w = state.cells.shape
        x, y = int(query["x"]), int(query["y"])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        g = exp.dynamics.growth
        u = ref.potential(state.cells, state.kfft)
        growth = ref.GROWTH[g.function](u, g.m, g.s)
        u, growth = to_host(u), to_host(growth)
        r = max(1, min(int(query.get("radius", 3)), 32))
        ys = [(y + d) % h for d in range(-r, r + 1)]
        xs = [(x + d) % w for d in range(-r, r + 1)]
        cells = to_host(state.cells)
        patch = cells[np.ix_(ys, xs)]
        return {
            "substrate": "lenia",
            "tick": state.tick,
            "query": {"x": x, "y": y, "radius": r},
            "value": float(cells[y, x]),
            "potential": float(u[y, x]),
            "growth": float(growth[y, x]),
            "next_value": float(np.clip(cells[y, x] + growth[y, x] / exp.dynamics.T, 0, 1)),
            "patch_mass": float(patch.sum()),
            "patch_max": float(patch.max()),
        }

    def state_distance(self, a: LeniaState, b: LeniaState) -> dict[str, Any]:
        ca, cb = to_host(a.cells), to_host(b.cells)
        if ca.shape != cb.shape:
            return {"metric": "max_abs_difference", "value": None, "exact_equal": False}
        diff = np.abs(ca - cb)
        return {
            "metric": "max_abs_difference",
            "value": float(diff.max()),
            "exact_equal": bool(np.array_equal(ca, cb)),
            "mean_abs_difference": float(diff.mean()),
        }

    def serialize_state(self, state: LeniaState) -> dict[str, Any]:
        return {
            "arrays": {"cells": to_host(state.cells)},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "dtype": "float64",
                "tracker": {
                    "last": list(state.last_centroid) if state.last_centroid else None,
                    "dx": state.dx,
                    "dy": state.dy,
                },
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> LeniaState:
        exp = _lenia(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible lenia checkpoint serialize_version")
        cells = np.asarray(payload["arrays"]["cells"], dtype=np.float64).copy()
        if cells.shape != (exp.world.height, exp.world.width):
            raise ValueError("lenia checkpoint shape does not match the spec")
        if not np.isfinite(cells).all() or cells.min() < 0 or cells.max() > 1:
            raise ValueError("lenia checkpoint values must be finite and within [0, 1]")
        tr = meta.get("tracker") or {}
        last = tr.get("last")
        return self._device(
            LeniaState(
                cells=cells,
                tick=int(meta["tick"]),
                kfft=_kfft(exp),
                last_centroid=(float(last[0]), float(last[1])) if last else None,
                dx=float(tr.get("dx", 0.0)),
                dy=float(tr.get("dy", 0.0)),
            )
        )
