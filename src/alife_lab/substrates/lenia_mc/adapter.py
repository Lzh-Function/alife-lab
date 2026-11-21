"""Multichannel Lenia adapter (CPU, NumPy rFFT, float64). See ``model.py`` for the rule and its
provenance; ``docs/reference_fidelity/lenia_mc.md`` for the comparison with SakanaAI/asal."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, LeniaMCExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.lenia.lenia import _circular_centroid
from alife_lab.substrates.lenia_mc import model

F64 = npt.NDArray[np.float64]
SERIALIZE_VERSION = 1
ALIVE_THRESHOLD = 0.1  # on the channel sum (Leniabreeder's emptiness test uses 0.1)


def _mc(spec: ExperimentSpec) -> LeniaMCExperiment:
    exp = spec.experiment
    if not isinstance(exp, LeniaMCExperiment):
        raise TypeError("lenia_mc adapter received a different spec")
    return exp


@dataclass
class LeniaMCState:
    cells: F64  # (C, H, W) float64 in [0, 1]
    tick: int
    kfft: Any = field(repr=False)
    last_centroid: tuple[float, float] | None = None
    dx: float = 0.0
    dy: float = 0.0

    def track(self) -> None:
        current = _circular_centroid(self.cells.sum(axis=0))
        if current is not None and self.last_centroid is not None:
            h, w = self.cells.shape[1:]
            self.dx += (current[0] - self.last_centroid[0] + w / 2) % w - w / 2
            self.dy += (current[1] - self.last_centroid[1] + h / 2) % h - h / 2
        self.last_centroid = current


_M = MetricDefinition
_METRICS = (
    _M("mass", "metric.mass", "sum of states", "tick", "Sum over all cells and channels.",
       "none", "every tick", 1, "never null"),
    _M("alive_fraction", "metric.alive_fraction", "fraction", "tick",
       f"Fraction of cells whose channel sum exceeds {ALIVE_THRESHOLD}.", "none", "every tick", 1,
       "never null"),
    _M("max_value", "metric.max_value", "state", "tick", "Maximum value over cells and channels.",
       "none", "every tick", 1, "never null"),
    _M("centroid_displacement", "metric.centroid_displacement", "cells", "tick",
       "Norm of the unwrapped displacement of the circular centroid of the channel sum since "
       "tick 0 (reset by interventions). Not a creature tracker.", "none", "every tick", 1,
       "null when the world is empty"),
)  # fmt: skip

LENIA_MC_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.lenia_mc",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="lenia_mc",
    display_name_key="substrate.lenia_mc",
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
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Multichannel Lenia (Chan 2020, 'Lenia and Expanded Universe') with the "
        "update of SakanaAI/asal lenia_impl.py (from maxencefaldor/Leniabreeder); creatures "
        "5N7KKM / VT049W by Bert Chan.",
        "paper_urls": ["https://arxiv.org/abs/2005.03742", "https://arxiv.org/abs/2412.17799"],
        "implementation": "https://github.com/SakanaAI/asal",
        "implementation_commit": "677ba0ea4d3b3ca78273c9906c6e84d2b1481ce7",
    },
)


class LeniaMCAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in LENIA_MC_MANIFEST.backends:
            raise ValueError(f"unsupported lenia_mc backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return LENIA_MC_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> LeniaMCState:
        exp = _mc(spec)
        st = LeniaMCState(cells=model.initial_cells(exp), tick=0, kfft=model.kernels_fft(exp))
        st.track()
        return st

    def step(self, state: LeniaMCState, spec: ExperimentSpec) -> LeniaMCState:
        state.cells = model.step(state.cells, state.kfft, _mc(spec))
        state.tick += 1
        state.track()
        return state

    def metrics(self, state: LeniaMCState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_mc(spec).observations.metrics)
        total = state.cells.sum(axis=0)
        out = {
            "mass": float(state.cells.sum()),
            "alive_fraction": float((total > ALIVE_THRESHOLD).mean()),
            "max_value": float(state.cells.max()),
            "centroid_displacement": (
                math.hypot(state.dx, state.dy) if state.last_centroid is not None else math.nan
            ),
        }
        return {k: v for k, v in out.items() if k in wanted}

    @staticmethod
    def rgb(cells: F64) -> F64:
        """(H, W, 3) in [0, 1]: channels 0, 1, 2 as red, green, blue (missing ones black)."""
        c, h, w = cells.shape
        out = np.zeros((h, w, 3))
        out[..., : min(c, 3)] = np.moveaxis(cells[:3], 0, -1)
        return np.clip(out, 0.0, 1.0)

    def render(self, state: LeniaMCState) -> tuple[dict[str, Any], bytes]:
        img = self.rgb(state.cells)
        h, w = img.shape[:2]
        data = (img * 255).round().astype(np.uint8).tobytes()
        return {"codec": "field_u8", "width": w, "height": h, "channels": 3,
                "scale": "channels 0/1/2 as R/G/B, linear 0..1"}, data  # fmt: skip

    def apply_intervention(
        self, state: LeniaMCState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[LeniaMCState, dict[str, Any]]:
        c, h, w = state.cells.shape
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 4))
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
        dyy = (yy - y + h // 2) % h - h // 2
        dxx = (xx - x + w // 2) % w - w // 2
        mask = dxx * dxx + dyy * dyy <= radius * radius
        chans = slice(None) if channel is None else slice(channel, channel + 1)
        before = float(state.cells.sum())
        cells = state.cells.copy()
        if kind == "paint":
            value = float(payload.get("value", 1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError("value must be within 0..1")
            cells[chans, mask] = value
        elif kind == "erase":
            cells[chans, mask] = 0.0
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        state.cells = cells
        state.last_centroid, state.dx, state.dy = None, 0.0, 0.0
        state.track()
        return state, {"cells": int(mask.sum()), "mass_before": before,
                       "mass_after": float(state.cells.sum())}  # fmt: skip

    def inspect(
        self, state: LeniaMCState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _mc(spec)
        h, w = state.cells.shape[1:]
        x, y = int(query["x"]), int(query["y"])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        u = model.potentials(state.cells, state.kfft, exp)
        g = model.growth(u, exp)
        nxt = model.step(state.cells, state.kfft, exp)
        return {
            "substrate": "lenia_mc",
            "tick": state.tick,
            "query": {"x": x, "y": y},
            "values": [float(v) for v in state.cells[:, y, x]],
            "next_values": [float(v) for v in nxt[:, y, x]],
            "kernels": [
                {"index": i, "c0": k.c0, "c1": k.c1, "potential": float(u[i, y, x]),
                 "growth": float(g[i, y, x]), "m": k.m, "s": k.s, "h": k.h}
                for i, k in enumerate(exp.dynamics.kernels)
            ],
        }  # fmt: skip

    def state_distance(self, a: LeniaMCState, b: LeniaMCState) -> dict[str, Any]:
        if a.cells.shape != b.cells.shape:
            return {"metric": "max_abs_difference", "value": None, "exact_equal": False}
        diff = np.abs(a.cells - b.cells)
        return {"metric": "max_abs_difference", "value": float(diff.max()),
                "exact_equal": bool(np.array_equal(a.cells, b.cells)),
                "mean_abs_difference": float(diff.mean())}  # fmt: skip

    def serialize_state(self, state: LeniaMCState) -> dict[str, Any]:
        return {
            "arrays": {"cells": state.cells},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "dtype": "float64",
                "tracker": {"last": list(state.last_centroid) if state.last_centroid else None,
                            "dx": state.dx, "dy": state.dy},
            },
        }  # fmt: skip

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> LeniaMCState:
        exp = _mc(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible lenia_mc checkpoint serialize_version")
        cells = np.asarray(payload["arrays"]["cells"], dtype=np.float64).copy()
        n = exp.world.size
        if cells.shape != (exp.dynamics.channels, n, n):
            raise ValueError("lenia_mc checkpoint shape does not match the spec")
        if not np.isfinite(cells).all() or cells.min() < 0 or cells.max() > 1:
            raise ValueError("lenia_mc checkpoint values must be finite and within [0, 1]")
        tr = meta.get("tracker") or {}
        last = tr.get("last")
        return LeniaMCState(
            cells=cells, tick=int(meta["tick"]), kfft=model.kernels_fft(exp),
            last_centroid=(float(last[0]), float(last[1])) if last else None,
            dx=float(tr.get("dx", 0.0)), dy=float(tr.get("dy", 0.0)),
        )  # fmt: skip
