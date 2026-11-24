"""Stage 2 / S1 substrate adapter: C. elegans brain <-> RFT crawling body (``worm_embodied``)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.connectome.graded import GradedCircuit
from alife_lab.connectome.lif import LIFState
from alife_lab.connectome.schema import CELL_ROLE_NAMES
from alife_lab.connectome.worm_body import (
    N_JOINTS,
    WORM_FRAMING_NOTE,
    WormBodyState,
    WormEmbodiedState,
    WormFoodSpot,
    _local_nodes,
    make_worm_world,
    render_worm_frame,
    step_worm_tick,
)
from alife_lab.core.spec import ExperimentSpec, WormEmbodiedExperiment
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1

_M = MetricDefinition
_METRICS = (
    _M(
        "distance_mm",
        "metric.distance_mm",
        "mm",
        "tick",
        "Cumulative planar path length crawled by the C. elegans worm body.",
        "sum",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "forward_speed_mm_s",
        "metric.forward_speed_mm_s",
        "mm/s",
        "tick",
        "Instantaneous signed velocity projected onto the head tangent (+ forward, - reverse).",
        "mean",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "body_wave_power",
        "metric.body_wave_power",
        "rad",
        "tick",
        "RMS joint bend amplitude across the 11 interior body segments.",
        "mean",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "reversals",
        "metric.reversals",
        "count",
        "tick",
        "Cumulative forward-to-backward escape reversal transitions.",
        "sum",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "avb_mean_mv",
        "metric.avb_mean_mv",
        "mV",
        "tick",
        "Mean membrane potential of forward command interneurons AVBL/AVBR.",
        "mean",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "ava_mean_mv",
        "metric.ava_mean_mv",
        "mV",
        "tick",
        "Mean membrane potential of reverse command interneurons AVAL/AVAR.",
        "mean",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_v_mv",
        "metric.mean_v_mv",
        "mV",
        "tick",
        "Mean membrane potential across all 302 C. elegans neurons.",
        "mean",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "chemotaxis_index",
        "metric.chemotaxis_index",
        "norm",
        "tick",
        "Time-averaged proximity index toward attractant odor spots [-1, 1].",
        "mean",
        "every tick",
        1,
        "never null",
    ),
)

WORM_EMBODIED_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.worm_embodied",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="worm_embodied",
    display_name_key="substrate.worm_embodied",
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
        intervention_types=(
            "place_food",
            "place_repellent",
            "clear_food",
            "poke_nose",
            "poke_tail",
            "toggle_gap_junctions",
            "stimulate",
            "silence",
            "clear_stimuli",
        ),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": WORM_FRAMING_NOTE,
        "implementation": "this repository",
    },
)


def _we(spec: ExperimentSpec) -> WormEmbodiedExperiment:
    exp = spec.experiment
    if not isinstance(exp, WormEmbodiedExperiment):
        raise TypeError("worm_embodied adapter received a different spec")
    return exp


class WormEmbodiedAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return WORM_EMBODIED_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> WormEmbodiedState:
        exp = _we(spec)
        ds_dir = load_settings().resolved_datasets_dir
        foods = [WormFoodSpot(x=f.x, y=f.y, radius=f.radius, kind=f.kind) for f in exp.arena.foods]
        state = make_worm_world(
            dataset_id=exp.dataset_id,
            wiring=exp.wiring,
            seed=int(exp.seed),
            gap_junctions_enabled=exp.dynamics.gap_junctions_enabled,
            arena_half_mm=exp.arena.half_mm,
            explore_drive=exp.dynamics.explore_drive,
            foods=foods,
            datasets_dir=ds_dir,
        )
        for port_id in exp.silenced_ports:
            if port_id not in state.pack.io_ports:
                known = ", ".join(sorted(state.pack.io_ports))
                raise ValueError(f"unknown silenced port {port_id!r} (available: {known})")
            state.lif.silenced[state.pack.io_ports[port_id].cell_indices] = np.uint8(1)
        return state

    def step(self, state: WormEmbodiedState, spec: ExperimentSpec) -> WormEmbodiedState:
        step_worm_tick(state)
        return state

    def metrics(self, state: WormEmbodiedState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_we(spec).observations.metrics)
        v = state.lif.v
        avb_mv = (
            float(v[state.pools.avb].mean()) if state.pools.avb.size else state.params.e_leak_mv
        )
        ava_mv = (
            float(v[state.pools.ava].mean()) if state.pools.ava.size else state.params.e_leak_mv
        )
        out = {
            "distance_mm": float(state.distance_mm),
            "forward_speed_mm_s": float(state.body.v_forward),
            "body_wave_power": float(state.body_wave_power),
            "reversals": float(state.reversals),
            "avb_mean_mv": avb_mv,
            "ava_mean_mv": ava_mv,
            "mean_v_mv": float(v.mean()) if v.size else state.params.e_leak_mv,
            "chemotaxis_index": float(state.chemotaxis_index),
        }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: WormEmbodiedState) -> tuple[dict[str, Any], bytes]:
        return render_worm_frame(state)

    def apply_intervention(
        self, state: WormEmbodiedState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[WormEmbodiedState, dict[str, Any]]:
        if kind in ("place_food", "place_repellent"):
            fkind = (
                "repellent" if kind == "place_repellent" else str(payload.get("kind", "attractant"))
            )
            if fkind not in ("attractant", "repellent"):
                raise ValueError(f"worm food kind must be attractant or repellent, got {fkind!r}")
            lim = state.arena_half_mm - 0.5
            # WorldView onPoint passes pixel-space grid coords or arena coords; accept both safely.
            raw_x = payload.get("x")
            raw_y = payload.get("y")
            if raw_x is not None and raw_y is not None:
                px, py = float(raw_x), float(raw_y)
                if 0.0 <= px <= 320.0 and 0.0 <= py <= 240.0 and ("radius" in payload):
                    # Convert pixel click if mapped in positive quadrant vs centered arena
                    if abs(px) > lim or abs(py) > lim:
                        fx = (px / 320.0 * 2.0 - 1.0) * state.arena_half_mm
                        fy = (1.0 - min(py, 208.0) / 208.0 * 2.0) * state.arena_half_mm
                    else:
                        fx, fy = px, py
                else:
                    fx, fy = px, py
            else:
                fx = state.body.x + 1.8 * float(np.cos(state.body.heading))
                fy = state.body.y + 1.8 * float(np.sin(state.body.heading))
            fx = max(-lim, min(lim, fx))
            fy = max(-lim, min(lim, fy))
            spot = WormFoodSpot(x=fx, y=fy, radius=float(payload.get("radius", 1.5)), kind=fkind)  # type: ignore[arg-type]
            if len(state.foods) >= 12:
                state.foods.pop(0)
            state.foods.append(spot)
            return state, {"placed": spot.to_dict(), "food_count": len(state.foods)}

        if kind == "clear_food":
            state.foods.clear()
            return state, {"food_count": 0}

        if kind == "poke_nose":
            dur = max(5, min(200, int(payload.get("duration_ticks", 25))))
            state.poke_nose_ticks = max(state.poke_nose_ticks, dur)
            return state, {"poke_nose_ticks": state.poke_nose_ticks}

        if kind == "poke_tail":
            dur = max(5, min(200, int(payload.get("duration_ticks", 25))))
            state.poke_tail_ticks = max(state.poke_tail_ticks, dur)
            return state, {"poke_tail_ticks": state.poke_tail_ticks}

        if kind == "toggle_gap_junctions":
            if "enabled" in payload:
                state.gap_junctions_enabled = bool(payload["enabled"])
            else:
                state.gap_junctions_enabled = not state.gap_junctions_enabled
            return state, {"gap_junctions_enabled": state.gap_junctions_enabled}

        if kind == "stimulate":
            port = str(payload.get("port", ""))
            if port not in state.pack.io_ports:
                raise ValueError(f"unknown port {port!r}")
            rate = max(0.0, min(1000.0, float(payload.get("rate_hz", 150.0))))
            idx = state.pack.io_ports[port].cell_indices
            state.lif.stim_rate_hz[idx] = np.float32(rate)
            return state, {"port": port, "rate_hz": rate, "cells": int(idx.size)}

        if kind == "silence":
            port = str(payload.get("port", "forward_command"))
            if port not in state.pack.io_ports:
                raise ValueError(f"unknown port {port!r}")
            sil = bool(payload.get("silenced", True))
            idx = state.pack.io_ports[port].cell_indices
            state.lif.silenced[idx] = np.uint8(1 if sil else 0)
            return state, {"port": port, "silenced": sil, "cells": int(idx.size)}

        if kind == "clear_stimuli":
            state.lif.stim_rate_hz.fill(np.float32(0.0))
            state.lif.silenced.fill(np.uint8(0))
            state.poke_nose_ticks = 0
            state.poke_tail_ticks = 0
            return state, {"cleared": True}

        raise ValueError(f"unsupported worm_embodied intervention {kind!r}")

    def inspect(
        self, state: WormEmbodiedState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        b = state.body
        v = state.lif.v
        avb_mv = (
            float(v[state.pools.avb].mean()) if state.pools.avb.size else state.params.e_leak_mv
        )
        ava_mv = (
            float(v[state.pools.ava].mean()) if state.pools.ava.size else state.params.e_leak_mv
        )
        db_mv = float(v[state.pools.db].mean()) if state.pools.db.size else state.params.e_leak_mv
        vb_mv = float(v[state.pools.vb].mean()) if state.pools.vb.size else state.params.e_leak_mv
        da_mv = float(v[state.pools.da].mean()) if state.pools.da.size else state.params.e_leak_mv
        va_mv = float(v[state.pools.va].mean()) if state.pools.va.size else state.params.e_leak_mv
        roles = {
            name: int((state.pack.cells.role == i).sum()) for i, name in enumerate(CELL_ROLE_NAMES)
        }
        return {
            "substrate": "worm_embodied",
            "tick": state.tick,
            "dataset_id": state.pack.dataset_id,
            "wiring": state.wiring,
            "gap_junctions_enabled": state.gap_junctions_enabled,
            "worm_pose": {
                "x_mm": round(b.x, 4),
                "y_mm": round(b.y, 4),
                "heading_rad": round(b.heading, 4),
                "v_forward_mm_s": round(b.v_forward, 4),
                "v_lateral_mm_s": round(b.v_lateral, 4),
                "omega_rad_s": round(b.omega, 4),
            },
            "worm_pools_mv": {
                "avb": round(avb_mv, 2),
                "ava": round(ava_mv, 2),
                "db": round(db_mv, 2),
                "vb": round(vb_mv, 2),
                "da": round(da_mv, 2),
                "va": round(va_mv, 2),
            },
            "reversals": state.reversals,
            "distance_mm": round(state.distance_mm, 4),
            "chemotaxis_index": round(state.chemotaxis_index, 4),
            "roles": roles,
            "framing_note": WORM_FRAMING_NOTE,
        }

    def state_distance(self, a: WormEmbodiedState, b: WormEmbodiedState) -> dict[str, Any]:
        dx = math.hypot(a.body.x - b.body.x, a.body.y - b.body.y)
        dv = float(np.abs(a.lif.v - b.lif.v).max())
        exact = bool(
            a.tick == b.tick
            and a.gap_junctions_enabled == b.gap_junctions_enabled
            and a.body.x == b.body.x
            and a.body.y == b.body.y
            and a.body.heading == b.body.heading
            and np.array_equal(a.body.phi, b.body.phi)
            and np.array_equal(a.lif.v.view(np.uint32), b.lif.v.view(np.uint32))
            and np.array_equal(a.lif.g.view(np.uint32), b.lif.g.view(np.uint32))
            and np.array_equal(a.lif.silenced, b.lif.silenced)
        )
        return {
            "metric": "position_difference_mm",
            "value": float(dx),
            "max_v_diff_mv": dv,
            "exact_equal": exact,
        }

    def serialize_state(self, state: WormEmbodiedState) -> dict[str, Any]:
        lif = state.lif
        b = state.body
        return {
            "arrays": {
                "v": lif.v.copy(),
                "g": lif.g.copy(),
                "refrac": lif.refrac.copy(),
                "delay_ring": lif.delay_ring.copy(),
                "stim_rate_hz": lif.stim_rate_hz.copy(),
                "silenced": lif.silenced.copy(),
                "last_tick_spikes": lif.last_tick_spikes.copy(),
                "total_spikes": lif.total_spikes.copy(),
                "phi": b.phi.copy(),
                "d_act": b.d_act.copy(),
                "v_act": b.v_act.copy(),
                "nodes_xy": b.nodes_xy.copy(),
            },
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "dataset_id": state.pack.dataset_id,
                "pack_digest": state.pack.pack_digest,
                "substep": int(lif.substep),
                "tick": int(lif.tick),
                "gap_junctions_enabled": bool(state.gap_junctions_enabled),
                "x": float(b.x),
                "y": float(b.y),
                "heading": float(b.heading),
                "v_forward": float(b.v_forward),
                "v_lateral": float(b.v_lateral),
                "omega": float(b.omega),
                "prev_attract": float(b.prev_attract),
                "prev_repel": float(b.prev_repel),
                "poke_nose_ticks": int(state.poke_nose_ticks),
                "poke_tail_ticks": int(state.poke_tail_ticks),
                "reversals": int(state.reversals),
                "prev_direction": int(state.prev_direction),
                "distance_mm": float(state.distance_mm),
                "signed_forward_mm": float(state.signed_forward_mm),
                "body_wave_power": float(state.body_wave_power),
                "chemotaxis_index": float(state.chemotaxis_index),
                "attract_samples": int(state.attract_samples),
                "attract_sum": float(state.attract_sum),
                "foods": [f.to_dict() for f in state.foods],
                "trail_xy": [list(pt) for pt in state.trail_xy],
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> WormEmbodiedState:
        meta = payload["meta"]
        if meta.get("serialize_version") != SERIALIZE_VERSION:
            raise ValueError("incompatible worm_embodied checkpoint version")
        state = self.initialize(spec)
        if meta.get("pack_digest") and meta["pack_digest"] != state.pack.pack_digest:
            raise ValueError("worm_embodied checkpoint pack_digest mismatch")
        arr = payload["arrays"]
        state.lif = LIFState(
            v=np.ascontiguousarray(arr["v"], dtype=np.float32),
            g=np.ascontiguousarray(arr["g"], dtype=np.float32),
            refrac=np.ascontiguousarray(arr["refrac"], dtype=np.int16),
            delay_ring=np.ascontiguousarray(arr["delay_ring"], dtype=np.int32),
            substep=int(meta["substep"]),
            tick=int(meta["tick"]),
            stim_rate_hz=np.ascontiguousarray(arr["stim_rate_hz"], dtype=np.float32),
            silenced=np.ascontiguousarray(arr["silenced"], dtype=np.uint8),
            last_tick_spikes=np.ascontiguousarray(arr["last_tick_spikes"], dtype=np.int32),
            total_spikes=np.ascontiguousarray(arr["total_spikes"], dtype=np.int32),
        )
        phi = np.ascontiguousarray(arr["phi"], dtype=np.float64)
        nodes_xy = np.ascontiguousarray(arr["nodes_xy"], dtype=np.float64)
        if phi.shape != (N_JOINTS,):
            nodes_loc, _ = _local_nodes(phi, 1.0)
            nodes_xy = nodes_loc
        state.body = WormBodyState(
            x=float(meta["x"]),
            y=float(meta["y"]),
            heading=float(meta["heading"]),
            phi=phi,
            d_act=np.ascontiguousarray(arr["d_act"], dtype=np.float64),
            v_act=np.ascontiguousarray(arr["v_act"], dtype=np.float64),
            v_forward=float(meta["v_forward"]),
            v_lateral=float(meta["v_lateral"]),
            omega=float(meta["omega"]),
            prev_attract=float(meta["prev_attract"]),
            prev_repel=float(meta["prev_repel"]),
            nodes_xy=nodes_xy,
        )
        state.gap_junctions_enabled = bool(meta["gap_junctions_enabled"])
        state.poke_nose_ticks = int(meta["poke_nose_ticks"])
        state.poke_tail_ticks = int(meta["poke_tail_ticks"])
        state.reversals = int(meta["reversals"])
        state.prev_direction = int(meta["prev_direction"])
        state.distance_mm = float(meta["distance_mm"])
        state.signed_forward_mm = float(meta["signed_forward_mm"])
        state.body_wave_power = float(meta["body_wave_power"])
        state.chemotaxis_index = float(meta["chemotaxis_index"])
        state.attract_samples = int(meta["attract_samples"])
        state.attract_sum = float(meta["attract_sum"])
        state.foods = [
            WormFoodSpot(
                x=float(f["x"]),
                y=float(f["y"]),
                radius=float(f["radius"]),
                kind=str(f["kind"]),  # type: ignore[arg-type]
                strength=float(f.get("strength", 1.0)),
            )
            for f in meta.get("foods", [])
        ]
        state.trail_xy = [list(pt) for pt in meta.get("trail_xy", [])]
        state.circuit = GradedCircuit.from_csr(state.csr)
        return state
