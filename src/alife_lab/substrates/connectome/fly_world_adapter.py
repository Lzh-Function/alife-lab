"""Phase F12: ``fly_world`` Substrate Adapter (Autonomous Artificial Life From Fly Connectome).

Streams compact ``fly_world_v1`` binary frames (``[N_alive x 12 float32]``: pose, sex, energy,
fly_distance, body_scale, leg_length, wing_span, sig0, sig1, feeding) at 60 FPS to the browser
3D evolutionary world renderer, with live interventions (toggle signal channel, trigger looming
predator shadow, drop food patch, freeze/resume mutations) and exact checkpoint/restore.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.connectome import ConnectomePack, load_connectome_pack
from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
from alife_lab.connectome.fly_world import (
    FlyWorldState,
    _mutual_information_bits,
    _spawn_food_patch,
    build_signal_dictionary,
    compute_fly_distance,
    decompose_fly_distance,
    make_fly_world,
    step_fly_world_tick,
)
from alife_lab.core.spec import ExperimentSpec, FlyWorldExperiment
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
_PACK_CACHE: dict[tuple[str, str], ConnectomePack] = {}


def _fw(spec: ExperimentSpec) -> FlyWorldExperiment:
    exp = spec.experiment
    if not isinstance(exp, FlyWorldExperiment):
        raise TypeError("fly_world adapter received a different spec")
    return exp


def _get_pack(dataset_id: str) -> ConnectomePack:
    ds_dir = load_settings().resolved_datasets_dir
    key = (dataset_id, str(ds_dir))
    cached = _PACK_CACHE.get(key)
    if cached is not None:
        return cached
    if dataset_id == "synthetic_fly_embodied":
        pack = make_synthetic_fly_embodied_pack()
    else:
        pack = load_connectome_pack(dataset_id, ds_dir, verify_hashes=True)
    _PACK_CACHE[key] = pack
    return pack


_M = MetricDefinition
_METRICS = (
    _M("population", "metric.population", "count", "tick",
       "Current number of living flies in the evolutionary world.", "last",
       "every tick", 1, "never null"),
    _M("mean_generation", "metric.mean_generation", "gen", "tick",
       "Mean generation index across living descendants.", "mean",
       "every tick", 1, "never null"),
    _M("fly_distance", "metric.fly_distance", "L2", "tick",
       "Mean quantitative genetic distance from the ancestral biological fly (0 = exact ancestor).",
       "mean", "every tick", 1, "never null"),
    _M("signal_mi_bits", "metric.signal_mi_bits", "bits", "tick",
       "Maximum empirical Mutual Information I(Signal; Context) across evolved channels.",
       "last", "every tick", 1, "never null"),
    _M("total_births", "metric.total_births", "count", "tick",
       "Cumulative autonomous sexual births.", "last",
       "every tick", 1, "never null"),
    _M("total_deaths", "metric.total_deaths", "count", "tick",
       "Cumulative starvation, predation, and senescence deaths.", "last",
       "every tick", 1, "never null"),
)  # fmt: skip

FLY_WORLD_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.fly_world",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="fly_world",
    display_name_key="substrate.fly_world",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="particles",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=True,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("toggle_signals", "toggle_mutations", "spawn_predator", "add_food"),
    ),
    backends=("numpy", "cupy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": (
            "Phase F12 open-ended artificial life starting from the adult Drosophila connectome "
            "(shared ancestral brain + per-individual differential genome, morphology, and "
            "evolvable communication signals; ZERO external fitness score)."
        ),
        "paper_urls": ["https://www.nature.com/articles/s41586-024-07763-9"],
    },
)


class FlyWorldAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in FLY_WORLD_MANIFEST.backends:
            raise ValueError(f"unsupported fly_world backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return FLY_WORLD_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> FlyWorldState:
        exp = _fw(spec)
        pack = _get_pack(exp.dataset_id)
        dyn = exp.dynamics
        return make_fly_world(
            pack,
            initial_population=exp.initial_population,
            max_population=exp.max_population,
            seed=int(exp.seed),
            backend=self.backend,
            arena_half_size=dyn.arena_half_size,
            signal_channel_enabled=dyn.signal_channel_enabled,
            predator_enabled=dyn.predator_enabled,
            predator_interval=dyn.predator_interval,
            signal_cost=dyn.signal_cost,
            mutation_rate=dyn.mutation_rate,
            mutation_sigma=dyn.mutation_sigma,
            food_patch_count=dyn.food_patch_count,
            kin_cluster_init=dyn.kin_cluster_init,
        )

    def step(self, state: FlyWorldState, spec: ExperimentSpec) -> FlyWorldState:
        step_fly_world_tick(state)
        return state

    def metrics(self, state: FlyWorldState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_fw(spec).observations.metrics)
        dist = decompose_fly_distance(state.genomes, state.alive)
        mi = max(
            _mutual_information_bits(state.mi_sig0_food),
            _mutual_information_bits(state.mi_sig1_pred),
        )
        out = {
            "population": float(state.population),
            "mean_generation": float(state.mean_generation),
            "fly_distance": float(dist["total"]),
            "signal_mi_bits": float(mi),
            "total_births": float(state.total_births),
            "total_deaths": float(state.total_deaths),
        }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: FlyWorldState) -> tuple[dict[str, Any], bytes]:
        """``fly_world_v1`` binary frame: ``[N_alive x 12 float32]``."""
        idx = np.flatnonzero(state.alive > 0)
        n_a = int(idx.size)
        buf = np.empty((n_a, 12), dtype=np.float32)
        if n_a > 0:
            g = state.genomes[idx]
            buf[:, 0] = state.x[idx].astype(np.float32)
            buf[:, 1] = state.y[idx].astype(np.float32)
            buf[:, 2] = state.yaw[idx].astype(np.float32)
            buf[:, 3] = state.speed[idx].astype(np.float32)
            buf[:, 4] = state.sex[idx].astype(np.float32)
            buf[:, 5] = state.energy[idx]
            buf[:, 6] = np.sqrt(np.mean(g * g, axis=1)).astype(np.float32)
            buf[:, 7] = np.exp(np.clip(g[:, 8], -0.45, 0.45)).astype(np.float32)
            buf[:, 8] = np.exp(np.clip(g[:, 9], -0.45, 0.45)).astype(np.float32)
            buf[:, 9] = np.exp(np.clip(g[:, 10], -0.45, 0.45)).astype(np.float32)
            buf[:, 10] = state.sig0_emit[idx]
            buf[:, 11] = state.sig1_emit[idx]
        dist_info = decompose_fly_distance(state.genomes, state.alive)
        sig_dict = build_signal_dictionary(state)
        header: dict[str, Any] = {
            "codec": "fly_world_v1",
            "tick": state.tick,
            "dataset_id": state.brain.dataset_id,
            "resolution": "cell_type_compressed",
            "source_cells": state.brain.source_cells,
            "compressed_nodes": state.brain.num_nodes,
            "num_flies": n_a,
            "max_population": state.max_population,
            "arena_half_size": state.arena_half_size,
            "signal_channel_enabled": state.signal_channel_enabled,
            "predator_enabled": state.predator_enabled,
            "mutations_enabled": state.mutation_rate > 0.0,
            "mean_generation": round(state.mean_generation, 2),
            "max_generation": state.max_generation,
            "total_births": state.total_births,
            "total_deaths": state.total_deaths,
            "total_predator_kills": state.total_predator_kills,
            "fly_distance": dist_info,
            "signal_dictionary": sig_dict,
            "foods": state.foods,
            "predator": {
                "active": state.predator_active,
                "x": round(state.predator_x, 3),
                "y": round(state.predator_y, 3),
                "r": round(state.predator_r, 3),
            },
            "feeding": [round(float(v), 2) for v in state.feeding[idx].tolist()] if n_a > 0 else [],
            "recent_births": state.recent_births[-8:],
        }
        return header, buf.tobytes()

    def apply_intervention(
        self,
        state: FlyWorldState,
        spec: ExperimentSpec,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[FlyWorldState, dict[str, Any]]:
        if kind == "toggle_signals":
            state.signal_channel_enabled = bool(
                payload.get("enabled", not state.signal_channel_enabled)
            )
            return state, {"signal_channel_enabled": state.signal_channel_enabled}
        if kind == "toggle_mutations":
            if state.mutation_rate > 0.0:
                state.mutation_rate = 0.0
            else:
                state.mutation_rate = _fw(spec).dynamics.mutation_rate or 0.14
            return state, {"mutation_rate": state.mutation_rate}
        if kind == "spawn_predator":
            alive_idx = np.flatnonzero(state.alive > 0)
            state.predator_active = True
            if alive_idx.size > 0:
                a = int(alive_idx[0])
                state.predator_x = float(state.x[a])
                state.predator_y = float(state.y[a])
            else:
                state.predator_x, state.predator_y = 0.0, 0.0
            state.predator_r = 1.75
            state.predator_timer = 65
            return state, {"predator_active": True}
        if kind == "add_food":
            if len(state.foods) < 8:
                state.foods.append(_spawn_food_patch(state.rng, state.arena_half_size))
            else:
                state.foods[0] = _spawn_food_patch(state.rng, state.arena_half_size)
            return state, {"food_count": len(state.foods)}
        raise ValueError(f"unsupported fly_world intervention {kind!r}")

    def inspect(
        self, state: FlyWorldState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        alive_idx = np.flatnonzero(state.alive > 0)
        if alive_idx.size == 0:
            return {"substrate": "fly_world", "population": 0}
        pos = max(0, min(int(alive_idx.size) - 1, int(query.get("fly_index", 0))))
        i = int(alive_idx[pos])
        g = state.genomes[i]
        return {
            "substrate": "fly_world",
            "tick": state.tick,
            "fly_id": int(state.fly_id[i]),
            "parents": [int(state.parent_a[i]), int(state.parent_b[i])],
            "generation": int(state.generation[i]),
            "age": int(state.age[i]),
            "sex": "male" if int(state.sex[i]) == 1 else "female",
            "energy": round(float(state.energy[i]), 3),
            "fly_distance": round(compute_fly_distance(g), 4),
            "morphology": {
                "body_scale": round(float(np.exp(np.clip(g[8], -0.45, 0.45))), 3),
                "leg_length": round(float(np.exp(np.clip(g[9], -0.45, 0.45))), 3),
                "wing_span": round(float(np.exp(np.clip(g[10], -0.45, 0.45))), 3),
                "antenna_length": round(float(np.exp(np.clip(g[11], -0.45, 0.45))), 3),
            },
            "signal_dictionary": build_signal_dictionary(state),
        }

    def serialize_state(self, state: FlyWorldState) -> dict[str, Any]:
        if state.brain.w_gpu is not None:
            import cupy as cp

            br_host = cp.asnumpy(state.brain_r)
        else:
            br_host = np.asarray(state.brain_r, dtype=np.float32).copy()
        return {
            "arrays": {
                "alive": state.alive.copy(),
                "fly_id": state.fly_id.copy(),
                "parent_a": state.parent_a.copy(),
                "parent_b": state.parent_b.copy(),
                "generation": state.generation.copy(),
                "age": state.age.copy(),
                "sex": state.sex.copy(),
                "x": state.x.copy(),
                "y": state.y.copy(),
                "yaw": state.yaw.copy(),
                "speed": state.speed.copy(),
                "energy": state.energy.copy(),
                "repro_cooldown": state.repro_cooldown.copy(),
                "genomes": state.genomes.copy(),
                "sig0_emit": state.sig0_emit.copy(),
                "sig1_emit": state.sig1_emit.copy(),
                "feeding": state.feeding.copy(),
                "brain_r": br_host,
                "mi_sig0_food": state.mi_sig0_food.copy(),
                "mi_sig1_pred": state.mi_sig1_pred.copy(),
            },
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": int(state.tick),
                "next_fly_id": int(state.next_fly_id),
                "signal_channel_enabled": bool(state.signal_channel_enabled),
                "predator_enabled": bool(state.predator_enabled),
                "mutation_rate": float(state.mutation_rate),
                "predator_active": bool(state.predator_active),
                "predator_x": float(state.predator_x),
                "predator_y": float(state.predator_y),
                "predator_r": float(state.predator_r),
                "predator_timer": int(state.predator_timer),
                "total_births": int(state.total_births),
                "total_deaths": int(state.total_deaths),
                "total_predator_kills": int(state.total_predator_kills),
                "total_food_eaten": float(state.total_food_eaten),
                "recv0_toward_sum": float(state.recv0_toward_sum),
                "recv0_count": int(state.recv0_count),
                "recv1_flee_sum": float(state.recv1_flee_sum),
                "recv1_count": int(state.recv1_count),
                "foods_json": json.dumps(state.foods),
                "recent_births_json": json.dumps(state.recent_births),
                "rng_state_json": json.dumps(state.rng.bit_generator.state),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> FlyWorldState:
        st = self.initialize(spec)
        arrs = payload["arrays"]
        meta = payload["meta"]
        st.alive[...] = arrs["alive"]
        st.fly_id[...] = arrs["fly_id"]
        st.parent_a[...] = arrs["parent_a"]
        st.parent_b[...] = arrs["parent_b"]
        st.generation[...] = arrs["generation"]
        st.age[...] = arrs["age"]
        st.sex[...] = arrs["sex"]
        st.x[...] = arrs["x"]
        st.y[...] = arrs["y"]
        st.yaw[...] = arrs["yaw"]
        st.speed[...] = arrs["speed"]
        st.energy[...] = arrs["energy"]
        st.repro_cooldown[...] = arrs["repro_cooldown"]
        st.genomes[...] = arrs["genomes"]
        st.sig0_emit[...] = arrs["sig0_emit"]
        st.sig1_emit[...] = arrs["sig1_emit"]
        st.feeding[...] = arrs["feeding"]
        st.mi_sig0_food[...] = arrs["mi_sig0_food"]
        st.mi_sig1_pred[...] = arrs["mi_sig1_pred"]
        if st.brain.w_gpu is not None:
            st.brain_r.set(np.ascontiguousarray(arrs["brain_r"], dtype=np.float32))
        else:
            st.brain_r[...] = arrs["brain_r"]
        st.tick = int(meta["tick"])
        st.next_fly_id = int(meta["next_fly_id"])
        st.signal_channel_enabled = bool(meta["signal_channel_enabled"])
        st.predator_enabled = bool(meta["predator_enabled"])
        st.mutation_rate = float(meta["mutation_rate"])
        st.predator_active = bool(meta["predator_active"])
        st.predator_x = float(meta["predator_x"])
        st.predator_y = float(meta["predator_y"])
        st.predator_r = float(meta["predator_r"])
        st.predator_timer = int(meta["predator_timer"])
        st.total_births = int(meta["total_births"])
        st.total_deaths = int(meta["total_deaths"])
        st.total_predator_kills = int(meta["total_predator_kills"])
        st.total_food_eaten = float(meta["total_food_eaten"])
        st.recv0_toward_sum = float(meta["recv0_toward_sum"])
        st.recv0_count = int(meta["recv0_count"])
        st.recv1_flee_sum = float(meta["recv1_flee_sum"])
        st.recv1_count = int(meta["recv1_count"])
        st.foods = json.loads(meta["foods_json"])
        st.recent_births = json.loads(meta["recent_births_json"])
        st.rng.bit_generator.state = json.loads(meta["rng_state_json"])
        return st

    def state_distance(self, a: FlyWorldState, b: FlyWorldState) -> dict[str, Any]:
        g_diff = float(np.max(np.abs(a.genomes - b.genomes)))
        p_diff = float(max(np.max(np.abs(a.x - b.x)), np.max(np.abs(a.y - b.y))))
        return {
            "exact_equal": bool(
                a.tick == b.tick
                and np.array_equal(a.alive, b.alive)
                and g_diff == 0.0
                and p_diff == 0.0
            ),
            "genome_max_abs_diff": g_diff,
            "pose_max_abs_diff": p_diff,
        }
