"""Phase F11 Stage ③: Collective Whole-Brain Fly Swarm & Dual-Sex Ecosystem Adapter (``fly_swarm``).

Supports all four F11 Stage ③ collective social modes inside ONE shared 3D arena:
- ``ramdya_odor_avoidance`` (24 × FlyWire = 3.33M neurons, Ramdya 2015 leg-touch escape)
- ``mixed_courtship_swarm`` (single-pack mixed-sex courtship pursuit & song)
- ``male_aggression_arena`` (12 × MaleCNS = 1.98M neurons, cVA + LC10a -> hg3/4 wing threat/lunge)
- ``dual_sex_ecosystem`` (8 × MaleCNS + 8 × BANC = 2.58M neurons: feeding + odor/touch avoidance +
  male-male territorial aggression + real male-to-female acoustic courtship all simultaneously!)
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.connectome import ConnectomePack, load_connectome_pack
from alife_lab.connectome.embodied import (
    make_synthetic_cns_embodied_pack,
    make_synthetic_fly_embodied_pack,
)
from alife_lab.connectome.fly_ecosystem import (
    ECOSYSTEM_ODOR_X_MAX,
    DualSexEcosystemWorld,
    make_dual_sex_ecosystem_world,
    make_synthetic_ecosystem_female_pack,
    make_synthetic_ecosystem_male_pack,
    step_dual_sex_ecosystem_tick,
)
from alife_lab.connectome.fly_swarm import (
    FlySwarmWorld,
    make_fly_swarm_world,
    make_synthetic_male_aggression_pack,
    step_fly_swarm_tick,
)
from alife_lab.core.spec import ExperimentSpec, FlySwarmExperiment
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
_PACK_CACHE: dict[tuple[str, str], ConnectomePack] = {}
SwarmState = FlySwarmWorld | DualSexEcosystemWorld


def _fs(spec: ExperimentSpec) -> FlySwarmExperiment:
    exp = spec.experiment
    if not isinstance(exp, FlySwarmExperiment):
        raise TypeError("fly_swarm adapter received a different spec")
    return exp


def _get_pack(dataset_id: str) -> ConnectomePack:
    ds_dir = load_settings().resolved_datasets_dir
    key = (dataset_id, str(ds_dir))
    cached = _PACK_CACHE.get(key)
    if cached is not None:
        return cached
    if dataset_id == "synthetic_fly_embodied":
        pack = make_synthetic_fly_embodied_pack()
    elif dataset_id == "synthetic_cns_embodied":
        pack = make_synthetic_cns_embodied_pack()
    elif dataset_id == "synthetic_male_aggression":
        pack = make_synthetic_male_aggression_pack()
    elif dataset_id == "synthetic_ecosystem_male":
        pack = make_synthetic_ecosystem_male_pack()
    elif dataset_id == "synthetic_ecosystem_female":
        pack = make_synthetic_ecosystem_female_pack()
    else:
        pack = load_connectome_pack(dataset_id, ds_dir, verify_hashes=True)
    _PACK_CACHE[key] = pack
    return pack


_M = MetricDefinition
_METRICS = (
    _M("clean_zone_fraction", "metric.clean_zone_fraction", "norm", "tick",
       "Fraction of flies currently in the clean-air zone.", "mean",
       "every tick", 1, "never null"),
    _M("encounter_rate", "metric.encounter_rate", "1/s", "tick",
       "Cumulative mechanosensory leg/body encounters per fly per simulated second.", "mean",
       "every tick", 1, "never null"),
    _M("mean_speed_cm_s", "metric.mean_speed_cm_s", "cm/s", "tick",
       "Mean walking speed across all flies in the swarm.", "mean",
       "every tick", 1, "never null"),
    _M("active_singers", "metric.active_singers", "count", "tick",
       "Number of courting males actively emitting wing song.", "mean",
       "every tick", 1, "never null"),
    _M("total_lunges", "metric.total_lunges", "count", "tick",
       "Cumulative male-male aggressive lunge clashes.", "last",
       "every tick", 1, "never null"),
    _M("active_threats", "metric.active_threats", "count", "tick",
       "Number of males displaying bilateral wing threat (hg3/hg4 MN).", "mean",
       "every tick", 1, "never null"),
    _M("active_feeders", "metric.active_feeders", "count", "tick",
       "Number of flies actively feeding via MN9 proboscis extension.", "mean",
       "every tick", 1, "never null"),
    _M("female_auditory_hz", "metric.female_auditory_hz", "Hz", "tick",
       "Mean female auditory relay firing rate (BANC SAD052/53 + CB1078 + AVLP082).", "mean",
       "every tick", 1, "never null"),
    _M("realtime_factor", "metric.realtime_factor", "x", "tick",
       "Measured simulation speed relative to 1 ms wall time (1.0x = real time).", "mean",
       "every tick", 1, "never null"),
)  # fmt: skip

FLY_SWARM_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.fly_swarm",
    plugin_version="0.2.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="fly_swarm",
    display_name_key="substrate.fly_swarm",
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
        intervention_types=("toggle_odor", "toggle_touch", "scatter", "wake_all"),
    ),
    backends=("numpy", "cupy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": (
            "F11 Stage 3 collective swarm & unified dual-sex whole-CNS ecosystem running N "
            "independent spiking Drosophila connectomes in ONE shared arena."
        ),
        "paper_urls": [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4359906/",
            "https://www.nature.com/articles/s41586-024-07763-9",
        ],
    },
)


def _upload_batch_gpu(batch: Any, arrs: dict[str, Any], prefix: str, substep: int) -> None:
    if batch.gpu_brain is not None:
        gb = batch.gpu_brain
        gb.d_v.set(np.ascontiguousarray(arrs[f"{prefix}v"], dtype=np.float32))
        gb.d_g.set(np.ascontiguousarray(arrs[f"{prefix}g"], dtype=np.float32))
        gb.d_refrac.set(np.ascontiguousarray(arrs[f"{prefix}refrac"], dtype=np.int16))
        gb.d_delay_ring.set(np.ascontiguousarray(arrs[f"{prefix}delay"], dtype=np.int32))
        gb.d_stim_rate_hz.set(np.ascontiguousarray(arrs[f"{prefix}stim"], dtype=np.float32))
        gb.d_silenced.set(np.ascontiguousarray(arrs[f"{prefix}sil"], dtype=np.uint8))
        gb.d_total_spikes.set(np.ascontiguousarray(arrs[f"{prefix}tot"], dtype=np.int32))
        gb.d_substep.set(np.array([substep], dtype=np.int64))
        gb.synchronize()
    else:
        for i, st in enumerate(batch.cpu_brains):
            st.v[...] = arrs[f"{prefix}v"][i]
            st.g[...] = arrs[f"{prefix}g"][i]
            st.refrac[...] = arrs[f"{prefix}refrac"][i]
            st.delay_ring[...] = arrs[f"{prefix}delay"][i]
            st.stim_rate_hz[...] = arrs[f"{prefix}stim"][i]
            st.silenced[...] = arrs[f"{prefix}sil"][i]
            st.total_spikes[...] = arrs[f"{prefix}tot"][i]
            st.substep = substep


class FlySwarmAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in FLY_SWARM_MANIFEST.backends:
            raise ValueError(f"unsupported fly_swarm backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return FLY_SWARM_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> SwarmState:
        exp = _fs(spec)
        dyn = exp.dynamics
        if dyn.social_mode == "dual_sex_ecosystem":
            male_pack = _get_pack(exp.dataset_id)
            fem_id = exp.female_dataset_id or (
                "synthetic_ecosystem_female"
                if exp.dataset_id.startswith("synthetic_")
                else "banc_888"
            )
            female_pack = _get_pack(fem_id)
            return make_dual_sex_ecosystem_world(
                male_pack,
                female_pack,
                num_flies=exp.num_flies,
                wiring=exp.wiring,
                seed=int(exp.seed),
                backend=self.backend,
                enable_touch=dyn.enable_touch,
                odor_enabled=dyn.odor_enabled,
                arena_half_size=dyn.arena_half_size,
                explore_drive_hz=dyn.explore_drive_hz,
            )
        pack = _get_pack(exp.dataset_id)
        return make_fly_swarm_world(
            pack,
            num_flies=exp.num_flies,
            wiring=exp.wiring,
            seed=int(exp.seed),
            backend=self.backend,
            social_mode=dyn.social_mode,
            enable_touch=dyn.enable_touch,
            odor_enabled=dyn.odor_enabled,
            arena_half_size=dyn.arena_half_size,
            explore_drive_hz=dyn.explore_drive_hz,
        )

    def step(self, state: SwarmState, spec: ExperimentSpec) -> SwarmState:
        if isinstance(state, DualSexEcosystemWorld):
            step_dual_sex_ecosystem_tick(state)
        else:
            step_fly_swarm_tick(state)
        return state

    def metrics(self, state: SwarmState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_fs(spec).observations.metrics)
        sim_s = max(1e-3, state.tick * 1e-3)
        if isinstance(state, DualSexEcosystemWorld):
            out = {
                "clean_zone_fraction": float(state.last_clean_fraction),
                "encounter_rate": float(state.total_encounters) / (state.num_flies * sim_s),
                "mean_speed_cm_s": float(state.last_mean_speed),
                "active_singers": float(state.last_active_singers),
                "total_lunges": float(state.total_lunges),
                "active_threats": float(state.last_active_threats),
                "active_feeders": float(state.last_active_feeders),
                "female_auditory_hz": float(state.last_female_auditory_hz),
                "realtime_factor": float(state.realtime_factor),
            }
        else:
            out = {
                "clean_zone_fraction": float(state.last_clean_fraction),
                "encounter_rate": float(state.total_encounters) / (state.num_flies * sim_s),
                "mean_speed_cm_s": float(state.last_mean_speed),
                "active_singers": float(state.last_active_singers),
                "total_lunges": float(state.total_lunges),
                "active_threats": float(state.last_active_threats),
                "active_feeders": 0.0,
                "female_auditory_hz": 0.0,
                "realtime_factor": float(state.realtime_factor),
            }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: SwarmState) -> tuple[dict[str, Any], bytes]:
        """``fly_swarm_v1`` binary frame: ``[N x 8 float32]`` per-fly pose/touch/song/threat."""
        n_f = state.num_flies
        buf = np.empty((n_f, 8), dtype=np.float32)
        buf[:, 0] = state.x.astype(np.float32)
        buf[:, 1] = state.y.astype(np.float32)
        buf[:, 2] = state.yaw.astype(np.float32)
        buf[:, 3] = state.speed.astype(np.float32)
        buf[:, 4] = state.sex.astype(np.float32)
        sim_s = max(1e-3, state.tick * 1e-3)

        if isinstance(state, DualSexEcosystemWorld):
            buf[:, 5] = (state.touch_timer_ms > 0).astype(np.float32)
            buf[:, 6] = state.song_intensity
            buf[:, 7] = state.threat_intensity
            header: dict[str, Any] = {
                "codec": "fly_swarm_v1",
                "tick": state.tick,
                "dataset_id": (
                    f"{state.male_batch.pack.dataset_id} + {state.female_batch.pack.dataset_id}"
                ),
                "wiring": state.wiring,
                "social_mode": state.social_mode,
                "num_flies": n_f,
                "n_males": state.n_males,
                "n_females": state.n_females,
                "cells_per_fly": state.male_batch.pack.cells.num_cells,
                "total_neurons": state.total_neurons,
                "arena_half_size": state.arena_half_size,
                "odor_enabled": state.odor_enabled,
                "odor_x_max": ECOSYSTEM_ODOR_X_MAX,
                "enable_touch": state.enable_touch,
                "clean_zone_fraction": round(state.last_clean_fraction, 3),
                "encounter_rate": round(state.total_encounters / (n_f * sim_s), 2),
                "total_encounters": int(state.total_encounters),
                "mean_speed": round(state.last_mean_speed, 2),
                "active_singers": int(state.last_active_singers),
                "total_lunges": int(state.total_lunges),
                "active_threats": int(state.last_active_threats),
                "active_feeders": int(state.last_active_feeders),
                "total_feeding_ms": int(state.total_feeding_ms),
                "female_auditory_hz": round(state.last_female_auditory_hz, 1),
                "mean_da1_pn_hz": round(float(state.male_batch.extra_rates[:, 0].mean()), 1),
                "mean_hg34_hz": round(float(state.male_batch.extra_rates[:, 3].mean()), 1),
                "realtime_factor": state.realtime_factor,
                "contacts": state.active_contacts,
                "clashes": state.active_clashes,
                "foods": list(state.foods),
                "feeding": [round(float(v), 2) for v in state.feeding_intensity.tolist()],
                "listening": [round(float(v), 2) for v in state.listening_intensity.tolist()],
            }
            return header, buf.tobytes()

        if state.social_mode == "male_aggression_arena":
            buf[:, 5] = (state.lunge_timer_ms > 0).astype(np.float32)
            buf[:, 6] = state.song_intensity
            buf[:, 7] = state.threat_intensity
        else:
            buf[:, 5] = (state.touch_timer_ms > 0).astype(np.float32)
            buf[:, 6] = state.song_intensity
            buf[:, 7] = state.brain_activity
        header = {
            "codec": "fly_swarm_v1",
            "tick": state.tick,
            "dataset_id": state.pack.dataset_id,
            "wiring": state.wiring,
            "social_mode": state.social_mode,
            "num_flies": n_f,
            "cells_per_fly": state.pack.cells.num_cells,
            "total_neurons": state.pack.cells.num_cells * n_f,
            "arena_half_size": state.arena_half_size,
            "odor_enabled": state.odor_enabled,
            "enable_touch": state.enable_touch,
            "clean_zone_fraction": round(state.last_clean_fraction, 3),
            "encounter_rate": round(state.total_encounters / (n_f * sim_s), 2),
            "total_encounters": int(state.total_encounters),
            "mean_speed": round(state.last_mean_speed, 2),
            "active_singers": int(state.last_active_singers),
            "total_lunges": int(state.total_lunges),
            "active_threats": int(state.last_active_threats),
            "mean_da1_pn_hz": round(float(state.agg_rates[:, 0].mean()), 1),
            "mean_hg34_hz": round(float(state.agg_rates[:, 3].mean()), 1),
            "realtime_factor": state.realtime_factor,
            "contacts": state.active_contacts,
            "clashes": state.active_clashes,
        }
        return header, buf.tobytes()

    def apply_intervention(
        self,
        state: SwarmState,
        spec: ExperimentSpec,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[SwarmState, dict[str, Any]]:
        if kind == "toggle_odor":
            state.odor_enabled = bool(payload.get("enabled", not state.odor_enabled))
            return state, {"odor_enabled": state.odor_enabled}
        if kind == "toggle_touch":
            state.enable_touch = bool(payload.get("enabled", not state.enable_touch))
            return state, {"enable_touch": state.enable_touch}
        if kind == "scatter":
            rng = state.rng_state_bitgen
            lim = state.arena_half_size * 0.78
            if isinstance(state, DualSexEcosystemWorld):
                state.x[...] = rng.uniform(-lim, lim, state.num_flies)
                state.y[...] = rng.uniform(-lim, lim, state.num_flies)
                state.yaw[...] = rng.uniform(-math.pi, math.pi, state.num_flies)
                state.retreat_timer_ms[...] = 0
                state.lunge_timer_ms[...] = 0
            elif state.social_mode == "male_aggression_arena":
                angs = np.linspace(0.0, 2.0 * math.pi, state.num_flies, endpoint=False)
                rads = rng.uniform(0.9, min(2.4, lim * 0.72), state.num_flies)
                state.x[...] = rads * np.cos(angs)
                state.y[...] = rads * np.sin(angs)
                state.yaw[...] = np.arctan2(-state.y, -state.x)
                state.retreat_timer_ms[...] = 0
                state.lunge_timer_ms[...] = 0
            else:
                state.x[...] = rng.uniform(
                    -lim, -0.35 if state.odor_enabled else lim, state.num_flies
                )
                state.y[...] = rng.uniform(-lim, lim, state.num_flies)
                state.yaw[...] = rng.uniform(-math.pi, math.pi, state.num_flies)
            return state, {"scattered": state.num_flies}
        if kind == "wake_all":
            state.active_state[...] = 1.0
            state.touch_timer_ms[...] = 250
            return state, {"woke": state.num_flies}
        raise ValueError(f"unsupported fly_swarm intervention {kind!r}")

    def inspect(
        self, state: SwarmState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        idx = max(0, min(state.num_flies - 1, int(query.get("fly_index", 0))))
        total_neurons = (
            state.total_neurons
            if isinstance(state, DualSexEcosystemWorld)
            else state.pack.cells.num_cells * state.num_flies
        )
        return {
            "substrate": "fly_swarm",
            "social_mode": state.social_mode,
            "tick": state.tick,
            "num_flies": state.num_flies,
            "total_neurons": total_neurons,
            "fly_index": idx,
            "pose": {
                "x": round(float(state.x[idx]), 3),
                "y": round(float(state.y[idx]), 3),
                "yaw": round(float(state.yaw[idx]), 3),
                "speed": round(float(state.speed[idx]), 3),
            },
            "sex": "male" if int(state.sex[idx]) == 1 else "female",
            "touch_timer_ms": int(state.touch_timer_ms[idx]),
            "song_intensity": round(float(state.song_intensity[idx]), 3),
            "threat_intensity": round(float(state.threat_intensity[idx]), 3),
            "clean_zone_fraction": round(state.last_clean_fraction, 3),
            "total_encounters": int(state.total_encounters),
            "total_lunges": int(state.total_lunges),
        }

    def serialize_state(self, state: SwarmState) -> dict[str, Any]:
        if isinstance(state, DualSexEcosystemWorld):
            m_brains, f_brains = state.ensure_cpu_brains()
            return {
                "arrays": {
                    "x": state.x.copy(),
                    "y": state.y.copy(),
                    "yaw": state.yaw.copy(),
                    "speed": state.speed.copy(),
                    "sex": state.sex.copy(),
                    "active_state": state.active_state.copy(),
                    "touch_timer_ms": state.touch_timer_ms.copy(),
                    "lunge_timer_ms": state.lunge_timer_ms.copy(),
                    "retreat_timer_ms": state.retreat_timer_ms.copy(),
                    "dominance_score": state.dominance_score.copy(),
                    "song_intensity": state.song_intensity.copy(),
                    "threat_intensity": state.threat_intensity.copy(),
                    "feeding_intensity": state.feeding_intensity.copy(),
                    "listening_intensity": state.listening_intensity.copy(),
                    "m_rates": state.male_batch.rates.copy(),
                    "m_extra": state.male_batch.extra_rates.copy(),
                    "m_prev": state.male_batch.prev_cmd_spikes.copy(),
                    "f_rates": state.female_batch.rates.copy(),
                    "f_extra": state.female_batch.extra_rates.copy(),
                    "f_prev": state.female_batch.prev_cmd_spikes.copy(),
                    "m_v": np.stack([b.v for b in m_brains], axis=0),
                    "m_g": np.stack([b.g for b in m_brains], axis=0),
                    "m_refrac": np.stack([b.refrac for b in m_brains], axis=0),
                    "m_delay": np.stack([b.delay_ring for b in m_brains], axis=0),
                    "m_stim": np.stack([b.stim_rate_hz for b in m_brains], axis=0),
                    "m_sil": np.stack([b.silenced for b in m_brains], axis=0),
                    "m_tot": np.stack([b.total_spikes for b in m_brains], axis=0),
                    "f_v": np.stack([b.v for b in f_brains], axis=0),
                    "f_g": np.stack([b.g for b in f_brains], axis=0),
                    "f_refrac": np.stack([b.refrac for b in f_brains], axis=0),
                    "f_delay": np.stack([b.delay_ring for b in f_brains], axis=0),
                    "f_stim": np.stack([b.stim_rate_hz for b in f_brains], axis=0),
                    "f_sil": np.stack([b.silenced for b in f_brains], axis=0),
                    "f_tot": np.stack([b.total_spikes for b in f_brains], axis=0),
                },
                "meta": {
                    "serialize_version": SERIALIZE_VERSION,
                    "is_dual_sex": True,
                    "tick": int(state.tick),
                    "substep": int(m_brains[0].substep if m_brains else 0),
                    "enable_touch": bool(state.enable_touch),
                    "odor_enabled": bool(state.odor_enabled),
                    "total_encounters": int(state.total_encounters),
                    "total_lunges": int(state.total_lunges),
                    "total_wing_threat_ms": int(state.total_wing_threat_ms),
                    "total_feeding_ms": int(state.total_feeding_ms),
                    "total_song_ms": int(state.total_song_ms),
                    "clean_zone_samples": int(state.clean_zone_samples),
                    "total_fly_samples": int(state.total_fly_samples),
                    "rng_state_json": json.dumps(state.rng_state_bitgen.bit_generator.state),
                },
            }

        brains = state.ensure_cpu_brains()
        return {
            "arrays": {
                "x": state.x.copy(),
                "y": state.y.copy(),
                "yaw": state.yaw.copy(),
                "speed": state.speed.copy(),
                "sex": state.sex.copy(),
                "active_state": state.active_state.copy(),
                "touch_timer_ms": state.touch_timer_ms.copy(),
                "song_intensity": state.song_intensity.copy(),
                "threat_intensity": state.threat_intensity.copy(),
                "lunge_timer_ms": state.lunge_timer_ms.copy(),
                "retreat_timer_ms": state.retreat_timer_ms.copy(),
                "dominance_score": state.dominance_score.copy(),
                "brain_activity": state.brain_activity.copy(),
                "rates": state.rates.copy(),
                "agg_rates": state.agg_rates.copy(),
                "prev_cmd_spikes": state.prev_cmd_spikes.copy(),
                "b_v": np.stack([b.v for b in brains], axis=0),
                "b_g": np.stack([b.g for b in brains], axis=0),
                "b_refrac": np.stack([b.refrac for b in brains], axis=0),
                "b_delay": np.stack([b.delay_ring for b in brains], axis=0),
                "b_stim": np.stack([b.stim_rate_hz for b in brains], axis=0),
                "b_sil": np.stack([b.silenced for b in brains], axis=0),
                "b_tot": np.stack([b.total_spikes for b in brains], axis=0),
            },
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "is_dual_sex": False,
                "tick": int(state.tick),
                "substep": int(brains[0].substep if brains else 0),
                "enable_touch": bool(state.enable_touch),
                "odor_enabled": bool(state.odor_enabled),
                "total_encounters": int(state.total_encounters),
                "total_lunges": int(state.total_lunges),
                "total_wing_threat_ms": int(state.total_wing_threat_ms),
                "clean_zone_samples": int(state.clean_zone_samples),
                "total_fly_samples": int(state.total_fly_samples),
                "rng_state_json": json.dumps(state.rng_state_bitgen.bit_generator.state),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> SwarmState:
        world = self.initialize(spec)
        arrs = payload["arrays"]
        meta = payload["meta"]
        substep = int(meta["substep"])

        world.x[...] = arrs["x"]
        world.y[...] = arrs["y"]
        world.yaw[...] = arrs["yaw"]
        world.speed[...] = arrs["speed"]
        world.sex[...] = arrs["sex"]
        world.active_state[...] = arrs["active_state"]
        world.touch_timer_ms[...] = arrs["touch_timer_ms"]
        world.song_intensity[...] = arrs["song_intensity"]
        world.tick = int(meta["tick"])
        world.enable_touch = bool(meta["enable_touch"])
        world.odor_enabled = bool(meta["odor_enabled"])
        world.total_encounters = int(meta["total_encounters"])
        world.total_lunges = int(meta.get("total_lunges", 0))
        world.total_wing_threat_ms = int(meta.get("total_wing_threat_ms", 0))
        world.clean_zone_samples = int(meta["clean_zone_samples"])
        world.total_fly_samples = int(meta["total_fly_samples"])
        world.rng_state_bitgen.bit_generator.state = json.loads(meta["rng_state_json"])

        if isinstance(world, DualSexEcosystemWorld):
            world.lunge_timer_ms[...] = arrs["lunge_timer_ms"]
            world.retreat_timer_ms[...] = arrs["retreat_timer_ms"]
            world.dominance_score[...] = arrs["dominance_score"]
            world.threat_intensity[...] = arrs["threat_intensity"]
            world.feeding_intensity[...] = arrs["feeding_intensity"]
            world.listening_intensity[...] = arrs["listening_intensity"]
            world.total_feeding_ms = int(meta.get("total_feeding_ms", 0))
            world.total_song_ms = int(meta.get("total_song_ms", 0))
            world.last_clean_fraction = (
                float(np.sum(world.x > ECOSYSTEM_ODOR_X_MAX)) / world.num_flies
            )
            world.male_batch.rates[...] = arrs["m_rates"]
            world.male_batch.extra_rates[...] = arrs["m_extra"]
            world.male_batch.prev_cmd_spikes[...] = arrs["m_prev"]
            world.female_batch.rates[...] = arrs["f_rates"]
            world.female_batch.extra_rates[...] = arrs["f_extra"]
            world.female_batch.prev_cmd_spikes[...] = arrs["f_prev"]
            _upload_batch_gpu(world.male_batch, arrs, "m_", substep)
            _upload_batch_gpu(world.female_batch, arrs, "f_", substep)
            for st in world.male_batch.cpu_brains:
                st.tick = world.tick
            for st in world.female_batch.cpu_brains:
                st.tick = world.tick
            return world

        if "threat_intensity" in arrs:
            world.threat_intensity[...] = arrs["threat_intensity"]
            world.lunge_timer_ms[...] = arrs["lunge_timer_ms"]
            world.retreat_timer_ms[...] = arrs["retreat_timer_ms"]
            world.dominance_score[...] = arrs["dominance_score"]
            world.agg_rates[...] = arrs["agg_rates"]
        world.brain_activity[...] = arrs["brain_activity"]
        world.rates[...] = arrs["rates"]
        world.prev_cmd_spikes[...] = arrs["prev_cmd_spikes"]
        world.last_clean_fraction = float(np.sum(world.x > 0.0)) / world.num_flies
        _upload_batch_gpu(world, arrs, "b_", substep)
        for st in world.cpu_brains:
            st.tick = world.tick
        return world

    def state_distance(self, a: SwarmState, b: SwarmState) -> dict[str, Any]:
        if isinstance(a, DualSexEcosystemWorld) and isinstance(b, DualSexEcosystemWorld):
            ma, fa = a.ensure_cpu_brains()
            mb, fb = b.ensure_cpu_brains()
            brain_eq = all(
                np.array_equal(x.v, y.v) and np.array_equal(x.total_spikes, y.total_spikes)
                for x, y in zip(ma + fa, mb + fb, strict=True)
            )
        elif isinstance(a, FlySwarmWorld) and isinstance(b, FlySwarmWorld):
            ba = a.ensure_cpu_brains()
            bb = b.ensure_cpu_brains()
            brain_eq = all(
                np.array_equal(x.v, y.v) and np.array_equal(x.total_spikes, y.total_spikes)
                for x, y in zip(ba, bb, strict=True)
            )
        else:
            brain_eq = False
        pos_diff = float(
            max(np.max(np.abs(a.x - b.x)), np.max(np.abs(a.y - b.y)), np.max(np.abs(a.yaw - b.yaw)))
        )
        return {
            "exact_equal": bool(a.tick == b.tick and brain_eq and pos_diff == 0.0),
            "brain_equal": brain_eq,
            "pose_max_abs_diff": pos_diff,
        }
