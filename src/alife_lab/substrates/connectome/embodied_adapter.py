"""Phase F8 Level 1: Real-time embodied Drosophila substrate adapter (``fly_embodied``).

Couples a spiking LIF connectome brain (``flywire_783`` on GPU ``cupy`` CUDA graphs or CPU
``numpy``, or built-in offline ``synthetic_fly_embodied``) with a CPU float64 MuJoCo 6-legged fly
body and Level 1 hand-crafted tripod CPG + stumble reflex.

Renders a single ``fly_embodied_v1`` binary frame containing BOTH:
- The 3D MuJoCo body geometry transforms (float32, 12 floats per geom: pos[3] + rotmat[9]), AND
- The brain's fired neuron indices (int32) so the browser 2-pane UI displays the 3D arena and the
  3D glowing brain synchronized on the exact same 1 ms tick!
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.connectome import ConnectomePack, load_connectome_pack
from alife_lab.connectome.cns_packs import LEG_NAMES, MUSCLE_CHANNELS
from alife_lab.connectome.embodied import (
    EMBODIED_LEVEL1_FRAMING,
    EmbodiedFlyWorld,
    make_embodied_world,
    make_synthetic_cns_embodied_pack,
    make_synthetic_fly_embodied_pack,
    step_embodied_tick,
)
from alife_lab.connectome.fly_body import (
    FoodSpot,
    MotorCommands,
    fly_pose_xy_yaw,
    make_fly_body,
)
from alife_lab.core.spec import ExperimentSpec, FlyEmbodiedExperiment
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}

_PACK_CACHE: dict[tuple[str, str], ConnectomePack] = {}


def _fe(spec: ExperimentSpec) -> FlyEmbodiedExperiment:
    exp = spec.experiment
    if not isinstance(exp, FlyEmbodiedExperiment):
        raise TypeError("fly_embodied adapter received a different spec")
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
    else:
        pack = load_connectome_pack(dataset_id, ds_dir, verify_hashes=True)
    _PACK_CACHE[key] = pack
    return pack


_M = MetricDefinition
_METRICS = (
    _M("cmd_forward", "metric.cmd_forward", "norm", "tick",
       "Normalized forward walking command from DNp09 descending neurons [0, 1].", "mean",
       "every tick", 1, "never null"),
    _M("cmd_turn", "metric.cmd_turn", "norm", "tick",
       "Normalized steering command from bilateral DNa01/DNa02 neurons [-1, +1].", "mean",
       "every tick", 1, "never null"),
    _M("cmd_proboscis", "metric.cmd_proboscis", "norm", "tick",
       "Normalized proboscis extension command from MN9 (CB0701) motor neurons [0, 1].", "mean",
       "every tick", 1, "never null"),
    _M("cmd_grooming", "metric.cmd_grooming", "norm", "tick",
       "Normalized antennal grooming command from DNg84 descending neurons [0, 1].", "mean",
       "every tick", 1, "never null"),
    _M("distance_cm", "metric.distance_cm", "cm", "tick",
       "Cumulative planar distance walked by the fly thorax in the arena.", "none",
       "every tick", 1, "never null"),
    _M("feeding_ms", "metric.feeding_ms", "ms", "tick",
       "Cumulative simulated time spent in active proboscis extension / feeding.", "sum",
       "every tick", 1, "never null"),
    _M("grooming_ms", "metric.grooming_ms", "ms", "tick",
       "Cumulative simulated time spent in active front-leg antennal grooming.", "sum",
       "every tick", 1, "never null"),
    _M("realtime_factor", "metric.realtime_factor", "x", "tick",
       "Measured closed-loop simulation speed relative to 1 ms wall time (1.0x = real time).",
       "mean", "every tick", 1, "never null"),
    _M("vnc_mn_mean_hz", "metric.vnc_mn_mean_hz", "Hz", "tick",
       "Mean firing rate across all 36 biological VNC leg motor neuron pools (Level 2).",
       "mean", "every tick", 1, "never null"),
    _M("vnc_tripod_alternation", "metric.vnc_tripod_alternation", "norm", "tick",
       "Tripod A {LF,RM,LH} vs Tripod B {RF,LM,RH} motor neuron alternation index [0, 1].",
       "mean", "every tick", 1, "never null"),
)  # fmt: skip

FLY_EMBODIED_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.fly_embodied",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="fly_embodied",
    display_name_key="substrate.fly_embodied",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="rigid_body",
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
            "clear_food",
            "poke_antenna",
            "stimulate",
            "silence",
            "clear_stimuli",
        ),
    ),
    backends=("numpy", "cupy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": EMBODIED_LEVEL1_FRAMING,
        "paper_urls": [
            "https://www.nature.com/articles/s41586-024-07763-9",
            "https://www.nature.com/articles/s41592-024-02497-y",
        ],
        "implementation": "https://github.com/NeLy-EPFL/flygym",
    },
)


class FlyEmbodiedAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in FLY_EMBODIED_MANIFEST.backends:
            raise ValueError(f"unsupported fly_embodied backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return FLY_EMBODIED_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> EmbodiedFlyWorld:
        exp = _fe(spec)
        pack = _get_pack(exp.dataset_id)
        ds_dir = load_settings().resolved_datasets_dir
        foods = [FoodSpot(x=f.x, y=f.y, radius=f.radius, kind=f.kind) for f in exp.arena.foods]
        dyn = exp.dynamics
        world = make_embodied_world(
            pack,
            wiring=exp.wiring,
            seed=int(exp.seed),
            backend=self.backend,
            foods=foods,
            arena_half_size=exp.arena.half_size,
            explore_drive_hz=dyn.explore_drive_hz,
            control_mode=dyn.control_mode,
            vnc_synapse_scale=dyn.vnc_synapse_scale,
            vnc_proprio_gain_hz=dyn.vnc_proprio_gain_hz,
            vnc_mn_norm_hz=dyn.vnc_mn_norm_hz,
            vnc_coxa_gain=dyn.vnc_coxa_gain,
            vnc_femur_gain=dyn.vnc_femur_gain,
            vnc_tibia_gain=dyn.vnc_tibia_gain,
            datasets_dir=ds_dir,
        )
        if dyn.vnc_profile:
            from alife_lab.connectome.vnc_learn import (
                apply_vnc_profile_to_world,
                load_vnc_profile,
                resolve_vnc_profile_path,
            )

            prof = load_vnc_profile(resolve_vnc_profile_path(dyn.vnc_profile))
            apply_vnc_profile_to_world(world, prof, datasets_dir=ds_dir)
            world.vnc_profile_name = dyn.vnc_profile
        for port_id in exp.silenced_ports:
            if port_id in pack.io_ports:
                world.lif.silenced[pack.io_ports[port_id].cell_indices] = np.uint8(1)
        if world.gpu_runner is not None and exp.silenced_ports:
            world.gpu_runner.upload_controls(world.lif)
        return world

    def step(self, state: EmbodiedFlyWorld, spec: ExperimentSpec) -> EmbodiedFlyWorld:
        step_embodied_tick(state)
        return state

    def metrics(self, state: EmbodiedFlyWorld, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_fe(spec).observations.metrics)
        cmd = state.last_commands
        out = {
            "cmd_forward": float(cmd.forward),
            "cmd_turn": float(cmd.turn),
            "cmd_proboscis": float(cmd.proboscis),
            "cmd_grooming": float(cmd.grooming),
            "distance_cm": float(state.distance_traveled),
            "feeding_ms": float(state.total_feeding_ms),
            "grooming_ms": float(state.total_grooming_ms),
            "realtime_factor": float(state.timing.realtime_factor),
            "vnc_mn_mean_hz": float(state.vnc_mn_mean_hz),
            "vnc_tripod_alternation": float(state.last_vnc_tripod_alt),
        }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: EmbodiedFlyWorld) -> tuple[dict[str, Any], bytes]:
        """``fly_embodied_v1`` binary packet: [geom_bytes (ngeom*12*f32)][fired_bytes (k*i32)]."""
        m, d = state.body.model, state.body.data
        geoms = getattr(state.body, "_cached_geom_schema", None)
        if geoms is None:
            geoms = [
                {
                    "name": m.geom(g).name,
                    "type": GEOM_TYPES.get(int(m.geom_type[g]), "box"),
                    "size": [round(float(v), 5) for v in m.geom_size[g]],
                    "rgba": [round(float(v), 3) for v in m.geom_rgba[g]],
                }
                for g in range(m.ngeom)
            ]
            state.body._cached_geom_schema = geoms  # type: ignore[attr-defined]
        body_ids = getattr(state.body, "_cached_body_ids", None)
        if body_ids is None:
            body_ids = {bname: int(m.body(bname).id) for bname in ("thorax", "head", "proboscis")}
            state.body._cached_body_ids = body_ids  # type: ignore[attr-defined]
        # Exact unrotated MuJoCo body frames (+X anterior, +Y left, +Z dorsal) so CT-scanned
        # anatomical STL meshes (thorax+abdomen+wings, head+eyes+antennae, proboscis) sit cleanly
        # horizontal rather than inheriting the +90 deg capsule axis rotation of g_thorax!
        body_poses = {
            bname: {
                "pos": [round(float(v), 5) for v in d.xpos[bid]],
                "mat": [round(float(v), 5) for v in d.xmat[bid]],
            }
            for bname, bid in body_ids.items()
        }
        geom_bytes = np.concatenate([d.geom_xpos, d.geom_xmat], axis=1).astype(np.float32).tobytes()
        if state.gpu_runner is not None:
            fired = state.gpu_runner.read_fired_indices()
        else:
            fired = np.flatnonzero(state.lif.last_tick_spikes > 0).astype(np.int32)
        fired_bytes = fired.tobytes()

        x, y, yaw = fly_pose_xy_yaw(state.body)
        mn9_idx = state.ports.proboscis_mn
        mn9_silenced = bool(mn9_idx.size > 0 and np.all(state.lif.silenced[mn9_idx] == 1))
        header: dict[str, Any] = {
            "codec": "fly_embodied_v1",
            "dataset_id": state.pack.dataset_id,
            "wiring": state.wiring,
            "num_cells": state.pack.cells.num_cells,
            "geom_count": int(m.ngeom),
            "geom_bytes": len(geom_bytes),
            "fired_count": int(fired.size),
            "geoms": geoms,
            "body_poses": body_poses,
            "torso": [round(float(v), 4) for v in d.qpos[:3]],
            "yaw": round(yaw, 4),
            "arena_half_size": state.arena.arena_half_size,
            "foods": [f.to_dict() for f in state.arena.foods],
            "trail": state.arena.trail[-180:],
            "poke_remaining_ms": state.arena.poke_remaining_ms,
            "commands": state.last_commands.to_dict(),
            "sensors": {
                "sugar": round(state.last_sensors.sugar_contact, 3),
                "bitter": round(state.last_sensors.bitter_contact, 3),
                "antennal_touch": round(state.last_sensors.antennal_touch, 3),
                "odor_left": round(state.last_sensors.odor_left, 3),
                "odor_right": round(state.last_sensors.odor_right, 3),
                "contacts": list(state.last_sensors.tarsus_contacts),
            },
            "timing_ms": {
                "brain": round(state.timing.brain_ms, 3),
                "body": round(state.timing.body_ms, 3),
                "transfer": round(state.timing.transfer_ms, 3),
                "total": round(state.timing.total_ms, 3),
                "realtime_factor": state.timing.realtime_factor,
            },
            "mn9_silenced": mn9_silenced,
            "control_mode": state.control_mode,
            "vnc_profile": state.vnc_profile_name,
            "alif_enabled": state.lif.has_alif,
            "has_vnc_ports": state.ports.has_vnc_ports,
            "vnc_mn_mean_hz": round(state.vnc_mn_mean_hz, 2),
            "vnc_tripod_alternation": round(state.last_vnc_tripod_alt, 3),
            "proprio": [round(float(v), 3) for v in state.last_proprio],
            "leg_mn_hz": (
                {
                    leg: {
                        ch: round(float(state.leg_mn_rates[li, mi]), 1)
                        for mi, ch in enumerate(MUSCLE_CHANNELS)
                    }
                    for li, leg in enumerate(LEG_NAMES)
                }
                if state.ports.has_vnc_ports
                else None
            ),
            "has_stl_meshes": bool(state.body.asset_info.get("installed", False)),
            "body_asset_source": state.body.asset_info.get("source", "procedural_fallback"),
            "level_note": state.framing_note,
            "pos_xy": [round(x, 3), round(y, 3)],
        }
        return header, geom_bytes + fired_bytes

    def apply_intervention(
        self,
        state: EmbodiedFlyWorld,
        spec: ExperimentSpec,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[EmbodiedFlyWorld, dict[str, Any]]:
        if kind == "place_food":
            fkind = str(payload.get("kind", "sugar"))
            if fkind not in ("sugar", "bitter"):
                raise ValueError(f"food kind must be 'sugar' or 'bitter', got {fkind!r}")
            x, y, yaw = fly_pose_xy_yaw(state.body)
            fx = float(payload["x"]) if "x" in payload else x + 0.55 * float(np.cos(yaw))
            fy = float(payload["y"]) if "y" in payload else y + 0.55 * float(np.sin(yaw))
            rad = float(payload.get("radius", 0.36))
            lim = state.arena.arena_half_size - 0.1
            fx = max(-lim, min(lim, fx))
            fy = max(-lim, min(lim, fy))
            spot = FoodSpot(x=fx, y=fy, radius=max(0.15, min(1.0, rad)), kind=fkind)  # type: ignore[arg-type]
            if len(state.arena.foods) >= 12:
                state.arena.foods.pop(0)
            state.arena.foods.append(spot)
            return state, {"placed_food": spot.to_dict(), "food_count": len(state.arena.foods)}

        if kind == "clear_food":
            state.arena.foods.clear()
            return state, {"food_count": 0}

        if kind == "poke_antenna":
            dur = max(20, min(4000, int(payload.get("duration_ms", 1200))))
            state.arena.poke_remaining_ms = max(state.arena.poke_remaining_ms, dur)
            return state, {"poke_remaining_ms": state.arena.poke_remaining_ms}

        if kind == "stimulate":
            port = str(payload.get("port", ""))
            if port not in state.pack.io_ports:
                raise ValueError(f"unknown port {port!r}")
            rate = max(0.0, min(1000.0, float(payload.get("rate_hz", 150.0))))
            idx = state.pack.io_ports[port].cell_indices
            state.manual_stim_hz[idx] = np.float32(rate)
            state.lif.stim_rate_hz[idx] = np.float32(rate)
            if state.gpu_runner is not None:
                state.gpu_runner.upload_controls(state.lif)
            return state, {"port": port, "rate_hz": rate, "cells": int(idx.size)}

        if kind == "silence":
            port = str(payload.get("port", "mn9_proboscis_extension"))
            if port not in state.pack.io_ports:
                raise ValueError(f"unknown port {port!r}")
            silenced = bool(payload.get("silenced", True))
            idx = state.pack.io_ports[port].cell_indices
            state.lif.silenced[idx] = np.uint8(1 if silenced else 0)
            if state.gpu_runner is not None:
                state.gpu_runner.upload_controls(state.lif)
            return state, {"port": port, "silenced": silenced, "cells": int(idx.size)}

        if kind == "clear_stimuli":
            state.manual_stim_hz.fill(np.float32(0.0))
            state.lif.stim_rate_hz.fill(np.float32(0.0))
            state.lif.silenced.fill(np.uint8(0))
            state.arena.poke_remaining_ms = 0
            if state.gpu_runner is not None:
                state.gpu_runner.upload_controls(state.lif)
            return state, {"cleared": True}

        raise ValueError(f"unsupported fly_embodied intervention {kind!r}")

    def inspect(
        self, state: EmbodiedFlyWorld, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        part = str(query.get("body_part", "summary"))
        x, y, yaw = fly_pose_xy_yaw(state.body)
        part_circuits = {
            "proboscis": {
                "label": "Proboscis (mouthpart)",
                "sensory_ports": ["sugar_grn", "bitter_grn"],
                "motor_ports": ["mn9_proboscis_extension"],
                "cell_types": ["CB0701"],
            },
            "antennae": {
                "label": "Antennae (smell & touch)",
                "sensory_ports": ["olfactory_sensory", "antennal_grooming_mechanosensory"],
                "motor_ports": ["grooming_dn"],
                "cell_types": ["DNg84", "DNa01", "DNa02"],
            },
            "legs": {
                "label": "Six legs (Level 1 hand-crafted tripod CPG + descending drive)",
                "sensory_ports": ["sugar_grn", "bitter_grn"],
                "motor_ports": ["forward_dn"],
                "cell_types": ["DNp09", "DNa01", "DNa02", "MDN"],
            },
        }
        return {
            "substrate": "fly_embodied",
            "tick": state.tick,
            "dataset_id": state.pack.dataset_id,
            "wiring": state.wiring,
            "pose": {"x": round(x, 4), "y": round(y, 4), "yaw_rad": round(yaw, 4)},
            "commands": state.last_commands.to_dict(),
            "sensors": {
                "sugar": round(state.last_sensors.sugar_contact, 3),
                "bitter": round(state.last_sensors.bitter_contact, 3),
                "antennal_touch": round(state.last_sensors.antennal_touch, 3),
                "odor_left": round(state.last_sensors.odor_left, 3),
                "odor_right": round(state.last_sensors.odor_right, 3),
            },
            "timing_ms": {
                "brain_ms": round(state.timing.brain_ms, 3),
                "body_ms": round(state.timing.body_ms, 3),
                "transfer_ms": round(state.timing.transfer_ms, 3),
                "total_ms": round(state.timing.total_ms, 3),
                "realtime_factor": state.timing.realtime_factor,
            },
            "body_asset": state.body.asset_info,
            "body_part_link": part_circuits.get(part),
            "framing_note": EMBODIED_LEVEL1_FRAMING,
        }

    @staticmethod
    def _mj_state_vec(state: EmbodiedFlyWorld) -> np.ndarray[Any, np.dtype[np.float64]]:
        import mujoco

        sp = mujoco.mjtState.mjSTATE_INTEGRATION
        out = np.empty(mujoco.mj_stateSize(state.body.model, sp), dtype=np.float64)
        mujoco.mj_getState(state.body.model, state.body.data, out, sp)
        return out

    def serialize_state(self, state: EmbodiedFlyWorld) -> dict[str, Any]:
        """Exact deterministic checkpoint of BOTH spiking LIF/ALIF brain AND MuJoCo body + arena."""
        lif = state.ensure_host()
        arrays: dict[str, Any] = {
            "v": lif.v.copy(),
            "g": lif.g.copy(),
            "refrac": lif.refrac.copy(),
            "delay_ring": lif.delay_ring.copy(),
            "stim_rate_hz": lif.stim_rate_hz.copy(),
            "manual_stim_hz": state.manual_stim_hz.copy(),
            "silenced": lif.silenced.copy(),
            "last_tick_spikes": lif.last_tick_spikes.copy(),
            "total_spikes": lif.total_spikes.copy(),
            "mj_state": self._mj_state_vec(state),
            "stumble_lift": state.body.stumble_lift.copy(),
            "leg_mn_rates": state.leg_mn_rates.copy(),
        }
        if lif.adapt is not None:
            arrays["adapt"] = lif.adapt.copy()
        return {
            "arrays": arrays,
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": int(lif.tick),
                "substep": int(lif.substep),
                "cpg_phase": float(state.body.cpg_phase),
                "groom_phase": float(state.body.groom_phase),
                "foods_json": json.dumps([f.to_dict() for f in state.arena.foods]),
                "trail_json": json.dumps(state.arena.trail),
                "poke_remaining_ms": int(state.arena.poke_remaining_ms),
                "rates": [
                    state.rate_forward,
                    state.rate_turn_left,
                    state.rate_turn_right,
                    state.rate_backward,
                    state.rate_proboscis,
                    state.rate_grooming,
                ],
                "last_commands": state.last_commands.to_dict(),
                "total_feeding_ms": int(state.total_feeding_ms),
                "total_grooming_ms": int(state.total_grooming_ms),
                "distance_traveled": float(state.distance_traveled),
                "prev_xy": list(state.prev_xy),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> EmbodiedFlyWorld:
        import mujoco

        world = self.initialize(spec)
        arrs = payload["arrays"]
        meta = payload["meta"]

        lif = world.lif
        lif.v[...] = arrs["v"]
        lif.g[...] = arrs["g"]
        if "adapt" in arrs and lif.adapt is not None:
            lif.adapt[...] = arrs["adapt"]
        lif.refrac[...] = arrs["refrac"]
        lif.delay_ring[...] = arrs["delay_ring"]
        lif.stim_rate_hz[...] = arrs["stim_rate_hz"]
        lif.silenced[...] = arrs["silenced"]
        lif.last_tick_spikes[...] = arrs["last_tick_spikes"]
        lif.total_spikes[...] = arrs["total_spikes"]
        lif.tick = int(meta["tick"])
        lif.substep = int(meta["substep"])
        world.manual_stim_hz[...] = arrs["manual_stim_hz"]
        if world.gpu_runner is not None:
            world.gpu_runner.upload_state(lif)
            world.gpu_synced_to_host = True

        # Restore exact MuJoCo integration state (qpos, qvel, act, qacc_warmstart, ctrl, time).
        world.body = make_fly_body(datasets_dir=load_settings().resolved_datasets_dir)
        d = world.body.data
        vec = np.asarray(arrs["mj_state"], dtype=np.float64)
        mujoco.mj_setState(world.body.model, d, vec, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(world.body.model, d)
        world.body.cpg_phase = float(meta["cpg_phase"])
        world.body.groom_phase = float(meta["groom_phase"])
        world.body.stumble_lift[...] = arrs["stumble_lift"]
        if "leg_mn_rates" in arrs:
            world.leg_mn_rates[...] = arrs["leg_mn_rates"]

        foods_raw = json.loads(meta["foods_json"])
        world.arena.foods = [
            FoodSpot(x=f["x"], y=f["y"], radius=f["radius"], kind=f["kind"]) for f in foods_raw
        ]
        world.arena.trail = json.loads(meta["trail_json"])
        world.arena.poke_remaining_ms = int(meta["poke_remaining_ms"])
        rates = meta["rates"]
        (
            world.rate_forward,
            world.rate_turn_left,
            world.rate_turn_right,
            world.rate_backward,
            world.rate_proboscis,
            world.rate_grooming,
        ) = [float(r) for r in rates]
        lc = meta["last_commands"]
        world.last_commands = MotorCommands(
            forward=float(lc["forward"]),
            turn=float(lc["turn"]),
            backward=float(lc["backward"]),
            proboscis=float(lc["proboscis"]),
            grooming=float(lc["grooming"]),
        )
        world.total_feeding_ms = int(meta["total_feeding_ms"])
        world.total_grooming_ms = int(meta["total_grooming_ms"])
        world.distance_traveled = float(meta["distance_traveled"])
        world.prev_xy = (float(meta["prev_xy"][0]), float(meta["prev_xy"][1]))
        return world

    def state_distance(self, a: EmbodiedFlyWorld, b: EmbodiedFlyWorld) -> dict[str, Any]:
        la = a.ensure_host()
        lb = b.ensure_host()
        adapt_eq = (la.adapt is None and lb.adapt is None) or (
            la.adapt is not None
            and lb.adapt is not None
            and bool(np.array_equal(la.adapt, lb.adapt))
        )
        brain_eq = bool(
            la.tick == lb.tick
            and la.substep == lb.substep
            and np.array_equal(la.v, lb.v)
            and np.array_equal(la.g, lb.g)
            and adapt_eq
            and np.array_equal(la.delay_ring, lb.delay_ring)
            and np.array_equal(la.total_spikes, lb.total_spikes)
        )
        qpos_diff = float(np.max(np.abs(a.body.data.qpos - b.body.data.qpos)))
        qvel_diff = float(np.max(np.abs(a.body.data.qvel - b.body.data.qvel)))
        body_eq = qpos_diff == 0.0 and qvel_diff == 0.0
        # Keep struct imported only if used? Not needed; return clean dict.
        _ = struct
        return {
            "exact_equal": bool(brain_eq and body_eq),
            "brain_equal": brain_eq,
            "qpos_max_abs_diff": qpos_diff,
            "qvel_max_abs_diff": qvel_diff,
        }
