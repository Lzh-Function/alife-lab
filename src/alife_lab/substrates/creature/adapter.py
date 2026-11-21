"""Embodied creature adapter (MuJoCo 3, CPU, float64). Project model, exploratory only."""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import CreatureExperiment, ExperimentSpec
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.creature import model

SERIALIZE_VERSION = 1
MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}


def _cr(spec: ExperimentSpec) -> CreatureExperiment:
    exp = spec.experiment
    if not isinstance(exp, CreatureExperiment):
        raise TypeError("creature adapter received a different spec")
    return exp


def _mujoco() -> Any:
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo is not installed (uv sync --extra embodied)")
    import mujoco

    return mujoco


_M = MetricDefinition
_METRICS = (
    _M("displacement", "metric.displacement", "m", "tick",
       "Planar distance of the torso from its start (interventions included).", "none",
       "every tick", 1, "never null"),
    _M("forward", "metric.forward", "m", "tick", "Torso displacement along +x since the start.",
       "none", "every tick", 1, "never null"),
    _M("speed", "metric.speed", "m/s", "tick",
       "Planar torso speed over the last 2 simulated seconds.", "none", "every tick", 1,
       "never null"),
    _M("height", "metric.height", "m", "tick", "Height of the torso centre.", "none",
       "every tick", 1, "never null"),
    _M("upright", "metric.upright", "cosine", "tick",
       "Cosine between the torso's and the world's z axes (1 upright, -1 upside down).", "none",
       "every tick", 1, "never null"),
    _M("energy", "metric.energy", "J", "tick",
       "Cumulative mechanical work magnitude of the motors, sum |tau * qdot| dt.", "sum",
       "every tick", 1, "never null"),
    _M("time", "metric.time", "s", "tick", "Simulated time.", "none", "every tick", 1,
       "never null"),
)  # fmt: skip

CREATURE_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.creature",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="creature",
    display_name_key="substrate.creature",
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
        intervention_types=("push",),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Project model: torso + legs in MuJoCo with a CPG controller "
        "(cf. Sims 1994 for morphology/control co-evolution). MuJoCo is the physics engine.",
        "paper_urls": ["https://doi.org/10.1145/192161.192167"],
        "implementation": "https://github.com/google-deepmind/mujoco",
    },
)


class CreatureAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in CREATURE_MANIFEST.backends:
            raise ValueError(f"unsupported creature backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return CREATURE_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> model.World:
        _mujoco()
        return model.make_world(_cr(spec))

    def step(self, state: model.World, spec: ExperimentSpec) -> model.World:
        model.step(state, _cr(spec))
        return state

    def metrics(self, state: model.World, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_cr(spec).observations.metrics)
        q = state.data.qpos
        out = {
            "displacement": math.hypot(q[0] - state.start_xy[0], q[1] - state.start_xy[1]),
            "forward": float(q[0] - state.start_xy[0]),
            "speed": model.speed(state),
            "height": float(q[2]),
            "upright": model.upright(state),
            "energy": state.energy,
            "time": float(state.data.time),
        }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: model.World) -> tuple[dict[str, Any], bytes]:
        """scene3d_f32: per geom position (3) and rotation matrix (9), float32 row-major; the
        header lists the geoms (type, size, rgba) and the torso position for the camera."""
        m, d = state.model, state.data
        geoms = []
        for g in range(m.ngeom):
            geoms.append({
                "type": GEOM_TYPES.get(int(m.geom_type[g]), "box"),
                "size": [round(float(v), 5) for v in m.geom_size[g]],
                "rgba": [round(float(v), 3) for v in m.geom_rgba[g]],
            })  # fmt: skip
        data = np.concatenate([d.geom_xpos, d.geom_xmat], axis=1).astype(np.float32).tobytes()
        return {
            "codec": "scene3d_f32",
            "count": int(m.ngeom),
            "geoms": geoms,
            "torso": [round(float(v), 4) for v in d.qpos[:3]],
            "start": [round(state.start_xy[0], 4), round(state.start_xy[1], 4)],
            "scale": "metres, z up",
        }, data

    def apply_intervention(
        self, state: model.World, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[model.World, dict[str, Any]]:
        """``push``: an instantaneous change of the torso's linear velocity (m/s), i.e. a kick."""
        if kind != "push":
            raise ValueError(f"unsupported intervention kind {kind!r}")
        try:
            dv = [float(payload.get("vx", 0.0)), float(payload.get("vy", 0.0)),
                  float(payload.get("vz", 0.0))]  # fmt: skip
        except (TypeError, ValueError) as exc:
            raise ValueError("push needs numeric vx, vy, vz (m/s)") from exc
        if any(not math.isfinite(v) or abs(v) > 10 for v in dv):
            raise ValueError("push components must be within +-10 m/s")
        before = [float(v) for v in state.data.qvel[:3]]
        state.data.qvel[:3] += np.asarray(dv)
        _mujoco().mj_forward(state.model, state.data)
        return state, {"velocity_before": before,
                       "velocity_after": [float(v) for v in state.data.qvel[:3]]}  # fmt: skip

    def inspect(
        self, state: model.World, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _cr(spec)
        m, d = state.model, state.data
        legs = []
        for i in range(len(exp.dynamics.body.legs)):
            hip, knee = m.joint(f"hip{i}").id, m.joint(f"knee{i}").id
            foot = m.geom(f"foot{i}").id
            touching = any(
                d.contact[c].geom1 == foot or d.contact[c].geom2 == foot for c in range(d.ncon)
            )
            legs.append({
                "leg": i,
                "hip_deg": math.degrees(float(d.qpos[m.jnt_qposadr[hip]])),
                "knee_deg": math.degrees(float(d.qpos[m.jnt_qposadr[knee]])),
                "hip_ctrl": float(d.ctrl[2 * i]),
                "knee_ctrl": float(d.ctrl[2 * i + 1]),
                "foot_on_ground": bool(touching),
            })  # fmt: skip
        return {
            "substrate": "creature",
            "tick": state.tick,
            "time": float(d.time),
            "query": {k: query.get(k) for k in ("x", "y")},
            "torso": {
                "position": [float(v) for v in d.qpos[:3]],
                "velocity": [float(v) for v in d.qvel[:3]],
                "upright": model.upright(state),
            },
            "legs": legs,
            "mass_kg": float(m.body_subtreemass[1]),
            "controller": exp.dynamics.controller.kind,
        }

    def state_distance(self, a: model.World, b: model.World) -> dict[str, Any]:
        sa, sb = self._state_vec(a), self._state_vec(b)
        same = sa.shape == sb.shape
        diff = float(np.abs(sa - sb).max()) if same else None
        exact = bool(same and np.array_equal(sa, sb) and a.energy == b.energy)
        return {"metric": "max_abs_state_difference", "value": diff, "exact_equal": exact}

    @staticmethod
    def _state_vec(w: model.World) -> np.ndarray[Any, Any]:
        mj = _mujoco()
        spec = mj.mjtState.mjSTATE_INTEGRATION
        out = np.empty(mj.mj_stateSize(w.model, spec))
        mj.mj_getState(w.model, w.data, out, spec)
        return out

    def serialize_state(self, state: model.World) -> dict[str, Any]:
        return {
            "arrays": {"mj_state": self._state_vec(state),
                       "recent": np.asarray(state.recent, dtype=np.float64).reshape(-1, 3)},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "start_xy": list(state.start_xy),
                "energy": state.energy,
                "mujoco": _mujoco().__version__,
                "state_spec": "mjSTATE_INTEGRATION",
            },
        }  # fmt: skip

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> model.World:
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible creature checkpoint serialize_version")
        mj = _mujoco()
        w = model.make_world(_cr(spec))
        vec = np.asarray(payload["arrays"]["mj_state"], dtype=np.float64)
        if vec.size != mj.mj_stateSize(w.model, mj.mjtState.mjSTATE_INTEGRATION):
            raise ValueError("creature checkpoint does not match the model")
        mj.mj_setState(w.model, w.data, vec, mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(w.model, w.data)
        w.tick = int(meta["tick"])
        w.start_xy = (float(meta["start_xy"][0]), float(meta["start_xy"][1]))
        w.energy = float(meta["energy"])
        recent = np.asarray(payload["arrays"]["recent"], dtype=np.float64).reshape(-1, 3)
        w.recent = [(float(r[0]), float(r[1]), float(r[2])) for r in recent]
        return w
