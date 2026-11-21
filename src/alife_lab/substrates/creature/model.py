"""Embodied creature numerics: MJCF generation from the spec and a MuJoCo (CPU, float64) world.

Morphology: a spherical torso with legs on its equator. Leg i at angle a: a thigh of length
``upper`` pointing outwards (hip joint: hinge about the vertical axis, range +-45 deg), then a
shin of length ``lower`` pointing outwards and down (knee joint: hinge about the leg's tangent
axis, range +-60 deg). Every part has the same density; collisions only with the floor.
Control: a CPG, ctrl_j = A_j sin(2 pi (f t + phi_j)), motor torque = gear * ctrl.
Terrain (optional): a height field of smooth random bumps generated from the experiment seed.
Reflex (optional): tilt and contact feedback added to the knee commands (see CreatureReflex);
a positive knee command pushes the foot down (towards a vertical shin).
Determinism: MuJoCo is single-threaded here; the same spec and version give identical states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import CreatureExperiment

F64 = npt.NDArray[np.float64]
HIP_RANGE = 45.0
KNEE_RANGE = 60.0
SHIN_DROP = 0.8  # shin points outwards by (1 - SHIN_DROP) of its length and down by SHIN_DROP


def _f(v: float) -> str:
    return f"{v:.6g}"


def terrain_heights(exp: CreatureExperiment) -> F64 | None:
    """(n, n) heights in metres (row = y, column = x), or None for a flat floor."""
    t = exp.terrain
    if t is None:
        return None
    n = round(2 * t.half_size / t.resolution) + 1
    coarse = math.ceil(2 * t.half_size / t.wavelength) + 2
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([exp.seed, 7])))
    noise = rng.random((coarse, coarse))
    xs = np.linspace(0, 2 * t.half_size / t.wavelength, n)
    i0 = np.minimum(np.floor(xs).astype(int), coarse - 2)
    f = xs - i0
    rows = noise[i0] * (1 - f)[:, None] + noise[i0 + 1] * f[:, None]  # interpolate along y
    h = rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]  # then along x
    coords = np.linspace(-t.half_size, t.half_size, n)
    r = np.hypot(coords[None, :], coords[:, None])
    ramp = np.clip((r - t.flat_radius) / max(t.wavelength, 1e-9), 0.0, 1.0)
    out: F64 = h * t.amplitude * ramp
    return out


def ground_height(exp: CreatureExperiment, x: float, y: float) -> float:
    h = terrain_heights(exp)
    if h is None or exp.terrain is None:
        return 0.0
    t = exp.terrain
    j = round((x + t.half_size) / t.resolution)
    i = round((y + t.half_size) / t.resolution)
    if 0 <= i < h.shape[0] and 0 <= j < h.shape[1]:
        return float(h[i, j])
    return 0.0


def build_xml(exp: CreatureExperiment) -> str:
    d = exp.dynamics
    b = d.body
    r = b.torso_radius
    parts = []
    acts = []
    for i, leg in enumerate(b.legs):
        a = math.radians(leg.angle_deg)
        c, s = math.cos(a), math.sin(a)
        ox, oy = (1 - SHIN_DROP) * leg.lower, -SHIN_DROP * leg.lower
        parts.append(
            f'<body name="leg{i}" pos="{_f(r * c)} {_f(r * s)} 0">'
            f'<joint name="hip{i}" class="limb" type="hinge" axis="0 0 1" '
            f'range="-{HIP_RANGE} {HIP_RANGE}"/>'
            f'<geom name="thigh{i}" type="capsule" rgba="0.33 0.6 0.9 1" '
            f'fromto="0 0 0 {_f(leg.upper * c)} '
            f'{_f(leg.upper * s)} 0" size="{_f(leg.radius)}"/>'
            f'<body name="shin{i}" pos="{_f(leg.upper * c)} {_f(leg.upper * s)} 0">'
            f'<joint name="knee{i}" class="limb" type="hinge" axis="{_f(-s)} {_f(c)} 0" '
            f'range="-{KNEE_RANGE} {KNEE_RANGE}"/>'
            f'<geom name="foot{i}" type="capsule" rgba="0.2 0.38 0.75 1" '
            f'fromto="0 0 0 {_f(ox * c)} {_f(ox * s)} '
            f'{_f(oy)}" size="{_f(leg.radius)}"/>'
            "</body></body>"
        )
        acts.append(f'<motor joint="hip{i}" gear="{_f(d.gear)}" ctrlrange="-1 1"/>')
        acts.append(f'<motor joint="knee{i}" gear="{_f(d.gear)}" ctrlrange="-1 1"/>')
    return (
        '<mujoco model="creature">'
        '<compiler angle="degree"/>'
        f'<option timestep="{_f(d.timestep)}" integrator="implicitfast"/>'
        # Damping, armature and springs apply to the leg joints only (class "limb"); the free
        # root joint must stay undamped, otherwise it would act like air drag.
        f'<default><default class="limb"><joint limited="true" damping="1" armature="0.05" '
        f'stiffness="{_f(d.joint_stiffness)}"/></default>'
        f'<geom density="{_f(d.density)}" friction="{_f(d.friction)} 0.1 0.1" '
        'contype="1" conaffinity="0" condim="3"/></default>'
        + _terrain_asset(exp)
        + "<worldbody>"
        + _terrain_geom(exp)
        + '<geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="1" '
        'rgba="0.86 0.87 0.89 1" '
        f'friction="{_f(d.friction)} 0.1 0.1"/>'
        f'<body name="torso" pos="0 0 {_f(exp.initialization.height)}"><freejoint name="root"/>'
        f'<geom name="torso" type="sphere" size="{_f(r)}" rgba="0.93 0.55 0.2 1"/>'
        + "".join(parts)
        + "</body></worldbody><actuator>"
        + "".join(acts)
        + "</actuator></mujoco>"
    )


def _terrain_asset(exp: CreatureExperiment) -> str:
    t = exp.terrain
    if t is None:
        return ""
    n = round(2 * t.half_size / t.resolution) + 1
    elev = max(t.amplitude, 1e-4)
    return (f'<asset><hfield name="terrain" nrow="{n}" ncol="{n}" '
            f'size="{_f(t.half_size)} {_f(t.half_size)} {_f(elev)} 0.05"/></asset>')  # fmt: skip


def _terrain_geom(exp: CreatureExperiment) -> str:
    if exp.terrain is None:
        return ""
    friction = _f(exp.dynamics.friction)
    return ('<geom name="terrain" type="hfield" hfield="terrain" contype="0" conaffinity="1" '
            f'rgba="0.78 0.74 0.66 1" friction="{friction} 0.1 0.1"/>')  # fmt: skip


@dataclass
class World:
    model: Any = field(repr=False)
    data: Any = field(repr=False)
    tick: int = 0
    start_xy: tuple[float, float] = (0.0, 0.0)
    energy: float = 0.0  # cumulative mechanical work magnitude, sum |tau . qdot| dt (J)
    recent: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    path_length: float = 0.0  # odometer of the torso's planar path (m)
    contacts: list[bool] = field(default_factory=list)  # per leg, after the last physics step
    reflex: list[float] = field(default_factory=list)  # per leg knee reflex command
    tilt_deg: float = 0.0
    foot_ids: list[int] = field(default_factory=list, repr=False)
    sensed_rot: F64 = field(default_factory=lambda: np.eye(3).ravel(), repr=False)
    last_xy: tuple[float, float] = (0.0, 0.0)


def make_world(exp: CreatureExperiment) -> World:
    import mujoco

    model = mujoco.MjModel.from_xml_string(build_xml(exp))
    heights = terrain_heights(exp)
    if heights is not None:
        assert exp.terrain is not None
        model.hfield_data[:] = (heights / max(exp.terrain.amplitude, 1e-4)).ravel()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    w = World(model=model, data=data)
    w.start_xy = (float(data.qpos[0]), float(data.qpos[1]))
    w.last_xy = w.start_xy
    legs = len(exp.dynamics.body.legs)
    w.foot_ids = [model.geom(f"foot{i}").id for i in range(legs)]
    w.contacts = [False] * legs
    w.reflex = [0.0] * legs
    w.sensed_rot = data.xmat[1].copy()
    return w


def foot_contacts(w: World) -> list[bool]:
    d = w.data
    touching = set()
    for c in range(d.ncon):
        touching.add(int(d.contact[c].geom1))
        touching.add(int(d.contact[c].geom2))
    return [f in touching for f in w.foot_ids]


def reflex_commands(w: World, exp: CreatureExperiment) -> tuple[F64, float]:
    """Per-leg knee reflex and the torso tilt (deg) from what the body senses.

    Sensing has a one-physics-step delay: the torso orientation and the foot contacts are the
    ones MuJoCo computed at the start of the previous step (``sensed_rot``, ``contacts``). Both
    are part of the state and are checkpointed, so restored runs continue bit-for-bit."""
    r = exp.dynamics.reflex
    legs = exp.dynamics.body.legs
    rot = w.sensed_rot.reshape(3, 3)  # torso orientation
    up = rot[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    if r is None:
        return np.zeros(len(legs)), tilt
    out = np.zeros(len(legs))
    for i, leg in enumerate(legs):
        a = math.radians(leg.angle_deg)
        direction = rot @ np.array([math.cos(a), math.sin(a), 0.0])  # leg direction in the world
        lean = float(up[0] * direction[0] + up[1] * direction[1])  # top leans towards the leg
        out[i] = r.tilt_gain * lean + (0.0 if w.contacts[i] else r.contact_gain)
    return out, tilt


def control(exp: CreatureExperiment, t: float) -> F64:
    c = exp.dynamics.controller
    n = 2 * len(exp.dynamics.body.legs)
    if c.kind == "off":
        return np.zeros(n)
    amp = np.asarray(c.amplitudes, dtype=np.float64)
    ph = np.asarray(c.phases, dtype=np.float64)
    out: F64 = amp * np.sin(2 * math.pi * (c.frequency_hz * t + ph))
    return out


def step(w: World, exp: CreatureExperiment) -> None:
    """One tick = ``substeps`` physics steps with the CPG evaluated at every physics step."""
    import mujoco

    m, d = w.model, w.data
    for _ in range(exp.dynamics.substeps):
        u = control(exp, float(d.time))
        reflex, w.tilt_deg = reflex_commands(w, exp)
        if exp.dynamics.reflex is not None and exp.dynamics.controller.kind != "off":
            u[1::2] += reflex  # knees
            w.reflex = [float(v) for v in reflex]
        d.ctrl[:] = np.clip(u, -1.0, 1.0)
        mujoco.mj_step(m, d)
        w.contacts = foot_contacts(w)
        w.sensed_rot = d.xmat[1].copy()
        w.energy += float(np.abs(d.actuator_force * d.actuator_velocity).sum()) * m.opt.timestep
    w.tick += 1
    x, y = float(d.qpos[0]), float(d.qpos[1])
    w.path_length += math.hypot(x - w.last_xy[0], y - w.last_xy[1])
    w.last_xy = (x, y)
    w.recent.append((float(d.time), x, y))
    if len(w.recent) > 200:
        w.recent.pop(0)


def speed(w: World, window_s: float = 2.0) -> float:
    """Planar torso speed (m/s) over the last ``window_s`` seconds (or what exists)."""
    if len(w.recent) < 2:
        return 0.0
    t1, x1, y1 = w.recent[-1]
    old = [r for r in w.recent if r[0] >= t1 - window_s]
    t0, x0, y0 = old[0]
    return math.hypot(x1 - x0, y1 - y0) / (t1 - t0) if t1 > t0 else 0.0


def upright(w: World) -> float:
    """Cosine between the torso's z axis and the world's z axis (1 = upright)."""
    q = w.data.qpos[3:7]  # w, x, y, z
    return float(1 - 2 * (q[1] ** 2 + q[2] ** 2))
