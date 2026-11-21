"""Embodied creature numerics: MJCF generation from the spec and a MuJoCo (CPU, float64) world.

Morphology: a spherical torso with legs on its equator. Leg i at angle a: a thigh of length
``upper`` pointing outwards (hip joint: hinge about the vertical axis, range +-45 deg), then a
shin of length ``lower`` pointing outwards and down (knee joint: hinge about the leg's tangent
axis, range +-60 deg). Every part has the same density; collisions only with the floor.
Control: a CPG, ctrl_j = A_j sin(2 pi (f t + phi_j)), motor torque = gear * ctrl.
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
        "<worldbody>"
        '<geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="1" '
        'rgba="0.86 0.87 0.89 1" '
        f'friction="{_f(d.friction)} 0.1 0.1"/>'
        f'<body name="torso" pos="0 0 {_f(exp.initialization.height)}"><freejoint name="root"/>'
        f'<geom name="torso" type="sphere" size="{_f(r)}" rgba="0.93 0.55 0.2 1"/>'
        + "".join(parts)
        + "</body></worldbody><actuator>"
        + "".join(acts)
        + "</actuator></mujoco>"
    )


@dataclass
class World:
    model: Any = field(repr=False)
    data: Any = field(repr=False)
    tick: int = 0
    start_xy: tuple[float, float] = (0.0, 0.0)
    energy: float = 0.0  # cumulative mechanical work magnitude, sum |tau . qdot| dt (J)
    recent: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)


def make_world(exp: CreatureExperiment) -> World:
    import mujoco

    model = mujoco.MjModel.from_xml_string(build_xml(exp))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    w = World(model=model, data=data)
    w.start_xy = (float(data.qpos[0]), float(data.qpos[1]))
    return w


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
        d.ctrl[:] = control(exp, float(d.time))
        mujoco.mj_step(m, d)
        w.energy += float(np.abs(d.actuator_force * d.actuator_velocity).sum()) * m.opt.timestep
    w.tick += 1
    w.recent.append((float(d.time), float(d.qpos[0]), float(d.qpos[1])))
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
