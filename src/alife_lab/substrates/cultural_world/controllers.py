"""Agent sensory computation, policy inference (scripted / MLP), and lifetime learning."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.agents import N_ACT, N_IN, CulturalAgents
from alife_lab.substrates.cultural_world.ecology import ResourceField, regime_phase

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def grid_cells(x: F64, y: F64, exp: CulturalWorldExperiment) -> tuple[I64, I64]:
    r = exp.world.resolution
    ci = np.floor(y / exp.world.height * r).astype(np.int64) % r
    cj = np.floor(x / exp.world.width * r).astype(np.int64) % r
    return ci, cj


def compute_sensors(
    agents: CulturalAgents,
    eco: ResourceField,
    exp: CulturalWorldExperiment,
    tick: int,
    phase_offset: float = 0.0,
) -> F64:
    """Build (n, 12) float64 observation matrix for all living agents."""
    n = agents.n
    x = np.zeros((n, N_IN), dtype=np.float64)
    if n == 0:
        return x
    sensor_dist = max(1.0, exp.world.width / exp.world.resolution)
    ang = math.radians(40.0)
    for k, off in enumerate((ang, 0.0, -ang)):
        sx = (agents.x + sensor_dist * np.cos(agents.heading + off)) % exp.world.width
        sy = (agents.y + sensor_dist * np.sin(agents.heading + off)) % exp.world.height
        ci, cj = grid_cells(sx, sy, exp)
        x[:, k] = eco.food[ci, cj] / max(1e-6, exp.dynamics.resources.food_capacity)
    ci0, cj0 = grid_cells(agents.x, agents.y, exp)
    x[:, 3] = eco.material_a[ci0, cj0] / max(1e-6, exp.dynamics.resources.material_a_capacity)
    x[:, 4] = eco.material_b[ci0, cj0] / max(1e-6, exp.dynamics.resources.material_b_capacity)
    x[:, 5] = np.clip(agents.inv_a, 0.0, 2.0) * 0.5
    x[:, 6] = np.clip(agents.inv_b, 0.0, 2.0) * 0.5
    x[:, 7] = np.clip(agents.energy / max(1e-6, exp.dynamics.reproduce_threshold), 0.0, 2.0)
    x[:, 8] = np.array([1.0 if sl else 0.0 for sl in agents.skills], dtype=np.float64)
    phase = regime_phase(tick, exp, phase_offset)
    x[:, 9] = math.sin(2.0 * math.pi * phase)
    x[:, 10] = math.cos(2.0 * math.pi * phase)
    x[:, 11] = 1.0
    return x


def choose_actions(
    agents: CulturalAgents,
    x: F64,
    exp: CulturalWorldExperiment,
    rng_policy: np.random.Generator,
) -> tuple[F64, F64, I64]:
    """Compute (hidden, action_probs, sampled_actions) from policy."""
    n = agents.n
    h_dim = exp.dynamics.controller.hidden
    if n == 0:
        return (
            np.zeros((0, h_dim), dtype=np.float64),
            np.zeros((0, N_ACT), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    ctrl = exp.dynamics.controller
    if ctrl.kind == "scripted":
        hidden = np.zeros((n, h_dim), dtype=np.float64)
        probs = np.zeros((n, N_ACT), dtype=np.float64)
        for i in range(n):
            has_skill = x[i, 8] > 0.5
            has_mat = (agents.inv_a[i] >= 0.45) or (agents.inv_b[i] >= 0.45)
            local_res = x[i, 1] + 0.7 * (x[i, 3] + x[i, 4])
            if has_skill and has_mat:
                probs[i, 4] = 0.85  # apply_skill
                probs[i, 5] = 0.15  # teach
            elif (
                (not has_skill)
                and has_mat
                and exp.dynamics.technology.allow_invention
                and rng_policy.random() < 0.25
            ):
                probs[i, 4] = 1.0  # experiment / invent
            elif local_res > 0.35:
                probs[i, 3] = 0.85  # gather
                probs[i, 1] = 0.15
            else:
                best = int(np.argmax([x[i, 0], x[i, 1], x[i, 2]]))
                act_idx = (0, 1, 2)[best]
                probs[i, act_idx] = 0.85
                probs[i, 3] = 0.15
        u = rng_policy.random(n)
        act = np.minimum((np.cumsum(probs, axis=1) <= u[:, None]).sum(axis=1), N_ACT - 1)
        return hidden, probs, act

    hidden = np.tanh(np.einsum("ni,nih->nh", x, agents.w1) + agents.b1)
    logits = np.einsum("nh,nha->na", hidden, agents.w2 + agents.dw2) + agents.b2
    z = logits / ctrl.temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    u = rng_policy.random(n)
    act = np.minimum((np.cumsum(probs, axis=1) <= u[:, None]).sum(axis=1), N_ACT - 1)
    return hidden, probs, act


def apply_lifetime_learning(
    agents: CulturalAgents,
    hidden: F64,
    probs: F64,
    act: I64,
    reward: F64,
    exp: CulturalWorldExperiment,
    rng_learn: np.random.Generator,
) -> None:
    """REINFORCE update on non-heritable ``dw2`` using internal energy change ONLY."""
    if not exp.dynamics.lifetime_learning or exp.dynamics.controller.kind != "mlp" or agents.n == 0:
        return
    # Draw small exploratory scalar from rng_learn to maintain independent stream state
    _ = float(rng_learn.random())
    ctrl = exp.dynamics.controller
    lr = ctrl.learning_rate
    if lr <= 0.0:
        return
    adv = np.clip(reward - agents.rbase, -2.0, 2.0)
    onehot = np.eye(N_ACT, dtype=np.float64)[act]
    grad = np.einsum("nh,na->nha", hidden, (onehot - probs) / ctrl.temperature)
    agents.dw2 = np.clip(agents.dw2 + lr * adv[:, None, None] * grad, -4.0, 4.0)
    agents.rbase = 0.9 * agents.rbase + 0.1 * reward
