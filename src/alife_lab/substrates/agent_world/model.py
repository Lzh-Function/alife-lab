"""AgentWorld numerics (NumPy, float64, vectorised over agents).

Per tick (fixed order):
1. Sensors: food and signal sampled at three points (left / front / right, ``sensor_angle``
   apart, ``sensor_distance`` ahead) on the periodic grid; energy / reproduce_threshold; bias.
   Without communication the signal inputs read zero.
2. Policy: ``baseline`` turns towards the largest food reading; ``random`` is uniform;
   ``mlp`` computes h = tanh(x W1 + b1), p = softmax((h (W2 + dW2) + b2) / T). Actions are
   sampled from p with one uniform draw per agent: 0 turn left, 1 forward, 2 turn right,
   3 emit signal (stay). Turning also moves half a step.
3. Movement (cost proportional to distance), emission (cost; adds to the signal grid).
4. Feeding: every agent asks for ``bite`` from its cell; contested cells are shared in
   proportion, so a cell never gives more than it holds.
5. Metabolism, reward = change of the agent's own energy this tick.
6. Lifetime learning (mode ``lifetime``): REINFORCE on the output layer,
   dW2 += lr * (r - baseline) * h^T (onehot(a) - p) / T; the baseline is a running mean of r.
7. Death at energy <= 0; reproduction at energy >= threshold (child gets half, placed nearby;
   genetic weights mutated with N(0, sigma) in ``evolution`` mode, copied otherwise; learned
   deltas are never inherited).
8. Food regrowth towards capacity; signal diffusion and decay.
Policy shuffle (control): before the policy step the controllers are permuted among agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import AgentWorldDynamics, AgentWorldExperiment

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
N_IN = 8
N_ACT = 4
INPUTS = ("food_left", "food_front", "food_right", "signal_left", "signal_front",
          "signal_right", "energy", "bias")  # fmt: skip
ACTIONS = ("left", "forward", "right", "emit")
GENOME_KEYS = ("w1", "b1", "w2", "b2")


@dataclass
class Agents:
    ids: I64
    parent: I64
    birth: I64
    x: F64
    y: F64
    heading: F64
    energy: F64
    age: I64
    w1: F64  # (n, IN, H) genetic
    b1: F64
    w2: F64  # (n, H, A) genetic
    b2: F64
    dw2: F64  # (n, H, A) learned during life (lifetime mode)
    rbase: F64  # reward baseline
    hist: F64  # (n, T, IN + H + A + 2): inputs, hidden, probs, action, reward

    @property
    def n(self) -> int:
        return int(self.ids.size)

    def fields(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def take(self, idx: Any) -> Agents:
        return Agents(**{k: v[idx] for k, v in self.fields().items()})

    @staticmethod
    def concat(parts: list[Agents]) -> Agents:
        keys = parts[0].fields().keys()
        return Agents(**{k: np.concatenate([p.fields()[k] for p in parts]) for k in keys})


def init_agents(exp: AgentWorldExperiment, n: int, rng: np.random.Generator,
                start_id: int = 0) -> Agents:  # fmt: skip
    h = exp.dynamics.controller.hidden
    t = exp.observations.history
    return Agents(
        ids=np.arange(start_id, start_id + n, dtype=np.int64),
        parent=np.full(n, -1, np.int64),
        birth=np.zeros(n, np.int64),
        x=rng.random(n) * exp.world.width,
        y=rng.random(n) * exp.world.height,
        heading=rng.random(n) * 2 * math.pi,
        energy=np.full(n, exp.dynamics.body.initial_energy),
        age=np.zeros(n, np.int64),
        w1=rng.standard_normal((n, N_IN, h)) / math.sqrt(N_IN),
        b1=np.zeros((n, h)),
        w2=rng.standard_normal((n, h, N_ACT)) / math.sqrt(h),
        b2=np.zeros((n, N_ACT)),
        dw2=np.zeros((n, h, N_ACT)),
        rbase=np.zeros(n),
        hist=np.zeros((n, t, N_IN + h + N_ACT + 2)),
    )


def _cell(x: Any, y: Any, exp: AgentWorldExperiment) -> tuple[I64, I64]:
    r = exp.world.resolution
    i = np.floor(y / exp.world.height * r).astype(np.int64) % r
    j = np.floor(x / exp.world.width * r).astype(np.int64) % r
    return i, j


def sensors(a: Agents, food: F64, signal: F64, exp: AgentWorldExperiment) -> F64:
    b = exp.dynamics.body
    ang = math.radians(b.sensor_angle_deg)
    x = np.zeros((a.n, N_IN))
    for k, off in enumerate((ang, 0.0, -ang)):  # left, front, right
        sx = a.x + b.sensor_distance * np.cos(a.heading + off)
        sy = a.y + b.sensor_distance * np.sin(a.heading + off)
        i, j = _cell(sx, sy, exp)
        x[:, k] = food[i, j]
        if exp.dynamics.communication:
            x[:, 3 + k] = signal[i, j]
    x[:, 6] = a.energy / b.reproduce_threshold
    x[:, 7] = 1.0
    return x


def policy(a: Agents, x: F64, d: AgentWorldDynamics) -> tuple[F64, F64]:
    """Hidden activations (n, H) and action probabilities (n, A)."""
    c = d.controller
    n = a.n
    hidden = np.zeros((n, c.hidden))
    if c.kind == "random":
        return hidden, np.full((n, N_ACT), 1.0 / N_ACT)
    if c.kind == "baseline":
        probs = np.zeros((n, N_ACT))
        best = np.argmax(x[:, [1, 0, 2]], axis=1)  # prefer front on ties
        probs[np.arange(n), np.array([1, 0, 2])[best]] = 1.0
        return hidden, probs
    hidden = np.tanh(np.einsum("ni,nih->nh", x, a.w1) + a.b1)
    logits = np.einsum("nh,nha->na", hidden, a.w2 + a.dw2) + a.b2
    z = logits / c.temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return hidden, e / e.sum(axis=1, keepdims=True)


@dataclass
class TickStats:
    births: int = 0
    deaths: int = 0
    emitted: int = 0
    forward: int = 0
    reward_sum: float = 0.0
    agents: int = 0


StepResult = tuple[Agents, F64, F64, int, "TickStats"]


def step(
    a: Agents,
    food: F64,
    signal: F64,
    exp: AgentWorldExperiment,
    tick: int,
    rng: np.random.Generator,
    next_id: int,
) -> StepResult:
    d = exp.dynamics
    b = d.body
    c = d.controller
    st = TickStats(agents=a.n)
    if a.n:
        if d.shuffle_policies:
            perm = rng.permutation(a.n)
            for k in (*GENOME_KEYS, "dw2", "rbase"):
                setattr(a, k, getattr(a, k)[perm])
        x = sensors(a, food, signal, exp)
        hidden, probs = policy(a, x, d)
        u = rng.random(a.n)
        act = np.minimum((np.cumsum(probs, axis=1) <= u[:, None]).sum(axis=1), N_ACT - 1)
        e0 = a.energy.copy()
        turn = math.radians(b.turn_deg)
        a.heading = (a.heading + np.where(act == 0, turn, 0.0) - np.where(act == 2, turn, 0.0)) % (
            2 * math.pi
        )
        dist = np.where(act == 1, b.speed, np.where(act == 3, 0.0, 0.5 * b.speed))
        a.x = (a.x + dist * np.cos(a.heading)) % exp.world.width
        a.y = (a.y + dist * np.sin(a.heading)) % exp.world.height
        a.energy -= b.move_cost * dist / b.speed
        emit = act == 3
        a.energy -= b.emit_cost * emit
        if d.communication and emit.any():
            i, j = _cell(a.x[emit], a.y[emit], exp)
            np.add.at(signal, (i, j), d.emit_amount)
        # feeding with proportional sharing of contested cells
        i, j = _cell(a.x, a.y, exp)
        demand = np.zeros_like(food)
        np.add.at(demand, (i, j), d.food.bite)
        ratio = np.where(demand > 0, np.minimum(1.0, food / np.maximum(demand, 1e-12)), 0.0)
        take = d.food.bite * ratio[i, j]
        food -= np.minimum(food, demand)
        a.energy += take
        a.energy -= b.metabolism
        a.age += 1
        reward = a.energy - e0
        if c.kind == "mlp" and c.mode == "lifetime" and c.learning_rate > 0:
            adv = reward - a.rbase
            onehot = np.eye(N_ACT)[act]
            a.dw2 += (
                c.learning_rate
                * adv[:, None, None]
                * np.einsum("nh,na->nha", hidden, (onehot - probs) / c.temperature)
            )
            a.rbase = 0.9 * a.rbase + 0.1 * reward
        row = np.concatenate([x, hidden, probs, act[:, None].astype(np.float64),
                              reward[:, None]], axis=1)  # fmt: skip
        a.hist = np.concatenate([a.hist[:, 1:], row[:, None, :]], axis=1)
        st.emitted = int(emit.sum())
        st.forward = int((act == 1).sum())
        st.reward_sum = float(reward.sum())
        # death
        alive = a.energy > 0
        st.deaths = int((~alive).sum())
        a = a.take(alive)
        # reproduction
        room = d.max_population - a.n
        ready = np.flatnonzero(a.energy >= b.reproduce_threshold)[: max(0, room)]
        if ready.size:
            parents = a.take(ready)
            a.energy[ready] /= 2
            child = parents.take(np.arange(ready.size))
            m = ready.size
            child.ids = np.arange(next_id, next_id + m, dtype=np.int64)
            next_id += m
            child.parent = parents.ids.copy()
            child.birth = np.full(m, tick, np.int64)
            child.energy = a.energy[ready].copy()
            child.age = np.zeros(m, np.int64)
            child.heading = rng.random(m) * 2 * math.pi
            child.x = (child.x + rng.standard_normal(m) * 0.5) % exp.world.width
            child.y = (child.y + rng.standard_normal(m) * 0.5) % exp.world.height
            if c.kind == "mlp" and c.mode == "evolution" and c.mutation_sigma > 0:
                for k in GENOME_KEYS:
                    v = getattr(child, k)
                    setattr(child, k, v + c.mutation_sigma * rng.standard_normal(v.shape))
            child.dw2 = np.zeros_like(child.dw2)
            child.rbase = np.zeros(m)
            child.hist = np.zeros_like(child.hist)
            a = Agents.concat([a, child])
            st.births = m
    # environment
    food += d.food.regrowth * (d.food.capacity - food)
    if d.signal_diffusion > 0:
        spread = signal * d.signal_diffusion
        signal = (
            signal
            - spread
            + sum(np.roll(spread, s, axis=ax) for ax in (0, 1) for s in (1, -1)) / 4.0
        )
    signal *= 1.0 - d.signal_decay
    return a, food, signal, next_id, st
