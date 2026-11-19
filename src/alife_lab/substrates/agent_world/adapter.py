"""``agent_world`` adapter: autonomous runs of sensing agents (see ``model.py``)."""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import AgentWorldExperiment, ExperimentSpec
from alife_lab.substrates.agent_world import model
from alife_lab.substrates.agent_world.model import Agents, TickStats
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
RENDER_SCALE = 6


def _aw(spec: ExperimentSpec) -> AgentWorldExperiment:
    exp = spec.experiment
    if not isinstance(exp, AgentWorldExperiment):
        raise TypeError("agent_world adapter received a different spec")
    return exp


@dataclass
class AWState:
    agents: Agents
    food: np.ndarray[Any, Any]
    signal: np.ndarray[Any, Any]
    tick: int
    rng: np.random.Generator = field(repr=False)
    next_id: int = 0
    births: int = 0
    deaths: int = 0
    last: TickStats = field(default_factory=TickStats)
    world: tuple[float, float] = (100.0, 100.0)


_M = MetricDefinition
_METRICS = (
    _M(
        "population",
        "metric.population",
        "agents",
        "tick",
        "Number of agents.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_energy",
        "metric.mean_energy",
        "energy",
        "tick",
        "Mean agent energy.",
        "none",
        "every tick",
        1,
        "null without agents",
    ),
    _M(
        "births",
        "metric.births",
        "agents",
        "tick",
        "Cumulative births.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "deaths",
        "metric.deaths",
        "agents",
        "tick",
        "Cumulative deaths.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "food_total",
        "metric.food_total",
        "food",
        "tick",
        "Food on the grid.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_age",
        "metric.mean_age",
        "ticks",
        "tick",
        "Mean agent age.",
        "none",
        "every tick",
        1,
        "null without agents",
    ),
    _M(
        "signal_rate",
        "metric.signal_rate",
        "fraction",
        "tick",
        "Fraction of agents that emitted a signal in the last tick.",
        "none",
        "every tick",
        1,
        "null without agents",
    ),
    _M(
        "forward_rate",
        "metric.forward_rate",
        "fraction",
        "tick",
        "Fraction of agents that moved straight in the last tick.",
        "none",
        "every tick",
        1,
        "null without agents",
    ),
    _M(
        "mean_reward",
        "metric.mean_reward",
        "energy",
        "tick",
        "Mean change of the agents' own energy in the last tick (internal, not a fitness).",
        "none",
        "every tick",
        1,
        "null without agents",
    ),
    _M(
        "genome_spread",
        "metric.genome_spread",
        "weight std",
        "tick",
        "Mean across weights of the population std of the genetic weights (diversity).",
        "none",
        "every tick",
        1,
        "null with fewer than 2 agents",
    ),
)

AGENT_WORLD_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.agent_world",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="agent_world",
    display_name_key="substrate.agent_world",
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
        intervention_types=("add_food", "remove"),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Project AgentWorld: 2D sensing agents with hand-coded, random, evolved "
        "or lifetime-learning (REINFORCE) controllers; autonomous run without external fitness.",
        "implementation": "this repository",
    },
)


class AgentWorldAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return AGENT_WORLD_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> AWState:
        exp = _aw(spec)
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        r = exp.world.resolution
        n = exp.initialization.count
        agents = model.init_agents(exp, n, rng)
        food = np.full((r, r), exp.dynamics.food.initial * exp.dynamics.food.capacity)
        return AWState(
            agents=agents,
            food=food,
            signal=np.zeros((r, r)),
            tick=0,
            rng=rng,
            next_id=n,
            world=(exp.world.width, exp.world.height),
        )

    def step(self, state: AWState, spec: ExperimentSpec) -> AWState:
        exp = _aw(spec)
        state.tick += 1
        state.agents, state.food, state.signal, state.next_id, st = model.step(
            state.agents, state.food, state.signal, exp, state.tick, state.rng, state.next_id
        )
        state.births += st.births
        state.deaths += st.deaths
        state.last = st
        return state

    def metrics(self, state: AWState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _aw(spec)
        wanted = set(exp.observations.metrics)
        a = state.agents
        n = a.n
        st = state.last
        acting = max(1, st.agents)
        spread = math.nan
        if n >= 2:
            flat = np.concatenate([getattr(a, k).reshape(n, -1) for k in model.GENOME_KEYS], 1)
            spread = float(flat.std(axis=0).mean())
        values = {
            "population": float(n),
            "mean_energy": float(a.energy.mean()) if n else math.nan,
            "births": float(state.births),
            "deaths": float(state.deaths),
            "food_total": float(state.food.sum()),
            "mean_age": float(a.age.mean()) if n else math.nan,
            "signal_rate": st.emitted / acting if st.agents else math.nan,
            "forward_rate": st.forward / acting if st.agents else math.nan,
            "mean_reward": st.reward_sum / acting if st.agents else math.nan,
            "genome_spread": spread,
        }
        return {k: v for k, v in values.items() if k in wanted}

    def render(self, state: AWState) -> tuple[dict[str, Any], bytes]:
        s = RENDER_SCALE
        r = state.food.shape[0]
        w, h = state.world
        img = np.zeros((r * s, r * s, 3), np.float32)
        img[..., 1] = np.kron(np.clip(state.food, 0, 1), np.ones((s, s))) * 0.5
        img[..., 2] = np.kron(np.clip(state.signal / 2.0, 0, 1), np.ones((s, s)))
        a = state.agents
        px = (a.x / w * r * s).astype(np.int64) % (r * s)
        py = (a.y / h * r * s).astype(np.int64) % (r * s)
        for k in range(a.n):
            col = colorsys.hsv_to_rgb(float(a.heading[k]) / (2 * math.pi), 0.8, 1.0)
            img[max(0, py[k] - 1) : py[k] + 2, max(0, px[k] - 1) : px[k] + 2] = col
        data = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes()
        return {
            "codec": "field_u8",
            "width": r * s,
            "height": r * s,
            "channels": 3,
            "scale": "food (green), signal (blue), agents coloured by heading",
        }, data

    def apply_intervention(
        self, state: AWState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[AWState, dict[str, Any]]:
        exp = _aw(spec)
        w, h = state.world
        try:
            x, y = float(payload["x"]), float(payload["y"])
            radius = float(payload.get("radius", 5))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs x, y (optional radius)") from exc
        if not (0 <= x < w and 0 <= y < h) or not 0 <= radius <= 512:
            raise ValueError("position or radius out of range")
        a = state.agents
        if kind == "add_food":
            r = state.food.shape[0]
            cy, cx = (np.arange(r) + 0.5) * h / r, (np.arange(r) + 0.5) * w / r
            dy = (cy[:, None] - y + h / 2) % h - h / 2
            dx = (cx[None, :] - x + w / 2) % w - w / 2
            mask = dx * dx + dy * dy <= radius * radius
            state.food[mask] = exp.dynamics.food.capacity
            return state, {"cells": int(mask.sum())}
        if kind == "remove":
            dx = (a.x - x + w / 2) % w - w / 2
            dy = (a.y - y + h / 2) % h - h / 2
            hit = dx * dx + dy * dy <= radius * radius
            state.agents = a.take(~hit)
            state.deaths += int(hit.sum())
            return state, {"removed": int(hit.sum())}
        raise ValueError(f"unsupported intervention kind {kind!r}")

    def inspect(
        self, state: AWState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _aw(spec)
        w, h = state.world
        x, y = float(query["x"]), float(query["y"])
        a = state.agents
        agent: dict[str, Any] | None = None
        if a.n:
            dx = (a.x - x + w / 2) % w - w / 2
            dy = (a.y - y + h / 2) % h - h / 2
            k = int(np.argmin(dx * dx + dy * dy))
            hid = exp.dynamics.controller.hidden
            hist = a.hist[k]  # rows before the agent's birth are not shown
            rows = hist[-min(int(a.age[k]), hist.shape[0]) :] if a.age[k] else hist[:0]
            agent = {
                "id": int(a.ids[k]),
                "parent": int(a.parent[k]),
                "birth_tick": int(a.birth[k]),
                "age": int(a.age[k]),
                "energy": float(a.energy[k]),
                "heading_deg": math.degrees(float(a.heading[k])),
                "inputs": list(model.INPUTS),
                "actions": list(model.ACTIONS),
                "history": {
                    "inputs": rows[:, : model.N_IN].round(4).tolist(),
                    "hidden": rows[:, model.N_IN : model.N_IN + hid].round(4).tolist(),
                    "probs": rows[:, model.N_IN + hid : model.N_IN + hid + model.N_ACT]
                    .round(4)
                    .tolist(),
                    "action": rows[:, -2].astype(int).tolist(),
                    "reward": rows[:, -1].round(5).tolist(),
                },
                "learned_delta_norm": float(np.linalg.norm(a.dw2[k])),
            }
        return {
            "substrate": "agent_world",
            "tick": state.tick,
            "query": {"x": x, "y": y},
            "agent": agent,
            "population": a.n,
        }

    def state_distance(self, a: AWState, b: AWState) -> dict[str, Any]:
        same = (
            np.array_equal(a.agents.ids, b.agents.ids)
            and np.array_equal(a.agents.x, b.agents.x)
            and np.array_equal(a.food, b.food)
        )
        return {
            "metric": "population_difference",
            "value": abs(a.agents.n - b.agents.n),
            "exact_equal": bool(same),
        }

    def serialize_state(self, state: AWState) -> dict[str, Any]:
        arrays = {f"a.{k}": v for k, v in state.agents.fields().items()}
        arrays.update({"food": state.food, "signal": state.signal})
        st = state.last
        return {
            "arrays": arrays,
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "next_id": state.next_id,
                "births": state.births,
                "deaths": state.deaths,
                "last": dict(st.__dict__),
                "rng_state": state.rng.bit_generator.state,
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> AWState:
        exp = _aw(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible agent_world checkpoint serialize_version")
        arr = payload["arrays"]
        fields = {k[2:]: np.asarray(v).copy() for k, v in arr.items() if k.startswith("a.")}
        agents = Agents(**fields)
        food = np.asarray(arr["food"], np.float64).copy()
        r = exp.world.resolution
        if food.shape != (r, r) or agents.w1.shape[1:] != (
            model.N_IN,
            exp.dynamics.controller.hidden,
        ):
            raise ValueError("agent_world checkpoint does not match the spec")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        return AWState(
            agents=agents,
            food=food,
            signal=np.asarray(arr["signal"], np.float64).copy(),
            tick=int(meta["tick"]),
            rng=rng,
            next_id=int(meta["next_id"]),
            births=int(meta["births"]),
            deaths=int(meta["deaths"]),
            last=TickStats(**meta.get("last", {})),
            world=(exp.world.width, exp.world.height),
        )
