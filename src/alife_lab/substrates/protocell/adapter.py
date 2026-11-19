"""``protocell`` adapter: abstract compartments with an exact matter ledger. See ``model.py``."""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, ProtocellExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.protocell import model
from alife_lab.substrates.protocell.model import Cells, Ledger, World

SERIALIZE_VERSION = 1
RENDER_SCALE = 3
MAX_ADD = 100_000


def _pc(spec: ExperimentSpec) -> ProtocellExperiment:
    exp = spec.experiment
    if not isinstance(exp, ProtocellExperiment):
        raise TypeError("protocell adapter received a different spec")
    return exp


@dataclass
class ProtoState:
    world: World
    tick: int
    rng: np.random.Generator = field(repr=False)
    density: float = 20.0  # membrane per area unit (from the spec; used for rendering)


_M = MetricDefinition
_METRICS = (
    _M(
        "population",
        "metric.population",
        "cells",
        "tick",
        "Number of protocells.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "divisions",
        "metric.divisions",
        "events",
        "tick",
        "Cumulative divisions.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "ruptures",
        "metric.ruptures",
        "events",
        "tick",
        "Cumulative ruptures (pressure).",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "deaths",
        "metric.deaths",
        "events",
        "tick",
        "Cumulative deaths (membrane too small).",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_radius",
        "metric.mean_radius",
        "cells",
        "tick",
        "Mean protocell radius.",
        "none",
        "every tick",
        1,
        "null without cells",
    ),
    _M(
        "total_matter",
        "metric.total_matter",
        "molecules",
        "tick",
        "All matter in the field and the cells.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "field_food",
        "metric.field_food",
        "molecules",
        "tick",
        "Food in the field.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "ledger_residual",
        "metric.ledger_residual",
        "molecules",
        "tick",
        "matter - (initial + inflow - degraded); must be exactly 0.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_k_replicate",
        "metric.mean_k_replicate",
        "rate",
        "tick",
        "Population mean of the heritable replication rate.",
        "none",
        "every tick",
        1,
        "null without cells",
    ),
    _M(
        "mean_permeability",
        "metric.mean_permeability",
        "rate",
        "tick",
        "Population mean of the heritable permeability.",
        "none",
        "every tick",
        1,
        "null without cells",
    ),
    _M(
        "partition_asymmetry",
        "metric.partition_asymmetry",
        "ratio",
        "tick",
        "Mean |C_a - C_b| / (C_a + C_b) of catalyst partition over the last 200 divisions.",
        "none",
        "every tick",
        1,
        "null before the first division",
    ),
)

PROTOCELL_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.protocell",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="protocell",
    display_name_key="substrate.protocell",
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
        intervention_types=("add_food", "kill", "add_cell"),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Abstract protocell model (compartment, metabolism, growth, division, "
        "inheritance) with integer molecule counts and an exact matter ledger. Exploratory only.",
        "implementation": "this repository",
    },
)


def _new_cells(exp: ProtocellExperiment, n: int, x: Any, y: Any, start_id: int) -> Cells:
    init = exp.initialization
    g = exp.dynamics.genome
    contents = np.tile(np.array([init.food, init.catalyst, init.membrane, 0], np.int64), (n, 1))
    genome = np.tile(np.array([g.k_lipid, g.k_replicate, g.permeability], np.float64), (n, 1))
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    return Cells(ids, np.asarray(x, np.float64), np.asarray(y, np.float64), contents, genome)


class ProtocellAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return PROTOCELL_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> ProtoState:
        exp = _pc(spec)
        h, w = exp.world.height, exp.world.width
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        n = exp.initialization.count
        cells = _new_cells(exp, n, rng.random(n) * w, rng.random(n) * h, 0)
        food = np.full((h, w), exp.dynamics.environment.initial_food_per_cell, np.int64)
        world = World(
            food=food,
            waste=np.zeros((h, w), np.int64),
            cells=cells,
            ledger=Ledger(),
            next_id=n,
            lineage={str(i): [-1, 0, -1] for i in range(n)},
        )
        world.ledger.initial = world.matter()
        return ProtoState(world=world, tick=0, rng=rng, density=exp.dynamics.membrane_density)

    def step(self, state: ProtoState, spec: ExperimentSpec) -> ProtoState:
        exp = _pc(spec)
        model.step(state.world, exp.dynamics, state.tick + 1, state.rng)
        state.tick += 1
        if state.world.cells.n > 4000:
            raise RuntimeError("protocell population exceeded 4000 (bounded by design)")
        return state

    def metrics(self, state: ProtoState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _pc(spec)
        wanted = set(exp.observations.metrics)
        wd = state.world
        c = wd.cells
        lg = wd.ledger
        n = c.n
        divs = wd.recent_divisions
        values = {
            "population": float(n),
            "divisions": float(lg.divisions),
            "ruptures": float(lg.ruptures),
            "deaths": float(lg.deaths),
            "mean_radius": float(
                model.radius(c.contents[:, 2], exp.dynamics.membrane_density).mean()
            )
            if n
            else math.nan,
            "total_matter": float(wd.matter()),
            "field_food": float(wd.food.sum()),
            "ledger_residual": float(wd.residual()),
            "mean_k_replicate": float(c.genome[:, 1].mean()) if n else math.nan,
            "mean_permeability": float(c.genome[:, 2].mean()) if n else math.nan,
            "partition_asymmetry": float(np.mean([d["asymmetry"] for d in divs]))
            if divs
            else math.nan,
        }
        return {k: v for k, v in values.items() if k in wanted}

    def render(self, state: ProtoState) -> tuple[dict[str, Any], bytes]:
        wd = state.world
        h, w = wd.food.shape
        s = RENDER_SCALE
        top = max(1.0, float(np.percentile(wd.food, 99)))
        bg = np.clip(wd.food / top, 0, 1)
        img = np.zeros((h * s, w * s, 3), np.float32)
        img[..., 1] = np.kron(bg, np.ones((s, s))) * 0.35
        img[..., 2] = np.kron(np.clip(wd.waste / max(1.0, top), 0, 1), np.ones((s, s))) * 0.25
        c = wd.cells
        yy, xx = np.mgrid[0 : h * s, 0 : w * s]
        for i in range(c.n):
            area = c.contents[i, 2] / state.density
            r = math.sqrt(area / math.pi) * s
            cx, cy = c.x[i] * s, c.y[i] * s
            conc = c.contents[i, 1] / max(area, 1e-9)
            hue = 0.55 - 0.55 * min(1.0, conc / 20.0)
            col = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            dx = (xx - cx + w * s / 2) % (w * s) - w * s / 2
            dy = (yy - cy + h * s / 2) % (h * s) - h * s / 2
            d2 = dx * dx + dy * dy
            inner = d2 <= (r - 1) ** 2
            ring = (d2 <= r * r) & ~inner
            img[inner] = np.asarray(col, np.float32) * 0.75
            img[ring] = 1.0
        data = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes()
        return {
            "codec": "field_u8",
            "width": w * s,
            "height": h * s,
            "channels": 3,
            "scale": "food (green), waste (blue), cells coloured by catalyst",
        }, data

    def apply_intervention(
        self, state: ProtoState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[ProtoState, dict[str, Any]]:
        exp = _pc(spec)
        wd = state.world
        h, w = wd.food.shape
        try:
            x, y = float(payload["x"]), float(payload["y"])
            radius = float(payload.get("radius", 3))
            count = int(payload.get("count", 200))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs x, y (optional radius, count)") from exc
        if not (0 <= x < w and 0 <= y < h) or not 0 <= radius <= 64:
            raise ValueError("position or radius out of range")
        if not 0 <= count <= MAX_ADD:
            raise ValueError(f"count must be within 0..{MAX_ADD}")
        if kind == "add_food":  # external input: booked as inflow so the ledger stays exact
            i, j = int(y) % h, int(x) % w
            wd.food[i, j] += count
            wd.ledger.inflow += count
            return state, {"added": count}
        if kind == "kill":
            c = wd.cells
            dx = (c.x - x + w / 2) % w - w / 2
            dy = (c.y - y + h / 2) % h - h / 2
            hit = dx * dx + dy * dy <= radius * radius
            for k in (int(v) for v in np.flatnonzero(hit)):
                model._release(
                    wd,
                    float(c.x[k]),
                    float(c.y[k]),
                    int(c.contents[k, 0]),
                    int(c.contents[k, 1:].sum()),
                )
                wd.lineage[str(int(c.ids[k]))][2] = state.tick
            wd.cells = c.take(~hit)
            return state, {"killed": int(hit.sum())}
        if kind == "add_cell":  # new matter from outside: booked as inflow
            new = _new_cells(exp, 1, [x], [y], wd.next_id)
            wd.lineage[str(wd.next_id)] = [-1, state.tick, -1]
            wd.next_id += 1
            wd.cells = Cells.concat([wd.cells, new])
            wd.ledger.inflow += int(new.contents.sum())
            return state, {"id": int(new.ids[0])}
        raise ValueError(f"unsupported intervention kind {kind!r}")

    def inspect(
        self, state: ProtoState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _pc(spec)
        wd = state.world
        h, w = wd.food.shape
        x, y = float(query["x"]), float(query["y"])
        c = wd.cells
        nearest: dict[str, Any] | None = None
        if c.n:
            dx = (c.x - x + w / 2) % w - w / 2
            dy = (c.y - y + h / 2) % h - h / 2
            k = int(np.argmin(dx * dx + dy * dy))
            cid = int(c.ids[k])
            ancestors: list[int] = []
            cur = wd.lineage.get(str(cid), [-1, 0, -1])[0]
            while cur >= 0 and len(ancestors) < 20:
                ancestors.append(cur)
                cur = wd.lineage.get(str(cur), [-1, 0, -1])[0]
            area = float(c.contents[k, 2]) / exp.dynamics.membrane_density
            nearest = {
                "id": cid,
                "birth_tick": wd.lineage.get(str(cid), [-1, 0, -1])[1],
                "ancestors": ancestors,
                "contents": dict(zip(model.CONTENTS, (int(v) for v in c.contents[k]), strict=True)),
                "genome": dict(zip(model.TRAITS, (float(v) for v in c.genome[k]), strict=True)),
                "radius": math.sqrt(area / math.pi),
                "pressure": float(c.contents[k, [0, 1, 3]].sum()) / max(area, 1e-9),
                "distance": math.hypot(float(dx[k]), float(dy[k])),
            }
        return {
            "substrate": "protocell",
            "tick": state.tick,
            "query": {"x": x, "y": y},
            "field": {
                "food": int(wd.food[int(y) % h, int(x) % w]),
                "waste": int(wd.waste[int(y) % h, int(x) % w]),
            },
            "protocell": nearest,
            "ledger": wd.ledger.to_json(),
            "ledger_residual": wd.residual(),
            "recent_divisions": wd.recent_divisions[-5:],
        }

    def state_distance(self, a: ProtoState, b: ProtoState) -> dict[str, Any]:
        same = (
            np.array_equal(a.world.food, b.world.food)
            and np.array_equal(a.world.cells.ids, b.world.cells.ids)
            and np.array_equal(a.world.cells.contents, b.world.cells.contents)
            and np.array_equal(a.world.cells.x, b.world.cells.x)
        )
        return {
            "metric": "population_difference",
            "value": abs(a.world.cells.n - b.world.cells.n),
            "exact_equal": bool(same),
        }

    def serialize_state(self, state: ProtoState) -> dict[str, Any]:
        wd = state.world
        c = wd.cells
        lin_ids = np.array([int(k) for k in wd.lineage], np.int64)
        lin = np.array([wd.lineage[str(i)] for i in lin_ids], np.int64).reshape(-1, 3)
        return {
            "arrays": {
                "food": wd.food,
                "waste": wd.waste,
                "ids": c.ids,
                "x": c.x,
                "y": c.y,
                "contents": c.contents,
                "genome": c.genome,
                "lineage_ids": lin_ids,
                "lineage": lin,
            },
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "next_id": wd.next_id,
                "ledger": wd.ledger.to_json(),
                "recent_divisions": wd.recent_divisions,
                "rng_state": state.rng.bit_generator.state,
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> ProtoState:
        exp = _pc(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible protocell checkpoint serialize_version")
        a = payload["arrays"]
        food = np.asarray(a["food"], np.int64).copy()
        if food.shape != (exp.world.height, exp.world.width):
            raise ValueError("protocell checkpoint field does not match the spec")
        cells = Cells(
            np.asarray(a["ids"], np.int64).copy(),
            np.asarray(a["x"], np.float64).copy(),
            np.asarray(a["y"], np.float64).copy(),
            np.asarray(a["contents"], np.int64).copy().reshape(-1, 4),
            np.asarray(a["genome"], np.float64).copy().reshape(-1, 3),
        )
        if (cells.contents < 0).any() or (food < 0).any():
            raise ValueError("negative molecule counts in checkpoint")
        lineage = {
            str(int(i)): [int(v) for v in row]
            for i, row in zip(a["lineage_ids"], a["lineage"], strict=True)
        }
        world = World(
            food=food,
            waste=np.asarray(a["waste"], np.int64).copy(),
            cells=cells,
            ledger=Ledger(**meta["ledger"]),
            next_id=int(meta["next_id"]),
            lineage=lineage,
            recent_divisions=list(meta["recent_divisions"]),
        )
        if world.residual() != 0:
            raise ValueError("protocell checkpoint fails the matter ledger")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        return ProtoState(
            world=world, tick=int(meta["tick"]), rng=rng, density=exp.dynamics.membrane_density
        )
