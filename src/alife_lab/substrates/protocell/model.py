"""Abstract protocell model: compartments on a 2D periodic field with integer molecule counts.

Every quantity of matter is an integer count and every process moves or converts whole
molecules (binomial / Poisson draws), so the matter ledger closes *exactly*:

    field food + field waste + sum over cells (food + catalyst + membrane + waste)
        == initial matter + inflow - degraded            (checked every tick)

Per tick (fixed order):
1. Food inflow (Poisson per grid cell); decay of field food and waste (binomial).
2. Field diffusion: each molecule hops to a random 4-neighbour with ``hop_probability``.
3. Uptake: each cell takes food from the grid cells it covers, binomial with
   ``1 - exp(-permeability)`` per molecule (cells in id order).
4. Internal chemistry (tau-leap with binomials, catalyst concentration = C / area):
   lipid synthesis F + C -> C + M, replication F + C -> 2 C, maintenance F -> W at a rate
   proportional to the membrane (energy expenditure), membrane decay M -> W, catalyst decay
   C -> W.
5. Export: all internal waste leaves to the grid cell under the centre (membrane permeable to
   waste). Without a membrane (control) food and catalyst leak too (released catalyst counts as
   field waste).
6. Rupture if (food + catalyst + waste) / area exceeds ``rupture_pressure``; death if the
   membrane falls below ``min_membrane``. Contents are released into the field.
7. Division when the membrane reaches ``division_membrane``: every content (including the
   membrane) is split binomially with p = 1/2 (or a uniform random fraction, control); daughters
   inherit the parent's genome with log-normal mutation of the heritable traits.
8. Mechanics: Brownian displacement and pairwise overlap repulsion (minimum image).

Area = membrane / membrane_density, radius = sqrt(area / pi). This is an abstract model: no
membrane biophysics or chemical realism is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import ProtocellDynamics

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]
TRAITS = ("k_lipid", "k_replicate", "permeability")
CONTENTS = ("food", "catalyst", "membrane", "waste")


@dataclass
class Ledger:
    initial: int = 0
    inflow: int = 0
    degraded: int = 0
    uptake: int = 0
    released: int = 0
    lipid: int = 0
    replication: int = 0
    maintenance: int = 0
    membrane_decay: int = 0
    catalyst_decay: int = 0
    divisions: int = 0
    ruptures: int = 0
    deaths: int = 0

    def to_json(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class Cells:
    ids: I64
    x: F64
    y: F64
    contents: I64  # (n, 4): food, catalyst, membrane, waste
    genome: F64  # (n, 3): k_lipid, k_replicate, permeability

    @property
    def n(self) -> int:
        return int(self.ids.size)

    def take(self, keep: npt.NDArray[np.bool_]) -> Cells:
        return Cells(self.ids[keep], self.x[keep], self.y[keep], self.contents[keep],
                     self.genome[keep])  # fmt: skip

    @staticmethod
    def concat(parts: list[Cells]) -> Cells:
        return Cells(
            np.concatenate([p.ids for p in parts]),
            np.concatenate([p.x for p in parts]),
            np.concatenate([p.y for p in parts]),
            np.concatenate([p.contents for p in parts]).reshape(-1, 4),
            np.concatenate([p.genome for p in parts]).reshape(-1, 3),
        )


@dataclass
class World:
    food: I64  # (H, W)
    waste: I64
    cells: Cells
    ledger: Ledger
    next_id: int
    lineage: dict[str, list[int]] = field(default_factory=dict)  # id -> parent/birth/death
    recent_divisions: list[dict[str, Any]] = field(default_factory=list)

    def matter(self) -> int:
        return int(self.food.sum() + self.waste.sum() + self.cells.contents.sum())

    def residual(self) -> int:
        lg = self.ledger
        return self.matter() - (lg.initial + lg.inflow - lg.degraded)


def area(membrane: Any, density: float) -> Any:
    return np.asarray(membrane, dtype=np.float64) / density


def radius(membrane: Any, density: float) -> Any:
    return np.sqrt(area(membrane, density) / math.pi)


def _hop(grid: I64, p: float, rng: np.random.Generator) -> I64:
    if p <= 0:
        return grid
    leave = rng.binomial(grid, p)
    stay = grid - leave
    out = stay.copy()
    rest = leave
    for k, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        share = rest if k == 3 else rng.binomial(rest, 1.0 / (4 - k))
        rest = rest - share
        out += np.roll(share, (dy, dx), axis=(0, 1))
    return out


def _cell_index(x: float, y: float, h: int, w: int) -> tuple[int, int]:
    return math.floor(y) % h, math.floor(x) % w


def _covered(x: float, y: float, r: float, h: int, w: int) -> tuple[I64, I64]:
    """Grid cells whose centres lie within r of (x, y), always including the centre cell."""
    rr = math.ceil(r) + 1
    ci, cj = _cell_index(x, y, h, w)
    di, dj = np.mgrid[-rr : rr + 1, -rr : rr + 1]
    ii, jj = (ci + di) % h, (cj + dj) % w
    cy = np.floor(y) + di + 0.5
    cx = np.floor(x) + dj + 0.5
    inside = (cx - x) ** 2 + (cy - y) ** 2 <= r * r
    inside[rr, rr] = True
    return ii[inside], jj[inside]


def _release(world: World, x: float, y: float, food: int, other: int) -> None:
    h, w = world.food.shape
    i, j = _cell_index(x, y, h, w)
    world.food[i, j] += food
    world.waste[i, j] += other
    world.ledger.released += food + other


def step(world: World, d: ProtocellDynamics, tick: int, rng: np.random.Generator) -> None:
    env = d.environment
    lg = world.ledger
    h, w = world.food.shape
    # 1. inflow and decay
    add = rng.poisson(env.inflow_per_cell, size=(h, w)).astype(np.int64)
    world.food += add
    lg.inflow += int(add.sum())
    df = rng.binomial(world.food, env.food_decay)
    dw = rng.binomial(world.waste, env.waste_decay)
    world.food -= df
    world.waste -= dw
    lg.degraded += int(df.sum() + dw.sum())
    # 2. diffusion
    world.food = _hop(world.food, env.hop_probability, rng)
    world.waste = _hop(world.waste, env.hop_probability, rng)
    cells = world.cells
    if cells.n == 0:
        return
    c = cells.contents
    # 3. uptake (id order)
    r = radius(c[:, 2], d.membrane_density)
    for i in range(cells.n):
        ii, jj = _covered(float(cells.x[i]), float(cells.y[i]), float(r[i]), h, w)
        p_up = 1.0 - math.exp(-float(cells.genome[i, 2]))
        take = rng.binomial(world.food[ii, jj], p_up)
        np.subtract.at(world.food, (ii, jj), take)
        c[i, 0] += int(take.sum())
        lg.uptake += int(take.sum())
    # 4. chemistry (vectorised over cells)
    conc = c[:, 1] / np.maximum(area(c[:, 2], d.membrane_density), 1e-9)
    if d.metabolism:
        lip = rng.binomial(c[:, 0], 1.0 - np.exp(-cells.genome[:, 0] * conc))
        c[:, 0] -= lip
        c[:, 2] += lip
        rep = rng.binomial(c[:, 0], 1.0 - np.exp(-cells.genome[:, 1] * conc))
        c[:, 0] -= rep
        c[:, 1] += rep
        lg.lipid += int(lip.sum())
        lg.replication += int(rep.sum())
    maint = rng.binomial(c[:, 0], 1.0 - np.exp(-d.maintenance * c[:, 2]))
    c[:, 0] -= maint
    c[:, 3] += maint
    md = rng.binomial(c[:, 2], d.membrane_decay)
    c[:, 2] -= md
    c[:, 3] += md
    cd = rng.binomial(c[:, 1], d.catalyst_decay)
    c[:, 1] -= cd
    c[:, 3] += cd
    lg.maintenance += int(maint.sum())
    lg.membrane_decay += int(md.sum())
    lg.catalyst_decay += int(cd.sum())
    # 5. export
    for i in range(cells.n):
        food_out = int(c[i, 0]) if not d.membrane else 0
        other_out = int(c[i, 3]) + (int(c[i, 1]) if not d.membrane else 0)
        if food_out or other_out:
            _release(world, float(cells.x[i]), float(cells.y[i]), food_out, other_out)
        c[i, 3] = 0
        if not d.membrane:
            c[i, 0] = 0
            c[i, 1] = 0
    # 6. rupture and death
    a = area(c[:, 2], d.membrane_density)
    pressure = (c[:, 0] + c[:, 1] + c[:, 3]) / np.maximum(a, 1e-9)
    ruptured = pressure > d.rupture_pressure
    dead = ~ruptured & (c[:, 2] < d.min_membrane)
    for k in (int(v) for v in np.flatnonzero(ruptured | dead)):
        _release(world, float(cells.x[k]), float(cells.y[k]), int(c[k, 0]),
                 int(c[k, 1] + c[k, 2] + c[k, 3]))  # fmt: skip
        world.lineage[str(int(cells.ids[k]))][2] = tick
    lg.ruptures += int(ruptured.sum())
    lg.deaths += int(dead.sum())
    cells = cells.take(~(ruptured | dead))
    # 7. division
    if d.division and cells.n:
        divide = cells.contents[:, 2] >= d.division_membrane
        if divide.any():
            cells = _divide(world, cells, divide, d, tick, rng)
    # 8. mechanics
    world.cells = _move(cells, d, w, h, rng)


def _divide(world: World, cells: Cells, divide: Any, d: ProtocellDynamics, tick: int,
            rng: np.random.Generator) -> Cells:  # fmt: skip
    keep = cells.take(~divide)
    par = cells.take(divide)
    n = par.n
    frac = np.full(n, 0.5) if d.partition == "binomial" else rng.random(n)
    a = rng.binomial(par.contents, frac[:, None])
    b = par.contents - a
    theta = rng.random(n) * 2 * math.pi
    off = radius(par.contents[:, 2], d.membrane_density) / 2
    ids_a = np.arange(world.next_id, world.next_id + n, dtype=np.int64)
    ids_b = ids_a + n
    world.next_id += 2 * n
    genomes = []
    for _ in range(2):
        g = par.genome.copy()
        for k, trait in enumerate(TRAITS):
            if trait in d.heritable and d.mutation_sigma > 0:
                g[:, k] *= np.exp(d.mutation_sigma * rng.standard_normal(n))
        genomes.append(g)
    for i in range(n):
        parent = int(par.ids[i])
        world.lineage[str(parent)][2] = tick
        for child in (int(ids_a[i]), int(ids_b[i])):
            world.lineage[str(child)] = [parent, tick, -1]
        ca, cb = int(a[i, 1]), int(b[i, 1])
        world.recent_divisions.append({
            "tick": tick, "parent": parent, "children": [int(ids_a[i]), int(ids_b[i])],
            "catalyst": [ca, cb],
            "asymmetry": abs(ca - cb) / (ca + cb) if ca + cb else 0.0,
        })  # fmt: skip
    world.recent_divisions = world.recent_divisions[-200:]
    world.ledger.divisions += n
    da = Cells(ids_a, par.x + off * np.cos(theta), par.y + off * np.sin(theta), a, genomes[0])
    db = Cells(ids_b, par.x - off * np.cos(theta), par.y - off * np.sin(theta), b, genomes[1])
    return Cells.concat([keep, da, db])


def _move(cells: Cells, d: ProtocellDynamics, w: int, h: int, rng: np.random.Generator) -> Cells:
    n = cells.n
    if n == 0:
        return cells
    x = cells.x + d.diffusion * rng.standard_normal(n)
    y = cells.y + d.diffusion * rng.standard_normal(n)
    if n > 1 and d.repulsion > 0:
        r = radius(cells.contents[:, 2], d.membrane_density)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dx -= w * np.round(dx / w)
        dy -= h * np.round(dy / h)
        dist = np.hypot(dx, dy)
        np.fill_diagonal(dist, np.inf)
        overlap = np.maximum(0.0, r[:, None] + r[None, :] - dist)
        push = d.repulsion * overlap / 2 / np.maximum(dist, 1e-9)
        x = x + (push * dx).sum(1)
        y = y + (push * dy).sum(1)
    return Cells(cells.ids, x % w, y % h, cells.contents, cells.genome)
