"""Agent container enforcing strict 3-layer separation: Genotype vs Lifetime vs Culture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.memory import SkillSlot

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

N_IN = 12
N_ACT = 6
INPUT_NAMES = (
    "food_left",
    "food_front",
    "food_right",
    "mat_a",
    "mat_b",
    "inv_a",
    "inv_b",
    "energy_norm",
    "has_skill",
    "regime_sin",
    "regime_cos",
    "bias",
)
ACTION_NAMES = ("left", "forward", "right", "gather", "apply_skill", "teach")
GENOME_ARRAY_KEYS = ("w1", "b1", "w2", "b2", "gene_lr", "gene_social")
LIFETIME_ARRAY_KEYS = (
    "energy",
    "inv_a",
    "inv_b",
    "dw2",
    "rbase",
    "private_best_x",
    "private_best_y",
    "private_best_val",
)
IDENTITY_ARRAY_KEYS = ("ids", "parent", "birth", "age", "x", "y", "heading")


@dataclass
class CulturalAgents:
    """SoA agents with explicit separation between heritable genome, lifetime state, and skills."""

    # Identity & kinematics
    ids: I64
    parent: I64
    birth: I64
    age: I64
    x: F64
    y: F64
    heading: F64
    # Genotype layer (copied / mutated on birth when genetic_evolution=True)
    w1: F64  # (n, N_IN, H)
    b1: F64  # (n, H)
    w2: F64  # (n, H, N_ACT)
    b2: F64  # (n, N_ACT)
    gene_lr: F64  # (n,)
    gene_social: F64  # (n,)
    # Lifetime layer (NEVER inherited by offspring)
    energy: F64
    inv_a: F64
    inv_b: F64
    dw2: F64  # (n, H, N_ACT)
    rbase: F64
    private_best_x: F64
    private_best_y: F64
    private_best_val: F64
    # Cultural layer (NEVER touched by genome mutation or lifetime dw2 learning)
    skills: list[list[SkillSlot]]

    @property
    def n(self) -> int:
        return int(self.ids.size)

    def array_fields(self) -> dict[str, npt.NDArray[Any]]:
        return {
            "ids": self.ids,
            "parent": self.parent,
            "birth": self.birth,
            "age": self.age,
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "gene_lr": self.gene_lr,
            "gene_social": self.gene_social,
            "energy": self.energy,
            "inv_a": self.inv_a,
            "inv_b": self.inv_b,
            "dw2": self.dw2,
            "rbase": self.rbase,
            "private_best_x": self.private_best_x,
            "private_best_y": self.private_best_y,
            "private_best_val": self.private_best_val,
        }

    def take(self, idx: Any) -> CulturalAgents:
        arrs = {k: v[idx] for k, v in self.array_fields().items()}
        if isinstance(idx, np.ndarray) and idx.dtype == np.bool_:
            sel = np.flatnonzero(idx).tolist()
        else:
            sel = np.asarray(idx, dtype=np.int64).tolist()
        skills_sub = [[SkillSlot(**sl.to_json()) for sl in self.skills[int(i)]] for i in sel]
        return CulturalAgents(**arrs, skills=skills_sub)

    @staticmethod
    def concat(parts: list[CulturalAgents]) -> CulturalAgents:
        keys = parts[0].array_fields().keys()
        arrs = {k: np.concatenate([p.array_fields()[k] for p in parts]) for k in keys}
        skills = [slot_list for p in parts for slot_list in p.skills]
        return CulturalAgents(**arrs, skills=skills)


def init_agents(
    exp: CulturalWorldExperiment, rng_birth: np.random.Generator, *, start_id: int = 0
) -> CulturalAgents:
    n = exp.initialization.count
    h = exp.dynamics.controller.hidden
    e0 = exp.dynamics.initial_energy
    return CulturalAgents(
        ids=np.arange(start_id, start_id + n, dtype=np.int64),
        parent=np.full(n, -1, dtype=np.int64),
        birth=np.zeros(n, dtype=np.int64),
        age=np.zeros(n, dtype=np.int64),
        x=rng_birth.random(n) * exp.world.width,
        y=rng_birth.random(n) * exp.world.height,
        heading=rng_birth.random(n) * 2.0 * math.pi,
        w1=rng_birth.standard_normal((n, N_IN, h)) / math.sqrt(N_IN),
        b1=np.zeros((n, h), dtype=np.float64),
        w2=rng_birth.standard_normal((n, h, N_ACT)) / math.sqrt(h),
        b2=np.zeros((n, N_ACT), dtype=np.float64),
        gene_lr=np.full(n, exp.dynamics.controller.learning_rate, dtype=np.float64),
        gene_social=np.full(n, 0.5, dtype=np.float64),
        energy=np.full(n, e0, dtype=np.float64),
        inv_a=np.zeros(n, dtype=np.float64),
        inv_b=np.zeros(n, dtype=np.float64),
        dw2=np.zeros((n, h, N_ACT), dtype=np.float64),
        rbase=np.zeros(n, dtype=np.float64),
        private_best_x=np.zeros(n, dtype=np.float64),
        private_best_y=np.zeros(n, dtype=np.float64),
        private_best_val=np.zeros(n, dtype=np.float64),
        skills=[[] for _ in range(n)],
    )


def make_offspring(
    parents: CulturalAgents,
    parent_indices: I64,
    *,
    tick: int,
    next_id: int,
    exp: CulturalWorldExperiment,
    rng_birth: np.random.Generator,
) -> tuple[CulturalAgents, int]:
    """Build newborn agents inheriting genotype only (lifetime & culture start zeroed)."""
    m = int(parent_indices.size)
    sub = parents.take(parent_indices)
    sub.ids = np.arange(next_id, next_id + m, dtype=np.int64)
    next_id += m
    sub.parent = parents.ids[parent_indices].copy()
    sub.birth = np.full(m, tick, dtype=np.int64)
    sub.age = np.zeros(m, dtype=np.int64)
    sub.heading = rng_birth.random(m) * 2.0 * math.pi
    sub.x = (sub.x + rng_birth.standard_normal(m) * 0.6) % exp.world.width
    sub.y = (sub.y + rng_birth.standard_normal(m) * 0.6) % exp.world.height

    # Lifetime state is NEVER inherited:
    sub.inv_a = np.zeros(m, dtype=np.float64)
    sub.inv_b = np.zeros(m, dtype=np.float64)
    sub.dw2 = np.zeros_like(sub.dw2)
    sub.rbase = np.zeros(m, dtype=np.float64)
    sub.private_best_x = np.zeros(m, dtype=np.float64)
    sub.private_best_y = np.zeros(m, dtype=np.float64)
    sub.private_best_val = np.zeros(m, dtype=np.float64)
    # Cultural skills start empty; only populated later if vertical transmission is enabled!
    sub.skills = [[] for _ in range(m)]

    # Mutate genotype weights & evolvable traits only when genetic_evolution=True
    sig = exp.dynamics.controller.mutation_sigma
    if exp.dynamics.genetic_evolution and sig > 0.0:
        for key in ("w1", "b1", "w2", "b2"):
            arr = getattr(sub, key)
            setattr(sub, key, arr + sig * rng_birth.standard_normal(arr.shape))
        sub.gene_lr = np.clip(sub.gene_lr + 0.01 * rng_birth.standard_normal(m), 0.0, 1.0)
        sub.gene_social = np.clip(sub.gene_social + 0.05 * rng_birth.standard_normal(m), 0.0, 1.0)
    return sub, next_id
