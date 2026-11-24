"""Safe transmissible procedural skills (``SkillProcedure``) and Knowledge DAG registry.

No Python ``eval``/``exec`` is permitted anywhere. Skills are structured finite operation
sequences from a whitelisted instruction set with explicit provenance and lineage tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ALLOWED_OPS: tuple[str, ...] = (
    "locate",
    "gather",
    "process_a",
    "process_b",
    "extract",
    "store",
)
MAX_EVENT_LOG = 512


@dataclass(frozen=True)
class SkillProcedure:
    """Immutable procedural technique in the cumulative-culture DAG."""

    skill_id: int
    parent_skill_id: int
    revision: int
    inventor_id: int
    origin_tick: int
    depth: int
    target_resource: int  # 1 = material_a, 2 = material_b
    ops: tuple[str, ...]
    phase: float  # tuned environmental regime phase in [0, 1)
    efficiency: float  # intrinsic craft multiplier [0.5, 2.6]

    def __post_init__(self) -> None:
        if any(op not in ALLOWED_OPS for op in self.ops):
            raise ValueError(f"disallowed operation in skill ops {self.ops!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "parent_skill_id": self.parent_skill_id,
            "revision": self.revision,
            "inventor_id": self.inventor_id,
            "origin_tick": self.origin_tick,
            "depth": self.depth,
            "target_resource": self.target_resource,
            "ops": list(self.ops),
            "phase": self.phase,
            "efficiency": self.efficiency,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SkillProcedure:
        return cls(
            skill_id=int(data["skill_id"]),
            parent_skill_id=int(data["parent_skill_id"]),
            revision=int(data["revision"]),
            inventor_id=int(data["inventor_id"]),
            origin_tick=int(data["origin_tick"]),
            depth=int(data["depth"]),
            target_resource=int(data["target_resource"]),
            ops=tuple(str(op) for op in data["ops"]),
            phase=float(data["phase"]),
            efficiency=float(data["efficiency"]),
        )


@dataclass
class SkillSlot:
    """Agent-local possession & origin metadata for one acquired skill."""

    skill_id: int
    teacher_id: int  # -1 if self-invented/self-improved; >= 0 if socially received
    mode: str  # "invent" | "improve" | "vertical" | "horizontal" | "oblique"
    acquired_tick: int
    uses: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "teacher_id": self.teacher_id,
            "mode": self.mode,
            "acquired_tick": self.acquired_tick,
            "uses": self.uses,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SkillSlot:
        return cls(
            skill_id=int(data["skill_id"]),
            teacher_id=int(data["teacher_id"]),
            mode=str(data["mode"]),
            acquired_tick=int(data["acquired_tick"]),
            uses=int(data.get("uses", 0)),
        )


@dataclass
class SkillRegistry:
    """Global knowledge ledger recording procedures, inventor mortality, and transfers."""

    procedures: dict[int, SkillProcedure] = field(default_factory=dict)
    inventor_death_tick: dict[int, int] = field(default_factory=dict)
    transmissions: list[dict[str, Any]] = field(default_factory=list)

    def register(self, proc: SkillProcedure) -> SkillProcedure:
        self.procedures[proc.skill_id] = proc
        return proc

    def mark_agent_died(self, agent_id: int, tick: int) -> None:
        self.inventor_death_tick[agent_id] = tick

    def is_inventor_dead(self, skill_id: int) -> bool:
        proc = self.procedures.get(skill_id)
        if proc is None:
            return False
        return proc.inventor_id in self.inventor_death_tick

    def log_transmission(
        self,
        *,
        teacher_id: int,
        learner_id: int,
        skill_id: int,
        revision: int,
        origin_seed: int,
        tick: int,
        mode: str,
        success: bool,
    ) -> None:
        self.transmissions.append(
            {
                "teacher_id": int(teacher_id),
                "learner_id": int(learner_id),
                "skill_id": int(skill_id),
                "revision": int(revision),
                "origin_seed": int(origin_seed),
                "tick": int(tick),
                "transmission_mode": mode,
                "success": bool(success),
            }
        )
        if len(self.transmissions) > MAX_EVENT_LOG:
            del self.transmissions[:-MAX_EVENT_LOG]

    def to_json(self) -> dict[str, Any]:
        return {
            "procedures": {str(k): v.to_json() for k, v in self.procedures.items()},
            "inventor_death_tick": {str(k): int(v) for k, v in self.inventor_death_tick.items()},
            "transmissions": list(self.transmissions),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SkillRegistry:
        return cls(
            procedures={int(k): SkillProcedure.from_json(v) for k, v in data["procedures"].items()},
            inventor_death_tick={int(k): int(v) for k, v in data["inventor_death_tick"].items()},
            transmissions=list(data.get("transmissions", [])),
        )


def _ops_efficiency(ops: tuple[str, ...], target_resource: int) -> float:
    """Intrinsic efficiency of an operation sequence (rewards coherent target pipelines)."""
    proc_op = "process_a" if target_resource == 1 else "process_b"
    score = 0.65
    if proc_op in ops:
        score += 0.35
    if "extract" in ops:
        score += 0.30
    if "locate" in ops or "gather" in ops:
        score += 0.15
    if "store" in ops:
        score += 0.15
    # Diversity of distinct valid ops without bloat:
    unique_bonus = 0.12 * len(set(ops))
    bloat_penalty = 0.08 * max(0, len(ops) - len(set(ops)))
    return float(max(0.5, min(2.6, score + unique_bonus - bloat_penalty)))


def evaluate_skill_utility(proc: SkillProcedure, env_phase: float) -> float:
    """Gross energy yield per 1.0 unit of target material consumed at ``env_phase``."""
    phase_match = 0.5 + 0.5 * math.cos(2.0 * math.pi * (proc.phase - env_phase))
    return float(1.45 * proc.efficiency * (0.45 + 0.55 * phase_match))


def invent_skill(
    *,
    skill_id: int,
    inventor_id: int,
    tick: int,
    target_resource: int,
    env_phase: float,
    max_len: int,
    rng_tech: np.random.Generator,
) -> SkillProcedure:
    """Synthesize a novel base procedure via individual trial and error (depth 0)."""
    length = int(rng_tech.integers(2, max(3, min(max_len, 4) + 1)))
    ops_idx = rng_tech.integers(0, len(ALLOWED_OPS), size=length)
    ops = tuple(ALLOWED_OPS[int(i)] for i in ops_idx)
    phase = float((env_phase + float(rng_tech.normal(0.0, 0.12))) % 1.0)
    eff = _ops_efficiency(ops, target_resource)
    return SkillProcedure(
        skill_id=skill_id,
        parent_skill_id=-1,
        revision=0,
        inventor_id=inventor_id,
        origin_tick=tick,
        depth=0,
        target_resource=target_resource,
        ops=ops,
        phase=phase,
        efficiency=eff,
    )


def improve_skill(
    parent: SkillProcedure,
    *,
    skill_id: int,
    improver_id: int,
    tick: int,
    env_phase: float,
    max_len: int,
    rng_tech: np.random.Generator,
    preserve_phase: bool = False,
) -> SkillProcedure:
    """Mutate/compose a parent procedure into a child revision (depth + 1)."""
    ops_list = list(parent.ops)
    proc_op = "process_a" if parent.target_resource == 1 else "process_b"
    roll = float(rng_tech.random())
    if roll < 0.45 and len(ops_list) < max_len:
        # Insert missing high-value step if not present
        candidates = [op for op in (proc_op, "extract", "store", "locate") if op not in ops_list]
        op_new = (
            candidates[0]
            if candidates
            else ALLOWED_OPS[int(rng_tech.integers(0, len(ALLOWED_OPS)))]
        )
        ops_list.append(op_new)
    elif roll < 0.80 and ops_list:
        pos = int(rng_tech.integers(0, len(ops_list)))
        ops_list[pos] = ALLOWED_OPS[int(rng_tech.integers(0, len(ALLOWED_OPS)))]
    elif len(ops_list) > 2:
        ops_list.pop(int(rng_tech.integers(0, len(ops_list))))

    ops = tuple(ops_list)
    base_eff = _ops_efficiency(ops, parent.target_resource)
    # Keep improvements bounded; preserve phase on exact/aligned refinement unless shifted.
    if preserve_phase:
        phase = float((parent.phase + float(rng_tech.normal(0.0, 0.04))) % 1.0)
    else:
        # Partial hill-climb toward current environmental phase + small noise
        diff = ((env_phase - parent.phase + 0.5) % 1.0) - 0.5
        phase = float((parent.phase + 0.35 * diff + float(rng_tech.normal(0.0, 0.03))) % 1.0)
    eff = float(max(0.5, min(2.6, 0.45 * parent.efficiency + 0.55 * base_eff + 0.06)))
    return SkillProcedure(
        skill_id=skill_id,
        parent_skill_id=parent.skill_id,
        revision=parent.revision + 1,
        inventor_id=improver_id,
        origin_tick=tick,
        depth=parent.depth + 1,
        target_resource=parent.target_resource,
        ops=ops,
        phase=phase,
        efficiency=eff,
    )
