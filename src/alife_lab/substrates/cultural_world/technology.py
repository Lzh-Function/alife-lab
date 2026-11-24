"""Stage 3 / P4: Safe compositional skill DSL & externalized physical artifacts (ADR 0052).

No Python ``eval``/``exec`` or pre-baked tech trees are allowed anywhere. All skill
procedures are finite sequences over whitelisted operations evaluated strictly by
``evaluate_pipeline``. Physical ``PlacedArtifact`` objects hold conserved material mass
on the 2D arena and enable asynchronous cross-generation learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from alife_lab.core.spec import CulturalWorldExperiment
    from alife_lab.substrates.cultural_world.agents import CulturalAgents
    from alife_lab.substrates.cultural_world.ecology import ResourceField
    from alife_lab.substrates.cultural_world.memory import SkillProcedure, SkillRegistry

P3_BASE_OPS: tuple[str, ...] = (
    "locate",
    "gather",
    "process_a",
    "process_b",
    "extract",
    "store",
)
COMPOSITION_OPS: tuple[str, ...] = ("combine", "transform")
EXTERNAL_OPS: tuple[str, ...] = ("place_object", "use_object")
ALLOWED_OPS: tuple[str, ...] = P3_BASE_OPS + COMPOSITION_OPS + EXTERNAL_OPS


@dataclass
class PlacedArtifact:
    """Physical externalized artifact anchored on the periodic 2D arena."""

    artifact_id: int
    x: float
    y: float
    skill_id: int
    creator_id: int
    created_tick: int
    remaining_ticks: int
    mass_a: float
    mass_b: float
    uses: int = 0
    learns: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_id": int(self.artifact_id),
            "x": float(self.x),
            "y": float(self.y),
            "skill_id": int(self.skill_id),
            "creator_id": int(self.creator_id),
            "created_tick": int(self.created_tick),
            "remaining_ticks": int(self.remaining_ticks),
            "mass_a": float(self.mass_a),
            "mass_b": float(self.mass_b),
            "uses": int(self.uses),
            "learns": int(self.learns),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PlacedArtifact:
        return cls(
            artifact_id=int(data["artifact_id"]),
            x=float(data["x"]),
            y=float(data["y"]),
            skill_id=int(data["skill_id"]),
            creator_id=int(data["creator_id"]),
            created_tick=int(data["created_tick"]),
            remaining_ticks=int(data["remaining_ticks"]),
            mass_a=float(data["mass_a"]),
            mass_b=float(data["mass_b"]),
            uses=int(data.get("uses", 0)),
            learns=int(data.get("learns", 0)),
        )


def evaluate_pipeline(ops: tuple[str, ...], target_resource: int) -> tuple[float, bool, bool]:
    """Order-sensitive interpreter returning ``(efficiency, is_compositional, is_valid)``.

    Preconditions matter:
    - ``combine`` unlocks dual-material synergy ONLY after both ``process_a`` and ``process_b``.
    - ``transform`` unlocks catalytic extraction ONLY after prior processing.
    - Misordered composition operators incur penalties instead of bonuses.
    """
    if any(op not in ALLOWED_OPS for op in ops):
        raise ValueError(f"disallowed operation in skill ops {ops!r}")

    has_a_prep = False
    has_b_prep = False
    dual_combined = False
    catalyzed = False
    extracted = False
    stored = False
    locate_gather = False
    ext_tagged = False
    invalid_order = False
    seen: set[str] = set()
    score = 0.65

    for op in ops:
        if op in ("locate", "gather"):
            if not locate_gather:
                score += 0.14
                locate_gather = True
        elif op == "process_a":
            if not has_a_prep:
                has_a_prep = True
                score += 0.34 if target_resource in (1, 3) else 0.12
        elif op == "process_b":
            if not has_b_prep:
                has_b_prep = True
                score += 0.34 if target_resource in (2, 3) else 0.12
        elif op == "combine":
            if has_a_prep and has_b_prep:
                if not dual_combined:
                    dual_combined = True
                    score += 0.68
            else:
                invalid_order = True
                score -= 0.12
        elif op == "transform":
            if has_a_prep or has_b_prep or dual_combined:
                if not catalyzed:
                    catalyzed = True
                    score += 0.36
            else:
                invalid_order = True
                score -= 0.10
        elif op == "extract":
            if not extracted:
                extracted = True
                score += 0.28
                if catalyzed:
                    score += 0.24
                if dual_combined:
                    score += 0.32
        elif op == "store":
            if not stored:
                stored = True
                score += 0.14 if extracted else 0.04
        elif op in ("place_object", "use_object") and not ext_tagged:
            ext_tagged = True
            score += 0.04
        seen.add(op)

    unique_bonus = 0.10 * len(seen)
    bloat_penalty = 0.10 * max(0, len(ops) - len(seen))
    raw_eff = score + unique_bonus - bloat_penalty
    is_compositional = dual_combined or catalyzed
    cap = 3.6 if is_compositional else 2.6
    efficiency = float(max(0.5, min(cap, raw_eff)))
    is_valid = (
        (not invalid_order)
        and (has_a_prep or has_b_prep)
        and (extracted or dual_combined or catalyzed)
    )
    return efficiency, is_compositional, is_valid


def compose_skills(
    parent_a: SkillProcedure,
    parent_b: SkillProcedure,
    *,
    skill_id: int,
    composer_id: int,
    tick: int,
    env_phase: float,
    max_len: int,
    allow_externalization: bool,
    rng_tech: np.random.Generator,
) -> SkillProcedure:
    """Compose two owned skills into a deeper child procedure in the dual-parent DAG."""
    from alife_lab.substrates.cultural_world.memory import SkillProcedure as ProcCls

    comp_ops = parent_a.target_resource != parent_b.target_resource or (
        3 in (parent_a.target_resource, parent_b.target_resource)
    )
    seq: list[str] = []
    if max_len >= 6:
        seq.append("locate" if float(rng_tech.random()) < 0.6 else "gather")
    if comp_ops:
        if float(rng_tech.random()) < 0.5:
            seq.extend(["process_a", "process_b"])
        else:
            seq.extend(["process_b", "process_a"])
        seq.append("combine")
        if max_len >= 5:
            seq.append("transform")
        seq.append("extract")
    else:
        proc_op = "process_a" if parent_a.target_resource == 1 else "process_b"
        seq.extend([proc_op, "transform", "extract"])

    if len(seq) < max_len:
        seq.append("store")
    if allow_externalization and len(seq) < max_len and float(rng_tech.random()) < 0.45:
        seq.append("place_object")

    ops = tuple(seq[:max_len])
    prelim_target = 3 if comp_ops and "combine" in ops else parent_a.target_resource
    eff, is_comp, _ = evaluate_pipeline(ops, prelim_target)
    target = 3 if is_comp and "combine" in ops else parent_a.target_resource

    # Circular mean of parent phases nudged gently toward current environmental phase.
    ang_a = 2.0 * math.pi * parent_a.phase
    ang_b = 2.0 * math.pi * parent_b.phase
    mean_ang = math.atan2(math.sin(ang_a) + math.sin(ang_b), math.cos(ang_a) + math.cos(ang_b))
    mean_phase = (mean_ang / (2.0 * math.pi)) % 1.0
    diff = ((env_phase - mean_phase + 0.5) % 1.0) - 0.5
    phase = float((mean_phase + 0.30 * diff + float(rng_tech.normal(0.0, 0.03))) % 1.0)

    pa_root = parent_a.root_efficiency if parent_a.root_efficiency > 0.0 else parent_a.efficiency
    pb_root = parent_b.root_efficiency if parent_b.root_efficiency > 0.0 else parent_b.efficiency
    root_eff = float(min(pa_root, pb_root))

    return ProcCls(
        skill_id=skill_id,
        parent_skill_id=parent_a.skill_id,
        secondary_parent_id=parent_b.skill_id,
        revision=max(parent_a.revision, parent_b.revision) + 1,
        inventor_id=composer_id,
        origin_tick=tick,
        depth=max(parent_a.depth, parent_b.depth) + 1,
        target_resource=target,
        ops=ops,
        phase=phase,
        efficiency=eff,
        root_efficiency=root_eff,
    )


def decay_artifacts(
    artifacts: list[PlacedArtifact],
    eco: ResourceField,
    exp: CulturalWorldExperiment,
) -> int:
    """Decrement artifact lifetimes and return conserved mass to local cells upon decay."""
    if not artifacts:
        return 0
    w, h = exp.world.width, exp.world.height
    r = exp.world.resolution
    kept: list[PlacedArtifact] = []
    expired = 0
    for art in artifacts:
        art.remaining_ticks -= 1
        if art.remaining_ticks <= 0:
            ci = math.floor(art.y / h * r) % r
            cj = math.floor(art.x / w * r) % r
            eco.material_a[ci, cj] += art.mass_a
            eco.material_b[ci, cj] += art.mass_b
            expired += 1
        else:
            kept.append(art)
    if expired:
        artifacts[:] = kept
    return expired


def learn_from_artifacts(
    *,
    agents: CulturalAgents,
    artifacts: list[PlacedArtifact],
    registry: SkillRegistry,
    exp: CulturalWorldExperiment,
    tick: int,
    next_skill_id: int,
    env_phase: float,
    rng_transmit: np.random.Generator,
    externalized_enabled: bool,
) -> tuple[int, int, int, int]:
    """Allow agents within observation_range of placed artifacts to learn skills asynchronously."""
    if (not externalized_enabled) or (not artifacts) or agents.n == 0:
        return next_skill_id, 0, 0, 0

    from alife_lab.substrates.cultural_world.transmission import _receive_skill

    tx = exp.dynamics.transmission
    w, h = exp.world.width, exp.world.height
    rng_r2 = tx.observation_range * tx.observation_range
    ax = np.asarray([art.x for art in artifacts], dtype=np.float64)
    ay = np.asarray([art.y for art in artifacts], dtype=np.float64)

    dx = ((agents.x[None, :] - ax[:, None] + 0.5 * w) % w) - 0.5 * w
    dy = ((agents.y[None, :] - ay[:, None] + 0.5 * h) % h) - 0.5 * h
    d2 = dx * dx + dy * dy

    attempts = 0
    successes = 0
    ext_learns = 0
    study_cost = tx.transmission_cost * 0.5
    p_base = min(0.95, float(tx.base_success_prob * 0.85))

    for j in range(agents.n):
        if agents.energy[j] <= study_cost + 0.2:
            continue
        col = d2[:, j]
        cand = np.flatnonzero(col <= rng_r2)
        if cand.size == 0:
            continue
        owned = {sl.skill_id for sl in agents.skills[j]}
        for a_idx in cand[np.argsort(col[cand])].tolist():
            art = artifacts[a_idx]
            if art.skill_id in owned:
                continue
            proc = registry.procedures.get(art.skill_id)
            if proc is None:
                continue
            attempts += 1
            agents.energy[j] -= study_cost
            p_succ = min(0.95, p_base * (0.5 + 0.5 * float(agents.gene_social[j])))
            ok = bool(rng_transmit.random() < p_succ)
            if ok:
                next_skill_id, stored = _receive_skill(
                    learner_slots=agents.skills[j],
                    learner_id=int(agents.ids[j]),
                    teacher_id=art.creator_id,
                    teacher_skill=proc,
                    mode="externalized",
                    tick=tick,
                    next_skill_id=next_skill_id,
                    registry=registry,
                    exp=exp,
                    env_phase=env_phase,
                    rng_transmit=rng_transmit,
                )
                ok = stored
            if ok:
                successes += 1
                ext_learns += 1
                art.learns += 1
            registry.log_transmission(
                teacher_id=art.creator_id,
                learner_id=int(agents.ids[j]),
                skill_id=proc.skill_id,
                revision=proc.revision,
                origin_seed=int(exp.seed),
                tick=tick,
                mode="externalized",
                success=ok,
            )
            break

    return next_skill_id, attempts, successes, ext_learns
