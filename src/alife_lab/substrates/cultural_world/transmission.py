"""Social & vertical transmission channels with explicit cost, noise, and provenance."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.agents import CulturalAgents
from alife_lab.substrates.cultural_world.memory import (
    SkillProcedure,
    SkillRegistry,
    SkillSlot,
    evaluate_skill_utility,
    improve_skill,
)

F64 = npt.NDArray[np.float64]


def resolve_transmission_flags(
    exp: CulturalWorldExperiment, overrides: dict[str, bool] | None = None
) -> tuple[bool, bool, bool]:
    """Resolve effective (vertical, horizontal, oblique) flags cleanly."""
    tx = exp.dynamics.transmission
    regime = exp.dynamics.transmission_regime
    if regime == "none":
        v, h, o = False, False, False
    elif regime == "vertical_only":
        v, h, o = True, False, False
    elif regime == "vertical_horizontal":
        v, h, o = True, True, False
    elif regime == "all":
        v, h, o = True, True, True
    else:
        v, h, o = tx.vertical, tx.horizontal, tx.oblique

    if overrides:
        if "vertical" in overrides:
            v = bool(overrides["vertical"])
        if "horizontal" in overrides:
            h = bool(overrides["horizontal"])
        if "oblique" in overrides:
            o = bool(overrides["oblique"])
    return v, h, o


def _best_skill_slot(
    slots: list[SkillSlot], registry: SkillRegistry, env_phase: float
) -> SkillSlot | None:
    if not slots:
        return None
    return max(
        slots,
        key=lambda sl: evaluate_skill_utility(registry.procedures[sl.skill_id], env_phase),
    )


def _receive_skill(
    *,
    learner_slots: list[SkillSlot],
    learner_id: int,
    teacher_id: int,
    teacher_skill: SkillProcedure,
    mode: str,
    tick: int,
    next_skill_id: int,
    registry: SkillRegistry,
    exp: CulturalWorldExperiment,
    env_phase: float,
    rng_transmit: np.random.Generator,
) -> tuple[int, bool]:
    """Attempt copying a skill from teacher to learner, respecting errors and slot limits."""
    max_slots = exp.dynamics.technology.max_skills_per_agent
    owned_ids = {s.skill_id for s in learner_slots}
    if teacher_skill.skill_id in owned_ids:
        return next_skill_id, False

    tx = exp.dynamics.transmission
    acquired_proc = teacher_skill
    if rng_transmit.random() < tx.error_probability:
        # Imprecise imitation creates a slightly noisy/mutated revision
        acquired_proc = improve_skill(
            teacher_skill,
            skill_id=next_skill_id,
            improver_id=learner_id,
            tick=tick,
            env_phase=env_phase,
            max_len=exp.dynamics.technology.max_program_length,
            rng_tech=rng_transmit,
            preserve_phase=True,
            allow_composition=exp.dynamics.technology.allow_composition,
            allow_externalization=exp.dynamics.technology.allow_externalization,
        )
        registry.register(acquired_proc)
        next_skill_id += 1

    slot = SkillSlot(
        skill_id=acquired_proc.skill_id,
        teacher_id=teacher_id,
        mode=mode,
        acquired_tick=tick,
        uses=0,
    )
    if len(learner_slots) < max_slots:
        learner_slots.append(slot)
    else:
        # Replace lowest-utility skill only if the incoming skill outperforms it
        worst_idx = min(
            range(len(learner_slots)),
            key=lambda idx: evaluate_skill_utility(
                registry.procedures[learner_slots[idx].skill_id], env_phase
            ),
        )
        worst_val = evaluate_skill_utility(
            registry.procedures[learner_slots[worst_idx].skill_id], env_phase
        )
        new_val = evaluate_skill_utility(acquired_proc, env_phase)
        if new_val <= worst_val:
            return next_skill_id, False
        learner_slots[worst_idx] = slot
    return next_skill_id, True


def apply_vertical_transmission(
    *,
    parents: CulturalAgents,
    parent_indices: Any,
    children: CulturalAgents,
    registry: SkillRegistry,
    exp: CulturalWorldExperiment,
    tick: int,
    next_skill_id: int,
    env_phase: float,
    rng_transmit: np.random.Generator,
    v_enabled: bool,
) -> tuple[int, int, int]:
    """Transfer up to one skill per newborn ONLY when vertical transmission is enabled."""
    if not v_enabled or len(parent_indices) == 0:
        return next_skill_id, 0, 0
    tx = exp.dynamics.transmission
    attempts = 0
    successes = 0
    for c_idx, p_idx in enumerate(parent_indices.tolist()):
        best = _best_skill_slot(parents.skills[p_idx], registry, env_phase)
        if best is None:
            continue
        if (
            parents.energy[p_idx] <= tx.transmission_cost
            or children.energy[c_idx] <= tx.transmission_cost
        ):
            continue
        attempts += 1
        parents.energy[p_idx] -= tx.transmission_cost * 0.5
        children.energy[c_idx] -= tx.transmission_cost * 0.5
        p_roll = float(tx.base_success_prob)
        ok = bool(rng_transmit.random() < p_roll)
        proc = registry.procedures[best.skill_id]
        if ok:
            next_skill_id, stored = _receive_skill(
                learner_slots=children.skills[c_idx],
                learner_id=int(children.ids[c_idx]),
                teacher_id=int(parents.ids[p_idx]),
                teacher_skill=proc,
                mode="vertical",
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
        registry.log_transmission(
            teacher_id=int(parents.ids[p_idx]),
            learner_id=int(children.ids[c_idx]),
            skill_id=proc.skill_id,
            revision=proc.revision,
            origin_seed=int(exp.seed),
            tick=tick,
            mode="vertical",
            success=ok,
        )
    return next_skill_id, attempts, successes


def apply_social_transmission(
    *,
    agents: CulturalAgents,
    registry: SkillRegistry,
    exp: CulturalWorldExperiment,
    tick: int,
    next_skill_id: int,
    env_phase: float,
    rng_transmit: np.random.Generator,
    teach_mask: npt.NDArray[np.bool_],
    actor_mask: npt.NDArray[np.bool_],
    h_enabled: bool,
    o_enabled: bool,
) -> tuple[int, int, int]:
    """Horizontal & oblique learning from nearby demonstrating/active agents."""
    if (not h_enabled and not o_enabled) or agents.n < 2:
        return next_skill_id, 0, 0

    tx = exp.dynamics.transmission
    w, h = exp.world.width, exp.world.height
    rng_r = tx.observation_range
    rng_r2 = rng_r * rng_r

    demonstrators = [
        i
        for i in range(agents.n)
        if agents.skills[i] and (bool(teach_mask[i]) or bool(actor_mask[i]))
    ]
    if not demonstrators:
        return next_skill_id, 0, 0

    demo_arr = np.asarray(demonstrators, dtype=np.int64)
    attempts = 0
    successes = 0
    dx_mat = ((agents.x[None, :] - agents.x[demo_arr, None] + 0.5 * w) % w) - 0.5 * w
    dy_mat = ((agents.y[None, :] - agents.y[demo_arr, None] + 0.5 * h) % h) - 0.5 * h
    d2_mat = dx_mat * dx_mat + dy_mat * dy_mat

    for j in range(agents.n):
        if agents.energy[j] <= tx.transmission_cost + 0.2:
            continue
        # Find nearest eligible teacher among active/teaching agents
        col = d2_mat[:, j]
        cand_pos = np.flatnonzero(col <= rng_r2)
        if cand_pos.size == 0:
            continue
        # Iterate nearest first so deterministic selection picks closest valid demonstrator
        for d_idx in cand_pos[np.argsort(col[cand_pos])].tolist():
            t_idx = int(demo_arr[d_idx])
            if t_idx == j or int(agents.parent[j]) == int(agents.ids[t_idx]):
                continue
            is_older = int(agents.age[t_idx]) > int(agents.age[j]) + 10
            if o_enabled and not h_enabled:
                if not is_older:
                    continue
                mode = "oblique"
            elif h_enabled and o_enabled:
                mode = "oblique" if is_older else "horizontal"
            else:
                mode = "horizontal"

            best = _best_skill_slot(agents.skills[t_idx], registry, env_phase)
            if best is None:
                continue
            proc = registry.procedures[best.skill_id]
            # Active teach gets higher probability than passive observation of apply_skill
            obs_bonus = 1.0 if bool(teach_mask[t_idx]) else 0.45
            p_succ = float(tx.base_success_prob * obs_bonus * (0.5 + 0.5 * agents.gene_social[j]))
            if agents.energy[t_idx] <= tx.transmission_cost * 0.5:
                continue
            attempts += 1
            agents.energy[j] -= tx.transmission_cost * 0.6
            agents.energy[t_idx] -= tx.transmission_cost * 0.4
            ok = bool(rng_transmit.random() < min(0.95, p_succ))
            if ok:
                next_skill_id, stored = _receive_skill(
                    learner_slots=agents.skills[j],
                    learner_id=int(agents.ids[j]),
                    teacher_id=int(agents.ids[t_idx]),
                    teacher_skill=proc,
                    mode=mode,
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
            registry.log_transmission(
                teacher_id=int(agents.ids[t_idx]),
                learner_id=int(agents.ids[j]),
                skill_id=proc.skill_id,
                revision=proc.revision,
                origin_seed=int(exp.seed),
                tick=tick,
                mode=mode,
                success=ok,
            )
            break  # at most 1 observation attempt per learner per tick
    return next_skill_id, attempts, successes
