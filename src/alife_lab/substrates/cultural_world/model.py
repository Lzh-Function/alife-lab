"""Authoritative state transition of ``cultural_world`` (ADR 0050)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.agents import CulturalAgents, init_agents, make_offspring
from alife_lab.substrates.cultural_world.controllers import (
    apply_lifetime_learning,
    choose_actions,
    compute_sensors,
    grid_cells,
)
from alife_lab.substrates.cultural_world.ecology import (
    ResourceField,
    init_ecology,
    regime_phase,
    step_ecology,
)
from alife_lab.substrates.cultural_world.memory import (
    SkillRegistry,
    SkillSlot,
    evaluate_skill_utility,
    improve_skill,
    invent_skill,
)
from alife_lab.substrates.cultural_world.transmission import (
    apply_social_transmission,
    apply_vertical_transmission,
    resolve_transmission_flags,
)

RNG_NAMES = ("env", "birth", "policy", "learn", "transmit", "tech")


@dataclass
class CulturalWorldState:
    eco: ResourceField
    agents: CulturalAgents
    registry: SkillRegistry
    tick: int
    next_id: int
    next_skill_id: int
    rngs: dict[str, np.random.Generator] = field(repr=False)
    births: int = 0
    deaths: int = 0
    tx_attempts: int = 0
    tx_successes: int = 0
    skill_uses_total: int = 0
    skill_uses_social: int = 0
    skill_uses_post_death: int = 0
    utility_sum: float = 0.0
    utility_count: int = 0
    env_phase_offset: float = 0.0
    tx_overrides: dict[str, bool] = field(default_factory=dict)


def init_world(exp: CulturalWorldExperiment) -> CulturalWorldState:
    ss = np.random.SeedSequence(int(exp.seed))
    children = ss.spawn(len(RNG_NAMES))
    rngs = {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(RNG_NAMES, children, strict=True)
    }
    eco = init_ecology(exp, rngs["env"])
    agents = init_agents(exp, rngs["birth"], start_id=0)
    return CulturalWorldState(
        eco=eco,
        agents=agents,
        registry=SkillRegistry(),
        tick=0,
        next_id=agents.n,
        next_skill_id=1,
        rngs=rngs,
    )


def step_world(state: CulturalWorldState, exp: CulturalWorldExperiment) -> CulturalWorldState:
    """Execute one tick in strict ADR 0050 order."""
    state.tick += 1
    d = exp.dynamics
    a = state.agents
    eco = state.eco
    env_phase = regime_phase(state.tick, exp, state.env_phase_offset)

    # 1. Environmental ecology step (inflow / decay logged to ResourceLedger).
    step_ecology(eco, exp, state.tick, state.rngs["env"], state.env_phase_offset)

    if a.n == 0:
        return state

    # 2. Sensors & policy choice.
    x = compute_sensors(a, eco, exp, state.tick, state.env_phase_offset)
    hidden, probs, act = choose_actions(a, x, exp, state.rngs["policy"])
    e0 = a.energy.copy()

    # 3. Movement, contested gathering, skill execution/invention, teach actions.
    turn = math.radians(32.0)
    a.heading = (a.heading + np.where(act == 0, turn, 0.0) - np.where(act == 2, turn, 0.0)) % (
        2.0 * math.pi
    )
    step_size = 1.0
    dist = np.where(act == 1, step_size, np.where(np.isin(act, (0, 2)), 0.45 * step_size, 0.0))
    a.x = (a.x + dist * np.cos(a.heading)) % exp.world.width
    a.y = (a.y + dist * np.sin(a.heading)) % exp.world.height
    a.energy -= d.move_cost * (dist / step_size)

    # Gathering (food + raw materials with proportional contention sharing)
    gather_mask = act == 3
    if gather_mask.any():
        ci_g, cj_g = grid_cells(a.x[gather_mask], a.y[gather_mask], exp)
        # 3a. Food -> direct metabolic energy
        bite_f = 0.42
        dem_f = np.zeros_like(eco.food)
        np.add.at(dem_f, (ci_g, cj_g), bite_f)
        rat_f = np.where(dem_f > 0.0, np.minimum(1.0, eco.food / np.maximum(dem_f, 1e-12)), 0.0)
        taken_f = np.minimum(eco.food, dem_f)
        eco.food -= taken_f
        eco.ledger.consumed["food"] += float(taken_f.sum())
        a.energy[gather_mask] += bite_f * rat_f[ci_g, cj_g]

        # 3b. Raw material_a & material_b -> stored into agent inventories (mass conserved!)
        bite_m = 0.45
        dem_a = np.zeros_like(eco.material_a)
        dem_b = np.zeros_like(eco.material_b)
        np.add.at(dem_a, (ci_g, cj_g), bite_m)
        np.add.at(dem_b, (ci_g, cj_g), bite_m)
        rat_a = np.where(
            dem_a > 0.0, np.minimum(1.0, eco.material_a / np.maximum(dem_a, 1e-12)), 0.0
        )
        rat_b = np.where(
            dem_b > 0.0, np.minimum(1.0, eco.material_b / np.maximum(dem_b, 1e-12)), 0.0
        )
        eco.material_a -= np.minimum(eco.material_a, dem_a)
        eco.material_b -= np.minimum(eco.material_b, dem_b)
        a.inv_a[gather_mask] += bite_m * rat_a[ci_g, cj_g]
        a.inv_b[gather_mask] += bite_m * rat_b[ci_g, cj_g]

        # Update private episodic best cell if rich
        val_local = eco.food[ci_g, cj_g] + eco.material_a[ci_g, cj_g] + eco.material_b[ci_g, cj_g]
        g_idx = np.flatnonzero(gather_mask)
        better = val_local > a.private_best_val[g_idx]
        if better.any():
            b_idx = g_idx[better]
            a.private_best_x[b_idx] = a.x[b_idx]
            a.private_best_y[b_idx] = a.y[b_idx]
            a.private_best_val[b_idx] = val_local[better]

    # Skill execution & de-novo trial-and-error invention / refinement
    tech = d.technology
    skill_actor_mask = act == 4
    used_mask = np.zeros(a.n, dtype=np.bool_)
    rng_tech = state.rngs["tech"]
    for i in np.flatnonzero(skill_actor_mask).tolist():
        slots = a.skills[i]
        # Find highest-utility applicable owned skill
        best_slot: SkillSlot | None = None
        best_util = -1e9
        for sl in slots:
            proc = state.registry.procedures[sl.skill_id]
            avail = a.inv_a[i] if proc.target_resource == 1 else a.inv_b[i]
            if avail >= 0.35:
                u_val = evaluate_skill_utility(proc, env_phase)
                if u_val > best_util:
                    best_util = u_val
                    best_slot = sl

        if best_slot is not None:
            proc = state.registry.procedures[best_slot.skill_id]
            res_name = "material_a" if proc.target_resource == 1 else "material_b"
            avail = float(a.inv_a[i] if proc.target_resource == 1 else a.inv_b[i])
            amt = min(avail, 0.55)
            if proc.target_resource == 1:
                a.inv_a[i] -= amt
            else:
                a.inv_b[i] -= amt
            eco.ledger.consumed[res_name] += amt
            gain = amt * best_util
            a.energy[i] += max(0.0, gain - tech.skill_energy_cost)
            best_slot.uses += 1
            used_mask[i] = True
            state.skill_uses_total += 1
            if best_slot.teacher_id >= 0:
                state.skill_uses_social += 1
            if state.registry.is_inventor_dead(best_slot.skill_id):
                state.skill_uses_post_death += 1
            state.utility_sum += gain
            state.utility_count += 1

            # Optional cumulative refinement of owned skill
            if tech.allow_improvement and rng_tech.random() < tech.invention_prob * 0.6:
                refined = improve_skill(
                    proc,
                    skill_id=state.next_skill_id,
                    improver_id=int(a.ids[i]),
                    tick=state.tick,
                    env_phase=env_phase,
                    max_len=tech.max_program_length,
                    rng_tech=rng_tech,
                )
                state.registry.register(refined)
                state.next_skill_id += 1
                new_slot = SkillSlot(
                    skill_id=refined.skill_id,
                    teacher_id=-1,
                    mode="improve",
                    acquired_tick=state.tick,
                    uses=0,
                )
                if len(slots) < tech.max_skills_per_agent:
                    slots.append(new_slot)
                else:
                    idx_r = slots.index(best_slot)
                    if evaluate_skill_utility(refined, env_phase) > best_util:
                        slots[idx_r] = new_slot

        elif tech.allow_invention and a.energy[i] > tech.skill_energy_cost + 0.3:
            # Individual exploration/trial to discover a new skill
            a.energy[i] -= tech.skill_energy_cost
            if rng_tech.random() < tech.invention_prob and len(slots) < tech.max_skills_per_agent:
                target = 1 if a.inv_a[i] >= a.inv_b[i] else 2
                created = invent_skill(
                    skill_id=state.next_skill_id,
                    inventor_id=int(a.ids[i]),
                    tick=state.tick,
                    target_resource=target,
                    env_phase=env_phase,
                    max_len=tech.max_program_length,
                    rng_tech=rng_tech,
                )
                state.registry.register(created)
                state.next_skill_id += 1
                slots.append(
                    SkillSlot(
                        skill_id=created.skill_id,
                        teacher_id=-1,
                        mode="invent",
                        acquired_tick=state.tick,
                        uses=0,
                    )
                )

    # 4. Horizontal / Oblique social transmission
    v_on, h_on, o_on = resolve_transmission_flags(exp, state.tx_overrides)
    teach_mask = act == 5
    state.next_skill_id, att, suc = apply_social_transmission(
        agents=a,
        registry=state.registry,
        exp=exp,
        tick=state.tick,
        next_skill_id=state.next_skill_id,
        env_phase=env_phase,
        rng_transmit=state.rngs["transmit"],
        teach_mask=teach_mask,
        actor_mask=used_mask,
        h_enabled=h_on,
        o_enabled=o_on,
    )
    state.tx_attempts += att
    state.tx_successes += suc

    # 5. Metabolism, age increment, and lifetime REINFORCE learning on dw2 ONLY
    a.energy -= d.metabolism
    a.age += 1
    reward = a.energy - e0
    apply_lifetime_learning(a, hidden, probs, act, reward, exp, state.rngs["learn"])

    # 6. Mortality: return unused inventories to local cells to preserve exact mass ledger.
    alive = (a.energy > 0.0) & (a.age < d.max_age)
    if not alive.all():
        dead_idx = np.flatnonzero(~alive)
        ci_d, cj_d = grid_cells(a.x[dead_idx], a.y[dead_idx], exp)
        np.add.at(eco.material_a, (ci_d, cj_d), a.inv_a[dead_idx])
        np.add.at(eco.material_b, (ci_d, cj_d), a.inv_b[dead_idx])
        for aid in a.ids[dead_idx].tolist():
            state.registry.mark_agent_died(int(aid), state.tick)
        state.deaths += int(dead_idx.size)
        a = a.take(alive)

    # 7. Reproduction + optional Vertical transmission
    room = max(0, d.max_population - a.n)
    ready = np.flatnonzero(a.energy >= d.reproduce_threshold)[:room]
    if ready.size > 0:
        a.energy[ready] *= 0.5
        children, state.next_id = make_offspring(
            a,
            ready,
            tick=state.tick,
            next_id=state.next_id,
            exp=exp,
            rng_birth=state.rngs["birth"],
        )
        state.next_skill_id, v_att, v_suc = apply_vertical_transmission(
            parents=a,
            parent_indices=ready,
            children=children,
            registry=state.registry,
            exp=exp,
            tick=state.tick,
            next_skill_id=state.next_skill_id,
            env_phase=env_phase,
            rng_transmit=state.rngs["transmit"],
            v_enabled=v_on,
        )
        state.tx_attempts += v_att
        state.tx_successes += v_suc
        a = a.concat([a, children])
        state.births += int(ready.size)

    state.agents = a
    return state
