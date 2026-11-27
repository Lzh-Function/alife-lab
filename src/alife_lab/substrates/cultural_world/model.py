"""Authoritative state transition of ``cultural_world`` (ADR 0050 & ADR 0052)."""

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
    SkillProcedure,
    SkillRegistry,
    SkillSlot,
    evaluate_skill_utility,
    improve_skill,
    invent_skill,
)
from alife_lab.substrates.cultural_world.social import (
    TradeEvent,
    apply_local_exchange,
    specialization_multiplier,
)
from alife_lab.substrates.cultural_world.technology import (
    PlacedArtifact,
    compose_skills,
    decay_artifacts,
    learn_from_artifacts,
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
    # Stage 3 (P4): compositional tech & externalized artifacts
    artifacts: list[PlacedArtifact] = field(default_factory=list)
    next_artifact_id: int = 1
    valid_skill_uses: int = 0
    net_utility_sum: float = 0.0
    externalized_learns: int = 0
    externalized_uses: int = 0
    compositions_created: int = 0
    # Stage 5 (P6): atomic barter, role specialization & division of labor
    exchange_override: bool | None = None
    trade_events: list[TradeEvent] = field(default_factory=list)
    partner_pairs: dict[str, int] = field(default_factory=dict)
    trade_count: int = 0
    trade_volume: float = 0.0
    repeat_trades: int = 0
    agent_role_uses: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    agent_trades: dict[int, int] = field(default_factory=dict)


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


def _nearest_artifact_map(
    agents: CulturalAgents,
    actor_indices: list[int],
    artifacts: list[PlacedArtifact],
    exp: CulturalWorldExperiment,
) -> dict[int, int]:
    """Map actor index -> nearest active PlacedArtifact index within observation_range."""
    if not actor_indices or not artifacts:
        return {}
    w, h = exp.world.width, exp.world.height
    rng_r2 = exp.dynamics.transmission.observation_range**2
    idx_arr = np.asarray(actor_indices, dtype=np.int64)
    ax = np.asarray([art.x for art in artifacts], dtype=np.float64)
    ay = np.asarray([art.y for art in artifacts], dtype=np.float64)
    dx = ((agents.x[idx_arr, None] - ax[None, :] + 0.5 * w) % w) - 0.5 * w
    dy = ((agents.y[idx_arr, None] - ay[None, :] + 0.5 * h) % h) - 0.5 * h
    d2 = dx * dx + dy * dy
    out: dict[int, int] = {}
    for row_i, agent_idx in enumerate(actor_indices):
        best_pos = int(np.argmin(d2[row_i]))
        if float(d2[row_i, best_pos]) <= rng_r2:
            out[agent_idx] = best_pos
    return out


def _available_for_skill(proc: SkillProcedure, inv_a: float, inv_b: float) -> float:
    if proc.target_resource == 3:
        return 2.0 * min(inv_a, inv_b) if (inv_a >= 0.18 and inv_b >= 0.18) else 0.0
    if proc.target_resource == 1:
        return inv_a if inv_a >= 0.35 else 0.0
    return inv_b if inv_b >= 0.35 else 0.0


def _pick_composition_pair(
    slots: list[SkillSlot], registry: SkillRegistry
) -> tuple[SkillProcedure, SkillProcedure] | None:
    """Select two distinct skills for synthesis (preferring complementary resource targets)."""
    procs = [registry.procedures[sl.skill_id] for sl in slots if sl.skill_id in registry.procedures]
    uniq: dict[int, SkillProcedure] = {p.skill_id: p for p in procs}
    distinct = list(uniq.values())
    if len(distinct) < 2:
        return None
    ones = [p for p in distinct if p.target_resource == 1]
    twos = [p for p in distinct if p.target_resource == 2]
    if ones and twos:
        pa = max(ones, key=lambda p: p.efficiency)
        pb = max(twos, key=lambda p: p.efficiency)
        return pa, pb
    distinct.sort(key=lambda p: p.efficiency, reverse=True)
    return distinct[0], distinct[1]


def step_world(state: CulturalWorldState, exp: CulturalWorldExperiment) -> CulturalWorldState:
    """Execute one tick in strict ADR 0050 / ADR 0052 order."""
    state.tick += 1
    d = exp.dynamics
    a = state.agents
    eco = state.eco
    env_phase = regime_phase(state.tick, exp, state.env_phase_offset)

    # 1. Environmental ecology step & physical artifact decay (conserving material mass).
    step_ecology(eco, exp, state.tick, state.rngs["env"], state.env_phase_offset)
    decay_artifacts(state.artifacts, eco, exp)

    if a.n == 0:
        return state

    # 2. Sensors & policy choice.
    x = compute_sensors(a, eco, exp, state.tick, state.env_phase_offset)
    hidden, probs, act = choose_actions(a, x, exp, state.rngs["policy"])
    e0 = a.energy.copy()

    # 3. Movement, contested gathering, skill execution/invention/composition, artifacts.
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

    # Skill execution, refinement, two-skill composition & externalized artifact creation
    tech = d.technology
    skill_actor_indices = np.flatnonzero(act == 4).tolist()
    nearby_art = (
        _nearest_artifact_map(a, skill_actor_indices, state.artifacts, exp)
        if tech.allow_externalization and state.artifacts
        else {}
    )
    used_mask = np.zeros(a.n, dtype=np.bool_)
    rng_tech = state.rngs["tech"]

    for i in skill_actor_indices:
        slots = a.skills[i]
        aid = int(a.ids[i])
        best_slot: SkillSlot | None = None
        best_util = -1e9
        for sl in slots:
            proc = state.registry.procedures[sl.skill_id]
            avail = _available_for_skill(proc, float(a.inv_a[i]), float(a.inv_b[i]))
            if avail > 0.0:
                u_val = evaluate_skill_utility(proc, env_phase) * specialization_multiplier(
                    state, exp, aid, proc
                )
                if u_val > best_util:
                    best_util = u_val
                    best_slot = sl

        if best_slot is not None:
            proc = state.registry.procedures[best_slot.skill_id]
            avail = _available_for_skill(proc, float(a.inv_a[i]), float(a.inv_b[i]))
            amt = min(avail, 0.55)
            if proc.target_resource == 3:
                half = 0.5 * amt
                a.inv_a[i] -= half
                a.inv_b[i] -= half
                eco.ledger.consumed["material_a"] += half
                eco.ledger.consumed["material_b"] += half
            elif proc.target_resource == 1:
                a.inv_a[i] -= amt
                eco.ledger.consumed["material_a"] += amt
            else:
                a.inv_b[i] -= amt
                eco.ledger.consumed["material_b"] += amt

            eff_util = best_util
            art_idx = nearby_art.get(i)
            if art_idx is not None:
                eff_util *= 1.18
                state.artifacts[art_idx].uses += 1
                state.externalized_uses += 1

            gain = amt * eff_util
            net_gain = gain - tech.skill_energy_cost
            a.energy[i] += max(0.0, net_gain)
            best_slot.uses += 1
            ua, ub, ud = state.agent_role_uses.get(aid, (0, 0, 0))
            if proc.target_resource == 1:
                state.agent_role_uses[aid] = (ua + 1, ub, ud)
            elif proc.target_resource == 2:
                state.agent_role_uses[aid] = (ua, ub + 1, ud)
            else:
                state.agent_role_uses[aid] = (ua, ub, ud + 1)
            used_mask[i] = True
            state.skill_uses_total += 1
            if net_gain > 0.0:
                state.valid_skill_uses += 1
            state.net_utility_sum += net_gain / max(amt, 1e-9)
            if best_slot.teacher_id >= 0:
                state.skill_uses_social += 1
            if state.registry.is_inventor_dead(best_slot.skill_id):
                state.skill_uses_post_death += 1
            state.utility_sum += gain
            state.utility_count += 1

            # Optional externalized artifact creation (strict material mass conservation)
            if (
                tech.allow_externalization
                and tech.artifact_material_cost > 0.0
                and len(state.artifacts) < tech.max_artifacts
            ):
                p_place = (
                    0.14 if "place_object" in proc.ops else (0.035 if proc.depth >= 1 else 0.0)
                )
                if p_place > 0.0:
                    cost = tech.artifact_material_cost
                    if proc.target_resource == 3:
                        need_a, need_b = 0.5 * cost, 0.5 * cost
                    elif proc.target_resource == 1:
                        need_a, need_b = cost, 0.0
                    else:
                        need_a, need_b = 0.0, cost
                    if (
                        float(a.inv_a[i]) >= need_a
                        and float(a.inv_b[i]) >= need_b
                        and float(rng_tech.random()) < p_place
                    ):
                        a.inv_a[i] -= need_a
                        a.inv_b[i] -= need_b
                        state.artifacts.append(
                            PlacedArtifact(
                                artifact_id=state.next_artifact_id,
                                x=float(a.x[i]),
                                y=float(a.y[i]),
                                skill_id=proc.skill_id,
                                creator_id=int(a.ids[i]),
                                created_tick=state.tick,
                                remaining_ticks=tech.artifact_decay_ticks,
                                mass_a=need_a,
                                mass_b=need_b,
                            )
                        )
                        state.next_artifact_id += 1

            # Optional P4 two-skill compositional synthesis
            composed_now = False
            if (
                tech.allow_composition
                and len(slots) >= 2
                and float(rng_tech.random()) < tech.invention_prob * 0.65
            ):
                pair = _pick_composition_pair(slots, state.registry)
                if pair is not None:
                    pa, pb = pair
                    composed = compose_skills(
                        pa,
                        pb,
                        skill_id=state.next_skill_id,
                        composer_id=int(a.ids[i]),
                        tick=state.tick,
                        env_phase=env_phase,
                        max_len=tech.max_program_length,
                        allow_externalization=tech.allow_externalization,
                        rng_tech=rng_tech,
                    )
                    comp_util = evaluate_skill_utility(
                        composed, env_phase
                    ) * specialization_multiplier(state, exp, aid, composed)
                    worst_slot = min(
                        slots,
                        key=lambda sl: (
                            evaluate_skill_utility(
                                state.registry.procedures[sl.skill_id], env_phase
                            )
                            * specialization_multiplier(
                                state, exp, aid, state.registry.procedures[sl.skill_id]
                            )
                        ),
                    )
                    worst_proc = state.registry.procedures[worst_slot.skill_id]
                    worst_u = evaluate_skill_utility(
                        worst_proc, env_phase
                    ) * specialization_multiplier(state, exp, aid, worst_proc)
                    if len(slots) < tech.max_skills_per_agent or comp_util > worst_u:
                        state.registry.register(composed)
                        state.next_skill_id += 1
                        state.compositions_created += 1
                        c_slot = SkillSlot(
                            skill_id=composed.skill_id,
                            teacher_id=-1,
                            mode="compose",
                            acquired_tick=state.tick,
                            uses=0,
                        )
                        if len(slots) < tech.max_skills_per_agent:
                            slots.append(c_slot)
                        else:
                            slots[slots.index(worst_slot)] = c_slot
                        composed_now = True

            # Optional single-parent refinement
            if (
                not composed_now
                and tech.allow_improvement
                and rng_tech.random() < tech.invention_prob * 0.6
            ):
                refined = improve_skill(
                    proc,
                    skill_id=state.next_skill_id,
                    improver_id=int(a.ids[i]),
                    tick=state.tick,
                    env_phase=env_phase,
                    max_len=tech.max_program_length,
                    rng_tech=rng_tech,
                    allow_composition=tech.allow_composition,
                    allow_externalization=tech.allow_externalization,
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
                    refined_util = evaluate_skill_utility(
                        refined, env_phase
                    ) * specialization_multiplier(state, exp, aid, refined)
                    if refined_util > best_util:
                        slots[idx_r] = new_slot

        elif tech.allow_invention and a.energy[i] > tech.skill_energy_cost + 0.3:
            # Individual exploration/trial to discover a new base skill
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
                    allow_composition=tech.allow_composition,
                    allow_externalization=tech.allow_externalization,
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

    # 4. Horizontal / Oblique social transmission & externalized artifact learning
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

    ext_on = (
        bool(state.tx_overrides["externalized"])
        if "externalized" in state.tx_overrides
        else (tech.allow_externalization and (v_on or h_on or o_on))
    )
    state.next_skill_id, e_att, e_suc, e_lrn = learn_from_artifacts(
        agents=a,
        artifacts=state.artifacts,
        registry=state.registry,
        exp=exp,
        tick=state.tick,
        next_skill_id=state.next_skill_id,
        env_phase=env_phase,
        rng_transmit=state.rngs["transmit"],
        externalized_enabled=ext_on,
    )
    state.tx_attempts += e_att
    state.tx_successes += e_suc
    state.externalized_learns += e_lrn

    # 4b. Atomic local barter (material_a <-> material_b) before metabolism
    apply_local_exchange(state, exp)

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
            aid_i = int(aid)
            state.registry.mark_agent_died(aid_i, state.tick)
            state.agent_role_uses.pop(aid_i, None)
            state.agent_trades.pop(aid_i, None)
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
