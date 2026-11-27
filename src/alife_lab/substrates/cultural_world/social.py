"""Stage 5 / P6: Atomic local barter, role specialization, division of labor, and graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from alife_lab.core.spec import CulturalWorldExperiment
    from alife_lab.substrates.cultural_world.memory import SkillProcedure, SkillRegistry, SkillSlot
    from alife_lab.substrates.cultural_world.model import CulturalWorldState

MAX_TRADE_EVENTS = 400
MAX_PARTNER_PAIRS = 2000
MIN_TRADE_QTY = 0.03
SURPLUS_MARGIN = 0.04
RECENT_TRADE_WINDOW = 40


@dataclass(frozen=True)
class TradeEvent:
    """Atomic bilateral material_a <-> material_b barter event."""

    tick: int
    agent_a: int
    agent_b: int
    qty_a_to_b: float
    qty_b_to_a: float
    role_a: str
    role_b: str

    def to_json(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "agent_a": int(self.agent_a),
            "agent_b": int(self.agent_b),
            "qty_a_to_b": float(self.qty_a_to_b),
            "qty_b_to_a": float(self.qty_b_to_a),
            "role_a": str(self.role_a),
            "role_b": str(self.role_b),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TradeEvent:
        return cls(
            tick=int(data["tick"]),
            agent_a=int(data["agent_a"]),
            agent_b=int(data["agent_b"]),
            qty_a_to_b=float(data["qty_a_to_b"]),
            qty_b_to_a=float(data["qty_b_to_a"]),
            role_a=str(data["role_a"]),
            role_b=str(data["role_b"]),
        )


def resolve_exchange_enabled(state: CulturalWorldState, exp: CulturalWorldExperiment) -> bool:
    """Resolve effective exchange state considering ``forbid_exchange`` intervention overrides."""
    if state.exchange_override is not None:
        return bool(state.exchange_override)
    return bool(exp.dynamics.social.allow_exchange)


def agent_role(
    uses_a: int,
    uses_b: int,
    uses_dual: int,
    inv_a: float,
    inv_b: float,
    slots: list[SkillSlot],
    registry: SkillRegistry,
) -> str:
    """Classify an agent as 'A', 'B', 'dual', or 'generalist' from usage, skills, and stock."""
    tot = uses_a + uses_b + uses_dual
    if tot > 0:
        if uses_dual > 0 and uses_dual >= max(uses_a, uses_b) and uses_dual >= 0.4 * tot:
            return "dual"
        ab = uses_a + uses_b
        if ab > 0:
            if uses_a >= 0.65 * ab and uses_a > uses_b:
                return "A"
            if uses_b >= 0.65 * ab and uses_b > uses_a:
                return "B"
            return "generalist"
        return "dual"

    has_a = 0
    has_b = 0
    has_dual = 0
    for sl in slots:
        proc = registry.procedures.get(sl.skill_id)
        if proc is None:
            continue
        if proc.target_resource == 1:
            has_a += 1
        elif proc.target_resource == 2:
            has_b += 1
        elif proc.target_resource == 3:
            has_dual += 1

    if has_dual > 0 and has_dual >= max(has_a, has_b):
        return "dual"
    if has_a > has_b:
        return "A"
    if has_b > has_a:
        return "B"
    if has_a > 0 and has_b > 0:
        return "generalist"

    if inv_a > inv_b + 0.15:
        return "A"
    if inv_b > inv_a + 0.15:
        return "B"
    return "generalist"


def specialization_multiplier(
    state: CulturalWorldState,
    exp: CulturalWorldExperiment,
    agent_id: int,
    proc: SkillProcedure,
) -> float:
    """Efficiency multiplier rewarding role specialization & trade-backed dual execution."""
    social = exp.dynamics.social
    if not social.interdependent_resources or social.shuffle_roles_control:
        return 1.0
    bonus = float(social.specialization_bonus)
    if bonus <= 0.0:
        return 1.0
    ua, ub, _ = state.agent_role_uses.get(agent_id, (0, 0, 0))
    ab = ua + ub
    if proc.target_resource == 1:
        if ab <= 0 or ua <= ub:
            return 1.0
        agent_spec = (ua - ub) / float(ab)
        return 1.0 + bonus * agent_spec
    if proc.target_resource == 2:
        if ab <= 0 or ub <= ua:
            return 1.0
        agent_spec = (ub - ua) / float(ab)
        return 1.0 + bonus * agent_spec
    # Dual target_resource == 3 benefit when backed by trades
    n_trades = state.agent_trades.get(agent_id, 0)
    if n_trades <= 0:
        return 1.0
    trade_factor = min(1.0, 0.5 + 0.5 * (n_trades / 2.0))
    return 1.0 + bonus * trade_factor


def _record_trade(
    state: CulturalWorldState,
    *,
    aid_a: int,
    aid_b: int,
    qty: float,
    role_a: str,
    role_b: str,
) -> None:
    ev = TradeEvent(
        tick=int(state.tick),
        agent_a=aid_a,
        agent_b=aid_b,
        qty_a_to_b=qty,
        qty_b_to_a=qty,
        role_a=role_a,
        role_b=role_b,
    )
    state.trade_events.append(ev)
    if len(state.trade_events) > MAX_TRADE_EVENTS:
        del state.trade_events[:-MAX_TRADE_EVENTS]

    pair_key = f"{min(aid_a, aid_b)}:{max(aid_a, aid_b)}"
    prev = state.partner_pairs.pop(pair_key, 0)
    if prev > 0:
        state.repeat_trades += 1
    state.partner_pairs[pair_key] = prev + 1
    while len(state.partner_pairs) > MAX_PARTNER_PAIRS:
        oldest = next(iter(state.partner_pairs))
        del state.partner_pairs[oldest]

    state.trade_count += 1
    state.trade_volume += 2.0 * qty
    state.agent_trades[aid_a] = state.agent_trades.get(aid_a, 0) + 1
    state.agent_trades[aid_b] = state.agent_trades.get(aid_b, 0) + 1


def _complementary_score(role_a: str, role_b: str) -> float:
    if (role_a == "A" and role_b == "B") or (role_a == "B" and role_b == "A"):
        return 2.0
    if role_a != role_b and ("dual" in (role_a, role_b)):
        return 1.5
    if role_a != role_b:
        return 1.0
    return 0.0


def apply_local_exchange(
    state: CulturalWorldState, exp: CulturalWorldExperiment
) -> tuple[int, float]:
    """Execute atomic all-or-nothing local barter of material_a <-> material_b.

    When ``exchange_enabled`` is False, returns immediately without consuming RNG draws.
    Strictly preserves ``inv_a.sum()`` and ``inv_b.sum()`` and non-negativity.
    """
    if not resolve_exchange_enabled(state, exp) or state.agents.n < 2:
        return 0, 0.0

    a = state.agents
    social = exp.dynamics.social
    r_ex = float(social.exchange_range)
    r_ex2 = r_ex * r_ex
    lot = float(social.exchange_lot)
    e_cost = float(social.exchange_energy_cost)
    rng_tx = state.rngs["transmit"]

    eligible_mask = (a.energy > e_cost + 0.15) & (
        np.maximum(a.inv_a, a.inv_b) >= 2.0 * MIN_TRADE_QTY
    )
    elig = np.flatnonzero(eligible_mask)
    if elig.size < 2:
        return 0, 0.0

    w, h = exp.world.width, exp.world.height
    ex = a.x[elig]
    ey = a.y[elig]
    dx = ((ex[:, None] - ex[None, :] + 0.5 * w) % w) - 0.5 * w
    dy = ((ey[:, None] - ey[None, :] + 0.5 * h) % h) - 0.5 * h
    d2 = dx * dx + dy * dy

    # Precompute roles for eligible agents once
    roles: dict[int, str] = {}
    for idx in elig.tolist():
        aid = int(a.ids[idx])
        ua, ub, ud = state.agent_role_uses.get(aid, (0, 0, 0))
        roles[idx] = agent_role(
            ua,
            ub,
            ud,
            float(a.inv_a[idx]),
            float(a.inv_b[idx]),
            a.skills[idx],
            state.registry,
        )

    trades_tick = 0
    volume_tick = 0.0
    used_pos: set[int] = set()

    if social.shuffle_roles_control:
        # Randomly shuffle candidate pairings and direction; reject un-coverable trades.
        order = rng_tx.permutation(elig.size).tolist()
        for pos_i in order:
            if pos_i in used_pos:
                continue
            i = int(elig[pos_i])
            nbrs = [
                pos_j
                for pos_j in np.flatnonzero(d2[pos_i] <= r_ex2).tolist()
                if pos_j != pos_i and pos_j not in used_pos
            ]
            if not nbrs:
                used_pos.add(pos_i)
                continue
            pos_j = nbrs[int(rng_tx.integers(0, len(nbrs)))]
            j = int(elig[pos_j])
            used_pos.add(pos_i)
            used_pos.add(pos_j)

            if float(rng_tx.random()) < 0.5:
                ga, gb = i, j
            else:
                ga, gb = j, i

            if float(a.inv_a[ga]) <= float(a.inv_b[ga]) + SURPLUS_MARGIN:
                continue
            if float(a.inv_b[gb]) <= float(a.inv_a[gb]) + SURPLUS_MARGIN:
                continue
            qty = min(lot, 0.5 * float(a.inv_a[ga]), 0.5 * float(a.inv_b[gb]))
            if qty < MIN_TRADE_QTY or float(a.inv_a[ga]) < qty or float(a.inv_b[gb]) < qty:
                continue
            if float(a.energy[ga]) <= e_cost or float(a.energy[gb]) <= e_cost:
                continue

            a.inv_a[ga] -= qty
            a.inv_a[gb] += qty
            a.inv_b[gb] -= qty
            a.inv_b[ga] += qty
            a.energy[ga] -= e_cost
            a.energy[gb] -= e_cost
            _record_trade(
                state,
                aid_a=int(a.ids[ga]),
                aid_b=int(a.ids[gb]),
                qty=qty,
                role_a=roles[ga],
                role_b=roles[gb],
            )
            trades_tick += 1
            volume_tick += 2.0 * qty
        return trades_tick, volume_tick

    # Complementary role-matched local barter: A-surplus paired with B-surplus
    a_pos_list = [
        pos
        for pos, idx in enumerate(elig.tolist())
        if float(a.inv_a[idx]) > float(a.inv_b[idx]) + SURPLUS_MARGIN
        and 0.5 * float(a.inv_a[idx]) >= MIN_TRADE_QTY
    ]
    b_pos_set = {
        pos
        for pos, idx in enumerate(elig.tolist())
        if float(a.inv_b[idx]) > float(a.inv_a[idx]) + SURPLUS_MARGIN
        and 0.5 * float(a.inv_b[idx]) >= MIN_TRADE_QTY
    }
    if not a_pos_list or not b_pos_set:
        return 0, 0.0

    perm = rng_tx.permutation(len(a_pos_list)).tolist()
    for p_idx in perm:
        pos_i = a_pos_list[p_idx]
        if pos_i in used_pos:
            continue
        i = int(elig[pos_i])
        aid_i = int(a.ids[i])
        role_i = roles[i]

        best_pos_j = -1
        best_score = -1e9
        best_qty = 0.0
        for pos_j in np.flatnonzero(d2[pos_i] <= r_ex2).tolist():
            if pos_j == pos_i or pos_j not in b_pos_set or pos_j in used_pos:
                continue
            j = int(elig[pos_j])
            qty_ij = min(lot, 0.5 * float(a.inv_a[i]), 0.5 * float(a.inv_b[j]))
            if qty_ij < MIN_TRADE_QTY:
                continue
            aid_j = int(a.ids[j])
            pair_key = f"{min(aid_i, aid_j)}:{max(aid_i, aid_j)}"
            repeat_bonus = 0.25 if pair_key in state.partner_pairs else 0.0
            comp_bonus = _complementary_score(role_i, roles[j])
            dist_pen = 0.02 * float(d2[pos_i, pos_j]) / max(r_ex2, 1e-6)
            tie = 1e-4 * float(rng_tx.random())
            score = comp_bonus + repeat_bonus - dist_pen + tie
            if score > best_score:
                best_score = score
                best_pos_j = pos_j
                best_qty = qty_ij

        if best_pos_j < 0:
            continue
        j = int(elig[best_pos_j])
        if float(a.inv_a[i]) < best_qty or float(a.inv_b[j]) < best_qty:
            continue
        if float(a.energy[i]) <= e_cost or float(a.energy[j]) <= e_cost:
            continue

        a.inv_a[i] -= best_qty
        a.inv_a[j] += best_qty
        a.inv_b[j] -= best_qty
        a.inv_b[i] += best_qty
        a.energy[i] -= e_cost
        a.energy[j] -= e_cost
        used_pos.add(pos_i)
        used_pos.add(best_pos_j)
        _record_trade(
            state,
            aid_a=aid_i,
            aid_b=int(a.ids[j]),
            qty=best_qty,
            role_a=role_i,
            role_b=roles[j],
        )
        trades_tick += 1
        volume_tick += 2.0 * best_qty

    return trades_tick, volume_tick


def compute_social_metrics(
    state: CulturalWorldState, exp: CulturalWorldExperiment
) -> dict[str, float]:
    """Read-only extraction of the 6 P6 social/division-of-labor metrics."""
    a = state.agents
    n = a.n
    trade_count = float(state.trade_count)
    trade_volume = float(state.trade_volume)
    repeat_rate = float(state.repeat_trades / state.trade_count) if state.trade_count > 0 else 0.0

    if n == 0:
        return {
            "trade_count": trade_count,
            "trade_volume": trade_volume,
            "specialization_index": 0.0,
            "division_of_labor_index": 0.0,
            "interdependence_index": 0.0,
            "repeat_partner_rate": repeat_rate,
        }

    specs: list[float] = []
    total_ua = 0
    total_ub = 0
    living_traders = 0
    living_ids = a.ids.tolist()

    for idx in range(n):
        aid = int(living_ids[idx])
        ua, ub, _ = state.agent_role_uses.get(aid, (0, 0, 0))
        ab = ua + ub
        if ab > 0:
            specs.append(abs(ua - ub) / float(ab))
            total_ua += ua
            total_ub += ub
        if state.agent_trades.get(aid, 0) > 0:
            living_traders += 1

    spec_index = float(sum(specs) / len(specs)) if specs else 0.0

    total_ab = total_ua + total_ub
    if total_ab > 0:
        pa = total_ua / float(total_ab)
        pb = total_ub / float(total_ab)
        ab_balance = 4.0 * pa * pb
    else:
        ab_balance = 0.0
    pop_comp = spec_index * ab_balance

    if state.trade_count == 0:
        interdep = 0.0
    else:
        recent = [ev for ev in state.trade_events if ev.tick > state.tick - RECENT_TRADE_WINDOW]
        recent_rate = min(1.0, len(recent) / max(8.0, 1.3 * n))
        recent_vol = sum(float(ev.qty_a_to_b + ev.qty_b_to_a) for ev in recent)
        recent_vol_rate = min(1.0, recent_vol / max(3.0, 0.35 * n))
        living_trader_frac = living_traders / float(n)
        partner_depth = 0.45 * living_trader_frac + 0.55 * repeat_rate
        if recent:
            comp_recent = sum(
                1
                for ev in recent
                if ev.role_a != ev.role_b and "generalist" not in (ev.role_a, ev.role_b)
            )
            comp_frac = comp_recent / float(len(recent))
        else:
            comp_frac = 0.0
        interdep = float(
            min(
                1.0,
                max(
                    0.0,
                    0.35 * recent_rate
                    + 0.25 * recent_vol_rate
                    + 0.25 * partner_depth
                    + 0.15 * comp_frac,
                ),
            )
        )

    dol = float(min(1.0, max(0.0, math.sqrt(max(0.0, pop_comp * interdep)))))
    return {
        "trade_count": trade_count,
        "trade_volume": trade_volume,
        "specialization_index": spec_index,
        "division_of_labor_index": dol,
        "interdependence_index": interdep,
        "repeat_partner_rate": repeat_rate,
    }


def _target_resource_label(tr: int) -> str:
    if tr == 3:
        return "dual_a_b"
    if tr == 1:
        return "material_a"
    return "material_b"


def extract_cultural_graph(
    state: CulturalWorldState,
    exp: CulturalWorldExperiment,
    *,
    max_nodes: int = 80,
    max_edges: int = 200,
) -> dict[str, Any]:
    """Build bounded read-only Knowledge DAG and Trade Network projections for GUI inspection."""
    del exp
    a = state.agents
    procs = state.registry.procedures

    # 1. Knowledge DAG
    holders: dict[int, int] = {}
    for slots in a.skills:
        for sl in slots:
            holders[sl.skill_id] = holders.get(sl.skill_id, 0) + 1

    # Prioritize living-held skills plus their ancestor lineage chains
    selected_sids: set[int] = set()
    active_sids = sorted(
        holders.keys(),
        key=lambda sid: (
            -holders[sid],
            -(procs[sid].depth if sid in procs else 0),
            -sid,
        ),
    )
    for sid in active_sids:
        if len(selected_sids) >= max_nodes:
            break
        queue = [sid]
        while queue and len(selected_sids) < max_nodes:
            cur = queue.pop(0)
            if cur in selected_sids or cur not in procs:
                continue
            selected_sids.add(cur)
            p = procs[cur]
            if p.parent_skill_id >= 0 and p.parent_skill_id not in selected_sids:
                queue.append(p.parent_skill_id)
            if p.secondary_parent_id >= 0 and p.secondary_parent_id not in selected_sids:
                queue.append(p.secondary_parent_id)

    if len(selected_sids) < max_nodes:
        remaining = sorted(
            (sid for sid in procs if sid not in selected_sids),
            key=lambda sid: (-procs[sid].depth, -procs[sid].efficiency, -sid),
        )
        for sid in remaining:
            if len(selected_sids) >= max_nodes:
                break
            selected_sids.add(sid)

    dag_nodes = []
    for sid in sorted(selected_sids, key=lambda s: (procs[s].depth, s)):
        p = procs[sid]
        dag_nodes.append(
            {
                "skill_id": int(p.skill_id),
                "parent_skill_id": int(p.parent_skill_id),
                "secondary_parent_id": int(p.secondary_parent_id),
                "depth": int(p.depth),
                "efficiency": round(float(p.efficiency), 4),
                "target_resource": _target_resource_label(p.target_resource),
                "ops": list(p.ops),
                "inventor_id": int(p.inventor_id),
                "inventor_dead": bool(state.registry.is_inventor_dead(p.skill_id)),
                "alive_holders": int(holders.get(p.skill_id, 0)),
            }
        )

    dag_edges: list[dict[str, Any]] = []
    for sid in sorted(selected_sids, key=lambda s: (procs[s].depth, s)):
        p = procs[sid]
        if p.parent_skill_id >= 0 and p.parent_skill_id in selected_sids:
            kind = "compose" if p.secondary_parent_id >= 0 else "refine"
            dag_edges.append(
                {"source": int(p.parent_skill_id), "target": int(p.skill_id), "kind": kind}
            )
        if p.secondary_parent_id >= 0 and p.secondary_parent_id in selected_sids:
            dag_edges.append(
                {
                    "source": int(p.secondary_parent_id),
                    "target": int(p.skill_id),
                    "kind": "compose",
                }
            )
        if len(dag_edges) >= max_edges:
            dag_edges = dag_edges[:max_edges]
            break

    # 2. Trade Network (living agents + aggregated recent TradeEvents between living endpoints)
    living_idx_by_id: dict[int, int] = {int(a.ids[idx]): idx for idx in range(a.n)}
    edge_agg: dict[tuple[int, int], tuple[int, float]] = {}
    for ev in state.trade_events:
        if ev.agent_a not in living_idx_by_id or ev.agent_b not in living_idx_by_id:
            continue
        if ev.agent_a == ev.agent_b:
            continue
        u, v = (ev.agent_a, ev.agent_b) if ev.agent_a < ev.agent_b else (ev.agent_b, ev.agent_a)
        cnt, vol = edge_agg.get((u, v), (0, 0.0))
        edge_agg[(u, v)] = (cnt + 1, vol + float(ev.qty_a_to_b + ev.qty_b_to_a))

    sorted_edges = sorted(
        edge_agg.items(),
        key=lambda item: (-item[1][1], -item[1][0], item[0][0], item[0][1]),
    )

    selected_aids: set[int] = set()
    for (u, v), _ in sorted_edges:
        if len(selected_aids) >= max_nodes:
            break
        if u not in selected_aids and v not in selected_aids and len(selected_aids) + 2 > max_nodes:
            continue
        selected_aids.add(u)
        if len(selected_aids) < max_nodes:
            selected_aids.add(v)

    if len(selected_aids) < max_nodes:
        rest_aids = sorted(
            (aid for aid in living_idx_by_id if aid not in selected_aids),
            key=lambda aid: (
                -state.agent_trades.get(aid, 0),
                -float(a.energy[living_idx_by_id[aid]]),
                aid,
            ),
        )
        for aid in rest_aids:
            if len(selected_aids) >= max_nodes:
                break
            selected_aids.add(aid)

    trade_nodes = []
    for aid in sorted(selected_aids):
        idx = living_idx_by_id[aid]
        ua, ub, ud = state.agent_role_uses.get(aid, (0, 0, 0))
        role = agent_role(
            ua,
            ub,
            ud,
            float(a.inv_a[idx]),
            float(a.inv_b[idx]),
            a.skills[idx],
            state.registry,
        )
        trade_nodes.append(
            {
                "agent_id": int(aid),
                "role": role,
                "energy": round(float(a.energy[idx]), 4),
                "inv_a": round(float(a.inv_a[idx]), 4),
                "inv_b": round(float(a.inv_b[idx]), 4),
                "trades": int(state.agent_trades.get(aid, 0)),
            }
        )

    trade_edges = []
    for (u, v), (cnt, vol) in sorted_edges:
        if u in selected_aids and v in selected_aids:
            trade_edges.append(
                {
                    "source": int(u),
                    "target": int(v),
                    "count": int(cnt),
                    "volume": round(float(vol), 4),
                }
            )
            if len(trade_edges) >= max_edges:
                break

    return {
        "knowledge_dag": {"nodes": dag_nodes, "edges": dag_edges},
        "trade_network": {"nodes": trade_nodes, "edges": trade_edges},
    }
