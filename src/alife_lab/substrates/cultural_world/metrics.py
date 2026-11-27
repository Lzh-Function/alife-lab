"""Passive metric extraction for ``cultural_world`` (must never mutate state)."""

from __future__ import annotations

import math

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.ecology import max_ledger_residual
from alife_lab.substrates.cultural_world.model import CulturalWorldState
from alife_lab.substrates.cultural_world.social import compute_social_metrics

_SOCIAL_METRIC_KEYS = frozenset(
    {
        "trade_count",
        "trade_volume",
        "specialization_index",
        "division_of_labor_index",
        "interdependence_index",
        "repeat_partner_rate",
    }
)


def compute_metrics(state: CulturalWorldState, exp: CulturalWorldExperiment) -> dict[str, float]:
    wanted = set(exp.observations.metrics)
    a = state.agents
    n = a.n
    art_mass_a = sum(art.mass_a for art in state.artifacts)
    art_mass_b = sum(art.mass_b for art in state.artifacts)
    residual = max_ledger_residual(
        state.eco,
        (float(a.inv_a.sum()) if n else 0.0) + art_mass_a,
        (float(a.inv_b.sum()) if n else 0.0) + art_mass_b,
    )

    if n > 0:
        skill_counts = [len(s) for s in a.skills]
        mean_skills = sum(skill_counts) / n
        unique_skills: set[int] = set()
        unique_motifs: set[tuple[int, tuple[str, ...]]] = set()
        depths: list[int] = []
        improvements: list[float] = []
        total_slots = 0
        ext_slots = 0
        comp_slots = 0
        for slots in a.skills:
            for sl in slots:
                total_slots += 1
                if sl.mode == "externalized":
                    ext_slots += 1
                unique_skills.add(sl.skill_id)
                proc = state.registry.procedures.get(sl.skill_id)
                if proc is not None:
                    depths.append(proc.depth)
                    unique_motifs.add((proc.target_resource, proc.ops))
                    if proc.parent_skill_id >= 0:
                        improvements.append(proc.efficiency - proc.effective_root_efficiency())
                    if proc.secondary_parent_id >= 0 or (
                        "combine" in proc.ops or "transform" in proc.ops
                    ):
                        comp_slots += 1
        k_div = float(len(unique_skills))
        max_depth = float(max(depths)) if depths else 0.0
        mean_e = float(a.energy.mean())
        tech_novelty = (
            float(len(unique_motifs)) / float(max(1, total_slots)) if total_slots else 0.0
        )
        tech_improvement = (
            float(sum(improvements)) / float(len(improvements)) if improvements else 0.0
        )
        ext_frac = float(ext_slots) / float(total_slots) if total_slots else 0.0
        comp_frac = float(comp_slots) / float(total_slots) if total_slots else 0.0
    else:
        mean_skills = 0.0
        k_div = 0.0
        max_depth = 0.0
        mean_e = math.nan
        tech_novelty = 0.0
        tech_improvement = 0.0
        ext_frac = 0.0
        comp_frac = 0.0

    tx_rate = state.tx_successes / state.tx_attempts if state.tx_attempts > 0 else 0.0
    retention = (
        state.skill_uses_post_death / state.skill_uses_total if state.skill_uses_total > 0 else 0.0
    )
    social_frac = (
        state.skill_uses_social / state.skill_uses_total if state.skill_uses_total > 0 else 0.0
    )
    util = state.utility_sum / state.utility_count if state.utility_count > 0 else 0.0
    tech_validity = (
        state.valid_skill_uses / state.skill_uses_total if state.skill_uses_total > 0 else 0.0
    )
    tech_utility = (
        state.net_utility_sum / state.skill_uses_total if state.skill_uses_total > 0 else 0.0
    )
    tech_transfer = (
        min(1.0, (state.skill_uses_social + state.externalized_uses) / state.skill_uses_total)
        if state.skill_uses_total > 0
        else 0.0
    )

    values: dict[str, float] = {
        "population": float(n),
        "births": float(state.births),
        "deaths": float(state.deaths),
        "mean_energy": mean_e,
        "resource_ledger_residual": float(residual),
        "mean_skills_per_agent": float(mean_skills),
        "knowledge_diversity": k_div,
        "successful_transmission_rate": float(tx_rate),
        "skill_retention_after_death": float(retention),
        "social_learning_fraction": float(social_frac),
        "improvement_depth": max_depth,
        "innovation_utility": float(util),
        "tech_novelty": float(tech_novelty),
        "tech_validity": float(tech_validity),
        "tech_utility": float(tech_utility),
        "tech_transfer": float(tech_transfer),
        "tech_improvement": float(tech_improvement),
        "active_artifacts": float(len(state.artifacts)),
        "externalized_learning_fraction": float(ext_frac),
        "compositional_skill_fraction": float(comp_frac),
    }
    if wanted & _SOCIAL_METRIC_KEYS:
        values.update(compute_social_metrics(state, exp))
    return {k: v for k, v in values.items() if k in wanted}
