"""Passive metric extraction for ``cultural_world`` (must never mutate state)."""

from __future__ import annotations

import math

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.ecology import max_ledger_residual
from alife_lab.substrates.cultural_world.model import CulturalWorldState


def compute_metrics(state: CulturalWorldState, exp: CulturalWorldExperiment) -> dict[str, float]:
    wanted = set(exp.observations.metrics)
    a = state.agents
    n = a.n
    residual = max_ledger_residual(
        state.eco,
        float(a.inv_a.sum()) if n else 0.0,
        float(a.inv_b.sum()) if n else 0.0,
    )

    if n > 0:
        skill_counts = [len(s) for s in a.skills]
        mean_skills = sum(skill_counts) / n
        unique_skills: set[int] = set()
        depths: list[int] = []
        for slots in a.skills:
            for sl in slots:
                unique_skills.add(sl.skill_id)
                proc = state.registry.procedures.get(sl.skill_id)
                if proc is not None:
                    depths.append(proc.depth)
        k_div = float(len(unique_skills))
        max_depth = float(max(depths)) if depths else 0.0
        mean_e = float(a.energy.mean())
    else:
        mean_skills = 0.0
        k_div = 0.0
        max_depth = 0.0
        mean_e = math.nan

    tx_rate = state.tx_successes / state.tx_attempts if state.tx_attempts > 0 else 0.0
    retention = (
        state.skill_uses_post_death / state.skill_uses_total if state.skill_uses_total > 0 else 0.0
    )
    social_frac = (
        state.skill_uses_social / state.skill_uses_total if state.skill_uses_total > 0 else 0.0
    )
    util = state.utility_sum / state.utility_count if state.utility_count > 0 else 0.0

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
    }
    return {k: v for k, v in values.items() if k in wanted}
