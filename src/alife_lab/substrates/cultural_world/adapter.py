"""``SubstrateAdapter`` implementation for ``cultural_world``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import CulturalWorldExperiment, ExperimentSpec
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.cultural_world import codecs, interventions, metrics, model
from alife_lab.substrates.cultural_world.ecology import ledger_residuals
from alife_lab.substrates.cultural_world.model import CulturalWorldState
from alife_lab.substrates.cultural_world.transmission import resolve_transmission_flags

_M = MetricDefinition
_METRICS = (
    _M(
        "population",
        "metric.population",
        "agents",
        "tick",
        "Current number of living agents.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "births",
        "metric.births",
        "count",
        "tick",
        "Cumulative births.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "deaths",
        "metric.deaths",
        "count",
        "tick",
        "Cumulative deaths.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_energy",
        "metric.mean_energy",
        "energy",
        "tick",
        "Mean agent energy.",
        "none",
        "every tick",
        1,
        "null without agents",
    ),
    _M(
        "resource_ledger_residual",
        "metric.resource_ledger_residual",
        "mass",
        "tick",
        "Max absolute deviation across resource conservation ledgers (0 = exact conservation).",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "mean_skills_per_agent",
        "metric.mean_skills_per_agent",
        "skills",
        "tick",
        "Mean number of transmissible skills carried per living agent.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "knowledge_diversity",
        "metric.knowledge_diversity",
        "skills",
        "tick",
        "Distinct executable skill procedures held by living agents.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "successful_transmission_rate",
        "metric.successful_transmission_rate",
        "fraction",
        "tick",
        "Successful transmissions divided by transmission attempts.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "skill_retention_after_death",
        "metric.skill_retention_after_death",
        "fraction",
        "tick",
        "Share of skill executions using procedures whose inventor has already died.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "social_learning_fraction",
        "metric.social_learning_fraction",
        "fraction",
        "tick",
        "Share of skill executions performed via socially acquired skills.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "improvement_depth",
        "metric.improvement_depth",
        "depth",
        "tick",
        "Maximum ancestor-refinement depth across skills currently held by living agents.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "innovation_utility",
        "metric.innovation_utility",
        "energy/unit",
        "tick",
        "Mean gross energy return per unit of processed material across skill executions.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "tech_novelty",
        "metric.tech_novelty",
        "fraction",
        "tick",
        "Distinct (target_resource, ops) procedure signatures per living skill slot.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "tech_validity",
        "metric.tech_validity",
        "fraction",
        "tick",
        "Share of skill executions producing strictly positive net energy after skill_energy_cost.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "tech_utility",
        "metric.tech_utility",
        "energy/unit",
        "tick",
        "Mean net energy return per unit material after deducting skill_energy_cost.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "tech_transfer",
        "metric.tech_transfer",
        "fraction",
        "tick",
        "Share of skill executions using socially or artifact-transferred procedures.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "tech_improvement",
        "metric.tech_improvement",
        "efficiency",
        "tick",
        "Mean efficiency gain over ancestral root skill across refined/composed living skills.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "active_artifacts",
        "metric.active_artifacts",
        "objects",
        "tick",
        "Number of active physical externalized artifacts currently on the arena.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "externalized_learning_fraction",
        "metric.externalized_learning_fraction",
        "fraction",
        "tick",
        "Share of living skill slots acquired asynchronously from PlacedArtifact objects.",
        "none",
        "every tick",
        1,
        "never null",
    ),
    _M(
        "compositional_skill_fraction",
        "metric.compositional_skill_fraction",
        "fraction",
        "tick",
        "Share of living skill slots formed by two-parent composition or combine/transform ops.",
        "none",
        "every tick",
        1,
        "never null",
    ),
)

CULTURAL_WORLD_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.cultural_world",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="cultural_world",
    display_name_key="substrate.cultural_world",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="particles",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=True,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=(
            "erase_culture",
            "destroy_objects",
            "set_transmission",
            "shock_environment",
            "add_resource",
        ),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=codecs.SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": (
            "Exploratory research world separating Genetic Evolution, Lifetime Learning, and "
            "Cultural Transmission with strict conservation ledgers and causal interventions."
        ),
        "implementation": "this repository",
    },
)


def _cw(spec: ExperimentSpec) -> CulturalWorldExperiment:
    exp = spec.experiment
    if not isinstance(exp, CulturalWorldExperiment):
        raise TypeError("cultural_world adapter received a different spec")
    return exp


class CulturalWorldAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return CULTURAL_WORLD_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> CulturalWorldState:
        return model.init_world(_cw(spec))

    def step(self, state: CulturalWorldState, spec: ExperimentSpec) -> CulturalWorldState:
        return model.step_world(state, _cw(spec))

    def metrics(self, state: CulturalWorldState, spec: ExperimentSpec) -> dict[str, float]:
        return metrics.compute_metrics(state, _cw(spec))

    def render(self, state: CulturalWorldState) -> tuple[dict[str, Any], bytes]:
        # Derive world geometry from field shape & scale cleanly without needing spec
        r = state.eco.food.shape[0]
        s = codecs.RENDER_SCALE
        # Build lightweight view compatible with render_frame expectations
        food_cap = max(float(state.eco.food.max()), 1.0)
        mata_cap = max(float(state.eco.material_a.max()), 0.8)
        matb_cap = max(float(state.eco.material_b.max()), 0.8)
        img = np.zeros((r * s, r * s, 3), dtype=np.float32)
        img[..., 0] = np.kron(
            np.clip(state.eco.material_a / mata_cap, 0.0, 1.0) * 0.35, np.ones((s, s))
        )
        img[..., 1] = np.kron(
            np.clip(state.eco.food / food_cap, 0.0, 1.0) * 0.55
            + np.clip(state.eco.material_a / mata_cap, 0.0, 1.0) * 0.15,
            np.ones((s, s)),
        )
        img[..., 2] = np.kron(
            np.clip(state.eco.material_b / matb_cap, 0.0, 1.0) * 0.55, np.ones((s, s))
        )

        a = state.agents
        if state.artifacts:
            w = max(64.0, math.ceil(float(a.x.max()))) if a.x.size else 64.0
            h = max(64.0, math.ceil(float(a.y.max()))) if a.y.size else 64.0
            codecs.draw_artifacts_overlay(img, state.artifacts, w, h, r, s)

        if a.n > 0:
            w = max(float(a.x.max()), 64.0) if a.x.size else 64.0
            h = max(float(a.y.max()), 64.0) if a.y.size else 64.0
            # Round domain to nearest power/standard size >= 16 if slightly below boundary
            w = max(64.0, math.ceil(w))
            h = max(64.0, math.ceil(h))
            px = np.floor(a.x / w * r * s).astype(np.int64) % (r * s)
            py = np.floor(a.y / h * r * s).astype(np.int64) % (r * s)
            for k in range(a.n):
                slots = a.skills[k]
                if not slots:
                    col = (0.75, 0.78, 0.82)
                elif any(sl.teacher_id >= 0 for sl in slots):
                    col = (0.95, 0.35, 0.90)
                else:
                    col = (0.98, 0.72, 0.18)
                y0, y1 = max(0, int(py[k]) - 1), min(r * s, int(py[k]) + 2)
                x0, x1 = max(0, int(px[k]) - 1), min(r * s, int(px[k]) + 2)
                img[y0:y1, x0:x1] = col
        data = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
        return {
            "codec": "field_u8",
            "width": r * s,
            "height": r * s,
            "channels": 3,
            "scale": (
                "food (green), material A/B, cyan=artifact; "
                "agents: grey=naive, gold=inventor, magenta=social learner"
            ),
        }, data

    def apply_intervention(
        self, state: CulturalWorldState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[CulturalWorldState, dict[str, Any]]:
        return interventions.apply_intervention(state, _cw(spec), kind, payload)

    def inspect(
        self, state: CulturalWorldState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _cw(spec)
        w, h = exp.world.width, exp.world.height
        x, y = float(query.get("x", 0.0)), float(query.get("y", 0.0))
        a = state.agents
        agent_info: dict[str, Any] | None = None
        if a.n > 0:
            dx = (a.x - x + 0.5 * w) % w - 0.5 * w
            dy = (a.y - y + 0.5 * h) % h - 0.5 * h
            k = int(np.argmin(dx * dx + dy * dy))
            skills_detail = []
            for sl in a.skills[k]:
                proc = state.registry.procedures[sl.skill_id]
                skills_detail.append(
                    {
                        **sl.to_json(),
                        "parent_skill_id": proc.parent_skill_id,
                        "secondary_parent_id": proc.secondary_parent_id,
                        "revision": proc.revision,
                        "depth": proc.depth,
                        "inventor_id": proc.inventor_id,
                        "target_resource": (
                            "dual_a_b"
                            if proc.target_resource == 3
                            else ("material_a" if proc.target_resource == 1 else "material_b")
                        ),
                        "ops": list(proc.ops),
                        "efficiency": round(proc.efficiency, 3),
                        "phase": round(proc.phase, 3),
                    }
                )
            agent_info = {
                "id": int(a.ids[k]),
                "parent": int(a.parent[k]),
                "birth_tick": int(a.birth[k]),
                "age": int(a.age[k]),
                "energy": round(float(a.energy[k]), 3),
                "inv_a": round(float(a.inv_a[k]), 3),
                "inv_b": round(float(a.inv_b[k]), 3),
                "learned_delta_norm": round(float(np.linalg.norm(a.dw2[k])), 4),
                "private_best_val": round(float(a.private_best_val[k]), 3),
                "skills": skills_detail,
            }
        v_on, h_on, o_on = resolve_transmission_flags(exp, state.tx_overrides)
        ext_on = (
            bool(state.tx_overrides["externalized"])
            if "externalized" in state.tx_overrides
            else (exp.dynamics.technology.allow_externalization and (v_on or h_on or o_on))
        )
        residuals = ledger_residuals(
            state.eco,
            (float(a.inv_a.sum()) if a.n else 0.0) + sum(art.mass_a for art in state.artifacts),
            (float(a.inv_b.sum()) if a.n else 0.0) + sum(art.mass_b for art in state.artifacts),
        )
        nearby_artifacts = []
        for art in state.artifacts:
            adx = (art.x - x + 0.5 * w) % w - 0.5 * w
            ady = (art.y - y + 0.5 * h) % h - 0.5 * h
            if adx * adx + ady * ady <= exp.dynamics.transmission.observation_range**2:
                aproc = state.registry.procedures.get(art.skill_id)
                item = art.to_json()
                if aproc is not None:
                    item["ops"] = list(aproc.ops)
                    item["efficiency"] = round(aproc.efficiency, 3)
                    item["target_resource"] = (
                        "dual_a_b"
                        if aproc.target_resource == 3
                        else ("material_a" if aproc.target_resource == 1 else "material_b")
                    )
                nearby_artifacts.append(item)
        return {
            "substrate": "cultural_world",
            "tick": state.tick,
            "query": {"x": x, "y": y},
            "population": a.n,
            "transmission": {"vertical": v_on, "horizontal": h_on, "oblique": o_on},
            "externalized_enabled": ext_on,
            "active_artifacts": len(state.artifacts),
            "nearby_artifacts": nearby_artifacts[:8],
            "ledger_residuals": {k: round(float(v), 12) for k, v in residuals.items()},
            "recent_transmissions": state.registry.transmissions[-8:],
            "cultural_agent": agent_info,
        }

    def state_distance(self, a: CulturalWorldState, b: CulturalWorldState) -> dict[str, Any]:
        same_arrs = (
            a.tick == b.tick
            and np.array_equal(a.agents.ids, b.agents.ids)
            and np.array_equal(a.agents.x, b.agents.x)
            and np.array_equal(a.agents.energy, b.agents.energy)
            and np.array_equal(a.agents.dw2, b.agents.dw2)
            and np.array_equal(a.eco.food, b.eco.food)
            and np.array_equal(a.eco.material_a, b.eco.material_a)
            and a.agents.skills == b.agents.skills
            and a.artifacts == b.artifacts
        )
        return {
            "metric": "population_difference",
            "value": abs(a.agents.n - b.agents.n),
            "exact_equal": bool(same_arrs),
        }

    def serialize_state(self, state: CulturalWorldState) -> dict[str, Any]:
        return codecs.serialize_state(state)

    def deserialize_state(
        self, payload: dict[str, Any], spec: ExperimentSpec
    ) -> CulturalWorldState:
        return codecs.deserialize_state(payload, _cw(spec))
