"""Rendering codec & exact deterministic checkpoint encoding for ``cultural_world``."""

from __future__ import annotations

from typing import Any

import numpy as np

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.agents import CulturalAgents
from alife_lab.substrates.cultural_world.ecology import ResourceField, ResourceLedger
from alife_lab.substrates.cultural_world.memory import SkillRegistry, SkillSlot
from alife_lab.substrates.cultural_world.model import RNG_NAMES, CulturalWorldState

SERIALIZE_VERSION = 1
RENDER_SCALE = 8


def render_frame(
    state: CulturalWorldState, exp: CulturalWorldExperiment
) -> tuple[dict[str, Any], bytes]:
    """Render 3-channel resource map + culturally color-coded agents into ``field_u8``."""
    s = RENDER_SCALE
    r = exp.world.resolution
    w, h = exp.world.width, exp.world.height
    res = exp.dynamics.resources

    img = np.zeros((r * s, r * s, 3), dtype=np.float32)
    food_norm = np.clip(state.eco.food / max(1e-6, res.food_capacity), 0.0, 1.0)
    mata_norm = np.clip(state.eco.material_a / max(1e-6, res.material_a_capacity), 0.0, 1.0)
    matb_norm = np.clip(state.eco.material_b / max(1e-6, res.material_b_capacity), 0.0, 1.0)
    # Green = food, Red/Amber tint = material_a, Blue = material_b
    img[..., 0] = np.kron(mata_norm * 0.35, np.ones((s, s)))
    img[..., 1] = np.kron(food_norm * 0.55 + mata_norm * 0.15, np.ones((s, s)))
    img[..., 2] = np.kron(matb_norm * 0.55, np.ones((s, s)))

    a = state.agents
    if a.n > 0:
        px = np.floor(a.x / w * r * s).astype(np.int64) % (r * s)
        py = np.floor(a.y / h * r * s).astype(np.int64) % (r * s)
        for k in range(a.n):
            slots = a.skills[k]
            if not slots:
                col = (0.75, 0.78, 0.82)  # grey: no acquired skills yet
            elif any(sl.teacher_id >= 0 for sl in slots):
                col = (0.95, 0.35, 0.90)  # magenta: socially learned skill holder
            else:
                col = (0.98, 0.72, 0.18)  # amber: self-invented/improved skill holder
            y0, y1 = max(0, int(py[k]) - 1), min(r * s, int(py[k]) + 2)
            x0, x1 = max(0, int(px[k]) - 1), min(r * s, int(px[k]) + 2)
            img[y0:y1, x0:x1] = col

    data = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
    header: dict[str, Any] = {
        "codec": "field_u8",
        "width": r * s,
        "height": r * s,
        "channels": 3,
        "scale": (
            "food (green), material A/B (amber/blue); "
            "agents: grey=naive, gold=inventor, magenta=social learner"
        ),
    }
    return header, data


def serialize_state(state: CulturalWorldState) -> dict[str, Any]:
    arrays = {k: v.copy() for k, v in state.agents.array_fields().items()}
    arrays.update(
        {
            "food": state.eco.food.copy(),
            "material_a": state.eco.material_a.copy(),
            "material_b": state.eco.material_b.copy(),
        }
    )
    return {
        "arrays": arrays,
        "meta": {
            "serialize_version": SERIALIZE_VERSION,
            "tick": int(state.tick),
            "next_id": int(state.next_id),
            "next_skill_id": int(state.next_skill_id),
            "births": int(state.births),
            "deaths": int(state.deaths),
            "tx_attempts": int(state.tx_attempts),
            "tx_successes": int(state.tx_successes),
            "skill_uses_total": int(state.skill_uses_total),
            "skill_uses_social": int(state.skill_uses_social),
            "skill_uses_post_death": int(state.skill_uses_post_death),
            "utility_sum": float(state.utility_sum),
            "utility_count": int(state.utility_count),
            "env_phase_offset": float(state.env_phase_offset),
            "tx_overrides": dict(state.tx_overrides),
            "ledger": state.eco.ledger.to_json(),
            "registry": state.registry.to_json(),
            "skills": [[sl.to_json() for sl in s_list] for s_list in state.agents.skills],
            "rng_states": {k: state.rngs[k].bit_generator.state for k in RNG_NAMES},
        },
    }


def deserialize_state(payload: dict[str, Any], exp: CulturalWorldExperiment) -> CulturalWorldState:
    meta = payload["meta"]
    if meta["serialize_version"] != SERIALIZE_VERSION:
        raise ValueError("incompatible cultural_world checkpoint serialize_version")
    arr = payload["arrays"]
    fields = {
        k: np.asarray(arr[k]).copy() for k in arr if k not in ("food", "material_a", "material_b")
    }
    skills = [[SkillSlot.from_json(d) for d in s_list] for s_list in meta["skills"]]
    agents = CulturalAgents(**fields, skills=skills)

    food = np.asarray(arr["food"], dtype=np.float64).copy()
    mat_a = np.asarray(arr["material_a"], dtype=np.float64).copy()
    mat_b = np.asarray(arr["material_b"], dtype=np.float64).copy()
    r = exp.world.resolution
    if food.shape != (r, r):
        raise ValueError("cultural_world checkpoint grid shape mismatch")

    eco = ResourceField(
        food=food,
        material_a=mat_a,
        material_b=mat_b,
        ledger=ResourceLedger.from_json(meta["ledger"]),
    )
    rngs = {}
    for k in RNG_NAMES:
        g = np.random.Generator(np.random.PCG64())
        g.bit_generator.state = meta["rng_states"][k]
        rngs[k] = g

    return CulturalWorldState(
        eco=eco,
        agents=agents,
        registry=SkillRegistry.from_json(meta["registry"]),
        tick=int(meta["tick"]),
        next_id=int(meta["next_id"]),
        next_skill_id=int(meta["next_skill_id"]),
        rngs=rngs,
        births=int(meta["births"]),
        deaths=int(meta["deaths"]),
        tx_attempts=int(meta["tx_attempts"]),
        tx_successes=int(meta["tx_successes"]),
        skill_uses_total=int(meta["skill_uses_total"]),
        skill_uses_social=int(meta["skill_uses_social"]),
        skill_uses_post_death=int(meta["skill_uses_post_death"]),
        utility_sum=float(meta["utility_sum"]),
        utility_count=int(meta["utility_count"]),
        env_phase_offset=float(meta["env_phase_offset"]),
        tx_overrides={str(k): bool(v) for k, v in meta.get("tx_overrides", {}).items()},
    )
