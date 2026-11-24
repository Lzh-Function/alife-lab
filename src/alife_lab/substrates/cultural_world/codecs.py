"""Rendering codec & exact deterministic checkpoint encoding for ``cultural_world``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.agents import CulturalAgents
from alife_lab.substrates.cultural_world.ecology import ResourceField, ResourceLedger
from alife_lab.substrates.cultural_world.memory import SkillRegistry, SkillSlot
from alife_lab.substrates.cultural_world.model import RNG_NAMES, CulturalWorldState
from alife_lab.substrates.cultural_world.technology import PlacedArtifact

SERIALIZE_VERSION = 2
RENDER_SCALE = 8


def draw_artifacts_overlay(
    img: np.ndarray[Any, np.dtype[np.float32]],
    artifacts: list[PlacedArtifact],
    w: float,
    h: float,
    r: int,
    s: int,
) -> None:
    """Draw physical externalized artifacts as crisp cyan 3x3 markers."""
    size = r * s
    for art in artifacts:
        px = math.floor(art.x / w * size) % size
        py = math.floor(art.y / h * size) % size
        y0, y1 = max(0, py - 1), min(size, py + 2)
        x0, x1 = max(0, px - 1), min(size, px + 2)
        img[y0:y1, x0:x1] = (0.15, 0.95, 0.98)


def render_frame(
    state: CulturalWorldState, exp: CulturalWorldExperiment
) -> tuple[dict[str, Any], bytes]:
    """Render 3-channel resource map, placed artifacts, and agents into ``field_u8``."""
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

    if state.artifacts:
        draw_artifacts_overlay(img, state.artifacts, w, h, r, s)

    a = state.agents
    if a.n > 0:
        px = np.floor(a.x / w * r * s).astype(np.int64) % (r * s)
        py = np.floor(a.y / h * r * s).astype(np.int64) % (r * s)
        for k in range(a.n):
            slots = a.skills[k]
            if not slots:
                col = (0.75, 0.78, 0.82)  # grey: no acquired skills yet
            elif any(sl.teacher_id >= 0 for sl in slots):
                col = (0.95, 0.35, 0.90)  # magenta: socially/artifact learned skill holder
            else:
                col = (0.98, 0.72, 0.18)  # amber: self-invented/composed skill holder
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
            "food (green), material A/B (amber/blue), cyan=artifact; "
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
            "artifacts": [art.to_json() for art in state.artifacts],
            "next_artifact_id": int(state.next_artifact_id),
            "valid_skill_uses": int(state.valid_skill_uses),
            "net_utility_sum": float(state.net_utility_sum),
            "externalized_learns": int(state.externalized_learns),
            "externalized_uses": int(state.externalized_uses),
            "compositions_created": int(state.compositions_created),
            "ledger": state.eco.ledger.to_json(),
            "registry": state.registry.to_json(),
            "skills": [[sl.to_json() for sl in s_list] for s_list in state.agents.skills],
            "rng_states": {k: state.rngs[k].bit_generator.state for k in RNG_NAMES},
        },
    }


def deserialize_state(payload: dict[str, Any], exp: CulturalWorldExperiment) -> CulturalWorldState:
    meta = payload["meta"]
    if meta["serialize_version"] not in (1, SERIALIZE_VERSION):
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
        artifacts=[PlacedArtifact.from_json(d) for d in meta.get("artifacts", [])],
        next_artifact_id=int(meta.get("next_artifact_id", 1)),
        valid_skill_uses=int(meta.get("valid_skill_uses", 0)),
        net_utility_sum=float(meta.get("net_utility_sum", 0.0)),
        externalized_learns=int(meta.get("externalized_learns", 0)),
        externalized_uses=int(meta.get("externalized_uses", 0)),
        compositions_created=int(meta.get("compositions_created", 0)),
    )
