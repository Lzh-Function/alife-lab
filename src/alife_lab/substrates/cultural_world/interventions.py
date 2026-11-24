"""Causal interventions for ``cultural_world`` at tick boundaries (ADR 0050)."""

from __future__ import annotations

from typing import Any

import numpy as np

from alife_lab.core.spec import CulturalWorldExperiment
from alife_lab.substrates.cultural_world.model import CulturalWorldState


def apply_intervention(
    state: CulturalWorldState,
    exp: CulturalWorldExperiment,
    kind: str,
    payload: dict[str, Any],
) -> tuple[CulturalWorldState, dict[str, Any]]:
    """Apply causal or ecological interventions cleanly without mutating unrelated layers."""
    if kind == "erase_culture":
        # Clear ONLY transmissible skills; genotypes, dw2, energy, inventories stay bit-identical.
        cleared_slots = sum(len(s) for s in state.agents.skills)
        state.agents.skills = [[] for _ in range(state.agents.n)]
        return state, {"erased_skill_slots": cleared_slots, "population": state.agents.n}

    if kind == "set_transmission":
        updated: dict[str, bool] = {}
        for ch in ("vertical", "horizontal", "oblique"):
            if ch in payload:
                val = bool(payload[ch])
                state.tx_overrides[ch] = val
                updated[ch] = val
        if not updated:
            # Convenience toggle: mute or restore all transmission channels together.
            mute = not any(
                state.tx_overrides.get(c, True) for c in ("vertical", "horizontal", "oblique")
            )
            for ch in ("vertical", "horizontal", "oblique"):
                state.tx_overrides[ch] = mute
                updated[ch] = mute
        return state, {"tx_overrides": dict(state.tx_overrides)}

    if kind == "shock_environment":
        delta = float(payload.get("phase_shift", 0.5))
        state.env_phase_offset = (state.env_phase_offset + delta) % 1.0
        return state, {"env_phase_offset": round(state.env_phase_offset, 4)}

    if kind == "add_resource":
        try:
            x = float(payload["x"])
            y = float(payload["y"])
            radius = float(payload.get("radius", 6.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("add_resource payload requires numeric x and y") from exc
        w, h = exp.world.width, exp.world.height
        if not (0.0 <= x < w and 0.0 <= y < h) or not (0.0 <= radius <= 512.0):
            raise ValueError("position or radius out of range")
        res_kind = str(payload.get("kind", "food"))
        grid = state.eco.grid(res_kind)
        r = exp.world.resolution
        cy = (np.arange(r) + 0.5) * h / r
        cx = (np.arange(r) + 0.5) * w / r
        dy = (cy[:, None] - y + 0.5 * h) % h - 0.5 * h
        dx = (cx[None, :] - x + 0.5 * w) % w - 0.5 * w
        mask = dx * dx + dy * dy <= radius * radius
        amount = float(payload.get("amount", 0.5))
        if amount < 0.0 or amount > 20.0:
            raise ValueError("amount must be within [0, 20]")
        grid[mask] += amount
        added = float(amount * int(mask.sum()))
        state.eco.ledger.inflow[res_kind] += added
        return state, {"kind": res_kind, "cells": int(mask.sum()), "added": added}

    raise ValueError(f"unsupported cultural_world intervention {kind!r}")
