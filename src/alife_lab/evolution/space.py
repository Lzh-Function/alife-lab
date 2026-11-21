"""Search space: maps normalised coordinates u in [0, 1]^d to concrete target-world specs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from alife_lab.core.spec import ExperimentSpec, SearchExperiment, search_path_kind

F64 = npt.NDArray[np.float64]


def decode(exp: SearchExperiment, u: F64) -> dict[str, float | int]:
    """Parameter values for normalised coordinates (clipped to the box)."""
    out: dict[str, float | int] = {}
    for p, ui in zip(exp.space, np.clip(u, 0.0, 1.0), strict=True):
        if p.scale == "log":
            value = math.exp(math.log(p.low) + float(ui) * (math.log(p.high) - math.log(p.low)))
        elif p.scale == "logit":
            lo, hi = _logit(p.low), _logit(p.high)
            value = 1.0 / (1.0 + math.exp(-(lo + float(ui) * (hi - lo))))
        else:
            value = p.low + float(ui) * (p.high - p.low)
        out[p.path] = round(value) if search_path_kind(exp.target, p.path) else value
    return out


def encode(exp: SearchExperiment, params: dict[str, float]) -> F64:
    """Inverse of :func:`decode` (for display; integer rounding is not inverted)."""
    u = []
    for p in exp.space:
        v = float(params[p.path])
        if p.scale == "log":
            u.append((math.log(v) - math.log(p.low)) / (math.log(p.high) - math.log(p.low)))
        elif p.scale == "logit":
            u.append((_logit(v) - _logit(p.low)) / (_logit(p.high) - _logit(p.low)))
        else:
            u.append((v - p.low) / (p.high - p.low))
    return np.asarray(u, dtype=np.float64)


def _logit(v: float) -> float:
    return math.log(v) - math.log1p(-v)


def bound_violation(u: F64) -> float:
    """Squared distance of u to the unit box (used as a CMA-ES penalty)."""
    return float(((u - np.clip(u, 0.0, 1.0)) ** 2).sum())


def candidate_spec(
    exp: SearchExperiment, params: dict[str, float | int], seed: int, name: str | None = None
) -> ExperimentSpec | None:
    """Full, validated target spec for one candidate and one seed; None if invalid."""
    body: dict[str, Any] = exp.target.model_dump(mode="json")
    for path, value in params.items():
        node = body
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        if isinstance(node, list):
            node[int(keys[-1])] = value
        else:
            node[keys[-1]] = value
    body["seed"] = int(seed)
    body["mode"] = "exploratory"
    body["name"] = name or f"{exp.name}-candidate"
    try:
        return ExperimentSpec.model_validate({"experiment": body})
    except ValidationError:
        return None


def seeds_for(exp: SearchExperiment, stream: int, count: int) -> list[int]:
    """Deterministic evaluation seeds: stream 1 = search seeds, stream 2 = holdout seeds."""
    if count == 0:
        return []
    raw = np.random.SeedSequence([exp.seed, stream]).generate_state(count, dtype=np.uint64)
    return [int(v >> np.uint64(1)) for v in raw]
