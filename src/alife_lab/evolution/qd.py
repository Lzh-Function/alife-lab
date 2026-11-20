"""Quality-diversity archive over a fixed two-descriptor grid (MAP-Elites, Mouret & Clune 2015).

The grid (descriptor names, bounds, bins) is pre-declared in the spec and never changes during
a run. A candidate enters only if it was viable on every search seed; its cell is given by its
mean descriptors (values outside the bounds fall into the edge bins); it replaces the cell's
elite only with a strictly higher quality (= objective score), so ties keep the earlier one.
The archive is a pure function of the evaluation history (recomputed on load).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import SearchQD

if TYPE_CHECKING:
    from alife_lab.substrates.search.adapter import SearchState

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def descriptors(state: SearchState, qd: SearchQD, idx: Any = slice(None)) -> F64:
    cols = [np.asarray(getattr(state, d.name))[idx] for d in qd.descriptors]
    return np.stack(cols, axis=-1).astype(np.float64)


def cells(desc: F64, qd: SearchQD) -> I64:
    """Flat cell index (row-major: descriptor 0 = column, descriptor 1 = row)."""
    out = []
    for k, d in enumerate(qd.descriptors):
        rel = (desc[..., k] - d.low) / (d.high - d.low)
        out.append(np.clip(np.floor(rel * d.bins), 0, d.bins - 1).astype(np.int64))
    return out[1] * qd.descriptors[0].bins + out[0]


def empty(qd: SearchQD) -> I64:
    return np.full(qd.descriptors[0].bins * qd.descriptors[1].bins, -1, dtype=np.int64)


def insert(elites: I64, state: SearchState, qd: SearchQD, start: int) -> int:
    """Insert candidates ``start..n-1`` in order; returns the number of cells changed."""
    n = state.score.size
    if start >= n:
        return 0
    idx = np.arange(start, n)
    ok = state.viable[idx] >= 1.0
    cell = cells(descriptors(state, qd, idx), qd)
    changed = 0
    for i, c, good in zip(idx, cell, ok, strict=True):
        if not good:
            continue
        cur = elites[c]
        if cur < 0 or state.score[i] > state.score[cur]:
            elites[c] = i
            changed += 1
    return changed


def build(state: SearchState, qd: SearchQD) -> I64:
    elites = empty(qd)
    insert(elites, state, qd, 0)
    return elites


def stats(elites: I64, state: SearchState) -> dict[str, float]:
    filled = elites[elites >= 0]
    return {
        "coverage": float(filled.size / elites.size),
        "qd_score": float(state.score[filled].sum()) if filled.size else 0.0,
        "archive_size": float(filled.size),
    }
