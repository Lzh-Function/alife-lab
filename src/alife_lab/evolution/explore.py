"""Novelty search and IMGEP over the evaluation history of a search run.

Behaviour space: with a ``qd`` section the two descriptors scaled to [0, 1] by their declared
bounds; otherwise (novelty only) the CLIP embedding of the last sampled frame. Candidates that
were not viable on every search seed have no behaviour (NaN) and are never parents.

* ``novelty`` (Lehman & Stanley 2011): novelty = mean distance to the k nearest neighbours
  among the archive and the current generation. Parents come from the last generation by
  binary tournaments on novelty; children are parents + N(0, sigma) in the normalised box
  (clipped). After each generation the ``archive_add`` most novel new candidates join the
  archive. Generation 1, and any generation without a viable parent, is sampled uniformly.
* ``imgep`` (goal exploration, Forestier et al. 2017; Reinke et al. 2020 for Lenia): every
  child gets a goal sampled uniformly in the behaviour box; its parent is the viable candidate
  so far whose behaviour is closest to the goal; the child is parent + N(0, sigma). Uniform
  samples until a viable candidate exists.

Both use only the run's RNG (deterministic, checkpointed) and record parents for lineage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import SearchExperiment

if TYPE_CHECKING:
    from alife_lab.substrates.search.adapter import SearchState

F64 = npt.NDArray[np.float64]


def behaviour(state: SearchState, exp: SearchExperiment, idx: Any = slice(None)) -> F64:
    """(n, d) behaviour vectors; rows of non-viable candidates are NaN."""
    if exp.qd is not None:
        cols = []
        for d in exp.qd.descriptors:
            v = np.asarray(getattr(state, d.name))[idx]
            cols.append((v - d.low) / (d.high - d.low))
        out: F64 = np.stack(cols, axis=-1).astype(np.float64)
    else:
        out = np.array(state.embedding[idx], dtype=np.float64)
    ok = np.asarray(state.viable)[idx] >= 1.0
    out[~ok] = np.nan
    return out


def knn_novelty(points: F64, reference: F64, k: int, self_in_reference: bool) -> F64:
    """Mean distance of each point to its k nearest reference points (NaN rows -> 0)."""
    valid_ref = reference[~np.isnan(reference).any(axis=1)]
    out = np.zeros(len(points))
    if len(valid_ref) == 0:
        return out
    for i, p in enumerate(points):
        if np.isnan(p).any():
            continue
        d = np.sqrt(((valid_ref - p) ** 2).sum(axis=1))
        d = np.sort(d)
        if self_in_reference:
            d = d[1:]  # drop the zero distance to itself
        if d.size:
            out[i] = float(d[: min(k, d.size)].mean())
    return out


def propose_novelty(
    state: SearchState, exp: SearchExperiment, rng: np.random.Generator, pop: int, dim: int
) -> tuple[F64, F64]:
    assert exp.explore is not None
    last = np.flatnonzero(state.generation == state.tick)
    bd = behaviour(state, exp)
    viable_last = last[~np.isnan(bd[last]).any(axis=1)]
    if state.tick == 0 or viable_last.size == 0:
        return rng.random((pop, dim)), np.full(pop, -1.0)
    ref = np.vstack([bd[state.novelty_archive], bd[last]]) if state.novelty_archive else bd[last]
    nov = knn_novelty(bd[viable_last], ref, exp.explore.k, self_in_reference=True)
    a = rng.integers(0, viable_last.size, size=pop)
    b = rng.integers(0, viable_last.size, size=pop)
    winners = np.where(nov[a] >= nov[b], viable_last[a], viable_last[b])
    noise = rng.standard_normal((pop, dim)) * exp.explore.mutation_sigma
    return np.clip(state.u[winners] + noise, 0.0, 1.0), winners.astype(np.float64)


def update_archive(state: SearchState, exp: SearchExperiment, start: int) -> float:
    """Add the most novel new candidates to the archive; returns their mean novelty."""
    assert exp.explore is not None
    new = np.arange(start, state.score.size)
    bd = behaviour(state, exp)
    ref = np.vstack([bd[state.novelty_archive], bd[new]]) if state.novelty_archive else bd[new]
    nov = knn_novelty(bd[new], ref, exp.explore.k, self_in_reference=True)
    viable = ~np.isnan(bd[new]).any(axis=1)
    order = [int(new[i]) for i in np.argsort(-nov, kind="stable") if viable[i]]
    state.novelty_archive.extend(order[: exp.explore.archive_add])
    return float(nov[viable].mean()) if viable.any() else 0.0


def propose_imgep(
    state: SearchState, exp: SearchExperiment, rng: np.random.Generator, pop: int, dim: int
) -> tuple[F64, F64]:
    assert exp.explore is not None
    bd = behaviour(state, exp)
    known = np.flatnonzero(~np.isnan(bd).any(axis=1)) if bd.size else np.zeros(0, dtype=int)
    if known.size == 0:
        return rng.random((pop, dim)), np.full(pop, -1.0)
    goals = rng.random((pop, bd.shape[1]))
    d2 = ((bd[known][None, :, :] - goals[:, None, :]) ** 2).sum(axis=2)  # (pop, known)
    parents = known[np.argmin(d2, axis=1)]
    noise = rng.standard_normal((pop, dim)) * exp.explore.mutation_sigma
    return np.clip(state.u[parents] + noise, 0.0, 1.0), parents.astype(np.float64)
