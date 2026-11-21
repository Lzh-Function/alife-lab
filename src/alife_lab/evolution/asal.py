"""ASAL scores (Kumar et al. 2024, "Automating the Search for Artificial Life with Foundation
Models"), NumPy ports of ``asal_metrics.py`` from ``SakanaAI/asal`` @ 677ba0e (Apache-2.0,
Copyright Sakana AI). The originals return losses to minimise; here every function returns a
*score to maximise* (the negated loss), as all ALife Lab objectives do.

Embeddings are L2-normalised rows (cosine similarity = dot product).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def supervised_target(z: F64, z_txt: F64) -> float:
    """Mean similarity of frame t to prompt ``t mod P``.

    ASAL tiles the prompts with ``repeat(z_txt, "T2 D -> (k T2) D")``, i.e. frame t is paired
    with prompt ``t % P`` (with the default ``time_sampling = P`` this is one frame per prompt).
    """
    t, p = z.shape[0], z_txt.shape[0]
    if t % p:
        raise ValueError("the number of frames must be a multiple of the number of prompts")
    paired = z_txt[np.arange(t) % p]
    return float(np.mean(np.sum(paired * z, axis=1)))


def open_endedness(z: F64) -> float:
    """Negated ASAL open-endedness loss: -mean_t max(tril(Z Z^T, -1)[t]).

    As in the original, entries on/above the diagonal are zeroed (not excluded), so the first
    frame contributes max(0, ...) = 0 and similarities are effectively clipped below at 0.
    Higher = later frames look less like any earlier frame.
    """
    k = np.tril(z @ z.T, k=-1)
    return -float(k.max(axis=-1).mean())


def illumination(zs: F64) -> float:
    """Negated ASAL illumination loss: -mean_i max_{j != i} sim(z_i, z_j) over a population."""
    k = zs @ zs.T
    np.fill_diagonal(k, -np.inf)
    return -float(k.max(axis=-1).mean())


def farthest_subset(zs: F64, n: int) -> F64:
    """Greedy max-min (farthest-point) selection of ``n`` rows, starting from row 0.

    Used to compare the diversity of searches with different numbers of viable candidates:
    the illumination of all candidates falls as their number grows (each gets closer
    neighbours), the illumination of a fixed-size spread-out subset does not.
    """
    if len(zs) <= n:
        return zs
    chosen = [0]
    best_sim = zs @ zs[0]
    for _ in range(n - 1):
        nxt = int(np.argmin(best_sim))
        chosen.append(nxt)
        best_sim = np.maximum(best_sim, zs @ zs[nxt])
    return zs[chosen]
