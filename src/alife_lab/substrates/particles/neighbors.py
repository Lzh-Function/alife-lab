"""Shared neighbour search for continuous 2D particle substrates (CPU / NumPy).

Uniform cell list with cell size >= radius; ``neighbor_pairs_bruteforce`` is the O(N²)
oracle used by tests. Results are directed pairs (i, j), i != j, with minimal-image offsets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Pairs:
    i: I64
    j: I64
    dx: F64
    dy: F64
    dist2: F64


def _wrap_delta(d: F64, size: float) -> F64:
    return np.asarray((d + size / 2.0) % size - size / 2.0, dtype=np.float64)


def _finish_pairs(
    pos: F64, i: I64, j: I64, radius: float, width: float, height: float, periodic: bool
) -> Pairs:
    keep = i != j
    i, j = i[keep], j[keep]
    dx = pos[j, 0] - pos[i, 0]
    dy = pos[j, 1] - pos[i, 1]
    if periodic:
        dx = _wrap_delta(dx, width)
        dy = _wrap_delta(dy, height)
    dist2 = dx * dx + dy * dy
    near = dist2 < radius * radius
    return Pairs(i[near], j[near], dx[near], dy[near], dist2[near])


def neighbor_pairs(pos: F64, radius: float, width: float, height: float, periodic: bool) -> Pairs:
    """Directed pairs (i, j), i != j, with minimal-image distance < radius (cell list)."""
    n = pos.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    if n == 0:
        z = np.zeros(0, dtype=np.float64)
        return Pairs(empty, empty, z, z, z)
    ncx = max(1, int(width // radius))
    ncy = max(1, int(height // radius))
    if periodic and (ncx < 3 or ncy < 3):
        # Neighbour cells would alias; fall back to the exact all-pairs search.
        return neighbor_pairs_bruteforce(pos, radius, width, height, periodic)
    cx = np.clip((pos[:, 0] / (width / ncx)).astype(np.int64), 0, ncx - 1)
    cy = np.clip((pos[:, 1] / (height / ncy)).astype(np.int64), 0, ncy - 1)
    cid = cy * ncx + cx
    order = np.argsort(cid, kind="stable")
    sorted_cid = cid[order]
    cells = np.arange(ncx * ncy)
    starts = np.searchsorted(sorted_cid, cells, side="left")
    ends = np.searchsorted(sorted_cid, cells, side="right")
    idx = np.arange(n, dtype=np.int64)
    all_i: list[I64] = []
    all_j: list[I64] = []
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            nx = cx + ox
            ny = cy + oy
            if periodic:
                nx %= ncx
                ny %= ncy
                valid = np.ones(n, dtype=bool)
            else:
                valid = (nx >= 0) & (nx < ncx) & (ny >= 0) & (ny < ncy)
            ncid = np.where(valid, ny * ncx + nx, 0)
            cnt = np.where(valid, ends[ncid] - starts[ncid], 0)
            total = int(cnt.sum())
            if total == 0:
                continue
            i_rep = np.repeat(idx, cnt)
            offsets = np.repeat(np.cumsum(cnt) - cnt, cnt)
            local = np.arange(total, dtype=np.int64) - offsets
            j_rep = order[np.repeat(starts[ncid], cnt) + local]
            all_i.append(i_rep)
            all_j.append(j_rep.astype(np.int64))
    if not all_i:
        z = np.zeros(0, dtype=np.float64)
        return Pairs(empty, empty, z, z, z)
    return _finish_pairs(
        pos, np.concatenate(all_i), np.concatenate(all_j), radius, width, height, periodic
    )


def neighbor_pairs_bruteforce(
    pos: F64, radius: float, width: float, height: float, periodic: bool
) -> Pairs:
    """O(N²) oracle (tests and tiny worlds only)."""
    n = pos.shape[0]
    i, j = np.meshgrid(np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), indexing="ij")
    return _finish_pairs(pos, i.ravel(), j.ravel(), radius, width, height, periodic)


def rms_id_distance(
    a_pos: F64, a_ids: I64, b_pos: F64, b_ids: I64, width: float, height: float, periodic: bool
) -> tuple[float | None, int, int, int]:
    """RMS (minimal-image) position difference over particles present in both states.

    Returns (value or None, common, only_in_first, only_in_second).
    """
    common, ia, ib = np.intersect1d(a_ids, b_ids, assume_unique=True, return_indices=True)
    value: float | None = None
    if common.size:
        diff = a_pos[ia] - b_pos[ib]
        if periodic and width > 0 and height > 0:
            diff[:, 0] = _wrap_delta(diff[:, 0], width)
            diff[:, 1] = _wrap_delta(diff[:, 1], height)
        value = float(np.sqrt((diff * diff).sum(axis=1).mean()))
    return value, int(common.size), int(a_ids.size - common.size), int(b_ids.size - common.size)
