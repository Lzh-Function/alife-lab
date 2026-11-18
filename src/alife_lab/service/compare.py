"""Run comparison: state divergence at common checkpoint ticks and aligned metrics.

Pure read-side logic shared by the controller (API) and the read-only CLI. It never
modifies runs. Divergence is measured only where every compared run has a checkpoint
at the same simulation tick (exact state), never from preview frames.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from alife_lab.artifacts.store import LocalArtifactStore, decode_checkpoint
from alife_lab.core.hashing import sha256_bytes

Query = Callable[[str, Sequence[Any]], list[dict[str, Any]]]
MAX_COMPARED_TICKS = 200


def _checkpoint_by_tick(query: Query, run_id: str) -> dict[int, dict[str, Any]]:
    """Latest-created checkpoint per tick (a later one at the same tick supersedes)."""
    rows = query("SELECT * FROM checkpoints WHERE run_id=? ORDER BY created_at, rowid", (run_id,))
    by_tick: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_tick[int(row["simulation_clock"])] = row
    return by_tick


def _load_grid(store: LocalArtifactStore, row: dict[str, Any]) -> np.ndarray[Any, Any]:
    data = store.resolve(row["run_id"], row["logical_path"]).read_bytes()
    if sha256_bytes(data) != row["sha256"]:
        raise ValueError(f"checksum mismatch for {row['run_id']}/{row['logical_path']}")
    grid: np.ndarray[Any, Any] = decode_checkpoint(data)["arrays"]["grid"]
    return grid


def state_divergence(
    query: Query, store: LocalArtifactStore, run_ids: Sequence[str]
) -> dict[str, Any]:
    """Hamming distance of each run to the first run at every common checkpoint tick."""
    if len(run_ids) < 2:
        raise ValueError("a comparison needs at least two runs")
    per_run = [_checkpoint_by_tick(query, rid) for rid in run_ids]
    common = sorted(set.intersection(*(set(m) for m in per_run)))
    sampled = common
    if len(common) > MAX_COMPARED_TICKS:
        idx = np.linspace(0, len(common) - 1, MAX_COMPARED_TICKS).round().astype(int)
        sampled = [common[i] for i in sorted(set(idx.tolist()))]
    points = []
    shape_mismatch = False
    for tick in sampled:
        grids = [_load_grid(store, m[tick]) for m in per_run]
        ref = grids[0]
        distances: list[int | None] = []
        for grid in grids:
            if grid.shape != ref.shape:
                shape_mismatch = True
                distances.append(None)
            else:
                distances.append(int(np.count_nonzero(grid != ref)))
        points.append(
            {
                "tick": tick,
                "hamming_to_first": distances,
                "alive": [int(g.sum()) for g in grids],
                "checkpoint_ids": [m[tick]["id"] for m in per_run],
                "reasons": [m[tick]["reason"] for m in per_run],
            }
        )
    return {
        "metric": "hamming_to_first",
        "definition": (
            "Number of cells whose state differs from the first run's state at the same tick, "
            "computed from checksum-verified checkpoints (exact states, not previews)."
        ),
        "clock_unit": "tick",
        "common_ticks": len(common),
        "sampled_ticks": len(sampled),
        "shape_mismatch": shape_mismatch,
        "points": points,
    }
