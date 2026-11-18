"""Run comparison: exact state distance at common checkpoint ticks.

Pure read-side logic shared by the controller (API) and the read-only CLI. It never
modifies runs. Distances are computed only where every compared run has a checkpoint
at the same simulation tick (exact states), never from preview frames. The distance
metric is defined by the substrate adapter (e.g. Hamming for Conway).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from alife_lab.artifacts.store import LocalArtifactStore, decode_checkpoint
from alife_lab.core.hashing import sha256_bytes
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates import get_adapter

Query = Callable[[str, Sequence[Any]], list[dict[str, Any]]]
MAX_COMPARED_TICKS = 200


def _checkpoint_by_tick(query: Query, run_id: str) -> dict[int, dict[str, Any]]:
    """Latest-created checkpoint per tick (a later one at the same tick supersedes)."""
    rows = query("SELECT * FROM checkpoints WHERE run_id=? ORDER BY created_at, rowid", (run_id,))
    by_tick: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_tick[int(row["simulation_clock"])] = row
    return by_tick


def _spec(query: Query, run_id: str) -> ExperimentSpec:
    rows = query(
        "SELECT e.spec_json FROM runs r JOIN experiments e ON e.id = r.experiment_id WHERE r.id=?",
        (run_id,),
    )
    if not rows:
        raise ValueError(f"run {run_id} not found")
    return ExperimentSpec.model_validate_json(rows[0]["spec_json"])


def _load_state(store: LocalArtifactStore, row: dict[str, Any], spec: ExperimentSpec) -> Any:
    data = store.resolve(row["run_id"], row["logical_path"]).read_bytes()
    if sha256_bytes(data) != row["sha256"]:
        raise ValueError(f"checksum mismatch for {row['run_id']}/{row['logical_path']}")
    adapter = get_adapter(spec.experiment.substrate)
    return adapter.deserialize_state(decode_checkpoint(data), spec)


def state_divergence(
    query: Query, store: LocalArtifactStore, run_ids: Sequence[str]
) -> dict[str, Any]:
    """Distance of each run's state to the first run's state at every common checkpoint tick."""
    if len(run_ids) < 2:
        raise ValueError("a comparison needs at least two runs")
    specs = [_spec(query, rid) for rid in run_ids]
    substrates = {s.experiment.substrate for s in specs}
    comparable = len(substrates) == 1
    adapter = get_adapter(specs[0].experiment.substrate)
    per_run = [_checkpoint_by_tick(query, rid) for rid in run_ids]
    common = sorted(set.intersection(*(set(m) for m in per_run)))
    sampled = common
    if len(common) > MAX_COMPARED_TICKS:
        idx = np.linspace(0, len(common) - 1, MAX_COMPARED_TICKS).round().astype(int)
        sampled = [common[i] for i in sorted(set(idx.tolist()))]
    points: list[dict[str, Any]] = []
    metric_name: str | None = None
    for tick in sampled:
        states = [_load_state(store, m[tick], spec) for m, spec in zip(per_run, specs, strict=True)]
        distances: list[float | int | None] = []
        for state in states:
            if not comparable:
                distances.append(None)
                continue
            result = adapter.state_distance(states[0], state)
            metric_name = result["metric"]
            distances.append(result["value"])
        points.append(
            {
                "tick": tick,
                "distance_to_first": distances,
                # Kept for clients of the Conway-only first version of this report.
                "hamming_to_first": distances if metric_name == "hamming" else None,
                "summary": [
                    adapter.metrics(st, spec) if comparable else None
                    for st, spec in zip(states, specs, strict=True)
                ],
                "checkpoint_ids": [m[tick]["id"] for m in per_run],
                "reasons": [m[tick]["reason"] for m in per_run],
            }
        )
    for point in points:
        summaries = point["summary"]
        point["alive"] = [
            int(s.get("alive_cell_count", s.get("population", 0))) if s else None for s in summaries
        ]
    return {
        "metric": metric_name,
        "definition": (
            "Substrate-defined exact state distance to the first run at the same tick, computed "
            "from checksum-verified checkpoints (never previews). conway: number of differing "
            "cells (Hamming); boids: see docs/reference_fidelity/boids.md."
        ),
        "comparable": comparable,
        "clock_unit": "tick",
        "common_ticks": len(common),
        "sampled_ticks": len(sampled),
        "shape_mismatch": not comparable,
        "points": points,
    }
