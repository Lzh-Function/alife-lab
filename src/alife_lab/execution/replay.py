"""Deterministic replay: reconstruct exact states between checkpoints.

Runs in a separate (spawned) process, never in the API process. Inputs are plain
data prepared and validated by the controller: the run spec, the base checkpoint
(path + expected sha256) and the ordered list of logged interventions to re-apply.

Semantics: the state "at tick T" is the state after the update of tick T and after
every intervention applied at tick T (in logged order). Replay is only offered for
substrates declaring ``supports_deterministic_cpu``; it is exact (bitwise) for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.hashing import sha256_bytes
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates import get_adapter

MAX_REPLAY_TICKS = 200_000
MAX_FRAMES = 200


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayInput:
    spec: dict[str, Any]
    base_path: str
    base_sha256: str
    base_tick: int
    base_expected_spec_digest: str | None
    # [{"tick": int, "kind": str, "payload": dict, "intervention_id": str}], sorted
    interventions: list[dict[str, Any]]


def _load(inp: ReplayInput) -> tuple[ExperimentSpec, Any, Any]:
    spec = ExperimentSpec.model_validate(inp.spec)
    adapter = get_adapter(spec.experiment.substrate)
    if not adapter.manifest.capabilities.supports_deterministic_cpu:
        raise ReplayError("substrate does not support deterministic replay")
    data = Path(inp.base_path).read_bytes()
    if sha256_bytes(data) != inp.base_sha256:
        raise ReplayError("base checkpoint failed checksum verification")
    payload = decode_checkpoint(data)
    expected = inp.base_expected_spec_digest or spec.digest()
    if payload["meta"].get("spec_digest") != expected:
        raise ReplayError("base checkpoint belongs to a different scientific spec")
    state = adapter.deserialize_state(payload, spec)
    if int(state.tick) != inp.base_tick:
        raise ReplayError("base checkpoint tick does not match the index")
    return spec, adapter, state


def _advance_to(
    spec: ExperimentSpec,
    adapter: Any,
    state: Any,
    target: int,
    pending: list[dict[str, Any]],
    on_tick: Any = None,
) -> tuple[Any, int]:
    """Apply interventions at the current tick, then step to ``target``."""
    applied = 0

    def apply_due(st: Any) -> Any:
        nonlocal applied
        while pending and int(pending[0]["tick"]) == int(st.tick):
            item = pending.pop(0)
            st, _ = adapter.apply_intervention(st, spec, item["kind"], dict(item["payload"]))
            applied += 1
        return st

    state = apply_due(state)
    if on_tick is not None:
        on_tick(state)
    while int(state.tick) < target:
        state = adapter.step(state, spec)
        state = apply_due(state)
        if on_tick is not None:
            on_tick(state)
    return state, applied


def _check_range(inp: ReplayInput, target: int) -> None:
    if target < inp.base_tick:
        raise ReplayError("target tick precedes the base checkpoint")
    if target - inp.base_tick > MAX_REPLAY_TICKS:
        raise ReplayError(f"replay span exceeds {MAX_REPLAY_TICKS} ticks")
    for item in inp.interventions:
        if int(item["tick"]) < inp.base_tick:
            raise ReplayError("intervention precedes the base checkpoint")


def replay_frames(inp: ReplayInput, start: int, count: int, stride: int) -> list[dict[str, Any]]:
    """Frames for ticks start, start+stride, … (count of them), bitpacked."""
    if not 1 <= count <= MAX_FRAMES or stride < 1:
        raise ReplayError(f"count must be 1..{MAX_FRAMES} and stride >= 1")
    last = start + (count - 1) * stride
    _check_range(inp, start)
    _check_range(inp, last)
    spec, adapter, state = _load(inp)
    wanted = set(range(start, last + 1, stride))
    pending = [i for i in inp.interventions if int(i["tick"]) <= last]
    frames: list[dict[str, Any]] = []

    def capture(st: Any) -> None:
        tick = int(st.tick)
        # A tick is captured once all its interventions were applied (no pending at tick).
        if tick in wanted and not (pending and int(pending[0]["tick"]) == tick):
            frames.append(
                {
                    "tick": tick,
                    "width": int(st.grid.shape[1]),
                    "height": int(st.grid.shape[0]),
                    "data": adapter.render_payload(st),
                    "steps_from_base": tick - inp.base_tick,
                    "interventions_replayed": len(inp.interventions) - len(pending),
                }
            )
            wanted.discard(tick)

    _advance_to(spec, adapter, state, last, pending, capture)
    return frames


def replay_state(inp: ReplayInput, target: int) -> dict[str, Any]:
    """Reconstruct the full state at ``target`` and encode it as a checkpoint payload."""
    _check_range(inp, target)
    spec, adapter, state = _load(inp)
    pending = [i for i in inp.interventions if int(i["tick"]) <= target]
    state, applied = _advance_to(spec, adapter, state, target, pending)
    serialized = adapter.serialize_state(state)
    meta = {
        **serialized["meta"],
        "spec_digest": inp.base_expected_spec_digest or spec.digest(),
        "clock_unit": "tick",
        "substrate": spec.experiment.substrate,
        "backend": "numpy",
        "backend_version": np.__version__,
        "reason": "replay_reconstruction",
        "replay": {
            "base_sha256": inp.base_sha256,
            "base_tick": inp.base_tick,
            "interventions_replayed": applied,
        },
    }
    data = encode_checkpoint(serialized["arrays"], meta)
    return {"tick": int(state.tick), "checkpoint_bytes": data, "interventions_replayed": applied}


def verify_segment(
    inp: ReplayInput, target_path: str, target_sha256: str, target_tick: int
) -> dict[str, Any]:
    """Re-simulate base -> target and compare with the stored target checkpoint state."""
    _check_range(inp, target_tick)
    spec, adapter, state = _load(inp)
    pending = list(inp.interventions)
    state, applied = _advance_to(spec, adapter, state, target_tick, pending)
    data = Path(target_path).read_bytes()
    if sha256_bytes(data) != target_sha256:
        return {"ok": False, "reason": "target checkpoint checksum mismatch"}
    stored = adapter.deserialize_state(decode_checkpoint(data), spec)
    expected = stored.grid
    differing = int(np.count_nonzero(expected != state.grid))
    return {
        "ok": differing == 0 and not pending,
        "differing_cells": differing,
        "interventions_replayed": applied,
        "unapplied_interventions": len(pending),
        "ticks": target_tick - inp.base_tick,
    }
