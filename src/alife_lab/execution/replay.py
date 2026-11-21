"""Deterministic replay: reconstruct exact states between checkpoints.

Runs in a separate (spawned) process, never in the API process. Inputs are plain
data prepared and validated by the controller: the run spec, the base checkpoint
(path + expected sha256) and the ordered list of logged interventions to re-apply.

Semantics: the state "at tick T" is the state after the update of tick T and after
every intervention applied at tick T (in logged order). Replay is only offered for
substrates declaring ``supports_deterministic_cpu``; it is exact (bitwise) for them.

Speed: the replay process keeps, per base checkpoint, the state it last reached (the
cursor) and serialized keyframes every ``KEYFRAME_EVERY`` ticks it passed. A request starts
from the latest of these at or before its target — only if it derives from the same base
checkpoint and the same interventions — instead of re-simulating from the checkpoint. This
memoises exactly the computation that would otherwise be repeated, so results are identical
(tested). ``verify_segment`` never uses the cache: verification stays an independent re-run.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.hashing import sha256_bytes
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates import get_adapter

MAX_REPLAY_TICKS = 200_000
MAX_FRAMES = 200
KEYFRAME_EVERY = 25
CACHE_BUDGET_BYTES = 512 * 1024 * 1024  # keyframes across all sessions
MAX_SESSIONS = 4
VERIFY_KEYFRAMES = 2


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
    backend: str = "numpy"


def _load(inp: ReplayInput) -> tuple[ExperimentSpec, Any, Any]:
    spec = ExperimentSpec.model_validate(inp.spec)
    adapter = get_adapter(spec.experiment.substrate)
    if hasattr(adapter, "set_backend"):
        adapter.set_backend(inp.backend)
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


@dataclass
class _Keyframe:
    tick: int
    applied: tuple[str, ...]  # ids of the interventions applied since the base (tick <= this)
    payload: dict[str, Any]  # serialized state (private copies)
    nbytes: int


@dataclass
class _Session:
    spec: ExperimentSpec
    adapter: Any
    keyframes: dict[int, _Keyframe] = field(default_factory=dict)
    cursor: Any = None  # the live state object last reached (taken while in use)
    cursor_tick: int = -1
    cursor_applied: tuple[str, ...] = ()
    # Whether a serialize -> deserialize round trip reproduces the display exactly. Some
    # substrates keep display-only history outside the checkpoint (e.g. the reaction network's
    # count history); for those, keyframes are not used (the cursor still is).
    keyframes_ok: bool | None = None
    verified: int = 0  # keyframes past the base whose round trip was checked


_SESSIONS: OrderedDict[tuple[str, ...], _Session] = OrderedDict()


def warm(substrate: str) -> bool:
    """Import the substrate's adapter in the replay process ahead of the first request."""
    get_adapter(substrate)
    return True


def clear_cache() -> None:
    _SESSIONS.clear()


def cache_stats() -> dict[str, int]:
    return {
        "sessions": len(_SESSIONS),
        "keyframes": sum(len(s.keyframes) for s in _SESSIONS.values()),
        "bytes": sum(k.nbytes for s in _SESSIONS.values() for k in s.keyframes.values()),
    }


def _session_key(inp: ReplayInput) -> tuple[str, ...]:
    spec_json = json.dumps(inp.spec, sort_keys=True)
    return (inp.base_sha256, str(inp.base_tick), inp.backend, sha256_bytes(spec_json.encode()))


def _applied_ids(inp: ReplayInput, tick: int) -> tuple[str, ...]:
    return tuple(str(i["intervention_id"]) for i in inp.interventions if int(i["tick"]) <= tick)


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Private copy, with meta passed through JSON exactly as a checkpoint file would be."""
    return {
        "arrays": {k: np.array(v, copy=True) for k, v in payload["arrays"].items()},
        "meta": json.loads(json.dumps(payload["meta"], sort_keys=True)),
    }


def _evict() -> None:
    total = sum(k.nbytes for s in _SESSIONS.values() for k in s.keyframes.values())
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    for sess in list(_SESSIONS.values()):  # least recently used sessions lose keyframes first
        for t in sorted(sess.keyframes):
            if total <= CACHE_BUDGET_BYTES:
                return
            total -= sess.keyframes.pop(t).nbytes


def _start(
    inp: ReplayInput, target: int, upto: int | None = None
) -> tuple[ExperimentSpec, Any, Any, list[dict[str, Any]], _Session]:
    """The latest usable state at or before ``target`` and the interventions still due up to
    ``upto`` (default ``target``)."""
    upto = target if upto is None else upto
    key = _session_key(inp)
    sess = _SESSIONS.get(key)
    if sess is not None:
        _SESSIONS.move_to_end(key)
        best: tuple[int, Any] | None = None
        if (
            sess.cursor is not None
            and sess.cursor_tick <= target
            and sess.cursor_applied == _applied_ids(inp, sess.cursor_tick)
        ):
            best = (sess.cursor_tick, "cursor")
        for t, kf in sess.keyframes.items():
            if t <= target and (best is None or t > best[0]) and kf.applied == _applied_ids(inp, t):
                best = (t, kf)
        if best is not None:
            tick, source = best
            if source == "cursor":
                state, sess.cursor = sess.cursor, None
            else:
                state = sess.adapter.deserialize_state(_copy_payload(source.payload), sess.spec)
            pending = [i for i in inp.interventions if tick < int(i["tick"]) <= upto]
            return sess.spec, sess.adapter, state, pending, sess
    spec, adapter, state = _load(inp)
    sess = _Session(spec=spec, adapter=adapter)
    _SESSIONS[key] = sess
    _evict()
    pending = [i for i in inp.interventions if int(i["tick"]) <= upto]
    return spec, adapter, state, pending, sess


def _keyframe_hook(inp: ReplayInput, sess: _Session) -> Any:
    def hook(st: Any) -> None:
        tick = int(st.tick)
        if sess.keyframes_ok is False or tick % KEYFRAME_EVERY or tick in sess.keyframes:
            return
        payload = _copy_payload(sess.adapter.serialize_state(st))
        # The first keyframes past the base are checked: restored, they must render exactly
        # like the original (display-only state outside the checkpoint shows up once ticks
        # have passed). After VERIFY_KEYFRAMES passes, the substrate is trusted.
        if sess.verified >= VERIFY_KEYFRAMES:
            nbytes = sum(a.nbytes for a in payload["arrays"].values())
            sess.keyframes[tick] = _Keyframe(tick, _applied_ids(inp, tick), payload, nbytes)
            _evict()
            return
        restored = sess.adapter.deserialize_state(_copy_payload(payload), sess.spec)
        h1, d1 = sess.adapter.render(st)
        h2, d2 = sess.adapter.render(restored)
        same = d1 == d2 and json.dumps(h1, sort_keys=True, default=str) == json.dumps(
            h2, sort_keys=True, default=str
        )
        if not same:
            sess.keyframes_ok = False
            sess.keyframes.clear()
            return
        sess.keyframes_ok = True
        if tick > inp.base_tick:
            sess.verified += 1
        nbytes = sum(a.nbytes for a in payload["arrays"].values())
        sess.keyframes[tick] = _Keyframe(tick, _applied_ids(inp, tick), payload, nbytes)
        _evict()

    return hook


def _finish(inp: ReplayInput, sess: _Session, state: Any) -> None:
    sess.cursor = state
    sess.cursor_tick = int(state.tick)
    sess.cursor_applied = _applied_ids(inp, sess.cursor_tick)


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
    """Display frames for ticks start, start+stride, … (count of them)."""
    if not 1 <= count <= MAX_FRAMES or stride < 1:
        raise ReplayError(f"count must be 1..{MAX_FRAMES} and stride >= 1")
    last = start + (count - 1) * stride
    _check_range(inp, start)
    _check_range(inp, last)
    spec, adapter, state, pending, sess = _start(inp, start, last)
    wanted = set(range(start, last + 1, stride))
    # Counted from the base checkpoint, as before the cache existed.
    scheduled = sum(1 for i in inp.interventions if int(i["tick"]) <= last)
    frames: list[dict[str, Any]] = []
    keyframe = _keyframe_hook(inp, sess)

    def capture(st: Any) -> None:
        tick = int(st.tick)
        # A tick is captured once all its interventions were applied (no pending at tick).
        if tick in wanted and not (pending and int(pending[0]["tick"]) == tick):
            header, data = adapter.render(st)
            frames.append(
                {
                    "tick": tick,
                    **header,
                    "data": data,
                    "steps_from_base": tick - inp.base_tick,
                    # Interventions applied between the base and this frame's tick.
                    "interventions_replayed": scheduled - len(pending),
                }
            )
            wanted.discard(tick)

    def on_tick(st: Any) -> None:
        capture(st)
        keyframe(st)

    state, _ = _advance_to(spec, adapter, state, last, pending, on_tick)
    _finish(inp, sess, state)
    return frames


def replay_state(inp: ReplayInput, target: int) -> dict[str, Any]:
    """Reconstruct the full state at ``target`` and encode it as a checkpoint payload."""
    _check_range(inp, target)
    spec, adapter, state, pending, sess = _start(inp, target)
    state, _ = _advance_to(spec, adapter, state, target, pending, _keyframe_hook(inp, sess))
    applied = len(_applied_ids(inp, target))
    serialized = adapter.serialize_state(state)
    meta = {
        **serialized["meta"],
        "spec_digest": inp.base_expected_spec_digest or spec.digest(),
        "clock_unit": "tick",
        "substrate": spec.experiment.substrate,
        "backend": inp.backend,
        "backend_version": None if inp.backend != "numpy" else np.__version__,
        "reason": "replay_reconstruction",
        "replay": {
            "base_sha256": inp.base_sha256,
            "base_tick": inp.base_tick,
            "interventions_replayed": applied,
        },
    }
    data = encode_checkpoint(serialized["arrays"], meta)
    _finish(inp, sess, state)
    return {"tick": int(state.tick), "checkpoint_bytes": data, "interventions_replayed": applied}


def inspect_at(inp: ReplayInput, target: int, query: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the state at ``target`` and run the adapter's read-only inspection."""
    _check_range(inp, target)
    spec, adapter, state, pending, sess = _start(inp, target)
    state, _ = _advance_to(spec, adapter, state, target, pending, _keyframe_hook(inp, sess))
    applied = len(_applied_ids(inp, target))
    result: dict[str, Any] = adapter.inspect(state, spec, query)
    _finish(inp, sess, state)
    result["source"] = {
        "method": "deterministic_replay",
        "base_tick": inp.base_tick,
        "interventions_replayed": applied,
    }
    return result


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
    distance = adapter.state_distance(stored, state)
    return {
        "ok": bool(distance["exact_equal"]) and not pending,
        "distance_metric": distance["metric"],
        "distance": distance["value"],
        "interventions_replayed": applied,
        "unapplied_interventions": len(pending),
        "ticks": target_tick - inp.base_tick,
    }
