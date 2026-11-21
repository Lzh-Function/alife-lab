"""The replay cache (cursor + keyframes) must give exactly the results of a fresh replay.

For each substrate: a checkpoint at tick 0, a few logged interventions, then a random mix of
forward / backward seeks, frame batches, inspections and state reconstructions. Every cached
answer is compared with the answer of a cold replay from the checkpoint.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.hashing import sha256_bytes
from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution import replay
from alife_lab.service.catalog import load_presets
from alife_lab.substrates import get_adapter

PRESETS = load_presets(Path(__file__).parents[2] / "configs")

# preset -> interventions [(tick, kind, payload)]
CASES: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
    "conway_glider": [
        (12, "brush", {"shape": "circle", "x": 20, "y": 20, "radius": 3, "value": 1})
    ],
    "lenia_orbium": [(30, "paint", {"x": 20, "y": 20, "radius": 5, "value": 0.8})],
    "lenia_mc_swarm": [(26, "paint", {"x": 30, "y": 30, "radius": 4, "value": 1.0})],
    "flow_lenia_example": [(25, "erase", {"x": 10, "y": 10, "radius": 4})],
    "gray_scott_coral": [(40, "seed", {"x": 10, "y": 10, "radius": 3})],
    "physarum_network": [(33, "attract", {"x": 20, "y": 20, "radius": 5})],
    "particle_life_clusters": [(20, "spawn", {"x": 50, "y": 50, "radius": 10, "count": 5})],
    "boids_flocking": [(20, "spawn", {"x": 50, "y": 50, "radius": 10, "count": 5})],
    "creature_rough_wave_reflex": [(30, "push", {"vx": 1.0})],
    "reaction_network_cycle": [],
    "protocell_default": [],
    "agent_world_evolution": [],
    "bff_reference": [],
}
END = 90
SMALLER: dict[str, dict[str, Any]] = {
    "bff_reference": {"soup.num_programs": 512},
    "flow_lenia_example": {"world.width": 64, "world.height": 64, "initialization.size": 24},
}


def _checkpoint(tmp: Path, spec: ExperimentSpec) -> replay.ReplayInput:
    adapter = get_adapter(spec.experiment.substrate)
    state = adapter.initialize(spec)
    ser = adapter.serialize_state(state)
    data = encode_checkpoint(ser["arrays"], {**ser["meta"], "spec_digest": spec.digest()})
    path = tmp / "base.npz"
    path.write_bytes(data)
    return replay.ReplayInput(
        spec=spec.model_dump(mode="json"),
        base_path=str(path),
        base_sha256=sha256_bytes(data),
        base_tick=0,
        base_expected_spec_digest=spec.digest(),
        interventions=[],
    )


def _frames_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x["data"] == y["data"]
        assert {k: v for k, v in x.items() if k != "data"} == {
            k: v for k, v in y.items() if k != "data"
        }


def _state_equal(a: dict[str, Any], b: dict[str, Any]) -> None:
    pa, pb = decode_checkpoint(a["checkpoint_bytes"]), decode_checkpoint(b["checkpoint_bytes"])
    assert pa["meta"] == pb["meta"]
    assert pa["arrays"].keys() == pb["arrays"].keys()
    for k in pa["arrays"]:
        np.testing.assert_array_equal(pa["arrays"][k], pb["arrays"][k])
    assert a["interventions_replayed"] == b["interventions_replayed"]


def _fresh(fn: Any, *args: Any) -> Any:
    replay.clear_cache()
    out = fn(*args)
    replay.clear_cache()
    return out


@pytest.mark.parametrize("preset", sorted(CASES))
def test_cached_replay_equals_fresh_replay(preset: str, tmp_path: Path) -> None:
    if preset not in PRESETS:
        pytest.skip(f"preset {preset} not available")
    raw = PRESETS[preset].spec.model_dump(mode="json")
    for path, value in SMALLER.get(preset, {}).items():  # keep the test fast
        node = raw["experiment"]
        *parents, leaf = path.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = value
    spec = ExperimentSpec.model_validate(raw)
    try:
        inp = _checkpoint(tmp_path, spec)
    except RuntimeError as exc:  # optional runtime (e.g. MuJoCo) missing
        pytest.skip(str(exc))
    interventions = [
        {"tick": t, "kind": k, "payload": p, "intervention_id": f"i{n}"}
        for n, (t, k, p) in enumerate(CASES[preset])
    ]
    inp = replay.ReplayInput(**{**inp.__dict__, "interventions": interventions})
    rng = random.Random(7)
    replay.clear_cache()
    # A history of requests the cache must survive: forward, backward, batches, repeats.
    ops: list[tuple[str, int, int]] = [("frames", 60, 1), ("frames", 61, 1), ("frames", 10, 1)]
    ops += [(rng.choice(["frames", "batch", "inspect", "state"]), rng.randint(0, END), 0)
            for _ in range(10)]  # fmt: skip
    cached_hits = 0
    for kind, tick, _ in ops:
        before = replay.cache_stats()["keyframes"]
        if kind in ("frames", "batch"):
            count = 1 if kind == "frames" else min(8, END - tick + 1)
            got = replay.replay_frames(inp, tick, count, 1)
            state_after = replay.cache_stats()
            replay_args = (inp, tick, count, 1)
            # Compare against a cold replay without disturbing the warm cache.
            saved = dict(replay._SESSIONS)
            want = _fresh(replay.replay_frames, *replay_args)
            replay._SESSIONS.update(saved)
            _frames_equal(got, want)
        elif kind == "inspect":
            got_i = replay.inspect_at(inp, tick, {"x": 1, "y": 0, "radius": 2})
            saved = dict(replay._SESSIONS)
            want_i = _fresh(replay.inspect_at, inp, tick, {"x": 1, "y": 0, "radius": 2})
            replay._SESSIONS.update(saved)
            assert json.dumps(got_i, sort_keys=True, default=str) == json.dumps(
                want_i, sort_keys=True, default=str
            )
            state_after = replay.cache_stats()
        else:
            got_s = replay.replay_state(inp, tick)
            saved = dict(replay._SESSIONS)
            want_s = _fresh(replay.replay_state, inp, tick)
            replay._SESSIONS.update(saved)
            _state_equal(got_s, want_s)
            state_after = replay.cache_stats()
        cached_hits += int(before > 0)
        assert state_after["bytes"] <= replay.CACHE_BUDGET_BYTES
    assert cached_hits > 0  # the cache was actually exercised
    replay.clear_cache()


def test_cache_is_not_used_across_different_interventions(tmp_path: Path) -> None:
    spec = PRESETS["conway_glider"].spec
    inp = _checkpoint(tmp_path, spec)
    replay.clear_cache()
    a = replay.replay_frames(inp, 31, 1, 1)
    brush = {"shape": "circle", "x": 20, "y": 20, "radius": 3, "value": 1}
    changed = replay.ReplayInput(
        **{**inp.__dict__, "interventions": [
            {"tick": 30, "kind": "brush", "payload": brush, "intervention_id": "x"}]}
    )  # fmt: skip
    b = replay.replay_frames(changed, 31, 1, 1)  # must not reuse the cursor / keyframes at >= 30
    want = _fresh(replay.replay_frames, changed, 31, 1, 1)
    _frames_equal(b, want)
    assert a[0]["data"] != b[0]["data"]
    replay.clear_cache()
