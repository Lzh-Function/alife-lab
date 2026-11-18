"""Inspector: read-only, consistent between live worker and replay, decomposition is exact."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller, ControllerError
from alife_lab.service.settings import Settings
from alife_lab.substrates.boids.boids import BoidsAdapter, neighbor_pairs, steering
from alife_lab.substrates.ca.conway import ConwayAdapter
from tests.integration.test_controller import arun, spec, state_of, wait_for
from tests.reference.test_boids import make_spec as boids_spec


def test_conway_inspect_glider_cells_and_passivity() -> None:
    adapter = ConwayAdapter()
    sp = spec("glider", offset_x=10, offset_y=10)
    state = adapter.initialize(sp)
    before = (state.grid.copy(), copy.deepcopy(state.tracker), state.tick)
    # Glider cells (1,0),(2,1),(0,2),(1,2),(2,2) at offset (10, 10).
    live = adapter.inspect(state, sp, {"x": 11, "y": 10, "radius": 1})
    assert live["cell"] == {"alive": True, "live_neighbors": 1, "next_tick": "dies"}
    born = adapter.inspect(state, sp, {"x": 10, "y": 11, "radius": 0})
    assert born["cell"] == {"alive": False, "live_neighbors": 3, "next_tick": "born"}
    assert live["patch"] == [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError):
        adapter.inspect(state, sp, {"x": 999, "y": 0})
    np.testing.assert_array_equal(state.grid, before[0])
    assert state.tracker == before[1] and state.tick == before[2]
    # The predicted outcome matches the actual next step for every cell of the patch.
    nxt = adapter.step(adapter.initialize(sp), sp).grid
    for y in range(8, 14):
        for x in range(8, 14):
            outcome = adapter.inspect(state, sp, {"x": x, "y": y, "radius": 0})["cell"]["next_tick"]
            assert (outcome in ("born", "survives")) == bool(nxt[y, x])


def test_boids_inspect_decomposition_matches_dynamics() -> None:
    sp = boids_spec(n=150, noise=0.0)
    adapter = BoidsAdapter()
    state = adapter.initialize(sp)
    for _ in range(40):
        state = adapter.step(state, sp)
    snapshot = (state.pos.copy(), state.vel.copy(), state.ids.copy(), state.rng.bit_generator.state)
    exp = sp.experiment
    target = state.pos[7]
    result = adapter.inspect(state, sp, {"x": target[0], "y": target[1], "radius": 1e-9})
    assert result["nearest"]["id"] == int(state.ids[7])
    st = result["nearest"]["steering"]
    pairs = neighbor_pairs(
        state.pos, exp.dynamics.perception_radius, exp.world.width, exp.world.height, True
    )
    acc, counts = steering(state.pos, state.vel, exp, pairs)
    total = np.array(
        [
            st["separation"]["x"] + st["alignment"]["x"] + st["cohesion"]["x"],
            st["separation"]["y"] + st["alignment"]["y"] + st["cohesion"]["y"],
        ]
    )
    np.testing.assert_allclose(total, acc[7], rtol=1e-9, atol=1e-12)
    assert st["neighbors"] == counts[7]
    np.testing.assert_array_equal(state.pos, snapshot[0])
    np.testing.assert_array_equal(state.vel, snapshot[1])
    np.testing.assert_array_equal(state.ids, snapshot[2])
    assert state.rng.bit_generator.state == snapshot[3]
    wide = adapter.inspect(state, sp, {"x": 60, "y": 48, "radius": 30})
    assert wide["count_in_radius"] >= len(wide["boids"]) and len(wide["boids"]) <= 20
    dists = [b["distance"] for b in wide["boids"]]
    assert dists == sorted(dists)


def test_live_and_replayed_inspection_agree(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            for sp in (
                spec("random", max_steps=10_000, interval=50, density=0.35),
                ExperimentSpec.model_validate(c.presets["boids_noisy"].spec.model_dump()),
            ):
                rid = c.create_run(sp, "cpu_dev", 0)["id"]
                await c.start_run(rid, start_paused=True)
                await wait_for(lambda rid=rid: state_of(c, rid) == "paused")
                await c.control(rid, "step", {"count": 73})
                q = {"x": 20, "y": 15, "radius": 6}
                n_commands = len(c.db.all("SELECT * FROM commands WHERE run_id=?", (rid,)))
                n_events = len(c.list_events(rid))
                live = await c.inspect(rid, q)
                replayed = await c.inspect(rid, q, tick=73)
                assert live["source"]["method"] == "live_worker"
                assert replayed["source"]["method"] == "deterministic_replay"
                strip = lambda r: {k: v for k, v in r.items() if k != "source"}  # noqa: E731
                assert strip(live) == strip(replayed)
                past = await c.inspect(rid, q, tick=10)
                assert past["tick"] == 10
                # Read-only: no command rows, no events, run unchanged and still verifiable.
                assert len(c.db.all("SELECT * FROM commands WHERE run_id=?", (rid,))) == n_commands
                assert len(c.list_events(rid)) == n_events
                assert c.run_summary(rid)["clock"]["value"] == 73
                with pytest.raises(ControllerError):
                    await c.inspect(rid, q, tick=500)
                await c.control(rid, "checkpoint")
                assert (await c.verify_replay(rid))["status"] == "verified"
                await c.cancel_run(rid)
        finally:
            await c.shutdown()

    arun(main)
