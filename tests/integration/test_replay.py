"""Deterministic replay, replay verification and fork-at-tick."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alife_lab.artifacts.store import decode_checkpoint
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller, ControllerError
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, spec, state_of, wait_for

B1 = {"kind": "brush", "payload": {"shape": "circle", "x": 10, "y": 10, "radius": 2, "value": 1}}
B2 = {"kind": "brush", "payload": {"shape": "circle", "x": 30, "y": 20, "radius": 3, "value": 0}}
B3 = {"kind": "brush", "payload": {"shape": "cells", "cells": [[1, 1], [2, 1], [3, 1]], "value": 1}}
# (tick, intervention, manual checkpoint after it?)
SCRIPT: list[tuple[int, dict[str, Any], bool]] = [(37, B1, False), (80, B2, True), (80, B3, False)]
END = 120


def unpack(frame: dict[str, Any]) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(frame["data"], dtype=np.uint8))
    return bits[: frame["width"] * frame["height"]].reshape(frame["height"], frame["width"])


async def new_paused(c: Controller) -> str:
    rid = c.create_run(
        spec("random", max_steps=100_000, interval=50, density=0.35), "cpu_dev", ticks_per_second=0
    )["id"]
    await c.start_run(rid, start_paused=True)
    await wait_for(lambda: state_of(c, rid) == "paused")
    return rid


async def scripted(c: Controller, rid: str, every_tick_checkpoint: bool) -> None:
    tick = 0
    for target, intervention, checkpoint_after in [*SCRIPT, (END, None, False)]:
        while tick < target:
            await c.control(rid, "step", {"count": 1 if every_tick_checkpoint else target - tick})
            tick = c.run_summary(rid)["clock"]["value"]
            if every_tick_checkpoint:
                await c.control(rid, "checkpoint")
        if intervention is not None:
            res = await c.control(rid, "intervene", intervention)
            assert res["status"] == "applied"
            if checkpoint_after or every_tick_checkpoint:
                await c.control(rid, "checkpoint")


def truth_grids(c: Controller, rid: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for cp in c._ordered_checkpoints(rid):  # latest per tick wins (post-intervention)
        data = c.store.resolve(rid, cp["logical_path"]).read_bytes()
        out[cp["simulation_clock"]] = decode_checkpoint(data)["arrays"]["grid"]
    return out


def test_replay_matches_ground_truth_and_verifies(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            a = await new_paused(c)
            b = await new_paused(c)
            await scripted(c, a, every_tick_checkpoint=False)
            await scripted(c, b, every_tick_checkpoint=True)
            truth = truth_grids(c, b)
            assert set(range(END + 1)) <= set(truth)
            frames = await c.replay_frames(a, 0, END + 1, 1)
            assert [f["tick"] for f in frames] == list(range(END + 1))
            for frame in frames:
                np.testing.assert_array_equal(unpack(frame), truth[frame["tick"]])
            sources = {f["tick"]: f["source"] for f in frames}
            assert sources[50] == "checkpoint_tick" and sources[51] == "deterministic_replay"
            # Strided request crossing interventions.
            strided = await c.replay_frames(a, 30, 10, 9)
            for frame in strided:
                np.testing.assert_array_equal(unpack(frame), truth[frame["tick"]])

            report = await c.verify_replay(a)
            # checkpoints of run a: 0 (initial), 50, 80 (manual, between B2 and B3), 100
            assert report["ok"] and report["segments_checked"] == report["segments_total"] == 3
            assert report["status"] == "verified"
            partial = await c.verify_replay(a, max_total_ticks=60)
            assert partial["status"] == "partial" and not partial["ok"]
            assert partial["segments_checked"] == 1  # 0->50 fits, 50->80 exceeds the budget
            assert "replay_verified" in [e["type"] for e in c.list_events(a)]

            timeline = c.timeline(a)
            assert timeline["range"] == [0, END]
            assert timeline["replay"]["mode"] == "exact_recompute"
            assert [i["applied_clock"] for i in timeline["interventions"]] == [37, 80, 80]

            with pytest.raises(ControllerError):
                await c.replay_frames(a, END + 1)
            fresh = await new_paused(c)
            nothing = await c.verify_replay(fresh)
            assert nothing["status"] == "nothing_to_verify" and not nothing["ok"]

            # Negative control: a falsified intervention log must fail verification.
            c.db.execute(
                "UPDATE interventions SET payload_json=? WHERE run_id=? AND applied_clock=37",
                ('{"shape":"circle","x":11,"y":10,"radius":2,"value":1}', a),
            )
            bad = await c.verify_replay(a)
            assert not bad["ok"] and bad["status"] == "mismatch"
            failed = [s for s in bad["segments"] if not s["ok"]]
            assert failed[0]["from_tick"] == 0 and failed[0]["differing_cells"] > 0
        finally:
            await c.shutdown()

    arun(main)


def test_fork_at_arbitrary_tick(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            parent = await new_paused(c)
            await scripted(c, parent, every_tick_checkpoint=False)
            # Base checkpoint: tick 80 (taken after B2); B3 at tick 80 must be re-applied.
            child = await c.fork_at_tick(parent, 85, label="fork@85")
            assert child["clock"]["value"] == 85
            origin = c.provenance(child["id"])["lineage"]["origin"]
            assert origin == {
                "method": "deterministic_replay",
                "base_checkpoint_id": child["parent_checkpoint_id"],
                "interventions_replayed": 1,
            }
            await c.start_run(child["id"], start_paused=True)
            await wait_for(lambda: state_of(c, child["id"]) == "paused")
            await c.control(child["id"], "step", {"count": 10})
            await c.control(child["id"], "checkpoint")
            parent_frames = {f["tick"]: unpack(f) for f in await c.replay_frames(parent, 85, 11)}
            child_truth = truth_grids(c, child["id"])
            for tick in (85, 95):
                np.testing.assert_array_equal(child_truth[tick], parent_frames[tick])
            with pytest.raises(ControllerError):
                await c.fork_at_tick(parent, END + 50)
        finally:
            await c.shutdown()

    arun(main)
