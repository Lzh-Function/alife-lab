"""Flow-Lenia through the shared platform: worker process, paint intervention, restart,
exact replay verification across a periodic checkpoint and past inspection."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

PAINT = {"kind": "paint", "payload": {"x": 5, "y": 5, "radius": 2, "value": 0.7, "channel": 1}}


@pytest.mark.integration
def test_flow_lenia_restart_replay_inspect(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        base = c.presets["flow_lenia_example"].spec.model_dump(mode="json")
        base["experiment"]["world"].update(width=64, height=64)
        base["experiment"]["initialization"]["size"] = 20
        base["experiment"]["runner"]["checkpoint_interval"] = 100
        rid = c.create_run(ExperimentSpec.model_validate(base), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 30})
        res = await c.control(rid, "intervene", PAINT)
        assert res["ack"]["result"]["cells"] == 13
        await c.control(rid, "step", {"count": 20})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 80})  # crosses the checkpoint at 100
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            frames = await c2.replay_frames(rid, 29, 2)
            assert [f["codec"] for f in frames] == ["field_u8"] * 2
            past = await c2.inspect(rid, {"x": 5, "y": 5}, tick=31)
            assert past["source"]["method"] == "deterministic_replay"
            assert len(past["channels"]) == 3
        finally:
            await c2.shutdown()

    arun(main)
