"""AgentWorld through the platform: worker, interventions, restart, replay verification and
past inspection of an agent's sensor/action history."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for


@pytest.mark.integration
def test_agent_world_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["agent_world_lifetime"].spec.model_dump(mode="json")
        raw["experiment"]["runner"]["checkpoint_interval"] = 100
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 50})
        rm = {"kind": "remove", "payload": {"x": 50, "y": 50, "radius": 10}}
        assert (await c.control(rid, "intervene", rm))["ack"]["status"] == "applied"
        await c.control(rid, "step", {"count": 20})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 60})  # crosses the checkpoint at 100
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 20, "y": 20}, tick=40)
            assert past["source"]["method"] == "deterministic_replay"
            assert len(past["agent"]["history"]["action"]) > 0
        finally:
            await c2.shutdown()

    arun(main)
