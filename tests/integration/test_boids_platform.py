"""Boids through the shared platform: worker process, interventions, restart/restore,
replay verification, paired fork and comparison."""

from __future__ import annotations

import pytest

from alife_lab.execution.executors import FakeExecutor, LocalExecutor
from alife_lab.service.controller import Controller, ControllerError
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

SPAWN = {"kind": "spawn", "payload": {"x": 50.0, "y": 50.0, "radius": 8.0, "count": 40}}


@pytest.mark.integration
def test_boids_lifecycle_restart_replay_and_compare(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        spec = c.presets["boids_noisy"].spec  # noise exercises the RNG in the checkpoint
        rid = c.create_run(spec, "cpu_dev", ticks_per_second=0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 150})
        res = await c.control(rid, "intervene", SPAWN, requested_clock=150)
        assert res["status"] == "applied"
        assert res["ack"]["result"] == {"population_before": 300, "population_after": 340}
        with pytest.raises(ControllerError):  # conway intervention is not offered for boids
            await c.control(rid, "intervene", {"kind": "brush", "payload": {}})
        await c.control(rid, "step", {"count": 100})  # periodic checkpoint at 200
        await c.shutdown()  # shutdown checkpoint at 250

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 150})  # periodic checkpoint at 400
            rows = c2.metrics(rid)["rows"]
            assert [r["tick"] for r in rows] == list(range(401))
            assert rows[-1]["values"]["population"] == 340
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            assert report["segments"][0]["distance_metric"] == "rms_position_distance"
            frames = await c2.replay_frames(rid, 149, 3)
            assert [f["codec"] for f in frames] == ["particles_f32"] * 3
            assert [f["count"] for f in frames] == [300, 340, 340]

            cmp = await c2.paired_fork(rid, "now", treatment_intervention=SPAWN)
            ctl_id, trt_id = (m["run"]["id"] for m in cmp["members"])
            await wait_for(lambda: all(state_of(c2, r) == "paused" for r in (ctl_id, trt_id)), 30)
            for arm in (ctl_id, trt_id):
                await c2.control(arm, "step", {"count": 50})
                await c2.control(arm, "checkpoint")
            div = c2.comparison_report(cmp["id"])["divergence"]
            assert div["metric"] == "rms_position_distance"
            first, last = div["points"][0], div["points"][-1]
            assert first["distance_to_first"] == [0.0, 0.0]
            assert last["distance_to_first"][1] > 0.0  # extra boids perturb the flock
            assert last["alive"] == [340, 380]
        finally:
            await c2.shutdown()

    arun(main)


def test_boids_rejects_reference_mode_and_bad_overrides(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            raw = c.presets["boids_flocking"].spec.model_dump(mode="json")
            raw["experiment"]["mode"] = "reference"
            from pydantic import ValidationError

            from alife_lab.core.spec import ExperimentSpec

            with pytest.raises(ValidationError):
                ExperimentSpec.model_validate(raw)
            raw["experiment"]["mode"] = "exploratory"
            raw["experiment"]["world"]["width"] = 20  # < 3 perception radii
            with pytest.raises(ValidationError):
                ExperimentSpec.model_validate(raw)
        finally:
            await c.shutdown()

    arun(main)
