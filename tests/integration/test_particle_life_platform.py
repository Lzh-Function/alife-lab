"""Particle Life through the shared platform: worker process, interventions, restart,
replay verification, inspection, and a sweep with the zero-matrix negative control."""

from __future__ import annotations

import pytest

from alife_lab.core.sweep import SweepSpec
from alife_lab.execution.executors import FakeExecutor, LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

SPAWN = {
    "kind": "spawn",
    "payload": {"x": 100.0, "y": 100.0, "radius": 6.0, "count": 30, "type": 1},
}


@pytest.mark.integration
def test_particle_life_restart_replay_inspect(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        rid = c.create_run(c.presets["particle_life_clusters"].spec, "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 120})
        res = await c.control(rid, "intervene", SPAWN)
        assert res["ack"]["result"] == {"population_before": 600, "population_after": 630}
        info = await c.inspect(rid, {"x": 100.0, "y": 100.0, "radius": 6.0})
        assert info["type_counts_in_radius"][1] >= 30 and len(info["matrix"]) == 4
        await c.control(rid, "step", {"count": 30})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 400})  # crosses the periodic checkpoint at 500
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            frames = await c2.replay_frames(rid, 119, 3)
            assert [f["codec"] for f in frames] == ["particles_typed_f32"] * 3
            assert [f["count"] for f in frames] == [600, 630, 630]
            past = await c2.inspect(rid, {"x": 100.0, "y": 100.0, "radius": 6.0}, tick=121)
            assert past["source"]["method"] == "deterministic_replay"
        finally:
            await c2.shutdown()

    arun(main)


def test_particle_life_sweep_negative_control(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            self_m = [[1.0 if i == j else -0.2 for j in range(4)] for i in range(4)]
            zero_m = [[0.0] * 4 for _ in range(4)]
            spec = SweepSpec.model_validate(
                {
                    "sweep": {
                        "name": "pl_sorting",
                        "base_preset": "particle_life_clusters",
                        "overrides": {
                            "initialization": {"count": 300},
                            "runner": {"max_steps": 400, "checkpoint_interval": 400},
                        },
                        "seeds": [1, 2, 3, 4],
                        "factors": [
                            {
                                "path": "dynamics.matrix",
                                "values": [zero_m, self_m],
                                "labels": ["zero", "self_attraction"],
                            }
                        ],
                        "endpoint": {"metric": "segregation_index"},
                    }
                }
            )
            sid = (await c.sweeps.create(spec))["id"]
            await wait_for(lambda: c.sweeps.summary(sid)["state"] != "running", 120)
            report = c.sweeps.report(sid)
            zero, self_attr = report["conditions"]
            assert zero["summary"]["n"] == 4 and self_attr["summary"]["n"] == 4
            assert min(self_attr["values"]) > max(zero["values"])
            assert report["comparisons"][0]["cliffs_delta"] == 1.0
        finally:
            await c.shutdown()

    arun(main)
