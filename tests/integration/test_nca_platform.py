"""NCA through the platform: an nca_train optimisation run in a worker, replay verification,
export of a checkpoint to an nca run spec, and damage/replay on that run. Skipped without JAX."""

from __future__ import annotations

import importlib.util

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX not installed"),
]


def test_train_export_run_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            raw = c.presets["nca_train_regenerating"].spec.model_dump(mode="json")
            raw["experiment"]["target"].update(preset="disc", size=12)
            raw["experiment"]["dynamics"].update(
                pad=2, batch_size=4, pool_size=8, steps_min=16, steps_max=24, damaged_per_batch=1
            )
            raw["experiment"]["runner"]["checkpoint_interval"] = 5
            rid = c.create_run(ExperimentSpec.model_validate(raw), "local_interactive", 0)["id"]
            await c.start_run(rid, start_paused=True)
            await wait_for(lambda: state_of(c, rid) == "paused", 120)
            await c.control(rid, "step", {"count": 8})  # crosses the checkpoint at 5
            assert c.run_summary(rid)["clock"]["unit"] == "optimizer_step"
            report = await c.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            await c.control(rid, "checkpoint", {})
            await wait_for(
                lambda: (c.latest_checkpoint(rid) or {}).get("simulation_clock") == 8, 30
            )

            exported = c.export_nca_spec(rid)
            assert exported["checkpoint"]["tick"] == 8
            nsp = ExperimentSpec.model_validate(exported["spec"])
            nsp_raw = nsp.model_dump(mode="json")
            nsp_raw["experiment"]["runner"]["checkpoint_interval"] = 20
            nrid = c.create_run(ExperimentSpec.model_validate(nsp_raw), "local_interactive", 0)[
                "id"
            ]
            await c.start_run(nrid, start_paused=True)
            await wait_for(lambda: state_of(c, nrid) == "paused", 120)
            await c.control(nrid, "step", {"count": 10})
            dmg = {"kind": "damage", "payload": {"x": 8, "y": 8, "radius": 3}}
            assert (await c.control(nrid, "intervene", dmg))["ack"]["status"] == "applied"
            await c.control(nrid, "step", {"count": 15})  # crosses the checkpoint at 20
            report = await c.verify_replay(nrid)
            assert report["status"] == "verified", report["segments"]
            prov = c.provenance(nrid)
            assert prov["execution"]["backend"] == "jax"
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.skipif(
    importlib.util.find_spec("jax_cuda13_plugin") is None, reason="JAX CUDA build not installed"
)
def test_gpu_profile_runs_nca_on_jax_cuda_and_replays(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            if "jax_cuda" not in c.target.capabilities.backends:
                pytest.skip("no CUDA device")
            sp = c.presets["nca_heart24_pretrained"].spec
            # The CPU-only numeric policy is rejected on the GPU (never silently changed).
            with pytest.raises(Exception, match="numeric_policy"):
                c.create_run(sp, "local_gpu", 0)
            raw = sp.model_dump(mode="json")
            raw["experiment"]["dynamics"]["numeric_policy"] = "float32_jax"
            raw["experiment"]["runner"]["checkpoint_interval"] = 20
            rid = c.create_run(ExperimentSpec.model_validate(raw), "local_gpu", 0)["id"]
            await c.start_run(rid, start_paused=True)
            await wait_for(lambda: state_of(c, rid) == "paused", 120)
            await c.control(rid, "step", {"count": 30})
            assert c.run_summary(rid)["backend"] == "jax_cuda"
            versions = c.provenance(rid)["execution"]["backend_versions"]
            assert versions["jax-cuda13-plugin"] and versions["gpu"]
            report = await c.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            await wait_for(
                lambda: (c.run_summary(rid)["live"] or {}).get("gpu_memory") is not None, 15
            )
            assert c.run_summary(rid)["live"]["gpu_memory"]["peak_pool_held_bytes"] > 0
        finally:
            await c.shutdown()

    arun(main)
