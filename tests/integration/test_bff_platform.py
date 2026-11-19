"""BFF through the platform: epochs, restore/replay exactness, interventions, inspection,
controls, and a substrate-specific clock unit."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import FakeExecutor, LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from alife_lab.substrates.bff.adapter import BFFAdapter, soup_entropy_metrics
from alife_lab.substrates.bff.soup import init_soup, run_epoch
from tests.integration.test_controller import arun, state_of, wait_for


def small(
    c: Controller, preset: str = "bff_reference", n: int = 256, **dyn: object
) -> ExperimentSpec:
    raw = c.presets[preset].spec.model_dump(mode="json")
    raw["experiment"]["soup"]["num_programs"] = n
    raw["experiment"]["runner"] = {"max_steps": 100_000, "checkpoint_interval": 10}
    raw["experiment"]["dynamics"].update(dyn)
    return ExperimentSpec.model_validate(raw)


def test_reference_mode_guards() -> None:
    base = {"name": "b", "substrate": "bff", "seed": 1}
    ExperimentSpec.model_validate({"experiment": base})
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({"experiment": {**base, "dynamics": {"step_limit": 100}}})
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({"experiment": {**base, "soup": {"num_programs": 3}}})
    ok = ExperimentSpec.model_validate(
        {"experiment": {**base, "mode": "exploratory", "dynamics": {"step_limit": 0}}}
    )
    assert ok.experiment.dynamics.step_limit == 0  # type: ignore[union-attr]


def test_entropy_metrics_known_cases() -> None:
    zeros = np.zeros((64, 64), dtype=np.uint8)
    m = soup_entropy_metrics(zeros)
    assert m["h0_bits_per_byte"] == 0.0 and m["brotli_bits_per_byte"] < 0.05
    rnd = init_soup(3, 1024)
    r = soup_entropy_metrics(rnd)
    assert 7.95 < r["h0_bits_per_byte"] <= 8.0
    assert r["brotli_bits_per_byte"] > 7.9  # random data is incompressible
    assert abs(r["higher_order_entropy"]) < 0.15
    repeated = np.tile(rnd[:1], (1024, 1))  # one program copied everywhere
    rep = soup_entropy_metrics(repeated)
    assert rep["higher_order_entropy"] > 5.0  # high h0 per byte but tiny compressed size


def test_non_execution_control_changes_only_by_mutation() -> None:
    n, seed, mut = 512, 9, 1 << 18
    soup = init_soup(seed, n)
    before = soup.copy()
    total = 0
    for e in range(40):
        total += run_epoch(soup, seed, e, mut, step_limit=0)["mutated_bytes"]
    # Without execution only mutation alters bytes (a few may be re-mutated or unchanged).
    assert 0 < int(np.count_nonzero(soup != before)) <= total
    assert sorted(map(bytes, soup.tolist())) != sorted(map(bytes, before.tolist()))


def test_bff_runs_restore_replay_inspect(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        sp = small(c)
        rid = c.create_run(sp, "cpu_dev", 0)["id"]
        assert c.run_summary(rid)["clock"]["unit"] == "epoch"
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 23})
        program = "[[{.>]-]" + "+" * 4
        res = await c.control(
            rid,
            "intervene",
            {"kind": "inject_program", "payload": {"program": program, "indices": [0, 5, 7]}},
        )
        assert res["status"] == "applied" and res["ack"]["result"]["slots"] == 3
        info = await c.inspect(rid, {"x": 5, "y": 0, "radius": 0})
        assert info["index"] == 5 and info["disassembly"].startswith(program)
        assert info["exact_copies_in_soup"] == 3
        assert "self_pair_test" in info
        assert c.run_summary(rid)["effective_mode"] == "exploratory"
        await c.control(rid, "step", {"count": 9})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 20})
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            assert report["segments"][0]["distance_metric"] == "differing_bytes"
            rows = c2.metrics(rid, max_points=1000)["rows"]
            assert [r["tick"] for r in rows] == list(range(53))
            assert rows[-1]["values"]["unique_programs"] <= 256
            events = c2.list_events(rid)
            assert {e["clock_unit"] for e in events} == {"epoch"}
            frames = await c2.replay_frames(rid, 22, 2)
            assert [f["codec"] for f in frames] == ["bff_soup", "bff_soup"]
            assert frames[1]["interventions_replayed"] == 1
        finally:
            await c2.shutdown()

    arun(main)


def test_platform_epochs_equal_direct_procedure(settings: Settings) -> None:
    """The worker's soup after k epochs equals the reference procedure called directly."""

    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            sp = small(c, n=128)
            rid = c.create_run(sp, "cpu_dev", 0)["id"]
            await c.start_run(rid, start_paused=True)
            await wait_for(lambda: state_of(c, rid) == "paused")
            await c.control(rid, "step", {"count": 30})  # checkpoints at 10, 20, 30
            from alife_lab.artifacts.store import decode_checkpoint

            latest = c.latest_checkpoint(rid)
            stored = decode_checkpoint(c.store.resolve(rid, latest["logical_path"]).read_bytes())
            direct = BFFAdapter().initialize(sp)
            for _ in range(30):
                direct = BFFAdapter().step(direct, sp)
            assert np.array_equal(stored["arrays"]["soup"], direct.soup)
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("numba") is None, reason="numba extra not installed"
)
def test_backend_choice_does_not_change_results(settings: Settings) -> None:
    """auto -> numba; identical soups to the numpy backend; recorded in the manifest."""
    from alife_lab.core.spec import ExecutionProfile

    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            base = c.profiles["cpu_dev"].model_dump(mode="json")
            base["execution"]["id"] = "cpu_auto"
            base["execution"]["target_selector"]["backend"] = "auto"
            c.profiles["cpu_auto"] = ExecutionProfile.model_validate(base)
            sp = small(c, n=512)
            runs = {}
            for profile in ("cpu_auto", "cpu_dev"):
                rid = c.create_run(sp, profile, 0)["id"]
                await c.start_run(rid, start_paused=True)
                await wait_for(lambda rid=rid: state_of(c, rid) == "paused", 30)
                await c.control(rid, "step", {"count": 40})
                runs[profile] = rid
            assert c.run_summary(runs["cpu_auto"])["backend"] == "numba"
            assert c.run_summary(runs["cpu_dev"])["backend"] == "numpy"
            manifest = c.provenance(runs["cpu_auto"])["execution"]
            assert manifest["backend"] == "numba" and manifest["backend_versions"]["numba"]
            cmp = c.create_comparison([runs["cpu_dev"], runs["cpu_auto"]], name="backends")
            points = c.comparison_report(cmp["id"])["divergence"]["points"]
            assert [p["tick"] for p in points] == [0, 10, 20, 30, 40]
            assert all(p["distance_to_first"] == [0, 0] for p in points)
            assert (await c.verify_replay(runs["cpu_auto"]))["status"] == "verified"
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("cupy") is None, reason="gpu extra not installed"
)
def test_gpu_backend_run_equals_cpu_and_replays(settings: Settings) -> None:
    """local_gpu profile -> cupy backend in a worker process; bit-identical soups to numpy;
    replay verification reproduces with the recorded backend; GPU memory is reported."""
    from alife_lab.substrates.bff.vm_cupy import available

    if not available():
        pytest.skip("no CUDA device")

    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            sp = small(c, n=512)
            runs = {}
            for profile in ("local_gpu", "cpu_dev"):
                rid = c.create_run(sp, profile, 0)["id"]
                await c.start_run(rid, start_paused=True)
                await wait_for(lambda rid=rid: state_of(c, rid) == "paused", 60)
                await c.control(rid, "step", {"count": 30})
                runs[profile] = rid
            assert c.run_summary(runs["local_gpu"])["backend"] == "cupy"
            manifest = c.provenance(runs["local_gpu"])["execution"]
            assert manifest["backend_versions"]["cupy"] and manifest["backend_versions"]["gpu"]
            cmp = c.create_comparison([runs["cpu_dev"], runs["local_gpu"]], name="gpu-vs-cpu")
            points = c.comparison_report(cmp["id"])["divergence"]["points"]
            assert all(p["distance_to_first"] == [0, 0] for p in points)
            assert (await c.verify_replay(runs["local_gpu"]))["status"] == "verified"
            await wait_for(
                lambda: (
                    (c.run_summary(runs["local_gpu"])["live"] or {}).get("gpu_memory") is not None
                ),
                15,
            )
            gm = c.run_summary(runs["local_gpu"])["live"]["gpu_memory"]
            assert gm["peak_pool_held_bytes"] > 0 and gm["device_total_mib"] > 1000
        finally:
            await c.shutdown()

    arun(main)
