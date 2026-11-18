"""Long runs write indexed, checksum-verified Parquet metric chunks."""

from __future__ import annotations

from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, spec, state_of, wait_for


def test_parquet_chunks_are_indexed_and_complete(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            rid = c.create_run(
                spec("random", max_steps=25_000, interval=5_000, density=0.3), "cpu_dev", 0
            )["id"]
            await c.start_run(rid)
            await wait_for(lambda: state_of(c, rid) == "completed", 120)
            chunks = c.db.all(
                "SELECT logical_path, size_bytes FROM artifacts WHERE run_id=? AND "
                "kind='metrics_chunk' ORDER BY logical_path",
                (rid,),
            )
            # 25 001 rows: two full chunks of 10 000 and a final chunk on completion.
            assert len(chunks) == 3
            run_dir = c.store.run_dir(rid)
            assert sorted(p.name for p in (run_dir / "metrics" / "chunks").iterdir()) == [
                r["logical_path"].split("/")[-1] for r in chunks
            ]
            assert (run_dir / "metrics" / "tail_e001.jsonl").read_text() == ""
            assert not (run_dir / "metrics" / "history.jsonl").exists()
            m = c.metrics(rid, max_points=100_000)
            assert m["total_points"] == 25_001
            assert [r["tick"] for r in m["rows"]] == list(range(25_001))
            window = c.metrics(rid, since_tick=12_344, until_tick=12_350)
            assert [r["tick"] for r in window["rows"]] == list(range(12_345, 12_351))
            assert c.verify_run(rid)["ok"]
            assert (await c.verify_replay(rid))["status"] == "verified"
        finally:
            await c.shutdown()

    arun(main)
