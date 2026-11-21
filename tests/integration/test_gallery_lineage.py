"""Gallery entries (PNG validation, storage, hiding) and the lineage table of a search run."""

from __future__ import annotations

import base64
import struct
import zlib

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller, ControllerError, NotFound
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, spec, state_of, wait_for


def png(w: int = 4, h: int = 3) -> bytes:
    raw = b"".join(b"\x00" + bytes([200, 30, 30] * w) for _ in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))  # fmt: skip


@pytest.mark.integration
def test_gallery(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            rid = c.create_run(spec(), "cpu_dev", 0)["id"]
            data = png()
            entry = c.gallery.add(rid, 3, "glider", "my favourite", base64.b64encode(data).decode())
            assert entry["width"] == 4 and entry["height"] == 3 and entry["substrate"] == "conway"
            assert c.gallery.image(entry["id"]) == data
            assert [e["id"] for e in c.gallery.list()] == [entry["id"]]
            for bad in (b"GIF89a....", data[:20], b"x" * 10):
                with pytest.raises(ControllerError):
                    c.gallery.add(rid, 0, "x", "", base64.b64encode(bad).decode())
            with pytest.raises(ControllerError):
                c.gallery.add(rid, 0, "x", "", "not base64!!")
            c.gallery.hide(entry["id"])
            assert c.gallery.list() == []
            with pytest.raises(NotFound):
                c.gallery.image(entry["id"])
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.integration
def test_lineage_table_of_a_map_elites_run(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            raw = c.presets["search_lenia_qd_map_elites"].spec.model_dump(mode="json")
            raw["experiment"]["dynamics"].update(population=6, eval_steps=80)
            raw["experiment"]["runner"] = {"max_steps": 4, "checkpoint_interval": 2}
            rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
            await c.start_run(rid)
            await wait_for(lambda: state_of(c, rid) == "completed", 60)
            lin = await c.inspect(rid, {"lineage": True})
            table = lin["lineage_table"]
            n = len(table["generation"])
            assert n == 24 and len(table["parent"]) == n and len(table["u"]) == n
            for child, parent in enumerate(table["parent"]):
                if parent >= 0:
                    assert table["generation"][parent] < table["generation"][child]
            assert set(lin["descriptors"]) == {"speed", "mass"}
            assert lin["best"]["index"] == max(range(n), key=lambda i: table["score"][i])
        finally:
            await c.shutdown()

    arun(main)
