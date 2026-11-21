"""Gallery: pictures the user liked (spec §3.10: a gallery separate from the discovery
ledger). An entry is a PNG of what the browser showed, the run and tick it came from, a title
and a note. It is a visual record (evidence E0), never a discovery claim.

Images are validated (PNG signature, IHDR size, byte limit) and stored as immutable artifacts
of the source run. Removing an entry only hides it; the artifact stays.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import secrets
import struct
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alife_lab.service.controller import Controller

MAX_PNG_BYTES = 2 * 1024 * 1024
MAX_SIDE = 4096
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from the IHDR chunk; raises ValueError if this is not a PNG."""
    if len(data) > MAX_PNG_BYTES:
        raise ValueError(f"image larger than {MAX_PNG_BYTES} bytes")
    if len(data) < 33 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise ValueError("not a PNG image")
    width, height = struct.unpack(">II", data[16:24])
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise ValueError("PNG dimensions out of range")
    return width, height


class GalleryService:
    def __init__(self, controller: Controller) -> None:
        self.c = controller

    def add(self, run_id: str, tick: int, title: str, note: str, png_base64: str) -> dict[str, Any]:
        from alife_lab.service.controller import ControllerError

        try:
            data = base64.b64decode(png_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ControllerError("image must be base64-encoded PNG") from exc
        try:
            width, height = png_size(data)
        except ValueError as exc:
            raise ControllerError(str(exc)) from exc
        run = self.c.run_summary(run_id)
        gid = "gal_" + secrets.token_hex(8)
        receipt = self.c.store.publish_bytes(run_id, f"gallery/{gid}.png", data, "gallery",
                                             "image/png")  # fmt: skip
        self.c.db.execute(
            "INSERT INTO gallery(id, run_id, tick, substrate, title, note, image_path, "
            "image_sha256, width, height, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (gid, run_id, int(tick), run["substrate"], title.strip()[:200] or run["name"],
             note.strip()[:2000], receipt.logical_path, hashlib.sha256(data).hexdigest(),
             width, height, time.time()),
        )  # fmt: skip
        self.c._emit_event(run_id, "gallery_added", {"gallery_id": gid, "tick": int(tick)})
        return self.get(gid)

    def _row(self, gid: str) -> dict[str, Any]:
        from alife_lab.service.controller import NotFound

        row = self.c.db.one("SELECT * FROM gallery WHERE id=? AND hidden_at IS NULL", (gid,))
        if row is None:
            raise NotFound(f"gallery entry {gid!r} not found")
        return dict(row)

    def get(self, gid: str) -> dict[str, Any]:
        r = self._row(gid)
        return {k: r[k] for k in ("id", "run_id", "tick", "substrate", "title", "note", "width",
                                  "height", "created_at", "image_sha256")}  # fmt: skip

    def list(self) -> list[dict[str, Any]]:
        rows = self.c.db.all(
            "SELECT id FROM gallery WHERE hidden_at IS NULL ORDER BY created_at DESC"
        )
        return [self.get(r["id"]) for r in rows]

    def image(self, gid: str) -> bytes:
        r = self._row(gid)
        data = self.c.store.resolve(r["run_id"], r["image_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != r["image_sha256"]:
            raise RuntimeError("gallery image hash mismatch")
        return data

    def hide(self, gid: str) -> dict[str, Any]:
        self._row(gid)
        self.c.db.execute("UPDATE gallery SET hidden_at=? WHERE id=?", (time.time(), gid))
        return {"id": gid, "hidden": True}
