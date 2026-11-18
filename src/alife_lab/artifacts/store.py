"""Local filesystem ArtifactStore.

Publication protocol (policy rule 6): write to a staging file inside the run
directory -> fsync -> sha256 -> re-read and verify -> atomic rename -> return a
receipt. Only the controller records receipts in the database. A crash at any
point before the rename leaves only a ``.staging`` file, which is never listed
as a valid artifact.

Paths are always derived from validated identifiers (``run_id``, logical name);
user-supplied paths are never opened.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.core.hashing import sha256_bytes, sha256_file

_RUN_ID_RE = re.compile(r"^run_[0-9a-f]{16}$")
_LOGICAL_RE = re.compile(r"^[a-z0-9_][a-z0-9_.\-]*(/[a-z0-9_][a-z0-9_.\-]*)*$")
CHECKPOINT_FORMAT = "alife_npz_v1"


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactReceipt:
    run_id: str
    kind: str
    logical_path: str
    storage_uri: str
    sha256: str
    size_bytes: int
    mime_type: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise ArtifactError(f"invalid run id {run_id!r}")
    return run_id


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = (self.root / validate_run_id(run_id)).resolve()
        if path.parent != self.root:
            raise ArtifactError("run directory escapes artifact root")
        return path

    def resolve(self, run_id: str, logical_path: str) -> Path:
        if not _LOGICAL_RE.match(logical_path) or ".." in logical_path or len(logical_path) > 200:
            raise ArtifactError(f"invalid logical artifact path {logical_path!r}")
        base = self.run_dir(run_id)
        path = (base / logical_path).resolve()
        if base not in path.parents:
            raise ArtifactError("artifact path escapes run directory")
        return path

    def storage_uri(self, run_id: str, logical_path: str) -> str:
        return f"artifact://{run_id}/{logical_path}"

    def publish_bytes(
        self, run_id: str, logical_path: str, data: bytes, kind: str, mime_type: str
    ) -> ArtifactReceipt:
        final = self.resolve(run_id, logical_path)
        if final.exists():
            raise ArtifactError(f"artifact {logical_path} already exists (immutable)")
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.with_name(final.name + ".staging")
        expected = sha256_bytes(data)
        with staging.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        actual = sha256_file(staging)
        if actual != expected:
            staging.unlink(missing_ok=True)
            raise ArtifactError(f"checksum mismatch while staging {logical_path}")
        os.replace(staging, final)
        _fsync_dir(final.parent)
        return ArtifactReceipt(
            run_id=run_id,
            kind=kind,
            logical_path=logical_path,
            storage_uri=self.storage_uri(run_id, logical_path),
            sha256=expected,
            size_bytes=len(data),
            mime_type=mime_type,
        )

    def write_json_atomic(self, run_id: str, logical_path: str, value: Any) -> Path:
        """Mutable JSON documents (e.g. indexes); replaced atomically."""
        final = self.resolve(run_id, logical_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.with_name(final.name + ".staging")
        with staging.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, final)
        return final

    def verify(self, run_id: str, logical_path: str, sha256: str) -> bool:
        path = self.resolve(run_id, logical_path)
        return path.is_file() and sha256_file(path) == sha256

    def disk_usage_bytes(self, run_id: str) -> int:
        base = self.run_dir(run_id)
        if not base.exists():
            return 0
        return sum(p.stat().st_size for p in base.rglob("*") if p.is_file())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# Checkpoint encoding (NPZ without pickle)
# --------------------------------------------------------------------------- #


def encode_checkpoint(arrays: dict[str, np.ndarray[Any, Any]], meta: dict[str, Any]) -> bytes:
    for name, array in arrays.items():
        if array.dtype == np.dtype("O"):
            raise ArtifactError(f"object arrays are not allowed in checkpoints ({name})")
    buffer = io.BytesIO()
    meta_json = json.dumps({"format": CHECKPOINT_FORMAT, **meta}, sort_keys=True)
    all_arrays: dict[str, Any] = {"__meta__": np.array(meta_json), **arrays}
    np.savez_compressed(buffer, **all_arrays)
    return buffer.getvalue()


def decode_checkpoint(data: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        meta = json.loads(str(npz["__meta__"]))
        if meta.get("format") != CHECKPOINT_FORMAT:
            raise ArtifactError("unknown checkpoint format")
        arrays = {name: npz[name] for name in npz.files if name != "__meta__"}
    return {"arrays": arrays, "meta": meta}
