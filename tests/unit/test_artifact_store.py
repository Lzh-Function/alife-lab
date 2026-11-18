"""ArtifactStore: atomic publish, immutability, checksum, path safety, no pickle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alife_lab.artifacts.store import (
    ArtifactError,
    LocalArtifactStore,
    decode_checkpoint,
    encode_checkpoint,
)

RUN = "run_0123456789abcdef"


def test_publish_and_verify(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    receipt = store.publish_bytes(RUN, "checkpoints/a.npz", b"hello", "checkpoint", "x/y")
    assert receipt.size_bytes == 5
    assert store.verify(RUN, "checkpoints/a.npz", receipt.sha256)
    assert not list(tmp_path.rglob("*.staging"))
    assert receipt.storage_uri == f"artifact://{RUN}/checkpoints/a.npz"


def test_published_artifacts_are_immutable(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.publish_bytes(RUN, "manifest.json", b"{}", "provenance", "application/json")
    with pytest.raises(ArtifactError):
        store.publish_bytes(RUN, "manifest.json", b"{1}", "provenance", "application/json")


def test_corruption_detected(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    receipt = store.publish_bytes(RUN, "x.bin", b"abc", "k", "m")
    (tmp_path / RUN / "x.bin").write_bytes(b"abd")
    assert not store.verify(RUN, "x.bin", receipt.sha256)


@pytest.mark.parametrize(
    "logical", ["../escape", "/abs", "a/../../b", ".hidden", "a//b", "A/UPPER", "a/b\\c"]
)
def test_path_traversal_rejected(tmp_path: Path, logical: str) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ArtifactError):
        store.resolve(RUN, logical)


@pytest.mark.parametrize("run_id", ["../x", "run_123", "run_0123456789ABCDEF", ""])
def test_invalid_run_ids_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ArtifactError):
        LocalArtifactStore(tmp_path).run_dir(run_id)


def test_checkpoint_codec_roundtrip_without_pickle() -> None:
    grid = (np.arange(64).reshape(8, 8) % 2).astype(np.uint8)
    data = encode_checkpoint({"grid": grid}, {"tick": 7})
    decoded = decode_checkpoint(data)
    np.testing.assert_array_equal(decoded["arrays"]["grid"], grid)
    assert decoded["meta"]["tick"] == 7
    with pytest.raises(ArtifactError):
        encode_checkpoint({"bad": np.array([object()], dtype=object)}, {})


def test_failed_staging_leaves_no_published_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alife_lab.artifacts.store as mod

    store = LocalArtifactStore(tmp_path)
    monkeypatch.setattr(mod, "sha256_file", lambda _p: "0" * 64)
    with pytest.raises(ArtifactError):
        store.publish_bytes(RUN, "checkpoints/b.npz", b"data", "checkpoint", "x")
    assert not (tmp_path / RUN / "checkpoints" / "b.npz").exists()
    assert not list(tmp_path.rglob("*.staging"))
