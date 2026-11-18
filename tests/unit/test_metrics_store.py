"""Metric storage: chunking, crash duplicates, legacy JSONL and equivalence with the Phase 1
reader semantics (property test against the original line-by-line implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import alife_lab.storage.metrics_store as ms
from alife_lab.artifacts.store import LocalArtifactStore
from alife_lab.storage.metrics_store import MetricsWriter, read_metrics

RUN = "run_00000000000000aa"


def reference_reader(
    rows: list[dict[str, Any]], start_ticks: list[tuple[int, int]], since: int = -1
) -> tuple[list[dict[str, Any]], int]:
    """The Phase 1 algorithm (file order, last row per tick wins)."""
    by_tick: dict[int, dict[str, Any]] = {}
    superseded = 0
    for row in rows:
        tick, epoch = int(row["tick"]), int(row["epoch"])
        later = [s for e, s in start_ticks if e > epoch]
        if later and tick > min(later):
            superseded += 1
            continue
        if tick > since:
            by_tick[tick] = row
    return [by_tick[t] for t in sorted(by_tick)], superseded


@st.composite
def histories(draw: st.DrawFn) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Epochs that restart from an earlier tick, with occasional post-intervention rows."""
    rows: list[dict[str, Any]] = []
    starts: list[tuple[int, int]] = []
    tick = 0
    for epoch in range(1, draw(st.integers(1, 4)) + 1):
        if epoch > 1:
            tick = draw(st.integers(0, tick))  # restore point
        starts.append((epoch, tick))
        if epoch == 1:
            rows.append({"tick": 0, "epoch": 1, "values": {"a": 0.0}})
        for _ in range(draw(st.integers(0, 40))):
            tick += 1
            rows.append({"tick": tick, "epoch": epoch, "values": {"a": float(tick * 10 + epoch)}})
            if draw(st.integers(0, 9)) == 0:
                rows.append(
                    {"tick": tick, "epoch": epoch, "values": {"a": -1.0}, "post_intervention": True}
                )
    return rows, starts


def write(tmp: Path, rows: list[dict[str, Any]], chunk: int) -> LocalArtifactStore:
    store = LocalArtifactStore(tmp)
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_epoch.setdefault(r["epoch"], []).append(dict(r))
    old = ms.CHUNK_ROWS
    ms.CHUNK_ROWS = chunk
    try:
        for epoch, part in by_epoch.items():
            w = MetricsWriter(store, RUN, epoch)
            for i in range(0, len(part), 3):
                w.append([dict(x) for x in part[i : i + 3]])
                w.compact()
            w.close()
    finally:
        ms.CHUNK_ROWS = old
    return store


@settings(max_examples=80, deadline=None)
@given(data=histories(), chunk=st.integers(1, 25), since=st.integers(-1, 30))
def test_equivalent_to_phase1_semantics(
    tmp_path_factory: pytest.TempPathFactory,
    data: tuple[list[dict[str, Any]], list[tuple[int, int]]],
    chunk: int,
    since: int,
) -> None:
    rows, starts = data
    tmp = tmp_path_factory.mktemp("m")
    store = write(tmp, rows, chunk)
    got = read_metrics(store.run_dir(RUN), starts, since_tick=since, max_points=10**6)
    expected, superseded = reference_reader(rows, starts, since)
    assert [(r["tick"], r["values"]["a"]) for r in got["rows"]] == [
        (r["tick"], r["values"]["a"]) for r in expected
    ]
    assert got["superseded_rows"] == superseded


def test_chunks_tail_crash_duplicates_and_legacy(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(ms, "CHUNK_ROWS", 5)
    store = LocalArtifactStore(tmp_path)
    run_dir = store.run_dir(RUN)
    (run_dir / "metrics").mkdir(parents=True)
    # Legacy Phase 1 file for ticks 0..2 (epoch 1).
    with (run_dir / "metrics" / "history.jsonl").open("w") as fh:
        for t in range(3):
            fh.write(json.dumps({"tick": t, "epoch": 1, "values": {"x": float(t)}}) + "\n")
    w = MetricsWriter(store, RUN, 2)
    w.append([{"tick": t, "epoch": 2, "values": {"x": float(t), "y": None}} for t in range(3, 10)])
    receipt = w.compact()
    assert receipt is not None and receipt.logical_path.startswith("metrics/chunks/e002_s")
    w.append([{"tick": 10, "epoch": 2, "values": {"x": 10.0, "y": 1.5}}])
    w.sync()
    # Simulate a crash after publishing a chunk but before truncating the tail.
    tail_copy = w.tail_path.read_text()
    w.compact(force=True)
    w.tail_path.write_text(tail_copy + '{"tick": 11, "epoch": 2, "seq": 8')  # + torn line
    w.close()
    out = read_metrics(run_dir, [(1, 0), (2, 2)], max_points=1000)
    assert [r["tick"] for r in out["rows"]] == list(range(11))
    assert out["rows"][10]["values"] == {"x": 10.0, "y": 1.5}
    assert out["rows"][5]["values"]["y"] is None  # null policy preserved
    assert out["metric_names"] == ["x", "y"]
    ranged = read_metrics(run_dir, [(1, 0), (2, 2)], since_tick=3, until_tick=6)
    assert [r["tick"] for r in ranged["rows"]] == [4, 5, 6]
    sampled = read_metrics(run_dir, [(1, 0), (2, 2)], max_points=3)
    assert sampled["rows"][-1]["tick"] == 10 and sampled["stride"] == 4


def test_writer_resumes_sequence_after_reopen(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(ms, "CHUNK_ROWS", 2)
    store = LocalArtifactStore(tmp_path)
    w = MetricsWriter(store, RUN, 1)
    w.append([{"tick": t, "epoch": 1, "values": {"x": 1.0}} for t in range(3)])
    w.compact()
    w.close()
    w2 = MetricsWriter(store, RUN, 1)  # e.g. a re-launched writer for the same epoch
    w2.append([{"tick": 3, "epoch": 1, "values": {"x": 2.0}}])
    assert w2.buffer[-1]["seq"] == 3
    w2.close()
    out = read_metrics(store.run_dir(RUN), [(1, 0)])
    assert [r["tick"] for r in out["rows"]] == [0, 1, 2, 3]
