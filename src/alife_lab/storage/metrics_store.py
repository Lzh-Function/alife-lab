"""Metric history storage: append-only JSONL tail + immutable Parquet chunks.

Writer (worker process):
* every row gets ``(epoch, seq)`` where ``seq`` increases monotonically within an epoch;
* rows are appended (and flushed) to ``metrics/tail_eEEE.jsonl`` — the crash-safe journal;
* every ``CHUNK_ROWS`` rows, and on checkpoint / exit, the buffered rows are written to an
  immutable Parquet chunk ``metrics/chunks/eEEE_sSSSSSSSSSSSS.parquet`` through the
  ArtifactStore (stage → fsync → sha256 → atomic rename), then the tail is truncated.
  A crash between publish and truncation duplicates rows; readers drop duplicate
  ``(epoch, seq)``.

Reader (controller): reads chunks + tails + the legacy ``metrics/history.jsonl`` (Phase 1
format) with the same semantics as before: rows of epoch *e* beyond the start tick of any
later epoch are superseded; for each tick the last row in (epoch, seq) order wins
(post-intervention rows follow the regular row of that tick).
"""

from __future__ import annotations

import io
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.artifacts.store import ArtifactReceipt, LocalArtifactStore

CHUNK_ROWS = 10_000


def _arrow() -> tuple[Any, Any]:
    """Import pyarrow lazily: it adds ~60 MB RSS and is only needed for chunk I/O."""
    import pyarrow as pa
    import pyarrow.compute  # registers pa.compute
    import pyarrow.parquet as pq

    return pa, pq


_LEGACY_SEQ_BASE = 0


def _value(v: Any) -> float:
    if v is None:
        return math.nan
    return float(v)


class MetricsWriter:
    def __init__(self, store: LocalArtifactStore, run_id: str, epoch: int) -> None:
        self.store = store
        self.run_id = run_id
        self.epoch = epoch
        run_dir = store.run_dir(run_id)
        (run_dir / "metrics" / "chunks").mkdir(parents=True, exist_ok=True)
        self.tail_path = run_dir / "metrics" / f"tail_e{epoch:03d}.jsonl"
        self.seq = self._resume_seq()
        self.buffer: list[dict[str, Any]] = self._read_tail()
        self.tail = self.tail_path.open("a", encoding="utf-8")

    def _resume_seq(self) -> int:
        """Continue the sequence if this epoch already wrote chunks (defensive)."""
        last = -1
        for f in (self.tail_path.parent / "chunks").glob(f"e{self.epoch:03d}_s*.parquet"):
            pa, pq = _arrow()
            table = pq.read_table(f, columns=["seq"])
            if table.num_rows:
                last = max(last, int(pa.compute.max(table["seq"]).as_py()))
        return last + 1

    def _read_tail(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.tail_path.is_file():
            for line in self.tail_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line
                rows.append(row)
                self.seq = max(self.seq, int(row.get("seq", -1)) + 1)
        return rows

    def append(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["seq"] = self.seq
            self.seq += 1
            self.tail.write(json.dumps(row, sort_keys=True) + "\n")
        self.tail.flush()
        self.buffer.extend(rows)

    def sync(self) -> None:
        self.tail.flush()
        os.fsync(self.tail.fileno())

    def compact(self, force: bool = False) -> ArtifactReceipt | None:
        if not self.buffer or (not force and len(self.buffer) < CHUNK_ROWS):
            return None
        pa, pq = _arrow()
        rows, first = self.buffer, int(self.buffer[0]["seq"])
        names = sorted({k for r in rows for k in r["values"]})
        columns: dict[str, Any] = {
            "tick": pa.array([int(r["tick"]) for r in rows], pa.int64()),
            "epoch": pa.array([int(r["epoch"]) for r in rows], pa.int32()),
            "seq": pa.array([int(r["seq"]) for r in rows], pa.int64()),
            "post_intervention": pa.array([bool(r.get("post_intervention")) for r in rows]),
        }
        for name in names:
            columns[f"m.{name}"] = pa.array(
                [r["values"].get(name) for r in rows], pa.float64(), from_pandas=False
            )
        sink = io.BytesIO()
        pq.write_table(pa.table(columns), sink, compression="zstd")
        receipt = self.store.publish_bytes(
            self.run_id,
            f"metrics/chunks/e{self.epoch:03d}_s{first:012d}.parquet",
            sink.getvalue(),
            kind="metrics_chunk",
            mime_type="application/vnd.apache.parquet",
        )
        # The chunk is durable: start a fresh journal.
        self.tail.close()
        self.tail = self.tail_path.open("w", encoding="utf-8")
        self.tail.flush()
        os.fsync(self.tail.fileno())
        self.buffer = []
        return receipt

    def close(self) -> None:
        try:
            self.tail.flush()
            os.fsync(self.tail.fileno())
        finally:
            self.tail.close()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


class _Columns:
    def __init__(self) -> None:
        self.tick: list[np.ndarray[Any, Any]] = []
        self.epoch: list[np.ndarray[Any, Any]] = []
        self.seq: list[np.ndarray[Any, Any]] = []
        self.post: list[np.ndarray[Any, Any]] = []
        self.metrics: dict[str, list[np.ndarray[Any, Any]]] = {}
        self.lengths: list[int] = []

    def add(
        self,
        tick: np.ndarray[Any, Any],
        epoch: np.ndarray[Any, Any],
        seq: np.ndarray[Any, Any],
        post: np.ndarray[Any, Any],
        metrics: dict[str, np.ndarray[Any, Any]],
    ) -> None:
        n = len(tick)
        if n == 0:
            return
        offset = sum(self.lengths)
        for name, values in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = [np.full(offset, np.nan)] if offset else []
            self.metrics[name].append(values.astype(np.float64))
        for name, parts in self.metrics.items():
            if name not in metrics:
                parts.append(np.full(n, np.nan))
        self.tick.append(tick.astype(np.int64))
        self.epoch.append(epoch.astype(np.int64))
        self.seq.append(seq.astype(np.int64))
        self.post.append(post.astype(bool))
        self.lengths.append(n)


def _rows_to_arrays(rows: list[dict[str, Any]], cols: _Columns, seq_base: int = 0) -> None:
    if not rows:
        return
    names = sorted({k for r in rows for k in r["values"]})
    cols.add(
        np.array([r["tick"] for r in rows], dtype=np.int64),
        np.array([r["epoch"] for r in rows], dtype=np.int64),
        np.array([r.get("seq", seq_base + i) for i, r in enumerate(rows)], dtype=np.int64),
        np.array([bool(r.get("post_intervention")) for r in rows]),
        {n: np.array([_value(r["values"].get(n)) for r in rows]) for n in names},
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line after a hard kill
    return rows


def read_metrics(
    run_dir: Path,
    epoch_start_ticks: Sequence[tuple[int, int]],
    since_tick: int = -1,
    until_tick: int | None = None,
    max_points: int = 2000,
) -> dict[str, Any]:
    cols = _Columns()
    metrics_dir = run_dir / "metrics"
    legacy = metrics_dir / "history.jsonl"
    if legacy.is_file():
        _rows_to_arrays(_read_jsonl(legacy), cols, _LEGACY_SEQ_BASE)
    for chunk in sorted((metrics_dir / "chunks").glob("*.parquet")):
        _, pq = _arrow()
        table = pq.read_table(chunk)
        names = [c for c in table.column_names if c.startswith("m.")]
        cols.add(
            table["tick"].to_numpy(),
            table["epoch"].to_numpy(),
            table["seq"].to_numpy(),
            table["post_intervention"].to_numpy(zero_copy_only=False),
            {n[2:]: table[n].to_numpy(zero_copy_only=False).astype(np.float64) for n in names},
        )
    for tail in sorted(metrics_dir.glob("tail_e*.jsonl")):
        _rows_to_arrays(_read_jsonl(tail), cols)
    empty = {"total_points": 0, "stride": 1, "superseded_rows": 0, "rows": [], "metric_names": []}
    if not cols.tick:
        return empty
    tick = np.concatenate(cols.tick)
    epoch = np.concatenate(cols.epoch)
    seq = np.concatenate(cols.seq)
    post = np.concatenate(cols.post)
    metrics = {k: np.concatenate(v) for k, v in cols.metrics.items()}
    # Drop duplicate (epoch, seq) produced by a crash between chunk publish and tail truncation.
    key = np.stack([epoch, seq], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.zeros(tick.size, dtype=bool)
    keep[first] = True
    # Superseded rows: tick beyond the start tick of any later epoch.
    limit = np.full(tick.size, np.iinfo(np.int64).max)
    for e, start in epoch_start_ticks:
        later = epoch < e
        limit = np.where(later, np.minimum(limit, start), limit)
    valid = keep & (tick <= limit)
    superseded = int((keep & ~valid).sum())
    window = valid & (tick > since_tick)
    if until_tick is not None:
        window &= tick <= until_tick
    idx = np.flatnonzero(window)
    if idx.size == 0:
        return {**empty, "superseded_rows": superseded, "metric_names": sorted(metrics)}
    order = idx[np.lexsort((seq[idx], epoch[idx]))]
    # Last row per tick in (epoch, seq) order.
    rev = order[::-1]
    _, last_pos = np.unique(tick[rev], return_index=True)
    chosen = rev[last_pos]  # sorted by tick ascending
    total = chosen.size
    stride = max(1, -(-total // max(1, max_points)))
    sampled = chosen[::stride]
    if sampled[-1] != chosen[-1]:
        sampled = np.append(sampled, chosen[-1])
    names = sorted(metrics)
    rows = []
    for i in sampled.tolist():
        values = {}
        for n in names:
            v = metrics[n][i]
            values[n] = None if math.isnan(v) else float(v)
        row: dict[str, Any] = {"tick": int(tick[i]), "epoch": int(epoch[i]), "values": values}
        if post[i]:
            row["post_intervention"] = True
        rows.append(row)
    return {
        "total_points": int(total),
        "stride": stride,
        "superseded_rows": superseded,
        "rows": rows,
        "metric_names": names,
    }
