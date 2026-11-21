"""SQLite metadata database (WAL) with forward-only versioned migrations.

Only the controller process opens this database for writing. Workers never
receive its path. See ADR 0003 for why stdlib sqlite3 is used instead of an ORM
in Phase 1.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            substrate TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at REAL NOT NULL,
            spec_hash TEXT NOT NULL,
            spec_json TEXT NOT NULL
        );
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL REFERENCES experiments(id),
            parent_run_id TEXT,
            parent_checkpoint_id TEXT,
            state TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 0,
            seed INTEGER NOT NULL,
            backend TEXT,
            target_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            profile_hash TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            effective_mode TEXT NOT NULL,
            intervened INTEGER NOT NULL DEFAULT 0,
            tick INTEGER NOT NULL DEFAULT 0,
            epoch INTEGER NOT NULL DEFAULT 0,
            ticks_per_second REAL NOT NULL,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            finish_reason TEXT,
            last_error TEXT,
            manifest_path TEXT NOT NULL
        );
        CREATE INDEX runs_state ON runs(state);
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            epoch INTEGER NOT NULL,
            lease_token TEXT NOT NULL,
            status TEXT NOT NULL,
            executor TEXT NOT NULL,
            worker_pid INTEGER,
            start_tick INTEGER NOT NULL,
            restored_from_checkpoint_id TEXT,
            created_at REAL NOT NULL,
            started_at REAL,
            heartbeat_at REAL,
            finished_at REAL,
            exit_code INTEGER,
            error_json TEXT,
            UNIQUE(run_id, epoch)
        );
        CREATE TABLE checkpoints (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            job_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            simulation_clock INTEGER NOT NULL,
            clock_unit TEXT NOT NULL,
            logical_path TEXT NOT NULL,
            storage_uri TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(run_id, logical_path)
        );
        CREATE INDEX checkpoints_run ON checkpoints(run_id, simulation_clock);
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id),
            seq INTEGER NOT NULL,
            clock_unit TEXT NOT NULL,
            clock_value INTEGER,
            observed_at_utc REAL NOT NULL,
            source TEXT NOT NULL,
            type TEXT NOT NULL,
            severity TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE(run_id, seq)
        );
        CREATE TABLE commands (
            command_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_clock INTEGER,
            created_at REAL NOT NULL,
            ack_json TEXT
        );
        CREATE TABLE interventions (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            command_id TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            author TEXT NOT NULL,
            spec_hash TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            requested_clock INTEGER,
            applied_clock INTEGER,
            epoch INTEGER NOT NULL,
            requested_at REAL NOT NULL,
            applied_at REAL,
            status TEXT NOT NULL,
            result_json TEXT
        );
        CREATE TABLE idempotency_keys (
            key TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            kind TEXT NOT NULL,
            logical_path TEXT NOT NULL,
            storage_uri TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(run_id, logical_path)
        );
        """,
    ),
    (
        2,
        """
        ALTER TABLE checkpoints ADD COLUMN spec_hash TEXT;
        ALTER TABLE runs ADD COLUMN label TEXT;
        CREATE TABLE comparisons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            design TEXT NOT NULL,
            parent_run_id TEXT,
            parent_checkpoint_id TEXT,
            created_at REAL NOT NULL,
            notes_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE comparison_members (
            comparison_id TEXT NOT NULL REFERENCES comparisons(id),
            run_id TEXT NOT NULL REFERENCES runs(id),
            arm TEXT NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (comparison_id, run_id)
        );
        CREATE INDEX runs_parent ON runs(parent_run_id);
        """,
    ),
    (
        3,
        """
        CREATE TABLE sweeps (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            design TEXT NOT NULL,
            spec_hash TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at REAL NOT NULL,
            finished_at REAL
        );
        CREATE TABLE sweep_members (
            sweep_id TEXT NOT NULL REFERENCES sweeps(id),
            run_id TEXT NOT NULL REFERENCES runs(id),
            condition_index INTEGER NOT NULL,
            condition_label TEXT NOT NULL,
            seed INTEGER NOT NULL,
            PRIMARY KEY (sweep_id, run_id)
        );
        CREATE INDEX sweep_members_run ON sweep_members(run_id);
        """,
    ),
    (
        4,
        """
        CREATE TABLE discoveries (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            claim_kind TEXT NOT NULL,
            substrate TEXT NOT NULL,
            source_run_id TEXT NOT NULL REFERENCES runs(id),
            source_tick INTEGER NOT NULL,
            source_ref_json TEXT NOT NULL,
            candidate_spec_json TEXT NOT NULL,
            candidate_spec_hash TEXT NOT NULL,
            recorded_json TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence_level TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE discovery_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovery_id TEXT NOT NULL REFERENCES discoveries(id),
            at REAL NOT NULL,
            kind TEXT NOT NULL,
            author TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX discovery_events_d ON discovery_events(discovery_id);
        CREATE TABLE assays (
            id TEXT PRIMARY KEY,
            discovery_id TEXT NOT NULL REFERENCES discoveries(id),
            plan_json TEXT NOT NULL,
            plan_hash TEXT NOT NULL,
            state TEXT NOT NULL,
            result_json TEXT,
            artifact_path TEXT,
            created_at REAL NOT NULL,
            finished_at REAL
        );
        CREATE INDEX assays_d ON assays(discovery_id);
        """,
    ),
    (
        5,
        """
        CREATE TABLE gallery (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id),
            tick INTEGER NOT NULL,
            substrate TEXT NOT NULL,
            title TEXT NOT NULL,
            note TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            created_at REAL NOT NULL,
            hidden_at REAL
        );
        CREATE INDEX gallery_run ON gallery(run_id);
        """,
    ),
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self.migrate()

    def migrate(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
            )
            applied = {
                int(r[0]) for r in self._conn.execute("SELECT version FROM schema_migrations")
            }
            pending = [v for v, _ in MIGRATIONS if v not in applied]
            if pending and applied:
                # Spec 10.5: keep a consistent pre-migration copy of existing metadata.
                backup = self.path.with_name(
                    f"{self.path.name}.pre-v{max(pending)}-{int(time.time())}.bak"
                )
                self.backup_to(backup)
            for version, sql in MIGRATIONS:
                if version in applied:
                    continue
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in sql.split(";"):
                        if statement.strip():
                            self._conn.execute(statement)
                    self._conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, time.time()),
                    )
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise

    @property
    def schema_version(self) -> int:
        row = self.one("SELECT MAX(version) AS v FROM schema_migrations")
        return int(row["v"]) if row and row["v"] is not None else 0

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def backup_to(self, target: Path) -> None:
        """Consistent online backup using the SQLite backup API."""
        with self._lock:
            dest = sqlite3.connect(target)
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
