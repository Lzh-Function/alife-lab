# ADR 0003 — stdlib sqlite3 with versioned migrations (deviation from SQLAlchemy/Alembic)

- Status: accepted for Phase 1; revisit in Phase 2
- Date: 2026-11-18

## Context
Spec §2.1 recommends SQLAlchemy/SQLModel and Alembic. Phase 1 has ~10 tables, one
writer and no ORM-heavy queries.

## Decision
Use Python's `sqlite3` in WAL mode with an ordered list of forward-only migrations
recorded in `schema_migrations` (`alife_lab/storage/db.py`). Access is isolated in
the controller so a repository layer / PostgreSQL migration remains possible.
Online backups must use the SQLite backup API (`Database.backup_to`).

## Consequences
- Fewer dependencies; explicit SQL.
- Pre-migration backup implemented in migration 2 (`*.pre-v<N>-<ts>.bak`).
