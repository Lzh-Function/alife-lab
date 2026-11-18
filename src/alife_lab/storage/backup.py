"""Independent backup, verification and restore of the metadata DB and run artifacts.

Policy (GIT_AND_ARTIFACT_POLICY rules 6–7, spec §10.11):
* The SQLite DB is copied with the SQLite online backup API (consistent while the
  controller is running); the live DB file is never copied directly.
* Run files are copied while computing SHA-256; every artifact/checkpoint row of the copied
  DB must resolve to a copied file with the recorded checksum, otherwise the backup fails.
* The backup is assembled in ``<id>.partial`` and atomically renamed when complete.
* Secrets and runtime files (session token, lock, WAL/SHM, test outputs) are never included.
* Restore verifies first and only writes into empty/non-existent target directories.

Scope: full backups (all runs). A partial backup would leave dangling DB references.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from alife_lab import __version__
from alife_lab.core.hashing import sha256_file
from alife_lab.provenance import git_info

MANIFEST = "backup_manifest.json"
DB_NAME = "alife.sqlite3"
SKIP_SUFFIXES = (".staging",)


class BackupError(RuntimeError):
    pass


def _copy_with_hash(src: Path, dst: Path) -> tuple[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        while chunk := fin.read(1 << 20):
            digest.update(chunk)
            fout.write(chunk)
            size += len(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    return digest.hexdigest(), size


def _inside(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def _db_references(db_path: Path) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT run_id, logical_path, sha256 FROM artifacts UNION ALL "
            "SELECT run_id, logical_path, sha256 FROM checkpoints"
        ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]
    finally:
        conn.close()


def _db_summary(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        runs = conn.execute("SELECT id, state FROM runs ORDER BY created_at").fetchall()
        return {
            "integrity_check": check,
            "schema_version": version,
            "runs": [{"id": r[0], "state": r[1]} for r in runs],
        }
    finally:
        conn.close()


def create_backup(db_path: Path, runs_root: Path, dest_root: Path) -> dict[str, Any]:
    if not db_path.is_file():
        raise BackupError(f"database {db_path} does not exist")
    if _inside(dest_root, runs_root) or _inside(dest_root, db_path.parent):
        raise BackupError("backup destination must be outside the runs directory and ALIFE_HOME")
    dest_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_id = f"bk_{stamp}_{secrets.token_hex(3)}"
    work = dest_root / f"{backup_id}.partial"
    final = dest_root / backup_id
    work.mkdir()
    try:
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        dst = sqlite3.connect(work / DB_NAME)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        db_sha, db_size = sha256_file(work / DB_NAME), (work / DB_NAME).stat().st_size
        summary = _db_summary(work / DB_NAME)
        if summary["integrity_check"] != "ok":
            raise BackupError(f"database integrity check failed: {summary['integrity_check']}")
        files: list[dict[str, Any]] = [{"path": DB_NAME, "sha256": db_sha, "size": db_size}]
        live_states = {"queued", "starting", "running", "pausing", "paused", "resuming"}
        live = [r["id"] for r in summary["runs"] if r["state"] in live_states]
        for run in summary["runs"]:
            run_dir = runs_root / run["id"]
            if not run_dir.is_dir():
                raise BackupError(f"run directory missing for {run['id']}")
            for f in sorted(p for p in run_dir.rglob("*") if p.is_file()):
                if f.name.endswith(SKIP_SUFFIXES):
                    continue
                rel = Path("runs") / run["id"] / f.relative_to(run_dir)
                sha, size = _copy_with_hash(f, work / rel)
                files.append({"path": rel.as_posix(), "sha256": sha, "size": size})
        by_path = {f["path"]: f["sha256"] for f in files}
        problems = []
        for run_id, logical, sha in _db_references(work / DB_NAME):
            got = by_path.get(f"runs/{run_id}/{logical}")
            if got != sha:
                problems.append(f"{run_id}/{logical}: {'missing' if got is None else 'checksum'}")
        if problems:
            raise BackupError(f"{len(problems)} referenced artifacts failed: {problems[:5]}")
        git = git_info()
        manifest = {
            "schema_version": 1,
            "backup_id": backup_id,
            "created_at_unix": time.time(),
            "alife_lab_version": __version__,
            "source_git_commit": git["git_commit"],
            "db_schema_version": summary["schema_version"],
            "runs": summary["runs"],
            "live_runs_at_backup": live,
            "note": (
                "Runs listed in live_runs_at_backup were active: their append-only logs and "
                "metrics are captured up to the copy time; checkpoints are immutable."
            ),
            "files": files,
            "total_bytes": sum(f["size"] for f in files),
        }
        with (work / MANIFEST).open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(work, final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {
        "backup": str(final),
        "backup_id": backup_id,
        "runs": len(summary["runs"]),
        "files": len(files),
        "total_bytes": manifest["total_bytes"],
        "live_runs": live,
    }


def verify_backup(backup: Path) -> dict[str, Any]:
    manifest_path = backup / MANIFEST
    if not manifest_path.is_file():
        return {"ok": False, "problems": ["backup_manifest.json missing (incomplete backup?)"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    listed = set()
    for f in manifest["files"]:
        path = (backup / f["path"]).resolve()
        listed.add(path)
        if not _inside(path, backup):
            problems.append(f"path escapes backup: {f['path']}")
        elif not path.is_file():
            problems.append(f"missing: {f['path']}")
        elif sha256_file(path) != f["sha256"]:
            problems.append(f"checksum mismatch: {f['path']}")
    extra = [
        p.relative_to(backup).as_posix()
        for p in backup.rglob("*")
        if p.is_file() and p.resolve() not in listed and p.name != MANIFEST
    ]
    db = backup / DB_NAME
    integrity = "missing"
    if db.is_file() and not any(p.endswith(DB_NAME) for p in problems):
        integrity = _db_summary(db)["integrity_check"]
        if integrity != "ok":
            problems.append(f"database integrity: {integrity}")
        by_path = {f["path"]: f["sha256"] for f in manifest["files"]}
        for run_id, logical, sha in _db_references(db):
            if by_path.get(f"runs/{run_id}/{logical}") != sha:
                problems.append(f"DB reference not satisfied: {run_id}/{logical}")
    return {
        "ok": not problems,
        "backup_id": manifest.get("backup_id"),
        "files_checked": len(manifest["files"]),
        "db_integrity": integrity,
        "unlisted_files": extra,
        "problems": problems,
    }


def _empty_or_absent(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def restore_backup(backup: Path, home: Path, runs_root: Path) -> dict[str, Any]:
    report = verify_backup(backup)
    if not report["ok"]:
        raise BackupError(f"backup failed verification: {report['problems'][:5]}")
    for target in (home, runs_root):
        if not _empty_or_absent(target):
            raise BackupError(f"restore target {target} is not empty; refusing to overwrite")
    manifest = json.loads((backup / MANIFEST).read_text(encoding="utf-8"))
    home.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    restored = 0
    for f in manifest["files"]:
        src = backup / f["path"]
        if f["path"] == DB_NAME:
            dst = home / DB_NAME
        else:
            dst = runs_root / Path(f["path"]).relative_to("runs")
        sha, _ = _copy_with_hash(src, dst)
        if sha != f["sha256"]:
            raise BackupError(f"checksum changed while restoring {f['path']}")
        restored += 1
    return {
        "restored_files": restored,
        "runs": len(manifest["runs"]),
        "backup_id": manifest["backup_id"],
        "home": str(home),
        "runs_root": str(runs_root),
    }
