"""Run provenance: source identity, lockfile digests and environment."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from alife_lab import __version__
from alife_lab.core.hashing import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_GLOBS = (
    "src/alife_lab/**/*.py",
    "pyproject.toml",
    "uv.lock",
    "frontend/src/**/*",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/index.html",
    "frontend/vite.config.ts",
)


def _git(*args: str) -> str | None:
    exe = shutil.which("git")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_info() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_available": shutil.which("git") is not None,
    }


@lru_cache(maxsize=1)
def source_digest() -> str:
    """SHA-256 over sorted (relative path, content digest) pairs of source files."""
    digest = hashlib.sha256()
    files: set[Path] = set()
    for pattern in _SOURCE_GLOBS:
        files.update(p for p in PROJECT_ROOT.glob(pattern) if p.is_file())
    for path in sorted(files):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if "__pycache__" in rel:
            continue
        digest.update(rel.encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _file_digest(rel: str) -> str | None:
    path = PROJECT_ROOT / rel
    return sha256_file(path) if path.is_file() else None


def environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("alife-lab", "fastapi", "pydantic", "numpy", "uvicorn", "starlette", "pyyaml"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "alife_lab_version": __version__,
        **git_info(),
        "source_digest": source_digest(),
        "uv_lock_sha256": _file_digest("uv.lock"),
        "frontend_lock_sha256": _file_digest("frontend/package-lock.json"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
