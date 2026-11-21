"""Process settings resolved from environment variables (never from Git-tracked secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from alife_lab.provenance import PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    project_root: Path
    runs_root: Path
    home: Path
    configs_dir: Path
    frontend_dist: Path
    max_active_jobs: int
    default_ticks_per_second: float
    datasets_dir: Path | None = None
    max_datasets_mb: int = 4096

    @property
    def db_path(self) -> Path:
        return self.home / "alife.sqlite3"

    @property
    def lock_path(self) -> Path:
        return self.home / "controller.lock"

    @property
    def token_path(self) -> Path:
        return self.home / "session_token"

    @property
    def resolved_datasets_dir(self) -> Path:
        return self.datasets_dir if self.datasets_dir is not None else self.home / "datasets"


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_settings() -> Settings:
    home = _path_env("ALIFE_HOME", PROJECT_ROOT / ".alife")
    return Settings(
        project_root=PROJECT_ROOT,
        runs_root=_path_env("ALIFE_RUNS_DIR", PROJECT_ROOT / "runs"),
        home=home,
        configs_dir=_path_env("ALIFE_CONFIGS_DIR", PROJECT_ROOT / "configs"),
        frontend_dist=_path_env("ALIFE_FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist"),
        max_active_jobs=int(os.environ.get("ALIFE_MAX_ACTIVE_JOBS", "4")),
        default_ticks_per_second=float(os.environ.get("ALIFE_DEFAULT_TPS", "10")),
        datasets_dir=_path_env("ALIFE_DATASETS_DIR", home / "datasets"),
        max_datasets_mb=int(os.environ.get("ALIFE_MAX_DATASETS_MB", "4096")),
    )
