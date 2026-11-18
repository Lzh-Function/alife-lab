"""Shared fixtures for integration tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from alife_lab.service.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        load_settings(),
        runs_root=tmp_path / "runs",
        home=tmp_path / "home",
        configs_dir=ROOT / "configs",
    )
