"""Presets (scientific scenarios + UI metadata) and execution profiles from configs/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from alife_lab.core.spec import ExecutionProfile, ExperimentSpec


@dataclass(frozen=True)
class Preset:
    id: str
    title_key: str
    description_key: str
    spec: ExperimentSpec
    expected_observation: str
    negative_control: str | None
    source_file: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title_key": self.title_key,
            "description_key": self.description_key,
            "substrate": self.spec.experiment.substrate,
            "spec": self.spec.model_dump(mode="json"),
            "spec_hash": self.spec.digest(),
            "expected_observation": self.expected_observation,
            "negative_control": self.negative_control,
            "source_file": self.source_file,
        }


def _safe_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_presets(configs_dir: Path) -> dict[str, Preset]:
    presets: dict[str, Preset] = {}
    for path in sorted((configs_dir / "presets").glob("*.yaml")):
        raw = _safe_load(path)
        meta = raw["preset"]
        scenario_path = (configs_dir / "scenarios" / meta["scenario"]).resolve()
        if (configs_dir / "scenarios").resolve() not in scenario_path.parents:
            raise ValueError(f"preset {path.name} references a scenario outside configs/scenarios")
        scenario = _safe_load(scenario_path)
        overrides = meta.get("overrides") or {}
        _deep_merge(scenario, overrides)
        spec = ExperimentSpec.model_validate(scenario)
        presets[meta["id"]] = Preset(
            id=meta["id"],
            title_key=meta["title_key"],
            description_key=meta["description_key"],
            spec=spec,
            expected_observation=meta["expected_observation"],
            negative_control=meta.get("negative_control"),
            source_file=path.relative_to(configs_dir.parent).as_posix(),
        )
    return presets


def load_profiles(configs_dir: Path) -> dict[str, ExecutionProfile]:
    profiles: dict[str, ExecutionProfile] = {}
    for path in sorted((configs_dir / "execution_profiles").glob("*.yaml")):
        profile = ExecutionProfile.model_validate(_safe_load(path))
        profiles[profile.execution.id] = profile
    return profiles


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
