"""ExperimentSpec / ExecutionProfile / state machine / brush contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.core.spec import ExperimentSpec, load_execution_profile, load_experiment_spec
from alife_lab.core.states import ALLOWED_TRANSITIONS, InvalidTransition, RunState, check_transition
from alife_lab.service.catalog import load_presets, load_profiles
from alife_lab.substrates.ca.conway import ConwayAdapter, brush_cells

ROOT = Path(__file__).resolve().parents[2]


def base() -> dict:
    return load_experiment_spec(ROOT / "configs/scenarios/conway_reference.yaml").model_dump(
        mode="json"
    )


def test_repository_configs_validate() -> None:
    spec = load_experiment_spec(ROOT / "configs/scenarios/conway_reference.yaml")
    assert spec.experiment.world.width == 128
    for path in (ROOT / "configs/execution_profiles").glob("*.yaml"):
        load_execution_profile(path)
    assert {"conway_glider", "conway_blinker", "conway_block", "conway_random_soup"} <= set(
        load_presets(ROOT / "configs")
    )
    assert {"cpu_dev", "local_interactive", "remote_compute", "cluster_batch"} <= set(
        load_profiles(ROOT / "configs")
    )


def test_spec_hash_is_stable_and_sensitive() -> None:
    a = ExperimentSpec.model_validate(base())
    b = ExperimentSpec.model_validate(base())
    assert a.digest() == b.digest()
    changed = base()
    changed["experiment"]["world"]["width"] = 64
    assert ExperimentSpec.model_validate(changed).digest() != a.digest()


def test_spec_and_profile_hashes_are_independent() -> None:
    spec = ExperimentSpec.model_validate(base())
    p1 = load_execution_profile(ROOT / "configs/execution_profiles/cpu_dev.yaml")
    p2 = load_execution_profile(ROOT / "configs/execution_profiles/local_interactive.yaml")
    assert p1.digest() != p2.digest()
    assert spec.digest() == ExperimentSpec.model_validate(base()).digest()


@pytest.mark.parametrize(
    "path,value",
    [
        (("world", "width"), 4),
        (("world", "width"), 100_000),
        (("world", "boundary"), "mirror"),
        (("seed",), -1),
        (("runner", "max_steps"), 0),
        (("dynamics", "rule"), "B3/S2x"),
        (("dynamics", "rule"), "B33/S23"),
        (("substrate",), "lenia"),
        (("name",), "../../etc/passwd"),
    ],
)
def test_invalid_specs_rejected(path: tuple[str, ...], value: object) -> None:
    raw = base()
    target = raw["experiment"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(raw)


def test_unknown_fields_rejected() -> None:
    raw = base()
    raw["experiment"]["reward"] = 1
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(raw)


def test_reference_mode_requires_conway_rule() -> None:
    raw = base()
    raw["experiment"]["dynamics"]["rule"] = "B36/S23"
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(raw)
    raw["experiment"]["mode"] = "exploratory"
    assert ExperimentSpec.model_validate(raw).experiment.dynamics.rule == "B36/S23"


def test_state_machine() -> None:
    check_transition(RunState.PENDING, RunState.QUEUED)
    check_transition(RunState.RUNNING, RunState.PAUSING)
    check_transition(RunState.INTERRUPTED, RunState.QUEUED)
    with pytest.raises(InvalidTransition):
        check_transition(RunState.COMPLETED, RunState.RUNNING)
    with pytest.raises(InvalidTransition):
        check_transition(RunState.PENDING, RunState.RUNNING)
    assert set(ALLOWED_TRANSITIONS) == set(RunState)


def test_brush_cells_bounds_and_shapes() -> None:
    assert brush_cells({"shape": "cells", "cells": [[1, 1], [1, 1], [2, 3]]}, 8, 8) == [
        (1, 1),
        (2, 3),
    ]
    assert len(brush_cells({"shape": "circle", "x": 4, "y": 4, "radius": 1}, 8, 8)) == 5
    assert len(brush_cells({"shape": "circle", "x": 0, "y": 0, "radius": 1}, 8, 8)) == 3
    for bad in (
        {"shape": "cells", "cells": [[8, 0]]},
        {"shape": "cells", "cells": [[-1, 0]]},
        {"shape": "circle", "x": 1, "y": 1, "radius": 1000},
        {"shape": "polygon"},
        {"shape": "cells", "cells": "1,2"},
    ):
        with pytest.raises(ValueError):
            brush_cells(bad, 8, 8)


def test_brush_intervention_and_serialization_roundtrip() -> None:
    adapter = ConwayAdapter()
    spec = ExperimentSpec.model_validate(base())
    state = adapter.initialize(spec)
    state, result = adapter.apply_intervention(
        state, spec, "brush", {"shape": "cells", "cells": [[0, 0], [1, 0]], "value": 1}
    )
    assert result["alive_after"] == result["alive_before"] + 2
    payload = adapter.serialize_state(state)
    restored = adapter.deserialize_state(payload, spec)
    np.testing.assert_array_equal(restored.grid, state.grid)
    assert restored.tick == state.tick
    with pytest.raises(ValueError):
        adapter.apply_intervention(state, spec, "poison", {})


def test_conway_spec_hashes_are_stable_across_schema_changes() -> None:
    """Spec hashes of existing runs must not change when new substrates are added."""
    presets = load_presets(ROOT / "configs")
    expected = {
        "conway_glider": "6b5dc2a4b18c",
        "conway_blinker": "67ae4015cc7e",
        "conway_block": "8b969e1ac05b",
        "conway_random_soup": "89daf47c92ef",
    }
    assert {k: presets[k].spec.digest()[:12] for k in expected} == expected
