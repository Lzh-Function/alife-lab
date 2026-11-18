"""Reference tests for Conway's Game of Life (known periods, translation, still lifes)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.ca.conway import (
    PATTERNS,
    ConwayAdapter,
    parse_rule,
    step_grid,
    step_grid_oracle,
)


def make_spec(
    preset: str = "glider", boundary: str = "periodic", size: int = 32, **kw: object
) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "t",
                "substrate": "conway",
                "seed": 1,
                "world": {"width": size, "height": size, "boundary": boundary},
                "initialization": {"preset": preset, **kw},
                "observations": {"metrics": ["alive_cell_count", "centroid_displacement"]},
            }
        }
    )


def run(adapter: ConwayAdapter, spec: ExperimentSpec, ticks: int):  # type: ignore[no-untyped-def]
    state = adapter.initialize(spec)
    for _ in range(ticks):
        state = adapter.step(state, spec)
    return state


def test_parse_rule() -> None:
    assert parse_rule("B3/S23") == (frozenset({3}), frozenset({2, 3}))
    assert parse_rule("B36/S23") == (frozenset({3, 6}), frozenset({2, 3}))


def test_glider_translates_one_cell_diagonally_every_four_ticks() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("glider", offset_x=2, offset_y=2)
    s0 = adapter.initialize(spec).grid.copy()
    for k in range(1, 9):
        s = run(adapter, spec, 4 * k)
        expected = np.roll(np.roll(s0, k, axis=0), k, axis=1)
        np.testing.assert_array_equal(s.grid, expected)
        assert int(s.grid.sum()) == 5


def test_glider_is_not_period_one_or_two() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("glider", offset_x=2, offset_y=2)
    s0 = adapter.initialize(spec).grid.copy()
    for t in (1, 2, 3):
        grid = run(adapter, spec, t).grid
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                assert not np.array_equal(grid, np.roll(np.roll(s0, dy, 0), dx, 1))


def test_glider_wraps_around_torus_and_returns() -> None:
    adapter = ConwayAdapter()
    size = 16
    spec = make_spec("glider", size=size, offset_x=1, offset_y=1)
    s0 = adapter.initialize(spec).grid.copy()
    s = run(adapter, spec, 4 * size)
    np.testing.assert_array_equal(s.grid, s0)


def test_glider_centroid_displacement_is_sqrt2_per_period() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("glider", size=24, offset_x=2, offset_y=2)
    state = adapter.initialize(spec)
    for k in range(1, 31):
        for _ in range(4):
            state = adapter.step(state, spec)
        value = adapter.metrics(state, spec)["centroid_displacement"]
        assert value == pytest.approx(k * math.sqrt(2), abs=1e-9)


def test_blinker_has_period_two() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("blinker")
    s0 = adapter.initialize(spec).grid.copy()
    s1 = run(adapter, spec, 1).grid
    s2 = run(adapter, spec, 2).grid
    assert not np.array_equal(s0, s1)
    np.testing.assert_array_equal(s0, s2)


@pytest.mark.parametrize("boundary", ["periodic", "dead"])
def test_block_is_still_life(boundary: str) -> None:
    adapter = ConwayAdapter()
    spec = make_spec("block", boundary=boundary)
    s0 = adapter.initialize(spec).grid.copy()
    np.testing.assert_array_equal(run(adapter, spec, 50).grid, s0)


def test_negative_control_block_has_zero_displacement() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("block")
    state = run(adapter, spec, 100)
    assert adapter.metrics(state, spec)["centroid_displacement"] == pytest.approx(0.0, abs=1e-12)


def test_dead_boundary_glider_becomes_block_at_corner() -> None:
    # Regression (not a literature claim): with this placement the glider ends as a 2x2 block
    # in the corner. The trajectory is cross-checked against the scalar oracle.
    adapter = ConwayAdapter()
    spec = make_spec("glider", boundary="dead", size=16, offset_x=8, offset_y=8)
    cells = adapter.initialize(spec).grid.tolist()
    for _ in range(200):
        cells = step_grid_oracle(cells, "B3/S23", "dead")
    state = run(adapter, spec, 200)
    assert state.grid.tolist() == cells
    assert int(state.grid.sum()) == 4
    ys, xs = np.nonzero(state.grid)
    assert set(xs) == {14, 15} and set(ys) == {14, 15}


def test_empty_grid_metric_null_policy() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("empty")
    state = adapter.initialize(spec)
    metrics = adapter.metrics(state, spec)
    assert metrics["alive_cell_count"] == 0
    assert math.isnan(metrics["centroid_displacement"])


def test_random_initialization_is_seed_deterministic() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("random", density=0.3)
    a = adapter.initialize(spec).grid
    b = adapter.initialize(spec).grid
    np.testing.assert_array_equal(a, b)
    other = ExperimentSpec.model_validate(
        {**spec.model_dump(), "experiment": {**spec.model_dump()["experiment"], "seed": 2}}
    )
    assert not np.array_equal(a, adapter.initialize(other).grid)


def test_numpy_matches_oracle_on_r_pentomino_long_run() -> None:
    adapter = ConwayAdapter()
    spec = make_spec("r_pentomino", size=24)
    grid = adapter.initialize(spec).grid
    cells = grid.tolist()
    for _ in range(60):
        grid = step_grid(grid, "B3/S23", "periodic")
        cells = step_grid_oracle(cells, "B3/S23", "periodic")
        assert grid.tolist() == cells


def test_patterns_are_well_formed() -> None:
    for name, cells in PATTERNS.items():
        assert len(set(cells)) == len(cells), name
