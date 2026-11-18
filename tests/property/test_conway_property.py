"""Differential / property tests: NumPy implementation vs scalar oracle."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from alife_lab.substrates.ca.conway import ConwayAdapter, step_grid, step_grid_oracle

rules = st.sampled_from(["B3/S23", "B36/S23", "B2/S", "B3678/S34678", "B/S012345678"])
boundaries = st.sampled_from(["periodic", "dead"])


@st.composite
def grids(draw: st.DrawFn) -> np.ndarray:
    h = draw(st.integers(3, 14))
    w = draw(st.integers(3, 14))
    bits = draw(st.lists(st.integers(0, 1), min_size=h * w, max_size=h * w))
    return np.array(bits, dtype=np.uint8).reshape(h, w)


@settings(max_examples=150, deadline=None)
@given(grid=grids(), rule=rules, boundary=boundaries)
def test_numpy_step_equals_oracle(grid: np.ndarray, rule: str, boundary: str) -> None:
    expected = step_grid_oracle(grid.tolist(), rule, boundary)
    assert step_grid(grid, rule, boundary).tolist() == expected


@settings(max_examples=60, deadline=None)
@given(grid=grids(), dx=st.integers(-5, 5), dy=st.integers(-5, 5))
def test_periodic_step_is_translation_equivariant(grid: np.ndarray, dx: int, dy: int) -> None:
    shifted = np.roll(np.roll(grid, dy, 0), dx, 1)
    a = np.roll(np.roll(step_grid(grid, "B3/S23", "periodic"), dy, 0), dx, 1)
    b = step_grid(shifted, "B3/S23", "periodic")
    np.testing.assert_array_equal(a, b)


@settings(max_examples=40, deadline=None)
@given(grid=grids())
def test_output_is_binary_and_input_untouched(grid: np.ndarray) -> None:
    before = grid.copy()
    out = step_grid(grid, "B3/S23", "periodic")
    np.testing.assert_array_equal(grid, before)
    assert set(np.unique(out).tolist()) <= {0, 1}
    assert out.dtype == np.uint8


def test_observation_does_not_change_dynamics() -> None:
    """Metrics/render are passive: interleaving them must not alter the trajectory."""
    from tests.reference.test_conway_reference import make_spec

    adapter = ConwayAdapter()
    spec = make_spec("random", density=0.35, size=40)
    a = adapter.initialize(spec)
    b = adapter.initialize(spec)
    for _ in range(100):
        a = adapter.step(a, spec)
        b = adapter.step(b, spec)
        adapter.metrics(b, spec)
        adapter.render_payload(b)
    np.testing.assert_array_equal(a.grid, b.grid)
