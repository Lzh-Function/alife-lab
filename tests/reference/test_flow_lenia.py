"""Flow-Lenia: Sobel / reintegration oracles, mass conservation, kernel construction,
rule sampling, negative control, interventions and checkpoint exactness."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.flow_lenia import core
from alife_lab.substrates.flow_lenia.flow_lenia import FlowLeniaAdapter

A = FlowLeniaAdapter()
EXAMPLE_M = [[3, 1, 0], [0, 3, 1], [1, 0, 3]]


def make(size: int = 64, **exp: Any) -> ExperimentSpec:
    dyn = exp.pop("dynamics", {"channels": 3, "rule": {"connectivity": EXAMPLE_M}})
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "fl",
                "substrate": "flow_lenia",
                "seed": exp.pop("seed", 0),
                "world": {"width": size, "height": size, **exp.pop("world", {})},
                "initialization": {"size": size // 3},
                "dynamics": dyn,
                **exp,
            }
        }
    )


def convolve2d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Textbook zero-padded 'same' 2D convolution (kernel flipped), 3x3 kernels."""
    h, w = a.shape
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            acc = 0.0
            for p in range(3):
                for q in range(3):
                    ii, jj = i + 1 - p, j + 1 - q
                    if 0 <= ii < h and 0 <= jj < w:
                        acc += k[p, q] * a[ii, jj]
            out[i, j] = acc
    return out


def test_sobel_matches_textbook_convolution() -> None:
    rng = np.random.default_rng(1)
    field = rng.random((9, 11, 2))
    kx = np.array([[1.0, 0, -1], [2, 0, -2], [1, 0, -1]])
    g = core.sobel(field)
    for c in range(2):
        np.testing.assert_allclose(g[:, :, 1, c], convolve2d_same(field[:, :, c], kx), atol=1e-14)
        np.testing.assert_allclose(g[:, :, 0, c], convolve2d_same(field[:, :, c], kx.T), atol=1e-14)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    sigma=st.sampled_from([0.3, 0.5, 0.65, 1.2]),
    boundary=st.sampled_from(["wall", "torus"]),
    scale=st.floats(0.0, 40.0),
)
def test_reintegration_matches_exact_overlap_oracle(
    seed: int, sigma: float, boundary: str, scale: float
) -> None:
    rng = np.random.default_rng(seed)
    cells = rng.random((10, 13, 2)) * (rng.random((10, 13, 1)) < 0.6)
    f = rng.normal(0, scale, (10, 13, 2, 2))
    mu = core.target_positions(cells.shape, f, 0.2, 3, sigma, boundary)
    got = core.reintegrate(cells, mu, 3, sigma, boundary)
    np.testing.assert_allclose(got, core.reintegrate_oracle(cells, mu, sigma, boundary), atol=1e-13)
    assert got.sum() == pytest.approx(cells.sum(), rel=1e-13)


@pytest.mark.parametrize("boundary", ["wall", "torus"])
def test_mass_is_conserved_over_many_steps(boundary: str) -> None:
    spec = make(world={"boundary": boundary})
    s = A.initialize(spec)
    m0 = s.cells.sum()
    for _ in range(150):
        s = A.step(s, spec)
    assert abs(s.cells.sum() - m0) / m0 < 1e-12
    assert s.cells.min() >= 0


def test_kernel_matches_official_grid_construction() -> None:
    rule = core.sample_rule(np.random.default_rng(3), [[4]], None)
    n = 32
    mid = n // 2
    grid = np.mgrid[-mid:mid, -mid:mid]
    dist = np.sqrt((grid**2).sum(0))
    for i in range(rule.k):
        d = dist / ((rule.R + 15) * rule.r[i])
        rings = (rule.b[i] * np.exp(-((d[..., None] - rule.a[i]) ** 2) / rule.w[i])).sum(-1)
        k = 0.5 * (np.tanh(-(d - 1) * 10 / 2) + 1) * rings
        expected = np.fft.fftshift(k / k.sum())
        np.testing.assert_allclose(core.kernels_spatial(n, n, rule)[:, :, i], expected, atol=1e-15)


def test_affinity_matches_direct_periodic_convolution() -> None:
    rule = core.sample_rule(np.random.default_rng(4), [[2]], 3.0)
    rule = core.rule_from_kernels(1.0, rule.to_json()["kernels"])  # small kernels: R + 15 = 16
    rule = core.Rule(**{**rule.__dict__, "r": np.array([0.3, 0.2])})
    h, w = 20, 24
    cells = np.random.default_rng(5).random((h, w, 1))
    ks = core.kernels_spatial(h, w, rule)
    u = core.affinity(cells, core.kernels_fft(h, w, rule), rule, 1)
    expected = np.zeros((h, w))
    for i in range(rule.k):
        pot = np.zeros((h, w))
        for dy in range(h):
            for dx in range(w):
                if ks[dy, dx, i] != 0:
                    pot += ks[dy, dx, i] * np.roll(cells[:, :, 0], (dy, dx), axis=(0, 1))
        expected += core.growth(pot, rule.m[i], rule.s[i]) * rule.h[i]
    np.testing.assert_allclose(u[:, :, 0], expected, atol=1e-12)


def test_conn_from_matrix_order_matches_official() -> None:
    assert core.conn_from_matrix([[2, 1], [0, 1]]) == ([0, 0, 0, 1], [0, 0, 1, 1])


def test_rule_sampling_is_seeded_and_in_range() -> None:
    a = core.sample_rule(np.random.default_rng(7), EXAMPLE_M, None)
    b = core.sample_rule(np.random.default_rng(7), EXAMPLE_M, None)
    assert a.to_json() == b.to_json() and a.k == 12
    for name, (lo, hi) in core.RANGES.items():
        v = np.asarray(getattr(a, name))
        assert v.min() >= lo and v.max() <= hi, name


def test_negative_control_diffuses_while_example_condenses() -> None:
    def final(spec: ExperimentSpec) -> dict[str, float]:
        s = A.initialize(spec)
        for _ in range(200):
            s = A.step(s, spec)
        return A.metrics(s, spec)

    example = final(make(size=96))
    control = final(
        make(size=96, dynamics={"channels": 3, "theta_A": 0.001,
                                "rule": {"connectivity": EXAMPLE_M}})
    )  # fmt: skip
    assert control["max_value"] < 0.5 < example["max_value"]
    assert control["alive_fraction"] > 2 * example["alive_fraction"]


def test_explicit_kernels_and_spec_validation() -> None:
    kern = {"r": 0.5, "m": 0.2, "s": 0.05, "h": 1.0, "a": [0.1, 0.5, 0.9],
            "b": [1, 0.5, 0.2], "w": [0.1, 0.1, 0.1]}  # fmt: skip
    spec = make(dynamics={"channels": 1, "rule": {"R": 5, "kernels": [kern]}})
    s = A.initialize(spec)
    assert s.rule.k == 1 and s.rule.R == 5
    bad = [
        {"channels": 1, "rule": {"kernels": [{**kern, "target": 1}]}},
        {"channels": 2},  # sampled multi-channel rule without connectivity
        {"channels": 2, "rule": {"connectivity": [[1]]}},
        {"channels": 1, "sigma": 5.0, "dd": 5},
    ]
    for dyn in bad:
        with pytest.raises(ValidationError):
            make(dynamics=dyn)


def test_checkpoint_roundtrip_continues_bitwise() -> None:
    spec = make()
    s = A.initialize(spec)
    for _ in range(13):
        s = A.step(s, spec)
    ser = A.serialize_state(s)
    restored = A.deserialize_state(
        decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), spec
    )
    for _ in range(20):
        s, restored = A.step(s, spec), A.step(restored, spec)
    assert A.state_distance(s, restored)["exact_equal"]
    assert A.metrics(s, spec) == A.metrics(restored, spec)


def test_checkpoint_with_a_different_rule_is_rejected() -> None:
    ser = A.serialize_state(A.initialize(make()))
    with pytest.raises(ValueError):
        A.deserialize_state(ser, make(seed=1))


def test_paint_channel_and_inspect() -> None:
    spec = make()
    s = A.initialize(spec)
    s, info = A.apply_intervention(s, spec, "paint", {"x": 3, "y": 3, "radius": 1, "channel": 2,
                                                      "value": 0.5})  # fmt: skip
    assert info["cells"] == 5 and s.cells[3, 3].tolist() == [0.0, 0.0, 0.5]
    assert A.metrics(s, spec)["mass_drift"] == 0.0  # baseline reset by the intervention
    with pytest.raises(ValueError):
        A.apply_intervention(s, spec, "paint", {"x": 3, "y": 3, "channel": 3})
    out = A.inspect(s, spec, {"x": 32, "y": 32})
    assert len(out["channels"]) == 3 and len(out["rule"]["kernels"]) == 12
    header, data = A.render(s)
    assert header["channels"] == 3 and len(data) == 64 * 64 * 3
