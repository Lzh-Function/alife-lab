"""Lenia: independent direct-convolution oracle, kernel/growth properties, RLE decoding,
Orbium survival, negative control, interventions and checkpoint/replay exactness."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.lenia import reference as ref
from alife_lab.substrates.lenia.lenia import LeniaAdapter

A = LeniaAdapter()


def make(size: int = 128, **exp: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "lenia",
                "substrate": "lenia",
                "seed": 0,
                "world": {"width": size, "height": size},
                **exp,
            }
        }
    )


def run(spec: ExperimentSpec, n: int) -> Any:
    s = A.initialize(spec)
    for _ in range(n):
        s = A.step(s, spec)
    return s


def direct_potential(cells: np.ndarray, R: int, b: list[float], core: str) -> np.ndarray:
    """O(N * K) periodic correlation with the normalised kernel (no FFT)."""
    offsets = np.arange(-R, R + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    k = ref.kernel_shell(np.sqrt(dx**2 + dy**2) / R, b, core)
    k = k / k.sum()
    out = np.zeros_like(cells)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if k[i, j] != 0:
                out += k[i, j] * np.roll(cells, (dy[i, j], dx[i, j]), axis=(0, 1))
    return out


@pytest.mark.parametrize("core", ["polynomial", "exponential", "step", "staircase"])
@pytest.mark.parametrize("b", [[1.0], [0.5, 1.0, 0.25]])
def test_fft_potential_matches_direct_convolution(core: str, b: list[float]) -> None:
    rng = np.random.default_rng(3)
    cells = rng.random((40, 48))
    kfft = ref.kernel_fft(40, 48, 7, b, core)
    np.testing.assert_allclose(
        ref.potential(cells, kfft), direct_potential(cells, 7, b, core), rtol=0, atol=1e-12
    )


def test_kernel_is_normalised_radial_and_zero_outside_r() -> None:
    k = np.real(np.fft.ifftn(ref.kernel_fft(64, 64, 13, [1.0], "polynomial")))
    assert k.sum() == pytest.approx(1.0, abs=1e-12)
    yy, xx = np.mgrid[0:64, 0:64]
    d = np.hypot(yy - 32, xx - 32)
    assert np.abs(k[d >= 13]).max() < 1e-12
    np.testing.assert_allclose(k, k[::-1, :][np.r_[63, 0:63], :], atol=1e-12)  # y mirror


def test_growth_functions_range_and_peak() -> None:
    n = np.linspace(0, 1, 2001)
    for name, g in ref.GROWTH.items():
        v = g(n, 0.15, 0.015)
        assert v.min() >= -1 and v.max() <= 1, name
        assert g(np.array([0.15]), 0.15, 0.015)[0] == 1.0, name


def test_orbium_rle_decodes_to_catalogue_shape() -> None:
    cells = ref.rle2arr(ref.ANIMALS["O2u"]["cells"])
    assert cells.shape == (20, 20)
    assert cells.min() == 0 and cells.max() == 1.0
    assert ref.rle2arr("2.o$A!").tolist() == [[0, 0, 1.0], [1 / 255, 0, 0]]


def test_adapter_step_equals_reference_update() -> None:
    spec = make()
    s = A.initialize(spec)
    cells = s.cells.copy()
    kfft = ref.kernel_fft(128, 128, 13, [1.0], "polynomial")
    for _ in range(50):
        cells = ref.calc_once(cells, kfft, 10, 0.15, 0.015, "polynomial")
        s = A.step(s, spec)
    assert np.array_equal(s.cells, cells)


def test_orbium_survives_and_moves() -> None:
    spec = make()
    s = A.initialize(spec)
    m0 = A.metrics(s, spec)["mass"]
    masses = []
    for t in range(1, 601):
        s = A.step(s, spec)
        if t % 50 == 0:
            masses.append(A.metrics(s, spec)["mass"])
    assert min(masses) > 0.85 * m0 and max(masses) < 1.15 * m0
    assert A.metrics(s, spec)["centroid_displacement"] > 50


def test_negative_control_off_niche_orbium_dissolves() -> None:
    spec = make(dynamics={"growth": {"m": 0.17, "s": 0.015}})
    m = A.metrics(run(spec, 500), spec)
    assert m["mass"] == 0.0
    assert np.isnan(m["centroid_displacement"])


def test_empty_world_stays_empty() -> None:
    spec = make(initialization={"preset": "empty"})
    s = run(spec, 20)
    assert s.cells.sum() == 0.0


def test_random_init_is_seeded() -> None:
    a = A.initialize(make(initialization={"preset": "random"}))
    b = A.initialize(make(initialization={"preset": "random"}))
    c = A.initialize(make(seed=1, initialization={"preset": "random"}))
    assert np.array_equal(a.cells, b.cells) and not np.array_equal(a.cells, c.cells)


def test_checkpoint_roundtrip_continues_bitwise() -> None:
    spec = make()
    s = run(spec, 37)
    payload = A.serialize_state(s)
    restored = A.deserialize_state(
        decode_checkpoint(encode_checkpoint(payload["arrays"], payload["meta"])), spec
    )
    for _ in range(40):
        s, restored = A.step(s, spec), A.step(restored, spec)
    assert A.state_distance(s, restored)["exact_equal"]
    assert A.metrics(s, spec) == A.metrics(restored, spec)


def test_deserialize_rejects_out_of_range() -> None:
    spec = make()
    payload = A.serialize_state(A.initialize(spec))
    payload["arrays"]["cells"] = payload["arrays"]["cells"] * 2
    with pytest.raises(ValueError):
        A.deserialize_state(payload, spec)


def test_paint_and_erase_interventions() -> None:
    spec = make(initialization={"preset": "empty"})
    s = A.initialize(spec)
    s, info = A.apply_intervention(s, spec, "paint", {"x": 0, "y": 0, "radius": 2, "value": 0.5})
    assert info["cells"] == 13 and s.cells.sum() == pytest.approx(6.5)
    assert s.cells[127, 127] == 0.5  # wraps around the torus
    s, _ = A.apply_intervention(s, spec, "erase", {"x": 1, "y": 0, "radius": 5})
    assert s.cells.sum() == 0
    for bad in ({"x": 200, "y": 0}, {"x": 0, "y": 0, "value": 2}, {"y": 0}):
        with pytest.raises(ValueError):
            A.apply_intervention(s, spec, "paint", bad)


def test_inspect_reports_consistent_next_value() -> None:
    spec = make()
    s = A.initialize(spec)
    info = A.inspect(s, spec, {"x": 64, "y": 64})
    nxt = A.step(A.deserialize_state(A.serialize_state(s), spec), spec)
    assert info["next_value"] == nxt.cells[64, 64]


def test_render_quantises_to_u8() -> None:
    header, data = A.render(A.initialize(make()))
    assert header["codec"] == "field_u8" and len(data) == 128 * 128


def test_spec_rejects_kernel_larger_than_world() -> None:
    with pytest.raises(ValidationError):
        make(size=16, dynamics={"kernel": {"R": 13}})


def test_potential_multiplies_kernel_first_like_leniand() -> None:
    # LeniaND computes kernel_FFT * world_FFT with world_FFT as a named array. Complex
    # multiplication is not bitwise commutative in NumPy, so the order is part of fidelity.
    rng = np.random.default_rng(5)
    cells = rng.random((64, 64))
    kfft = ref.kernel_fft(64, 64, 13, [1.0], "polynomial")
    world_fft = np.fft.fftn(cells)
    expected = np.fft.fftshift(np.real(np.fft.ifftn(kfft * world_fft)))
    assert np.array_equal(ref.potential(cells, kfft), expected)
