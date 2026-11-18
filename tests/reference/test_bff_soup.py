"""BFF soup procedure: vectorised == scalar reference; SplitMix64 known values."""

from __future__ import annotations

import numpy as np

from alife_lab.substrates.bff.soup import (
    init_soup,
    run_epoch,
    run_epoch_scalar,
    splitmix64,
    splitmix64_int,
)
from alife_lab.substrates.bff.vm import HALF, TAPE


def test_splitmix64_known_values() -> None:
    # Reference values of SplitMix64 (Vigna) for state increments from 0.
    assert splitmix64_int(0) == 0xE220A8397B1DCDAF
    assert splitmix64_int(0x9E3779B97F4A7C15 * 0 + 0) == 0xE220A8397B1DCDAF
    xs = np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64)
    assert splitmix64(xs).tolist() == [splitmix64_int(int(x)) for x in xs.tolist()]


def test_init_is_deterministic_and_uniformish() -> None:
    a = init_soup(7, 256)
    b = init_soup(7, 256)
    assert a.shape == (256, HALF) and np.array_equal(a, b)
    assert not np.array_equal(a, init_soup(8, 256))
    counts = np.bincount(a.ravel(), minlength=256)
    assert counts.min() > 20 and counts.max() < 110  # 16384 bytes over 256 values
    assert not init_soup(7, 4, zero=True).any()


def test_vectorised_epochs_equal_scalar_reference() -> None:
    n, seed, mut = 64, 123, 1 << 22  # high mutation rate to exercise the mutation path
    soup = init_soup(seed, n)
    ref = [bytearray(p.tobytes()) for p in soup]
    for epoch in range(6):
        stats = run_epoch(soup, seed, epoch, mut)
        run_epoch_scalar(ref, seed, epoch, mut)
        assert [bytes(p) for p in soup] == [bytes(p) for p in ref], f"epoch {epoch}"
        assert stats["mutated_bytes"] > 0
    assert TAPE == 2 * HALF


def test_bff_variant_epochs_equal_scalar_reference() -> None:
    n, seed = 32, 5
    soup = init_soup(seed, n)
    ref = [bytearray(p.tobytes()) for p in soup]
    for epoch in range(4):
        run_epoch(soup, seed, epoch, 1 << 18, variant="bff")
        run_epoch_scalar(ref, seed, epoch, 1 << 18, variant="bff")
    assert [bytes(p) for p in soup] == [bytes(p) for p in ref]
