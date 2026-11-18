"""Numba backend == scalar oracle == NumPy backend == CuBFF fixtures (skipped without numba)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from alife_lab.substrates.bff import vm_numba
from alife_lab.substrates.bff.soup import init_soup, run_epoch
from alife_lab.substrates.bff.vm import TAPE, evaluate_scalar

pytestmark = pytest.mark.skipif(not vm_numba.AVAILABLE, reason="numba extra not installed")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cubff_soups.json"


@pytest.mark.parametrize("variant", ["bff_noheads", "bff"])
def test_numba_equals_scalar_oracle(variant: str) -> None:
    rng = np.random.default_rng(1)
    instr = np.frombuffer(b"<>{}+-.,[]\x00\x00\x01", dtype=np.uint8)
    tapes = np.where(
        rng.random((600, TAPE)) < 0.7,
        rng.choice(instr, size=(600, TAPE)),
        rng.integers(0, 256, size=(600, TAPE)),
    ).astype(np.uint8)
    ref = [bytearray(t.tobytes()) for t in tapes]
    expected = [evaluate_scalar(t, 8192, variant) for t in ref]  # type: ignore[arg-type]
    ops = vm_numba.evaluate_batch_numba(tapes, 8192, variant)
    assert ops.tolist() == expected
    assert all(bytes(a) == bytes(b) for a, b in zip(tapes, ref, strict=True))


def test_numba_epochs_equal_numpy_epochs() -> None:
    a = init_soup(4, 512)
    b = a.copy()
    for e in range(25):
        sa = run_epoch(a, 4, e, 1 << 20, backend="numpy")
        sb = run_epoch(b, 4, e, 1 << 20, backend="numba")
        assert sa == sb
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: f"{c['lang']}-{c['num']}"
)
def test_numba_matches_cubff_binary(case: dict) -> None:
    soup = init_soup(case["seed"], case["num"])
    done = 0
    for snap in case["snapshots"]:
        while done < snap["epochs_completed"]:
            run_epoch(
                soup,
                case["seed"],
                done,
                case["mutation_prob_int"],
                variant=case["lang"],
                backend="numba",
            )
            done += 1
        assert hashlib.sha256(soup.tobytes()).hexdigest() == snap["soup_sha256"]
