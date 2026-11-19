"""CUDA (CuPy) BFF backend == scalar oracle == NumPy backend == CuBFF fixtures.

Skipped when CuPy or a CUDA device is unavailable (CPU-only CI)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from alife_lab.substrates.bff import vm_cupy
from alife_lab.substrates.bff.soup import init_soup, run_epoch
from alife_lab.substrates.bff.vm import TAPE, evaluate_scalar

pytestmark = pytest.mark.skipif(not vm_cupy.available(), reason="cupy / CUDA device unavailable")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cubff_soups.json"


@pytest.mark.parametrize("variant", ["bff_noheads", "bff"])
def test_cupy_equals_scalar_oracle(variant: str) -> None:
    rng = np.random.default_rng(2)
    instr = np.frombuffer(b"<>{}+-.,[]\x00\x00\x01", dtype=np.uint8)
    tapes = np.where(
        rng.random((800, TAPE)) < 0.7,
        rng.choice(instr, size=(800, TAPE)),
        rng.integers(0, 256, size=(800, TAPE)),
    ).astype(np.uint8)
    ref = [bytearray(t.tobytes()) for t in tapes]
    expected = [evaluate_scalar(t, 8192, variant) for t in ref]  # type: ignore[arg-type]
    ops = vm_cupy.evaluate_batch_cupy(tapes, 8192, variant)
    assert ops.tolist() == expected
    assert all(bytes(a) == bytes(b) for a, b in zip(tapes, ref, strict=True))


def test_cupy_handles_empty_batch_and_zero_step_limit() -> None:
    assert vm_cupy.evaluate_batch_cupy(np.zeros((0, TAPE), np.uint8), 8192, "bff").size == 0
    tapes = np.full((3, TAPE), ord("+"), np.uint8)
    assert vm_cupy.evaluate_batch_cupy(tapes, 0, "bff_noheads").tolist() == [0, 0, 0]
    assert (tapes == ord("+")).all()


def test_cupy_epochs_equal_numpy_epochs() -> None:
    a = init_soup(4, 512)
    b = a.copy()
    for e in range(25):
        assert run_epoch(a, 4, e, 1 << 20, backend="numpy") == run_epoch(
            b, 4, e, 1 << 20, backend="cupy"
        )
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: f"{c['lang']}-{c['num']}"
)
def test_cupy_matches_cubff_binary(case: dict) -> None:
    soup = init_soup(case["seed"], case["num"])
    done = 0
    for snap in case["snapshots"]:
        while done < snap["epochs_completed"]:
            run_epoch(
                soup, case["seed"], done, case["mutation_prob_int"], variant=case["lang"],
                backend="cupy",
            )  # fmt: skip
            done += 1
        assert hashlib.sha256(soup.tobytes()).hexdigest() == snap["soup_sha256"]
