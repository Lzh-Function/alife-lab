"""Primordial-soup epoch procedure, bit-compatible with CuBFF ``common_language.h``.

All randomness is stateless SplitMix64 of (params seed, epoch, indices), exactly as CuBFF:

* ``seed(x) = SM(SM(params_seed) ^ SM(x))``
* init: ``prog[k][i] = SM(64·N·seed(0) + 64·k + i) mod 256``
* each epoch: identity permutation, Fisher–Yates for i = N−1..0 with
  ``j = SM(seed(epoch·N + i)) mod (i + 1)``; pairs (s[2k], s[2k+1]), k < N/2
* mutation of the 128-byte pair tape: for byte i of pair k,
  ``r = SM((N·seed(epoch) + k)·128 + i)``; if ``(r >> 8) & (2^30 − 1) < mutation_prob``
  the byte becomes ``r & 0xFF``
* then ``Evaluate(tape, 8192)`` and the halves are written back.

All arithmetic is modulo 2^64 (C ``size_t`` / ``uint64_t``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.substrates.bff.vm import HALF, STEP_LIMIT, TAPE, Variant, evaluate_batch

U64 = npt.NDArray[np.uint64]
MASK64 = (1 << 64) - 1
MASK30 = np.uint64((1 << 30) - 1)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def splitmix64_int(x: int) -> int:
    """Scalar reference (Python ints, explicit 2^64 wrap)."""
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def splitmix64(x: U64) -> U64:
    """Vectorised SplitMix64 on uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        out: U64 = z ^ (z >> np.uint64(31))
    return out


def derived_seed(params_seed: int, x: int) -> int:
    return splitmix64_int(splitmix64_int(params_seed) ^ splitmix64_int(x & MASK64))


def init_soup(params_seed: int, num_programs: int, zero: bool = False) -> npt.NDArray[np.uint8]:
    if zero:
        return np.zeros((num_programs, HALF), dtype=np.uint8)
    s0 = derived_seed(params_seed, 0)
    base = np.uint64((HALF * num_programs * s0) & MASK64)
    idx = np.arange(num_programs * HALF, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = splitmix64(base + idx)
    return (values & np.uint64(0xFF)).astype(np.uint8).reshape(num_programs, HALF)


def derived_seeds(params_seed: int, xs: U64) -> U64:
    """Vectorised ``seed(x) = SM(SM(params_seed) ^ SM(x))`` for an array of x."""
    return splitmix64(np.uint64(splitmix64_int(params_seed)) ^ splitmix64(xs))


def _fisher_yates(js: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    s = np.arange(js.size, dtype=np.int64)
    for i in range(js.size - 1, -1, -1):  # sequential swaps (order matters)
        j = int(js[i])
        s[i], s[j] = s[j], s[i]
    return s


_FY_JIT: Any = None
_FY_TRIED = False


def _fisher_yates_fast(js: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """JIT-compiled swap loop if numba is installed (imported lazily, identical result)."""
    global _FY_JIT, _FY_TRIED
    if not _FY_TRIED:
        _FY_TRIED = True
        try:
            from numba import njit

            _FY_JIT = njit(cache=True)(_fisher_yates)
        except ImportError:  # pragma: no cover
            _FY_JIT = None
    if _FY_JIT is None:
        return _fisher_yates(js)
    result: npt.NDArray[np.int64] = _FY_JIT(js)
    return result


def shuffle_indices(params_seed: int, epoch: int, num_programs: int) -> npt.NDArray[np.int64]:
    n = num_programs
    with np.errstate(over="ignore"):
        xs = np.uint64((epoch * n) & MASK64) + np.arange(n, dtype=np.uint64)
    seeds = derived_seeds(params_seed, xs)
    js = (splitmix64(seeds) % (np.arange(n, dtype=np.uint64) + np.uint64(1))).astype(np.int64)
    return _fisher_yates_fast(js)


def mutate_pairs(
    tapes: npt.NDArray[np.uint8],
    params_seed: int,
    epoch: int,
    num_programs: int,
    mutation_prob: int,
    pair_offset: int = 0,
) -> int:
    """Apply CuBFF background mutation in place; returns the number of replaced bytes.

    Row r of ``tapes`` is pair ``pair_offset + r`` of the epoch (used to mutate one pair).
    """
    if mutation_prob <= 0:
        return 0
    seed_e = derived_seed(params_seed, epoch)
    pairs = tapes.shape[0]
    k = np.arange(pairs, dtype=np.uint64) + np.uint64(pair_offset)
    i = np.arange(TAPE, dtype=np.uint64)
    with np.errstate(over="ignore"):
        base = (np.uint64((num_programs * seed_e) & MASK64) + k) * np.uint64(TAPE)
        rng = splitmix64(base[:, None] + i[None, :])
    repl = (rng & np.uint64(0xFF)).astype(np.uint8)
    hit = ((rng >> np.uint64(8)) & MASK30) < np.uint64(mutation_prob)
    tapes[hit] = repl[hit]
    return int(hit.sum())


def run_epoch(
    soup: npt.NDArray[np.uint8],
    params_seed: int,
    epoch: int,
    mutation_prob: int,
    step_limit: int = STEP_LIMIT,
    variant: Variant = "bff_noheads",
    backend: str = "numpy",
) -> dict[str, Any]:
    """One CuBFF epoch in place on ``soup`` (N, 64). Returns per-epoch statistics."""
    n = soup.shape[0]
    if n % 2:
        raise ValueError("num_programs must be even")
    s = shuffle_indices(params_seed, epoch, n)
    p1, p2 = s[0::2], s[1::2]
    tapes = np.concatenate([soup[p1], soup[p2]], axis=1)
    mutated = mutate_pairs(tapes, params_seed, epoch, n, mutation_prob)
    if backend == "numba":
        from alife_lab.substrates.bff.vm_numba import evaluate_batch_numba

        ops = evaluate_batch_numba(tapes, step_limit, variant)
    elif backend == "numpy":
        ops = evaluate_batch(tapes, step_limit, variant)
    else:
        raise ValueError(f"unknown BFF backend {backend!r}")
    soup[p1] = tapes[:, :HALF]
    soup[p2] = tapes[:, HALF:]
    return {
        "ops_total": int(ops.sum()),
        "ops_per_pair": float(ops.mean()),
        "mutated_bytes": mutated,
    }


def run_epoch_scalar(
    soup: list[bytearray],
    params_seed: int,
    epoch: int,
    mutation_prob: int,
    step_limit: int = STEP_LIMIT,
    variant: Variant = "bff_noheads",
) -> None:
    """Pure-Python reference epoch (tests only)."""
    from alife_lab.substrates.bff.vm import evaluate_scalar

    n = len(soup)
    s = list(range(n))
    for i in range(n - 1, -1, -1):
        j = splitmix64_int(derived_seed(params_seed, epoch * n + i)) % (i + 1)
        s[i], s[j] = s[j], s[i]
    seed_e = derived_seed(params_seed, epoch)
    for k in range(n // 2):
        a, b = s[2 * k], s[2 * k + 1]
        tape = bytearray(soup[a] + soup[b])
        for i in range(TAPE):
            r = splitmix64_int((((n * seed_e) + k) * TAPE + i) & MASK64)
            if ((r >> 8) & ((1 << 30) - 1)) < mutation_prob:
                tape[i] = r & 0xFF
        evaluate_scalar(tape, step_limit, variant)
        soup[a] = bytearray(tape[:HALF])
        soup[b] = bytearray(tape[HALF:])
