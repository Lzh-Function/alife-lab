"""Optional CUDA backend for the BFF VM via CuPy (``uv sync --extra gpu``).

One CUDA thread evaluates one 128-byte pair tape, copied into thread-local memory. The loop is a
direct port of :func:`alife_lab.substrates.bff.vm.evaluate_scalar` (CuBFF ``bff.inc.h``
semantics); results are integers, so the backend must be bit-identical to the scalar oracle,
the NumPy and the Numba backends (tested). CuPy is imported lazily so the API process never
initialises CUDA.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import numpy.typing as npt

TAPE = 128

_SOURCE = r"""
__device__ long long eval_tape(unsigned char* tape, const int step_limit, const int heads) {
    int head0, head1, pos;
    if (heads) { head0 = tape[0] % 128; head1 = tape[1] % 128; pos = 2; }
    else { head0 = 128; head1 = 128; pos = 0; }
    int nskip = 0;
    int i = 0;
    while (i < step_limit) {
        head0 &= 127;
        head1 &= 127;
        const unsigned char cmd = tape[pos];
        switch (cmd) {
            case '<': head0 -= 1; break;
            case '>': head0 += 1; break;
            case '{': head1 -= 1; break;
            case '}': head1 += 1; break;
            case '+': tape[head0] = (unsigned char)(tape[head0] + 1); break;
            case '-': tape[head0] = (unsigned char)(tape[head0] - 1); break;
            case '.': tape[head1] = tape[head0]; break;
            case ',': tape[head0] = tape[head1]; break;
            case '[':
                if (tape[head0] == 0) {
                    int depth = 1;
                    pos += 1;
                    while (pos < 128 && depth > 0) {
                        if (tape[pos] == ']') depth -= 1;
                        if (tape[pos] == '[') depth += 1;
                        pos += 1;
                    }
                    pos -= 1;
                    if (depth != 0) pos = 128;
                }
                break;
            case ']':
                if (tape[head0] != 0) {
                    int depth = 1;
                    pos -= 1;
                    while (pos >= 0 && depth > 0) {
                        if (tape[pos] == ']') depth += 1;
                        if (tape[pos] == '[') depth -= 1;
                        pos -= 1;
                    }
                    pos += 1;
                    if (depth != 0) pos = -1;
                }
                break;
            default: nskip += 1;
        }
        if (pos < 0) { i += 1; break; }
        pos += 1;
        if (pos >= 128) { i += 1; break; }
        i += 1;
    }
    return (long long)(i - nskip);
}

__device__ unsigned long long splitmix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void bff_eval(unsigned char* tapes, long long* ops, const int n,
                                    const int step_limit, const int heads) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    unsigned char tape[128];
    unsigned char* src = tapes + (long long)p * 128;
    for (int k = 0; k < 128; ++k) tape[k] = src[k];
    ops[p] = eval_tape(tape, step_limit, heads);
    for (int k = 0; k < 128; ++k) src[k] = tape[k];
}

// One CuBFF epoch for pair k: gather soup[s[2k]] ++ soup[s[2k+1]], background mutation
// (r = SM((N*seed(epoch) + k)*128 + i); replace byte if ((r>>8) & (2^30-1)) < prob), evaluate,
// scatter back. `mul` = N * seed(epoch) mod 2^64 is computed on the host.
extern "C" __global__ void bff_epoch(unsigned char* soup, const long long* perm, long long* ops,
                                     int* mutated, const int pairs,
                                     const unsigned long long mul,
                                     const unsigned int mutation_prob, const int step_limit,
                                     const int heads) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pairs) return;
    unsigned char tape[128];
    unsigned char* a = soup + perm[2 * k] * 64;
    unsigned char* b = soup + perm[2 * k + 1] * 64;
    for (int i = 0; i < 64; ++i) { tape[i] = a[i]; tape[64 + i] = b[i]; }
    int hits = 0;
    if (mutation_prob > 0) {
        const unsigned long long base = (mul + (unsigned long long)k) * 128ULL;
        for (int i = 0; i < 128; ++i) {
            const unsigned long long r = splitmix64(base + (unsigned long long)i);
            if (((r >> 8) & 0x3FFFFFFFULL) < (unsigned long long)mutation_prob) {
                tape[i] = (unsigned char)(r & 0xFF);
                hits += 1;
            }
        }
    }
    ops[k] = eval_tape(tape, step_limit, heads);
    mutated[k] = hits;
    for (int i = 0; i < 64; ++i) { a[i] = tape[i]; b[i] = tape[64 + i]; }
}
"""

_MODULE: Any = None
BLOCK = 128


def available() -> bool:
    """True if CuPy is installed and a CUDA device is usable (initialises CUDA)."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def _kernel(name: str = "bff_eval") -> Any:
    global _MODULE
    if _MODULE is None:
        import cupy

        _MODULE = cupy.RawModule(code=_SOURCE)
    return _MODULE.get_function(name)


def evaluate_batch_cupy(
    tapes: npt.NDArray[np.uint8], step_limit: int, variant: str
) -> npt.NDArray[np.int64]:
    """Evaluate host tapes in place on the GPU (copy in, run, copy back)."""
    import cupy

    if tapes.ndim != 2 or tapes.shape[1] != TAPE or tapes.dtype != np.uint8:
        raise ValueError("tapes must be uint8 of shape (P, 128)")
    n = tapes.shape[0]
    d_tapes = cupy.asarray(np.ascontiguousarray(tapes))
    d_ops = cupy.empty(n, dtype=cupy.int64)
    if n:
        grid = ((n + BLOCK - 1) // BLOCK,)
        _kernel()(
            grid,
            (BLOCK,),
            (d_tapes, d_ops, np.int32(n), np.int32(step_limit), np.int32(variant == "bff")),
        )
    tapes[...] = cupy.asnumpy(d_tapes)
    ops: npt.NDArray[np.int64] = cupy.asnumpy(d_ops)
    return ops


def run_epoch_cupy(
    soup: npt.NDArray[np.uint8],
    params_seed: int,
    epoch: int,
    mutation_prob: int,
    step_limit: int,
    variant: str,
) -> dict[str, Any]:
    """One CuBFF epoch with gather, mutation, evaluation and scatter on the GPU.

    The Fisher-Yates permutation is sequential and stays on the host (shared code with the
    CPU backends). ``soup`` is updated in place; bit-identical to ``soup.run_epoch``.
    """
    import cupy

    from alife_lab.substrates.bff.soup import MASK64, derived_seed, shuffle_indices

    n = soup.shape[0]
    pairs = n // 2
    perm = shuffle_indices(params_seed, epoch, n)
    mul = (n * derived_seed(params_seed, epoch)) & MASK64
    d_soup = cupy.asarray(np.ascontiguousarray(soup))
    d_perm = cupy.asarray(perm.astype(np.int64))
    d_ops = cupy.empty(pairs, dtype=cupy.int64)
    d_mut = cupy.empty(pairs, dtype=cupy.int32)
    if pairs:
        _kernel("bff_epoch")(
            ((pairs + BLOCK - 1) // BLOCK,),
            (BLOCK,),
            (
                d_soup, d_perm, d_ops, d_mut, np.int32(pairs), np.uint64(mul),
                np.uint32(mutation_prob), np.int32(step_limit), np.int32(variant == "bff"),
            ),
        )  # fmt: skip
    soup[...] = cupy.asnumpy(d_soup)
    ops = d_ops.sum().item() if pairs else 0
    return {
        "ops_total": int(ops),
        "ops_per_pair": float(ops / pairs) if pairs else 0.0,
        "mutated_bytes": int(d_mut.sum().item()) if pairs else 0,
    }
