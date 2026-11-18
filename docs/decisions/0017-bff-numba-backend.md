# ADR 0017 — Optional Numba backend for BFF

- Status: accepted (Phase 3, slice 2)
- Date: 2026-11-18

## Context
The NumPy lockstep interpreter is ≈100× slower than CuBFF's C++ build (ADR 0016), which made
paper-scale soups (2^17 programs × thousands of epochs) impractical.

## Decision
- `numba` (≥ 0.68, with llvmlite) as an **optional extra** (`uv sync --extra fast`). Without it
  the NumPy backend is used; CPU-only setups and CI keep working.
- `vm_numba.py` is a direct port of the scalar oracle; pairs run in parallel with `prange`.
  The Fisher–Yates swap loop is JIT-compiled when available; shuffle seeds are vectorised.
- Backend selection is execution-only: the BFF manifest lists `("numba", "numpy")` (preference
  order for `auto`), targets advertise the backends that are importable, the resolved backend
  is stored in `runs.backend`, the run manifest (with the numba version) and checkpoints.
- Bit-identity is required and tested: Numba == scalar oracle (600 tapes, both variants),
  Numba epochs == NumPy epochs, Numba == CuBFF fixture digests, and a platform run with
  `auto` (numba) vs `numpy` has distance 0 at every checkpoint.

## Measurements (2026-11-18, 24-logical-core machine; early random soups)
| Programs | NumPy (1 core) | Numba 1 thread | Numba 24 threads | CuBFF CPU 1 thread |
|---|---|---|---|---|
| 4 096 | 210 ms | 4.7 ms | 6.2 ms | 2.0 ms |
| 32 768 | 5 300 ms | 41.7 ms | 30.1 ms | — |
| 131 072 | 8 300 ms | 184 ms | 119 ms | — |

## Cost
Installed size: numba 17 MB + llvmlite 172 MB. First use JIT-compiles (cached on disk under
`__pycache__`). Remaining gap to CuBFF is mainly mutation/shuffle bookkeeping in NumPy.
