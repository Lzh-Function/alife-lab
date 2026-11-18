# ADR 0019 — Flow-Lenia: clean-room NumPy implementation of the official semantics

Status: accepted (2026-11-18)

## Context
The authors' JAX implementation (erwanplantec/FlowLenia) has no licence. The project owner
allows code reuse, but unlicensed code cannot be redistributed, and the repository may be
published later. JAX is also not a dependency of the CPU-only baseline.

## Decision
- Implement Flow-Lenia in NumPy float64 from the paper, using the official code only to
  resolve unspecified details (kernel formula with `R + 15`, Sobel with zero padding, α with
  θ_A = C and n = 2, `conn_from_matrix` order, wall clipping, overlap formula). No code copied;
  if code ever is, it goes to `third_party/` with provenance and is excluded from publication.
- Independent oracles in tests: scalar exact-overlap reintegration, textbook convolution for
  Sobel, direct convolution for the affinity.
- Rules are explicit or sampled from the official ranges with NumPy PCG64; the realised rule is
  part of the checkpoint and is validated on load.
- `field_u8` gains a `channels` header field; multi-channel fields are shown as RGB.
- The JAX comparison runs only in a separate environment (`scripts/compare_flow_lenia_upstream.py`);
  run with the user's approval on 2026-11-18: float64 agreement ≤ 1.1e-13 after 200 steps.

## Consequences
- Results are reproducible on CPU without JAX but are not bitwise comparable with upstream runs.
- Reintegration costs (2dd+1)² array passes per tick; a Numba/GPU backend may be added later as
  a separate backend validated against this implementation.
