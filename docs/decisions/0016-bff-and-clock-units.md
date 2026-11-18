# ADR 0016 — BFF substrate (CuBFF-compatible) and substrate clock units

- Status: accepted (Phase 3, slice 1)
- Date: 2026-11-18

## Decision
- Implement BFF from the CuBFF source semantics (see `docs/reference_fidelity/bff.md`),
  including its stateless SplitMix64 randomness, so that a reference run is reproducible
  bit-for-bit and can later be checked against a CuBFF binary byte-for-byte.
- Two implementations: a scalar oracle and a NumPy batch interpreter, tested for exact
  equality; the batch hands the last ≤ 512 running tapes to the scalar interpreter.
- Clock units: the manifest's `clock_unit` (`tick` | `epoch` | …) is now used for run
  summaries, events, checkpoints, acks, timelines and GUI labels. The DB column `runs.tick`,
  message keys and metric row keys keep the name `tick` as a generic step index.
- New dependency: `brotli` (Google, MIT) for CuBFF's complexity metric.

## Measured performance (single core, 2026-11-18, early random soups)
| Programs | s / epoch |
|---|---|
| 4 096 | 0.21 |
| 32 768 | 5.3 |
| 131 072 (CuBFF default) | 8.3 |
The paper's takeover happens after thousands of epochs at 131 072 programs, i.e. days on one
CPU core with this backend. Larger soups need a compiled/GPU backend (Phase 3 GPU, after the
oracle comparison) — not attempted yet.
