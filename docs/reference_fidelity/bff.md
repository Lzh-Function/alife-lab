# Reference fidelity — BFF primordial soup (`bff`, plugin 0.1.0)

## References
- Agüera y Arcas, Alakuijala, Evans, Laurie, Mordvintsev, Niklasson, Randazzo, Versari (2024),
  *Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple
  Interaction*, arXiv:2406.19108.
- CuBFF, github.com/paradigms-of-intelligence/cubff, commit
  `f212e849027c98fcf4b242eccfb5fed435223e23` (2025-08-17), **Apache-2.0**. Source files read:
  `bff.inc.h`, `bff.cu`, `bff_noheads.cu`, `common_language.h`, `common.h`, `main.cc`.
  No CuBFF code is vendored; the algorithm was re-implemented from reading the source and
  then checked byte-for-byte against a CuBFF binary (see below).

## Where the paper is under-specified and the implementation decides
| Point | Paper | CuBFF (followed here) |
|---|---|---|
| Initial heads | not stated | `bff_noheads`: both 0 (`2·64` masked by 127); `bff`: `tape[0] % 128`, `tape[1] % 128`, pc = 2 |
| Head wrap | not stated | heads masked with 127 at the start of every step |
| Unmatched bracket | terminates | `[`: pc = 128 (stop); `]`: pc = −1 (stop) |
| Step limit | 2^13 | 8 · 1024 steps per pair |
| Mutation | 0.024 % | each of the 128 pair bytes independently: `r = SM((N·seed(e)+k)·128+i)`; replace with `r & 0xFF` if `(r>>8) & (2^30−1) < 2^18` |
| Pairing | random pairs | identity permutation + Fisher–Yates with `j = SM(seed(e·N+i)) mod (i+1)`; pairs (s[2k], s[2k+1]) |
| Seeds | — | `seed(x) = SM(SM(flag_seed) ^ SM(x))`, SM = SplitMix64; init byte `SM(64·N·seed(0)+64·p+i) mod 256` |
| Complexity | h0 − K/n | h0 (byte entropy, bits) − brotli(quality 2, lgwin 24, generic) bits/byte |

## Implementation
- `vm.py`: `evaluate_scalar` (line-by-line port, reference oracle) and `evaluate_batch`
  (NumPy lockstep interpreter; finishes the last ≤ 512 running tapes with the scalar
  interpreter from their exact state). Programs are data; nothing runs on the host.
- `soup.py`: vectorised SplitMix64 (uint64 wrap), init, shuffle, mutation, epoch; plus a pure
  Python reference epoch.

## Verified
| Property | Result |
|---|---|
| Instruction semantics (heads, wrap, ±, copies, loops, unmatched brackets, step limit, `bff` head init) | pass |
| Batch == scalar oracle: 300 Hypothesis tapes (instruction-heavy), 400 uniform random tapes, pure-vector and hybrid paths | pass (bytes and ops identical) |
| SplitMix64 first output for state 0 = `0xE220A8397B1DCDAF` | pass |
| Vectorised epochs == pure-Python reference epochs (6 epochs with high mutation; `bff` variant) | pass |
| Platform (worker, checkpoints) soup after 30 epochs == direct procedure | pass |
| Restore + replay verification with an injected program | pass |

| **Byte-identical soups vs. a CuBFF binary** (CPU build of the pinned commit): `bff_noheads` 1024 programs seed 7, `bff` 512 seed 3, `bff_noheads` 256 seed 11 with mutation 0.01 — after 1, 65, 129 and 193 epochs | **pass: 0 differing bytes** (fixture digests in `tests/fixtures/cubff_soups.json`, regenerate with `scripts/regenerate_cubff_fixtures.sh`) |

External check performed on 2026-11-18 with the user's approval to build and run CuBFF
(`make CUDA=0`, g++ + OpenMP + libbrotli 1.1.0) in an isolated scratch directory; only
SHA-256 digests of the resulting soups are stored in the repository.

### Speed relative to CuBFF (4096 programs, early soup, 2026-11-18)
CuBFF CPU build: ≈ 2.0 ms/epoch on one thread (7.8 ms with 24 OpenMP threads at this small
size). This NumPy backend: ≈ 210 ms/epoch on one core — about 100× slower. A compiled backend
(e.g. Numba or wrapping CuBFF as an external backend) would be needed for paper-scale runs.

## Deviations / options (explicit)
- `num_programs` defaults to 4096 here (CuBFF default 131072) for CPU interactivity; it is a
  CuBFF flag, so runs remain reference reproductions of the algorithm at that size.
- `step_limit = 0` (non-execution control) and `init = zero` are exploratory-only.
- Brotli 1.2.1 is used; the paper reports brotli 1.1.0 — compressed sizes may differ
  slightly, so `brotli_bits_per_byte` is not compared across library versions.
- Not implemented: `allowed_interactions` (spatial soups), `fixed_shuffle`, `reset_interval`,
  CuBFF's `eval_selfrep`, other languages (Forth, SUBLEQ).

## Evidence
Metrics are passive. `top_program_fraction` and the inspector's self-pair test are hints,
not replication evidence (spec §9.2); E-levels are assigned by review.

## Observation log (exploratory, not a claim)
2026-11-18, reference semantics, `bff_noheads`, 4096 programs, mutation 2^18/2^30, seeds 0 and
1, 5000 epochs each (≈ 17.6 min per run on one core):

| seed | max higher-order entropy (epoch) | final HOE | final distinct programs | ops/pair epoch 1 → 5000 |
|---|---|---|---|---|
| 0 | 0.373 (651) | 0.078 | 4096 / 4096 | 82 → 276 |
| 1 | 0.338 (1201) | 0.072 | 4096 / 4096 | 81 → 266 |

No takeover by copies (every program stays distinct; no sharp entropy transition) within
5000 epochs at this soup size. This is a null result for these two seeds only — the paper's
transitions were reported for 2^17 programs, and the waiting time is stochastic. Execution
activity (ops per pair) roughly tripled.

## VM debugger (trace)
`substrates/bff/trace.py` records every step of one pair evaluation (pc, instruction, heads at
the start of the step, write address/old/new value, next pc, halt reason). It is a view of the
oracle loop, not a second semantics:

| Property (tests/reference/test_bff_trace.py) | Result |
|---|---|
| Final tape and ops == `evaluate_scalar` (300 Hypothesis tapes, both variants, limits 0..2000) | pass |
| Replaying the write log from the initial tape reproduces the final tape; next_pc chains | pass |
| Halt reasons: step limit, end of tape, unmatched `[`, unmatched `]` | pass |
| `next_epoch_pair` tape (shuffle + per-pair mutation) traced == `run_epoch` result for that pair | pass |

The "next-epoch pair" trace shows exactly the evaluation the next epoch performs for the
selected program (including background mutation of that pair). The default program of the
stand-alone debugger (`80c05b2e3e7d5d`, `bff` variant) is a hand-written 7-byte copier, not
an evolved replicator.

## Replicator assay (`substrates/bff/assay.py`, assay_version 1)
Passive test on *copies* of a program (the run is never modified). Pre-registered rule, stored
with each result: `replicates_in_assay` iff in a strict majority of replicates the final exact
copy count of P is ≥ growth_factor × copies (default 4 × 16) while the byte-scrambled control's
is not; `inconclusive` if the scrambled program equals P or the control also grows; otherwise
`no_evidence`. All conditions of a replicate share the same random background soup (paired).
Defaults: soup 1024, 16 copies, 64 epochs, 3 replicates, mutation 0 for manual programs (a run's
own variant/step limit/mutation rate when assaying a run program), 256 random pair partners.

Measured 2026-11-18 (tests/reference/test_bff_assay.py, numba and numpy backends agree):

| program | outcome | soup assay finals (P / scrambled), 3 replicates |
|---|---|---|
| top program of run_2fdbc1db12984e53 at epoch 4795 (99 copies in 131072) | replicates_in_assay | 75/4, 102/1, 179/3 (from 16) |
| 2nd and 3rd most frequent programs (mutation 0) | replicates_in_assay | 164/0, 120/0, 109/0 · 206/3, 187/0, 117/1 |
| 2nd program with the run's mutation rate | no_evidence | 72/0, 26/0, 12/0 (threshold 64) |
| hand-written 7-byte copier (`bff` variant) | no_evidence | 0/0 in all replicates |
| all-NUL program | inconclusive | scrambled == program |

Observation (E1): the evolved programs do **not** copy themselves within a single pair
evaluation — with random partners (exact and near copy rate 0 %), with an all-NUL partner, or
with themselves (P+P leaves the second half at Hamming distance 62). Their growth in the soup
therefore depends on multi-interaction dynamics with other soup programs. The mechanism has not
been analysed; the debugger's next-epoch trace is the intended tool for that.

## GPU backend (`cupy`, `vm_cupy.py`)
CUDA kernels via CuPy 14.2.0 (CUDA runtime 13.2, pip toolkit), NVIDIA GeForce RTX 5070
(compute capability 12.0), driver 610.88. One thread per pair: gather `soup[s[2k]] ++ soup[s[2k+1]]`,
CuBFF background mutation (SplitMix64 on the device), evaluation, scatter. The Fisher–Yates
permutation stays on the host (sequential). Integer semantics ⇒ the backend must be exact:

| Property (tests/reference/test_bff_cupy.py, tests/integration/test_bff_platform.py) | Result |
|---|---|
| Kernel == scalar oracle (800 tapes × 2 variants) | bit-identical |
| 25 epochs with mutation == NumPy backend (soup, ops, mutated bytes) | bit-identical |
| CuBFF binary fixtures (3 configs, up to 193 epochs) | SHA-256 equal |
| Worker run on `local_gpu` vs `cpu_dev` (comparison) and replay verification | distance 0, `verified` |

Measured 2026-11-18, mutation 2^18 / 2^30, warm (single epoch wall time, host↔device copies
included): 4096 programs 3.94 ms/epoch (numba 5.49); 131072 programs **9.09 ms/epoch** (numba,
all cores, 96.6); 1048576 programs 57.1 ms/epoch. CuPy pool held 80 MiB at 2^20 programs.
