# Reference fidelity — Lenia (`lenia`, plugin 0.1.0)

## Reference
- B. W.-C. Chan, "Lenia — Biology of Artificial Life", Complex Systems 28(3), 2019
  (arXiv:1812.05433).
- Implementation: `Chakazul/Lenia`, `Python/LeniaND.py` at commit
  `adfc542939266de7f4bb7ebb552e8499701ee107` (MIT). Its core functions are ported in
  `src/alife_lab/substrates/lenia/reference.py` with the licence notice; Orbium unicaudatus
  (code `O2u`) and its parameters are copied from that commit's `animals.json`.

## Reference mode = LeniaND default CPU path (2D)
- Kernel: `K(d) = (d < 1) · core(B·d mod 1) · b[⌊B·d⌋]` with d = distance / R, centred at
  `MID = size // 2`, normalised to sum 1, transformed once with `np.fft.fftn`.
- Potential: `U = fftshift(real(ifftn(K̂ · fftn(A))))` (periodic boundary).
- Update: `A ← clip(A + G(U; m, s) / T, 0, 1)`, float64.
- Cores: polynomial `(4r(1−r))⁴`, exponential `exp(4 − 1/(r(1−r)))`, step, staircase.
  Growth: polynomial `max(0, 1 − (n−m)²/(9s²))⁴·2 − 1`, gaussian, step.
- Not ported (documented deviations, unavailable in reference mode): multi-step integration,
  soft clip, noise, update masks, value quantisation `P`, GPU (reikna, complex64), N-D worlds,
  the Analyzer statistics and creature tracking.
- LeniaND's GPU path computes in complex64/float32; this port follows its CPU fallback
  (complex128/float64). Results therefore do *not* match LeniaND runs with a GPU enabled.

## Verified (tests/reference/test_lenia.py, tests/integration/test_lenia_platform.py)
| Property | Result |
|---|---|
| FFT potential == independent direct periodic convolution (4 cores × 2 shell sets, 40×48) | pass (atol 1e-12) |
| Kernel sums to 1, zero for d ≥ R, mirror symmetric | pass |
| Growth functions within [−1, 1], G(m) = 1 | pass |
| Adapter step == ported `calc_once` for 50 steps | bitwise equal |
| Orbium (128², T=10) keeps mass within ±15 % for 600 ticks and moves | pass |
| Checkpoint round trip ⇒ bitwise identical continuation | pass |
| Worker restart + paint intervention + replay verification across a periodic checkpoint | `verified` |

**Not yet verified:** a direct run of the original `LeniaND.py` against this port. A script that
executes the upstream file with its GUI/GPU imports stubbed was prepared but not run, because
running downloaded third-party code needs the user's explicit approval. Until then the
bitwise claim is "equal to our transcription of LeniaND's update path", not "equal to LeniaND".

## Negative control
`lenia_orbium_off_niche`: the same Orbium with the growth centre m = 0.17 (s = 0.015) dissolves
completely (mass 0.0 at tick 500), while m = 0.15 keeps mass ≈ 74. Lenia with a pattern
initialisation has no stochasticity, so seeds do not vary this result. m = 0.13 still survived
for 500 ticks (mass ≈ 70) — Orbium's niche is not symmetric around 0.15 in m.

## Measured (2026-11-18, NumPy 2.5.4, x86_64, single process, polynomial kernel R = 13)
| world | ms/tick | ticks/s |
|---|---|---|
| 128² | 0.36 | 2806 |
| 256² | 1.11 | 901 |
| 512² | 7.96 | 126 |
| 1024² | 32.18 | 31 |

Worker RSS in the GUI run (128²): 53.5 MB. Orbium centroid speed in this setup is about
0.63 cells/tick (≈ 0.48 R per unit time T); not compared against a published value.

## Interventions / comparison / display
- `paint {x, y, radius, value}` and `erase {x, y, radius}`: a disc on the torus, applied at
  the tick boundary; the centroid tracker is reset.
- `state_distance`: `max_abs_difference` with `exact_equal` (array equality).
- Display codec `field_u8`: `round(A·255)`; display only, never used for metrics or state.
- `centroid_displacement` is the unwrapped motion of the mass-weighted circular mean over
  the whole world; it is not a creature tracker and is meaningless with several patterns.
