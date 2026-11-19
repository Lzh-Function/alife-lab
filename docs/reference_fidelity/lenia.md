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
| Adapter == original LeniaND.py CPU path (see below) | bitwise equal, 1000 steps |
| Orbium (128², T=10) keeps mass within ±15 % for 600 ticks and moves | pass |
| Checkpoint round trip ⇒ bitwise identical continuation | pass |
| Worker restart + paint intervention + replay verification across a periodic checkpoint | `verified` |

## Upstream comparison (scripts/compare_lenia_upstream.py, run 2026-11-18)
The original `LeniaND.py` (commit above) was executed locally with its GUI/GPU imports stubbed
and the GPU path disabled (CPU `numpy.fft`), side by side with the adapter, starting from the
same Orbium board:

| world | steps | initial state | kernel FFT | max abs difference (all steps) | final SHA-256 of float64 state (both) |
|---|---|---|---|---|---|
| 128² | 1000 | equal | equal | 0.0 | `3c5ee630b2eb566a5565bd038e430659f5221eb0a8fdcc4c0a0c0decc8b29a3c` |
| 256² | 500 | equal | equal | 0.0 | `a55f95d94e295439d0408fc967f8ea10bcb8872eaa16bd5b4b8d4bb3828b3fd4` |

NumPy 2.5.4, x86_64. Bitwise equality depends on the FFT implementation and SIMD dispatch, so
the digests are evidence for this environment, not portable fixtures.

Finding during the comparison: the first version of the port computed
`kfft * np.fft.fftn(A)`. NumPy elides the temporary and evaluates `fftn(A) * kfft` in place;
complex multiplication is not bitwise commutative in NumPy's SIMD loops, which produced 1-ulp
differences from step 1 (max 5.4e-6 after 1000 steps). The port now keeps `world_fft` as a
named array and multiplies `kfft * world_fft` like LeniaND; a test pins this order.

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

## GPU backend (`cupy`)
The same reference functions run on CuPy arrays (NumPy → CuPy dispatch; cuFFT, float64). The
field stays on the device; only marginals/metrics, previews and checkpoints are copied back.
**Not bitwise equal** to the NumPy oracle (different FFT library): measured max |difference|
on Orbium 128² was 7e-16 after 1 tick, 3e-15 after 10, 6e-14 after 100 and 3e-6 after 1000
ticks (sensitivity of the dynamics). Tests: ≤ 1e-13 after 10 ticks, ≤ 1e-10 after 100; GPU runs
are bitwise reproducible on the same device; Orbium survives 600 ticks; checkpoints continue
bitwise; worker run + replay verification with the recorded backend `verified`. Throughput
(float64, warm): 128² 0.28 ms/tick, 512² 0.41, 1024² 1.34 (CPU 32.2), 2048² 6.1.
A GPU run is therefore a numerically different execution of the same spec; the backend is
recorded in the run, its checkpoints and its provenance, and replay uses the same backend.
