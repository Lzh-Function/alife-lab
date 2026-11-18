# Reference fidelity — Flow-Lenia (`flow_lenia`, plugin 0.1.0)

## Reference
- E. Plantec, G. Hamon, M. Etcheverry, P.-Y. Oudeyer, C. Moulin-Frier, B. W.-C. Chan,
  "Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation
  and parameter localization", ALIFE 2023 (arXiv:2212.07906).
- Authors' JAX code: `erwanplantec/FlowLenia` at commit
  `dce428c6b0c5079a06e5606fb7b5ac1fe1323bc5` (`flowlenia/flowlenia.py`, `utils.py`,
  `reintegration_tracking.py`). **The repository has no licence**, so no code was copied: the
  NumPy implementation in `src/alife_lab/substrates/flow_lenia/core.py` was written from the
  paper and from reading that code to pin down details the paper leaves open.

## Reference mode = semantics of `FlowLenia.__call__` (no parameter embedding)
- Kernels: `D = |x| / ((R + 15) r_k)`,
  `K_k = sigmoid(-10 (D - 1)) · Σ_j b_kj exp(-(D - a_kj)² / w_kj)`, `sigmoid(x) = (tanh(x/2)+1)/2`,
  normalised to sum 1, origin at cell (0, 0), periodic FFT convolution.
- Affinity: `U_c = Σ_{k: target_k = c} h_k (2 exp(-((K_k * A_source_k − m_k)/s_k)²/2) − 1)`.
- Flow: `F = (1 − α) ∇U − α ∇(Σ_c A_c)`, `α = clip((A / θ_A)^n, 0, 1)`, θ_A = channels and n = 2
  as in the official code; ∇ = zero-padded Sobel ("same" convolution) **also on the torus**.
- Reintegration tracking: `μ = pos + clip(dt F, −(dd − σ), dd − σ)` (wall: μ clipped to
  `[σ, size − σ]`); each target cell receives `A · Π_axis clip(½ − |pos − μ| + σ, 0, min(1, 2σ))
  / (4σ²)` (torus: minimum-image distance). Defaults dt 0.2, dd 5, σ 0.65, wall.
- Kernel connectivity from a C×C count matrix in `conn_from_matrix` order.

## Documented deviations
- **Numerics:** float64 NumPy vs. float32 JAX upstream. Results are not expected to be
  bitwise equal to the JAX code.
- **Random rules:** parameters are drawn from the official uniform ranges with NumPy PCG64
  (SeedSequence(seed) → rule stream, init stream), not with JAX's RNG, so "seed 0" here is not
  "key 0" there. The realised rule is recorded in checkpoints and in the inspector.
- **Rectangular worlds** use per-axis sizes for the wall clip and the kernel grid (upstream uses
  `SX` for both axes and `mgrid[-mid:mid]`, i.e. assumes square, even worlds). For square even
  worlds the kernel grid is identical (tested).
- Not implemented: parameter embedding (`flowlenia_params.py`) and its mixing rules.

## Verified (tests/reference/test_flow_lenia.py, tests/integration/test_flow_lenia_platform.py)
| Property | Result |
|---|---|
| Sobel == textbook zero-padded 3×3 convolution with the official kernels | atol 1e-14 |
| Vectorised reintegration == scalar exact-interval-overlap oracle (25 Hypothesis cases, σ ∈ {0.3, 0.5, 0.65, 1.2}, wall/torus, 2 channels) | atol 1e-13 |
| Total mass conserved for 150 steps (wall and torus) | relative error < 1e-12 |
| Kernel grid == official `fftshift(mgrid[-mid:mid])` construction | atol 1e-15 |
| Affinity (FFT) == direct periodic convolution | atol 1e-12 |
| `conn_from_matrix` order; seeded rule sampling within the official ranges | pass |
| Checkpoint round trip ⇒ bitwise identical continuation; checkpoint of another rule rejected | pass |
| Worker restart + channel paint + replay verification across a periodic checkpoint | `verified` |

**Not yet verified:** side-by-side comparison with the official JAX code (needs JAX, Equinox
and jaxtyping in a separate environment and running the downloaded code — pending the user's
approval). Expected outcome is agreement within float32 tolerance for the same explicit rule.

## Negative control
`flow_lenia_diffusion_control`: θ_A = 0.001 makes α = 1 wherever mass exists, so the flow is
pure diffusion down the total-mass gradient. Same seed and rule as `flow_lenia_example`
(128², 3 channels), measured 2026-11-18:

| tick | example: alive_fraction / max_value / centroid | control: alive_fraction / max_value / centroid |
|---|---|---|
| 100 | 0.133 / 3.38 / 0.86 | 0.413 / 0.37 / 0.007 |
| 600 | 0.163 / 4.98 / 11.4 | 0.916 / 0.11 / 0.006 |

Mass drift in both: |Δm/m| ≤ 2e-14 after 600 ticks.

Single-channel scan (24 seeds, torus, 600 ticks): most sampled rules settle into a stationary
blob (centroid displacement < 1 cell); seed 19 drifts ~7 cells. Three-channel scan with the
official example connectivity: seeds 0 and 2 drift steadily (~12 and ~9 cells by tick 600).
These are observations (E0/E1), not claims about creatures.

## Measured (2026-11-18, NumPy 2.5.4, x86_64, single process, dd = 5)
| world | channels | ms/tick | ticks/s |
|---|---|---|---|
| 64² | 1 | 4.8 | 207 |
| 128² | 1 | 12.8 | 78 |
| 128² | 3 | 46.4 | 21.6 |
| 256² | 3 | 249.7 | 4.0 |

Worker RSS in the GUI (128², 3 channels): 61.3 MB. Cost is dominated by the (2dd+1)² = 121
shifted passes of reintegration tracking.

## Interventions / display
- `paint {x, y, radius, value, channel?}` / `erase {x, y, radius, channel?}` set values in a disc
  (all channels by default). They change the total mass, so `mass_drift` is re-based.
- Display codec `field_u8` with `channels`: `round(clip(A, 0, 1)·255)` per channel; 1 channel →
  colormap, 2–3 channels → RGB. Values above 1 (mass concentrates) saturate in the preview only.
