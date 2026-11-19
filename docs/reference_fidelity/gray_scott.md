# Reference fidelity — Gray–Scott (`gray_scott`, plugin 0.1.0)

## Reference
J. E. Pearson, "Complex Patterns in a Simple System", Science 261:189–192 (1993).
`du/dt = Du ∇²u − uv² + F(1 − u)`, `dv/dt = Dv ∇²v + uv² − (F + k)v` on a periodic
2.5 × 2.5 domain with 256 × 256 points, Du = 2·10⁻⁵, Dv = 10⁻⁵, forward Euler with dt = 1,
initially u = 1, v = 0 except a centred 20 × 20 square with u = 1/2, v = 1/4, perturbed by
±1 % random noise. Implemented directly from the paper (no code reused).

## Implementation choices (documented)
- 5-point Laplacian `((u[i−1] + u[i+1]) + u[j−1]) + u[j+1] − 4u) / dx²` with a fixed summation
  order; forward Euler on both fields from the same old state.
- The ±1 % noise is multiplicative and applied to the whole field
  (`x ← x · (1 + 0.01 · U(−1, 1))`, PCG64(seed)); Pearson does not specify the form.
- The spec rejects `dt > dx² / (4 max(Du, Dv))` (explicit-Euler diffusion stability; 1.19 for
  Pearson's grid) and non-square domains.
- `reaction: false` is the diffusion-only negative control.

## Verified (tests/reference/test_gray_scott.py, tests/integration/test_gray_scott_platform.py)
| Property | Result |
|---|---|
| Vectorised step == scalar per-cell oracle (with and without reaction) | bit-identical |
| Laplacian of a constant is 0; of a delta is the 5-point stencil, sums to 0 | pass |
| u = 1, v = 0 is an exact fixed point (50 ticks) | exact |
| Pure diffusion conserves Σu and Σv (500 ticks) | relative error < 1e-12 |
| Coral regime (128², 4000 ticks) forms a pattern; diffusion-only control does not | area 0.1+ vs 0.0 |
| Checkpoint round trip ⇒ bitwise continuation; inspection rates match the next step | pass |
| Worker restart + seed intervention + replay verification across a periodic checkpoint | `verified` |
| GPU (`cupy`) state vs NumPy, 2000 ticks | bit-identical (metric sums differ in the last ulp) |

Pattern regimes are reported as observations (E0/E1): with Pearson's setup the presets give
v-area fractions of 0.006 → 0.10 (F 0.0367, k 0.0649, ticks 1000 → 10000) and 0.02 → 0.61
(F 0.0545, k 0.062). Pearson's figures were not digitised for a quantitative comparison.

## Measured (2026-11-19, float64, warm)
| grid | CPU (NumPy, 1 process) | GPU (CuPy, RTX 5070) |
|---|---|---|
| 256² | 0.58 ms/tick (GUI: 668 ticks/s incl. metrics) | 0.49 ms/tick |
| 1024² | 29.2 ms/tick | 0.92 ms/tick |

## Interventions / display
`seed {x, y, radius}` sets u = 0.5, v = 0.25 in a disc (Pearson's perturbation values);
`erase` restores u = 1, v = 0. Display codec `field_u8`: `v / 0.4`, clipped (display only).
