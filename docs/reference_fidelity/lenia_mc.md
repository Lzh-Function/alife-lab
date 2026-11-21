# Model card — Multichannel Lenia (`lenia_mc`, plugin 0.1.0)

Rule of SakanaAI/asal `substrates/lenia_impl.py` @ 677ba0e (Apache-2.0), itself from
maxencefaldor/Leniabreeder (MIT), a version of Bert Chan's multichannel Lenia ("Lenia and
Expanded Universe", 2020; MIT). Creatures 5N7KKM (Aquarium swarm, its twin and gyrating
variants) and VT049W (self-replicating Aquarium) by Bert Chan; data extracted as literals into
`substrates/lenia_mc/patterns.json` with its source note.

## Rule
K_k: Gaussian-bump shells `b[floor(D_k)] · bell(D_k mod 1, 0.5, 0.15)`, D_k = |x|/R · len(b)/r,
normalised; U_k = K_k * A[c0] (periodic FFT); G_k = h_k (2 bell(U_k, m_k, s_k) − 1);
A[c] ← clip(A[c] + (1/T) Σ_{k: c1=c} G_k, 0, 1).
Differences from Chan's `LeniaNDKC.py` @ adfc542: kernel core exp(4 − 1/(r(1−r))) there,
and growth divided by Σh per target channel there. For the catalogued creatures Σh = 2 for
every channel, so Leniabreeder's update equals Chan's normalised one with the step doubled.
Leniabreeder re-centres the world each step (a cyclic roll); here the world stays fixed.

## Verified (tests/reference/test_lenia_mc.py, tests/integration/test_lenia_mc_platform.py)
| Property | Result |
|---|---|
| Initial world from the pattern equals upstream | bitwise |
| Normalised kernels vs upstream | ≤ 3.8e-17 |
| World vs upstream (JAX 0.12, float64, CPU), aligned by upstream's re-centring shift | 1e-15 after 1 tick; 5N7KKM 2.7e-9 / VT049W 2.8e-12 after 256 ticks |
| Divergence later (5N7KKM 0.14 after 500 ticks) matches the growth of a 1e-15 self-perturbation (2e-9 at 200, 0.6 at 500): chaotic amplification of rounding, not a rule difference | pass |
| Fixture (upstream masses every 16 ticks, 200 sampled cells at 50/100/200) | pass, CPU-only |
| Translation equivariance; values in [0, 1]; kernels sum to 1 | pass |
| Negative controls: no cross-channel kernels → spreads (alive fraction > 0.4); m × 1.15 → dissolves | pass |
| PyTorch float64 batch = NumPy within 1e-9 after 120 ticks (frames, mass, speed) | pass |
| Worker restart + channel intervention + replay verification | `verified` |

Upstream was run once with the owner's permission (scripts/compare_lenia_mc_upstream.py).

## Measured
NumPy CPU: 3.5 ms per tick (128², 3 channels, 15 kernels). Torch CUDA float64 batch:
16 candidates × 256 ticks in about 1.0 s after warm-up (RTX 5070) vs 9.5 s with NumPy.

## Limitations
Only catalogued creatures from the ASAL/Leniabreeder file; Chan's own normalised variant is not
implemented as an option; the batched torch path is not used for ordinary runs.
