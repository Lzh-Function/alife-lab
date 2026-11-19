# Reference fidelity — Petri Dish NCA (`pd_nca`, plugin 0.1.0)

## Reference
Sakana AI, "Petri Dish Neural Cellular Automata" (2025), https://pub.sakana.ai/pdnca/, and the
authors' PyTorch code `SakanaAI/petri-dish-nca` at commit
`875523f3228b92ed5669c88aa9a2012e25fbcc60`. **The repository has no licence**: no code was copied;
the JAX implementation (`substrates/pd_nca/model.py`) was written from reading it.

## Mechanics reproduced
Channels `[aliveness of sun + N NCAs | state (attack | defence) | hidden]`; per-NCA conv nets
(k×k conv → GELU, grouped residual k×k layers, grouped 1×1 → tanh, no biases); hidden-channel
write masks `(n + j) mod Hd`; a fixed random sun update (hidden part zero); 3×3 max-pooled alive
masks with threshold 0.4 (sun always alive); ordered-pair attack/defence cosine strengths for
pairs alive together somewhere; softmax over alive entities; cell update `x += Σ w_m u_m`;
aliveness = softmax(w / temp) over alive entities, re-thresholded and renormalised; clamp to
[−1, 1]. Learning: per-entity perspectives in which only the entity's own updates carry gradient,
loss `−Σ_m asinh(Σ aliveness_m + 1e−3)`, global gradient-norm clipping (1.0), SGD / RMSprop /
Adam with PyTorch default constants; the sun vector is not trained (reference default).

## Documented deviations
- **Output channel layout**: each NCA owns a contiguous block of the grouped 1×1 output. The
  reference reshapes the grouped output with the einops pattern `(oc m)`, which interleaves
  channels across groups (an NCA's update mixes channels produced by other groups' weights);
  we treat this as an implementation detail and keep per-NCA blocks, so perspectives isolate
  gradients exactly (tested).
- **Extinction revert** (`revert_on_extinction`) is implemented but **off by default**: with it on,
  our runs froze (the world reverted on every epoch after an NCA died; reference-example scale,
  seed 0, from epoch ~5). Whether the reference shows the same behaviour was not checked (that
  would require running the unlicensed PyTorch code).
- Batch = pool = 1 (one persistent dish, as in the reference example); no burn-in schedule; no
  dropout; JAX float32 instead of PyTorch (bfloat16 autocast on CUDA in the reference).
- Comparison with the reference code was **not** run; agreement is at the level of the described
  mechanics only.

## Verified (tests/reference/test_pd_nca.py, tests/integration/test_pd_nca_platform.py)
| Property | Result |
|---|---|
| Aliveness is a distribution per cell; values within [−1, 1]; territories sum to 1 | pass |
| Gradient isolation: in perspective p, other NCAs' weights get exactly zero gradient | pass |
| Hidden write mask (per_hid_upd, offset by NCA index) | pass |
| `frozen` keeps weights; `online` changes them; `uniform` gives a zero learning signal | pass |
| Deterministic runs; checkpoint (grid, weights, optimizer state) ⇒ bitwise continuation | pass |
| Worker restart + erase intervention + replay verification across a checkpoint | `verified` |

## Controls (48², 4 NCAs, epoch 300, seeds 0 / 1 / 2) — observations (E1)
| condition | alive NCAs | sun share | effective number of NCAs |
|---|---|---|---|
| online learning | 3 / 4 / 4 | 0.34 / 0.36 / 0.45 | 2.95 / 3.79 / 3.64 |
| frozen weights | 0 / 1 / 3 | 1.00 / 0.44 / 0.30 | — / 1.00 / 2.11 |
| no competition (uniform) | 4 / 4 / 4 | 0.50 / 0.49 / 0.49 | 3.85 / 3.66 / 3.82 |

## Measured (2026-11-19)
Default (48², 4 NCAs, 1 step per epoch): 59 ms/epoch on the CPU, 8.5 ms on the GPU
(`jax_cuda`); reference example (50², 15 NCAs, state 64, 9×9 kernels, one residual layer):
120 ms/epoch on the GPU, XLA peak 1.6 GB. Worker RSS on the CPU backend in the GUI: ~0.95 GB.
