# Reference fidelity — Growing NCA (`nca_train`, `nca`, plugins 0.1.0)

## Reference
A. Mordvintsev, E. Randazzo, E. Niklasson, M. Levin, "Growing Neural Cellular Automata",
Distill (2020), https://distill.pub/2020/growing-ca/. Implemented in JAX from the article; no
code reused. Targets are procedurally drawn shapes (heart, disc, ring, letter A) or images
drawn in the browser — the article's emoji targets are not bundled.

## Model as implemented
- 16 channels (RGBA + 12 hidden), depthwise perception [identity, Sobel-x/8, Sobel-y/8] with
  zero padding, dense 48→128 ReLU → 128→16 (output layer zero-initialised, first layer
  Glorot-uniform), stochastic update with probability 0.5 per cell, alive masking with the
  3×3 max of alpha > 0.1 before and after the update.
- Training: seed = one cell with channels 3.. set to 1; loss = RGBA MSE to the target padded
  by `pad` cells after n ~ U[steps_min, steps_max] steps (computed as a static scan of
  steps_max with a mask, so only one compilation is needed); per-variable gradient L2
  normalisation; Adam (β 0.9/0.999, ε 1e-7) with lr 2e-3 → ×0.1 after `lr_decay_step`.
  Pool (persistent/regenerating): sample a batch, sort by loss, replace the worst with the
  seed; regenerating: erase a random disc (radius 0.1–0.4 of the side, centre in the middle
  40 %) in the `damaged_per_batch` lowest-loss samples.

Documented deviations: the default world is 40 + 2×8 = 56 cells (article: 40 + 2×16 = 72);
pool 256 (article 1024) to keep CPU checkpoints small; float32 JAX on the CPU platform (forced),
not TensorFlow on a GPU. Results are not expected to match the article's numbers exactly.

## Verified (tests/reference/test_nca.py, tests/integration/test_nca_platform.py)
| Property | Result |
|---|---|
| Perception == NumPy identity/Sobel convolution; alive mask = 3×3 max-pool > threshold | atol 1e-6 |
| Untrained weights: zero output layer ⇒ state unchanged for 30 steps | exact |
| Training is deterministic (two runs, 40 steps: weights and pool identical) | bitwise |
| Loss decreases (tiny config, 40 steps: last-5 mean < 0.7 × first-5 mean) | pass |
| Training checkpoint round trip (weights, Adam moments, pool, both RNGs) ⇒ bitwise continuation | pass |
| Export → `nca` spec with weights + sha256; `nca` checkpoint round trip bitwise | pass |
| Platform: training in a worker + replay verification; export; nca run + damage + replay verification | `verified` |
| Bundled 24² heart model regrows after damage; untrained control never approaches the target | pass |

## Regeneration evidence (bundled model `nca_heart24_pretrained`, seed 0)
Trained with `regenerating`, 24 × 24 heart, pad 6, batch 8, pool 128, 48–64 steps, 1500
optimizer steps (CPU, 6 min; final batch loss 0.0083). Target MSE: 0.0049 after 200 ticks;
erasing a radius-6 disc (113 cells) raises it to 0.0415; after 20 / 50 / 100 / 300 ticks it is
0.0069 / 0.0053 / 0.0056 / 0.0071. Untrained control: 0.117 throughout. This is evidence of
learned regeneration of a trained shape (E2 within this assay), not of open-ended life.

## Measured (2026-11-19, JAX 0.12.0.1 CPU, x86_64)
- Training step (56², batch 8, scan of 96 steps): 0.74 s; (36², batch 8, 64 steps): ~0.24 s.
- Worker RSS with JAX loaded: ~413 MB. `nca` run (36²): hundreds of ticks per second.
- GPU (`jax_cuda`, `nca-gpu` extra: jax[cuda13] 0.12.0.1, RTX 5070): article scale (72²,
  pool 1024, batch 8, 64–96 steps, regenerating) 0.16 s/step (~21 min for 8000 steps), XLA peak
  3.3 GB (preallocation disabled; XLA's autotuner logged one failed 4 GiB scratch allocation and
  fell back). Checkpoint at step 21: 5 MB compressed.

## GPU backend (`jax_cuda`)
- Placement is explicit per adapter (`jax.default_device`), so installing the CUDA build does not
  move CPU runs. `--xla_gpu_deterministic_ops=true` is set; preallocation is disabled.
- The GPU is only allowed with `numeric_policy: float32_jax` (device-agnostic); a spec declaring
  `float32_jax_cpu` is rejected on the GPU instead of being silently run with other rounding.
- Verified (tests/reference/test_nca_gpu.py, tests/integration/test_nca_platform.py): two GPU
  training runs are bitwise identical; GPU vs CPU weights after 5 steps differ by < 5e-3
  (measured 2.9e-4 on the small config); the bundled model regrows on the GPU; a `local_gpu`
  run resolves to `jax_cuda`, records jax/jaxlib/plugin versions and the GPU, reports XLA memory
  statistics and passes replay verification with the recorded backend.
