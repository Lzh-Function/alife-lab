# ADR 0022 — JAX for Neural Cellular Automata (CPU and CUDA builds)

Status: accepted (2026-11-19)

## Context
Growing NCA needs automatic differentiation. Candidates were JAX, PyTorch or a hand-written
NumPy backward pass. Later phases (evosax, QDax, MJX) are JAX-based, and the project relies on
explicit RNG state for deterministic replay.

## Decision
- JAX, chosen by the project owner over PyTorch. Optional extras: `nca` (CPU build, ~450 MB with
  SciPy) and `nca-gpu` (`jax[cuda13]`, ~4 GB of CUDA libraries). Imported only in workers and
  replay processes.
- Training is a separate optimisation substrate (`nca_train`, clock `optimizer_step`) with an
  explicit objective; trained weights run in the `nca` substrate with the weights embedded in
  the spec (hashed). Export is NumPy-only in the controller.
- Backends `jax` (CPU) and `jax_cuda`; device placement is explicit per adapter. Profiles: the
  new selector value `gpu` picks the substrate's first GPU backend (`cupy` or `jax_cuda`); `auto`
  never picks a GPU backend.
- Substrates may restrict backends to numeric policies (`backend_numeric_policies`); NCA allows
  `jax_cuda` only with `float32_jax`, so GPU rounding is never applied to a spec that declared
  CPU-only numerics.
- Defaults are smaller than the article (56², pool 256) for CPU use; an article-scale preset
  (72², pool 1024, 8000 steps) is provided for the GPU.

## Consequences
- CPU-only CI keeps working (JAX tests skip); GPU tests run locally.
- Worker RSS with JAX is ~0.4 GB; XLA GPU memory is reported from `memory_stats`.
