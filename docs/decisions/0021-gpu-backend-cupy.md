# ADR 0021 — Optional CUDA backend via CuPy

Status: accepted (2026-11-19)

## Context
Phase 3 allows GPU backends only after oracle validation. The machine has an RTX 5070
(Blackwell, sm_120, 12 GB) under WSL2. CPU-only setup and CI must stay viable.

## Decision
- One optional extra `gpu = ["cupy-cuda13x[ctk]>=14.2"]` (CUDA 13 toolkit from pip, ~1.9 GB).
  CuPy covers cuFFT (Lenia, Flow-Lenia) and hand-written kernels (BFF VM, Flow-Lenia
  reintegration). JAX/CUDA was not chosen: larger, and BFF's divergent control flow maps
  poorly to it. CuPy is imported lazily, only in workers/replay processes — never in the API.
- Backend `cupy` is offered by the local target only when CuPy is installed and a GPU is
  probed. It is **never selected by `auto`**; the `local_gpu` profile selects it explicitly
  (validation rejects it without a GPU or without a GPU reservation).
- Fidelity rules: integer substrates (BFF) must be bit-identical to the CPU oracle; float
  substrates (Lenia, Flow-Lenia) are a numerically different execution — tolerances against the
  NumPy oracle are measured and tested, custom kernels are bit-identical where the inputs are.
  The backend is part of each run's record (run row, checkpoint meta, provenance with CuPy
  version and GPU), and replay/verification uses the run's backend.
- GPU memory: per-process values are unavailable under WSL2 (`nvidia-smi` reports [N/A]); the
  worker reports CuPy pool bytes (held, peak) and device-wide `cudaMemGetInfo` in heartbeats,
  the live run summary and the `job_finished` event.

## Consequences
- GPU tests skip automatically on machines without CuPy/CUDA (CI). They were run locally.
- Reference-mode statements of bitwise equality with upstream (LeniaND) hold for `numpy` only.
- Not done: precision changes (no float32 policy), multi-GPU, GPU Boids/Particle Life.
