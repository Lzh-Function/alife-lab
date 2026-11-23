# ADR 0046: Phase F9 — Parallel Multi-Fly Embodied Simulation (MuJoCo Warp + Batched GPU LIF)

- **Status:** Accepted
- **Date:** 2026-11-23
- **Related:** ADR 0043 (Embodied Fly Level 1), `docs/reference_fidelity/connectome_fly.md`, `docs/plans/connectome_plan.md`

## Context

Phase F8 runs one 138,639-cell FlyWire v783 brain coupled to one CPU float64 MuJoCo fly body at `1.8×–2.7×` real time. Large-scale embodied behavioral silencing screens, multi-seed sweeps, and evolutionary foraging searches require stepping tens of independent embodied flies concurrently while preserving CPU-only CI viability and exact CPU float64 verification of key findings.

## Decisions

1. **Optional dependency (`embodied-gpu` extra):**
   - Add `mujoco-warp>=3.17` (Apache-2.0, pulling `warp-lang 1.19.0`) strictly under `[project.optional-dependencies] embodied-gpu` in `pyproject.toml` so CPU-only CI and default installs remain unaffected.
   - Ignore optional `mujoco_warp` / `warp` imports in `mypy` overrides.

2. **Minimal-overhead batched brain/body coupling (`lif_cupy.py`, `embodied_batch.py`):**
   - Extend `CupyBatchedLIFRunner` with `< 4 KB` subset transfers (`upload_stim_subset_batch`, `read_total_spikes_subset`) and read per-tick command spike deltas from `d_total_spikes` masked by active (non-silenced) output neurons — zero CUDA kernel modifications required.
   - Capture 10 MuJoCo Warp substeps (`1.0 ms` tick) into a single CUDA Graph (`wp.ScopedCapture`) across `nworld = B` parallel 6-legged fly bodies.
   - Provide an identical `physics_backend="cpu"` + `brain_backend="numpy"` code path so CPU CI and deterministic re-verification share the exact same runner interface.

3. **Explicit GPU float32 vs CPU float64 verification gate (`connectome_plan.md §4 F9`):**
   - Measure both raw Warp repeatability (`max_diff = 1.49e-8` float32 ULP across two GPU runs, `nworld = 16`) and GPU-vs-CPU trajectory accuracy (`max|dqpos| = 1.97e-7`, `max|dqvel| = 4.05e-6` over 50 ms; `0 ms` feeding-time difference on FlyWire v783).
   - Automatically re-verify the strongest hit of every batched behavioral screen on exact CPU float64 MuJoCo (`verify_top_cpu=True`) and report `cpu_confirmation`.

4. **CLI entry point (`alife connectome embody-batch`):**
   - Exposes `--mode benchmark` (`B = 1, 8, 16, 32` throughput & CPU-vs-Warp parity), `--mode screen` (batched embodied cell-type silencing screen with permutation test + Holm correction + shuffled control + CPU confirmation), and `--mode search` (parallel evolutionary foraging parameter search).

## Measured Results (RTX 5070, CUDA 13.4, MuJoCo Warp 3.17.0, FlyWire v783 138,639 cells)

- Pure physics (`nworld = 64`): **`25.3 fly-s / wall-s`** (`2.53 ms` per batched 1 ms tick across 64 flies).
- Coupled `FlyWire v783 (138,639 cells) ↔ MuJoCo Warp`:
  - `B = 1`: `0.23 fly-s/s`, `128.8 MiB` brain VRAM
  - `B = 8`: `1.58 fly-s/s`, `220.4 MiB` brain VRAM
  - `B = 16`: `3.56 fly-s/s`, `325.1 MiB` brain VRAM
  - **`B = 32`**: **`5.14 fly-s/s`**, `534.5 MiB` brain VRAM, `cpu vs warp feeding_ms_diff = 0 ms`, `distance_cm_diff = 0.0 cm`.
- Behavioral silencing screen (20 full-brain embodied evaluations on `flywire_783`): `8.08 s` total; silencing `CB0701` (MN9, 2 cells) suppressed sugar feeding by `+100.0%` (`173.8 -> 0.0 ms`, `shuffled = 0.0 ms`, confirmed `218 -> 0 ms` on CPU float64 MuJoCo), while silencing `DNp09` (`173.5 ms`) and `DNg84` (`173.8 ms`) left feeding intact.
