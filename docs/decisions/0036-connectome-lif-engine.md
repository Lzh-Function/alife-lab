# ADR 0036 — Bit-identical CPU/GPU spiking LIF connectome engine (F1)

Status: accepted (2026-11-21)

## Context
Phase F1 of `docs/plans/connectome_plan.md` requires simulating the adult *Drosophila*
connectome (138,639 neurons, 15,091,983 directed edges) under the leaky integrate-and-fire
(LIF) equations of Shiu et al. (2024), Nature 634:210–219, with:
- an offline CPU oracle (`numpy`) for CI and analytic checks,
- a fast CUDA C++ GPU engine (`cupy`) using CUDA graphs for real-time and batch work,
- bit-identical continuation across CPU, GPU (eager and CUDA graph), and checkpoints,
- interventions (`stimulate`, `silence`, `clear_stimuli`) applied at safe tick boundaries.

## Decision
1. **Exact linear ODE integration (`dt = 0.1 ms`, 10 substeps = 1 ms tick):**
   Each substep integrates `dv/dt = (v_0 - v + g)/t_mbr`, `dg/dt = -g/tau` analytically via
   precomputed `float32` coefficients `(alpha_v, alpha_g, beta_vg)`, matching Brian2's `exact`
   integrator and passing analytic single-neuron tests within `1e-4 mV`.
2. **Exact bitwise CPU/GPU reproducibility:**
   - Incoming delayed synaptic inputs (`t_dly = 1.8 ms = 18 substeps`) accumulate as `int32`
     signed synapse counts in a `(ring_len, N)` ring buffer via `atomicAdd`. Because integer
     addition is associative and commutative, thread scheduling never alters results.
   - External Poisson stimulation uses a stateless SplitMix64 counter hash of
     `(seed, substep, neuron_index)` producing top-24-bit `float32` uniforms, identical on CPU
     and GPU with zero extra RNG checkpoint state.
   - The CUDA module compiles with `--fmad=false` and explicit `__fmul_rn` / `__fadd_rn` /
     `__fsub_rn` matching the CPU `float32` operation order `(v_0 + (alpha_v * (v - v_0))) +
     (beta_vg * g)`. Verified 0 ULP (`max|dv| = 0.0`, exact uint32 bit match across all
     arrays) on the full 138,639-neuron FlyWire v783 graph.
3. **CUDA-graph-compatible sparse scatter for skewed hubs:**
   Real FlyWire v783 has hub out-degrees up to 9,783. A fixed grid `(256,)` of 128-thread
   blocks strides over the dense `fired_ids` list, with each block's 128 threads scattering a
   firing neuron's CSR slice in parallel. Because grid dimensions and device substep pointer
   are fixed, 10-step (1 tick) and 100-step batches capture cleanly into reusable CUDA graphs.
4. **Degree-preserving shuffled control & built-in offline circuit:**
   - `wiring: "shuffled_control"` deterministically permutes postsynaptic targets while keeping
     per-neuron out-degree, weight multiset and Dale sign (breaking self-loops), providing a
     strict connectivity control.
   - `dataset_id: "synthetic_tiny"` provides a 4-neuron excitatory/inhibitory relay circuit so
     CPU-only CI runs full controller/worker/checkpoint/replay-verify tests without external
     data.

## Measured (RTX 5070 12 GiB, FlyWire v783, sugar 150 Hz, 1 simulated second)
- CPU (`numpy`): ~6.16 s (0.16× real-time)
- GPU eager (`cupy`, no graph): 0.235 s (4.3× real-time)
- GPU CUDA graph (10-step / tick): 0.060 s (16.7× real-time)
- GPU CUDA graph (100-step batch): 0.047–0.057 s (17.5–21.5× real-time), 129 MiB VRAM
- Batched GPU (`B=8` shared CSR): 0.216 s for 8 brain-seconds (**37.0 brain-s / wall-s**,
  220 MiB VRAM, 13.1 MiB per extra brain; `B=128` fits in 1.79 GiB at 13.0 brain-s / wall-s).
