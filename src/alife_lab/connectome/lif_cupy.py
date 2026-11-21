"""CUDA C++ LIF whole-brain engine via CuPy RawModule and CUDA graphs (F1).

Design goals (connectome plan sections 3.3 and 3.4):
- Bit-identical to ``alife_lab.connectome.lif.step_substep_cpu``:
  * compiled with ``--fmad=false`` and explicit IEEE-754 ``__fmul_rn`` / ``__fadd_rn`` /
    ``__fsub_rn`` in identical operation order;
  * exact ``int32`` ``atomicAdd`` accumulation into the delayed arrival ring buffer;
  * identical stateless SplitMix64 counter-based RNG for external Poisson stimulation.
- Event-driven sparse spike scatter with fixed-grid block cooperation so high-degree hubs
  (up to 9,783 outgoing edges in FlyWire v783) are parallelized across 128 threads per block
  AND the entire kernel sequence can be captured into reusable CUDA graphs.
- Supports both single-brain interactive execution (``CupyLIFRunner``) and shared-wiring
  batched multi-brain execution (``CupyBatchedLIFRunner``).

CuPy is imported lazily so the FastAPI controller process never initializes CUDA.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from typing import Any

import numpy as np

from alife_lab.connectome.lif import LIFParams, LIFState
from alife_lab.connectome.schema import CSRGraph

_CUDA_SOURCE = r"""
__device__ __forceinline__ float splitmix64_u01_f32(
    unsigned long long seed,
    unsigned long long substep,
    unsigned int neuron_idx
) {
    unsigned long long z =
        seed
        + substep * 0xD1B54A32D192ED03ULL
        + (unsigned long long)neuron_idx * 0x94D049BB133111EBULL
        + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    float top24 = (float)(unsigned int)(z >> 40);
    return __fmul_rn(top24, 5.960464477539063e-08f); // 2^-24 exact
}

extern "C" __global__ void lif_zero_tick_spikes(int* last_tick_spikes, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) last_tick_spikes[i] = 0;
}

extern "C" __global__ void lif_step_neurons(
    float* __restrict__ v,
    float* __restrict__ g,
    short* __restrict__ refrac,
    int* __restrict__ delay_ring,
    const float* __restrict__ stim_rate_hz,
    const unsigned char* __restrict__ silenced,
    int* __restrict__ last_tick_spikes,
    int* __restrict__ total_spikes,
    int* __restrict__ fired_ids,
    int* __restrict__ n_fired,
    const long long* __restrict__ d_substep,
    const int n,
    const int ring_len,
    const unsigned long long seed,
    const float v_0,
    const float v_rst,
    const float v_th,
    const float alpha_v,
    const float alpha_g,
    const float beta_vg,
    const float w_syn,
    const float stim_w,
    const float dt_s,
    const short refrac_steps,
    const int stim_no_refrac
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const long long s = d_substep[0];
    const int slot = (int)(s % (long long)ring_len);
    const long long ring_off = (long long)slot * (long long)n + (long long)i;

    float vi = v[i];
    float gi = g[i];
    short ri = refrac[i];

    // 1. Deliver queued integer synaptic sum arriving at this substep.
    const int acc = delay_ring[ring_off];
    if (acc != 0) {
        delay_ring[ring_off] = 0;
        gi = __fadd_rn(gi, __fmul_rn((float)acc, w_syn));
    }

    // 2. Counter-based Poisson stimulation (target_var='v').
    const float rate = stim_rate_hz[i];
    if (rate > 0.0f) {
        const float u = splitmix64_u01_f32(seed, (unsigned long long)s, (unsigned int)i);
        const float p = __fmul_rn(rate, dt_s);
        if (u < p) {
            vi = __fadd_rn(vi, stim_w);
            if (stim_no_refrac != 0) {
                ri = 0;
            }
        }
    }

    // 3. Spike check, reset, or exact float32 ODE update (non-FMA, matches CPU op-for-op).
    if (ri == 0 && vi > v_th) {
        v[i] = v_rst;
        g[i] = 0.0f;
        refrac[i] = (stim_no_refrac != 0 && rate > 0.0f) ? (short)0 : refrac_steps;
        last_tick_spikes[i] += 1;
        total_spikes[i] += 1;
        if (silenced[i] == 0) {
            const int pos = atomicAdd(n_fired, 1);
            fired_ids[pos] = i;
        }
    } else {
        if (ri == 0) {
            const float dv = __fsub_rn(vi, v_0);
            const float v_decay = __fmul_rn(alpha_v, dv);
            const float g_drive = __fmul_rn(beta_vg, gi);
            v[i] = __fadd_rn(__fadd_rn(v_0, v_decay), g_drive);
        } else {
            refrac[i] = (short)(ri - 1);
        }
        g[i] = __fmul_rn(alpha_g, gi);
    }
}

extern "C" __global__ void lif_scatter_spikes(
    const int* __restrict__ indptr,
    const int* __restrict__ post,
    const int* __restrict__ signed_weight,
    const int* __restrict__ fired_ids,
    const int* __restrict__ n_fired,
    const long long* __restrict__ d_substep,
    int* __restrict__ delay_ring,
    const int n,
    const int delay_steps,
    const int ring_len
) {
    const int nf = n_fired[0];
    if (nf <= 0) return;

    const long long s = d_substep[0];
    const int dst_slot = (int)((s + (long long)delay_steps) % (long long)ring_len);
    int* __restrict__ dst = delay_ring + (long long)dst_slot * (long long)n;

    for (int k = blockIdx.x; k < nf; k += gridDim.x) {
        const int u = fired_ids[k];
        const int start = indptr[u];
        const int end = indptr[u + 1];
        for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
            atomicAdd(&dst[post[e]], signed_weight[e]);
        }
    }
}

extern "C" __global__ void lif_finish_substep(
    long long* __restrict__ d_substep,
    int* __restrict__ n_fired
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_substep[0] += 1;
        n_fired[0] = 0;
    }
}

// ---------------------------------------------------------------------------
// Batched multi-brain kernels (shared CSR wiring, B independent brain states)
// ---------------------------------------------------------------------------

extern "C" __global__ void lif_batched_step_neurons(
    float* __restrict__ v,
    float* __restrict__ g,
    short* __restrict__ refrac,
    int* __restrict__ delay_ring,
    const float* __restrict__ stim_rate_hz,
    const unsigned char* __restrict__ silenced,
    int* __restrict__ total_spikes,
    int* __restrict__ fired_ids,
    int* __restrict__ n_fired,
    const long long* __restrict__ d_substep,
    const unsigned long long* __restrict__ seeds,
    const int b_count,
    const int n,
    const int ring_len,
    const float v_0,
    const float v_rst,
    const float v_th,
    const float alpha_v,
    const float alpha_g,
    const float beta_vg,
    const float w_syn,
    const float stim_w,
    const float dt_s,
    const short refrac_steps,
    const int stim_no_refrac
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (i >= n || b >= b_count) return;

    const long long s = d_substep[0];
    const int slot = (int)(s % (long long)ring_len);
    const long long cell_off = (long long)b * (long long)n + (long long)i;
    const long long ring_off =
        ((long long)b * (long long)ring_len + (long long)slot) * (long long)n + (long long)i;

    float vi = v[cell_off];
    float gi = g[cell_off];
    short ri = refrac[cell_off];

    const int acc = delay_ring[ring_off];
    if (acc != 0) {
        delay_ring[ring_off] = 0;
        gi = __fadd_rn(gi, __fmul_rn((float)acc, w_syn));
    }

    const float rate = stim_rate_hz[cell_off];
    if (rate > 0.0f) {
        const float u = splitmix64_u01_f32(seeds[b], (unsigned long long)s, (unsigned int)i);
        const float p = __fmul_rn(rate, dt_s);
        if (u < p) {
            vi = __fadd_rn(vi, stim_w);
            if (stim_no_refrac != 0) {
                ri = 0;
            }
        }
    }

    if (ri == 0 && vi > v_th) {
        v[cell_off] = v_rst;
        g[cell_off] = 0.0f;
        refrac[cell_off] = (stim_no_refrac != 0 && rate > 0.0f) ? (short)0 : refrac_steps;
        total_spikes[cell_off] += 1;
        if (silenced[cell_off] == 0) {
            const int pos = atomicAdd(&n_fired[b], 1);
            fired_ids[(long long)b * (long long)n + (long long)pos] = i;
        }
    } else {
        if (ri == 0) {
            const float dv = __fsub_rn(vi, v_0);
            const float v_decay = __fmul_rn(alpha_v, dv);
            const float g_drive = __fmul_rn(beta_vg, gi);
            v[cell_off] = __fadd_rn(__fadd_rn(v_0, v_decay), g_drive);
        } else {
            refrac[cell_off] = (short)(ri - 1);
        }
        g[cell_off] = __fmul_rn(alpha_g, gi);
    }
}

extern "C" __global__ void lif_batched_scatter_spikes(
    const int* __restrict__ indptr,
    const int* __restrict__ post,
    const int* __restrict__ signed_weight,
    const int* __restrict__ fired_ids,
    const int* __restrict__ n_fired,
    const long long* __restrict__ d_substep,
    int* __restrict__ delay_ring,
    const int b_count,
    const int n,
    const int delay_steps,
    const int ring_len
) {
    const int b = blockIdx.y;
    if (b >= b_count) return;
    const int nf = n_fired[b];
    if (nf <= 0) return;

    const long long s = d_substep[0];
    const int dst_slot = (int)((s + (long long)delay_steps) % (long long)ring_len);
    int* __restrict__ dst =
        delay_ring + ((long long)b * (long long)ring_len + (long long)dst_slot) * (long long)n;
    const int* __restrict__ b_fired = fired_ids + (long long)b * (long long)n;

    for (int k = blockIdx.x; k < nf; k += gridDim.x) {
        const int u = b_fired[k];
        const int start = indptr[u];
        const int end = indptr[u + 1];
        for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
            atomicAdd(&dst[post[e]], signed_weight[e]);
        }
    }
}

extern "C" __global__ void lif_batched_finish_substep(
    long long* __restrict__ d_substep,
    int* __restrict__ n_fired,
    const int b_count
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < b_count) {
        n_fired[b] = 0;
    }
    if (b == 0) {
        d_substep[0] += 1;
    }
}
"""

_MODULE: Any = None
NEURON_BLOCK = 256
SCATTER_BLOCKS = 256
SCATTER_THREADS = 128


def _ensure_cupy_cache_dir() -> None:
    if "CUPY_CACHE_DIR" not in os.environ:
        default = os.path.expanduser("~/.cupy")
        parent = os.path.dirname(default) or "/"
        if not os.access(default if os.path.exists(default) else parent, os.W_OK):
            cache = os.path.join(tempfile.gettempdir(), f"alife_cupy_cache_{os.getuid()}")
            os.environ["CUPY_CACHE_DIR"] = cache


def available() -> bool:
    """True if CuPy is installed and a CUDA GPU is usable (initialises CUDA on call)."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        _ensure_cupy_cache_dir()
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def _get_module() -> Any:
    global _MODULE
    if _MODULE is None:
        _ensure_cupy_cache_dir()
        import cupy

        _MODULE = cupy.RawModule(code=_CUDA_SOURCE, options=("--fmad=false",))
    return _MODULE


class CupyLIFRunner:
    """Resident GPU LIF simulator with reusable CUDA graphs for 1-tick and multi-tick batches."""

    def __init__(self, csr: CSRGraph, params: LIFParams, seed: int) -> None:
        _ensure_cupy_cache_dir()
        import cupy as cp

        self.cp = cp
        self.params = params
        self.seed = int(seed)
        self.num_cells = csr.num_cells
        self.ring_len = params.ring_len
        self.delay_steps = params.delay_steps
        self.refrac_steps = np.int16(params.refrac_steps)
        self.alpha_v, self.alpha_g, self.beta_vg = params.exact_coeffs_f32()
        self.v_0 = np.float32(params.v_0)
        self.v_rst = np.float32(params.v_rst)
        self.v_th = np.float32(params.v_th)
        self.w_syn = np.float32(params.w_syn_mv)
        self.stim_w = np.float32(params.stim_weight_mv)
        self.dt_s = np.float32(params.dt_s)
        self.stim_no_refrac = np.int32(1 if params.stim_no_refrac else 0)

        mod = _get_module()
        self._k_zero_tick = mod.get_function("lif_zero_tick_spikes")
        self._k_step = mod.get_function("lif_step_neurons")
        self._k_scatter = mod.get_function("lif_scatter_spikes")
        self._k_finish = mod.get_function("lif_finish_substep")

        # Resident CSR wiring on GPU.
        self.d_indptr = cp.asarray(np.ascontiguousarray(csr.indptr, dtype=np.int32))
        self.d_post = cp.asarray(np.ascontiguousarray(csr.post, dtype=np.int32))
        self.d_weight = cp.asarray(np.ascontiguousarray(csr.signed_weight, dtype=np.int32))

        # Resident state buffers on GPU.
        n = self.num_cells
        self.d_v = cp.full(n, self.v_0, dtype=cp.float32)
        self.d_g = cp.zeros(n, dtype=cp.float32)
        self.d_refrac = cp.zeros(n, dtype=cp.int16)
        self.d_delay_ring = cp.zeros((self.ring_len, n), dtype=cp.int32)
        self.d_stim_rate_hz = cp.zeros(n, dtype=cp.float32)
        self.d_silenced = cp.zeros(n, dtype=cp.uint8)
        self.d_last_tick_spikes = cp.zeros(n, dtype=cp.int32)
        self.d_total_spikes = cp.zeros(n, dtype=cp.int32)
        self.d_fired_ids = cp.zeros(n, dtype=cp.int32)
        self.d_n_fired = cp.zeros(1, dtype=cp.int32)
        self.d_substep = cp.zeros(1, dtype=cp.int64)

        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graphs: dict[tuple[int, bool], Any] = {}
        cp.cuda.get_current_stream().synchronize()

    def upload_state(self, state: LIFState) -> None:
        """Copy host LIFState into resident GPU buffers (exact bitwise copy)."""
        self.d_v.set(np.ascontiguousarray(state.v, dtype=np.float32))
        self.d_g.set(np.ascontiguousarray(state.g, dtype=np.float32))
        self.d_refrac.set(np.ascontiguousarray(state.refrac, dtype=np.int16))
        self.d_delay_ring.set(np.ascontiguousarray(state.delay_ring, dtype=np.int32))
        self.d_stim_rate_hz.set(np.ascontiguousarray(state.stim_rate_hz, dtype=np.float32))
        self.d_silenced.set(np.ascontiguousarray(state.silenced, dtype=np.uint8))
        self.d_last_tick_spikes.set(np.ascontiguousarray(state.last_tick_spikes, dtype=np.int32))
        self.d_total_spikes.set(np.ascontiguousarray(state.total_spikes, dtype=np.int32))
        self.d_n_fired.fill(0)
        self.d_substep.set(np.array([state.substep], dtype=np.int64))
        self.cp.cuda.get_current_stream().synchronize()

    def download_state(self, state: LIFState) -> LIFState:
        """Copy resident GPU state back into host LIFState (exact bitwise copy)."""
        self._stream.synchronize()
        state.v[...] = self.cp.asnumpy(self.d_v)
        state.g[...] = self.cp.asnumpy(self.d_g)
        state.refrac[...] = self.cp.asnumpy(self.d_refrac)
        state.delay_ring[...] = self.cp.asnumpy(self.d_delay_ring)
        state.stim_rate_hz[...] = self.cp.asnumpy(self.d_stim_rate_hz)
        state.silenced[...] = self.cp.asnumpy(self.d_silenced)
        state.last_tick_spikes[...] = self.cp.asnumpy(self.d_last_tick_spikes)
        state.total_spikes[...] = self.cp.asnumpy(self.d_total_spikes)
        state.substep = int(self.cp.asnumpy(self.d_substep)[0])
        return state

    def upload_controls(self, state: LIFState) -> None:
        """Fast upload of only stim_rate_hz and silenced after an intervention."""
        self._stream.synchronize()
        self.d_stim_rate_hz.set(np.ascontiguousarray(state.stim_rate_hz, dtype=np.float32))
        self.d_silenced.set(np.ascontiguousarray(state.silenced, dtype=np.uint8))
        self.cp.cuda.get_current_stream().synchronize()

    def _launch_substep(self) -> None:
        n = self.num_cells
        grid_n = ((n + NEURON_BLOCK - 1) // NEURON_BLOCK,)
        self._k_step(
            grid_n,
            (NEURON_BLOCK,),
            (
                self.d_v,
                self.d_g,
                self.d_refrac,
                self.d_delay_ring,
                self.d_stim_rate_hz,
                self.d_silenced,
                self.d_last_tick_spikes,
                self.d_total_spikes,
                self.d_fired_ids,
                self.d_n_fired,
                self.d_substep,
                np.int32(n),
                np.int32(self.ring_len),
                np.uint64(self.seed & 0xFFFFFFFFFFFFFFFF),
                self.v_0,
                self.v_rst,
                self.v_th,
                self.alpha_v,
                self.alpha_g,
                self.beta_vg,
                self.w_syn,
                self.stim_w,
                self.dt_s,
                self.refrac_steps,
                self.stim_no_refrac,
            ),
        )
        self._k_scatter(
            (SCATTER_BLOCKS,),
            (SCATTER_THREADS,),
            (
                self.d_indptr,
                self.d_post,
                self.d_weight,
                self.d_fired_ids,
                self.d_n_fired,
                self.d_substep,
                self.d_delay_ring,
                np.int32(n),
                np.int32(self.delay_steps),
                np.int32(self.ring_len),
            ),
        )
        self._k_finish((1,), (1,), (self.d_substep, self.d_n_fired))

    def _get_graph(self, n_substeps: int, zero_tick: bool) -> Any:
        key = (n_substeps, zero_tick)
        graph = self._graphs.get(key)
        if graph is not None:
            return graph
        n = self.num_cells
        grid_n = ((n + NEURON_BLOCK - 1) // NEURON_BLOCK,)
        self.cp.cuda.Device().synchronize()
        with self._stream:
            self._stream.begin_capture()
            if zero_tick:
                self._k_zero_tick(grid_n, (NEURON_BLOCK,), (self.d_last_tick_spikes, np.int32(n)))
            for _ in range(n_substeps):
                self._launch_substep()
            graph = self._stream.end_capture()
        self._stream.synchronize()
        self._graphs[key] = graph
        return graph

    def step_substeps(
        self, n_substeps: int, *, zero_tick_spikes: bool = False, use_graph: bool = True
    ) -> None:
        """Advance ``n_substeps`` on GPU using either a captured CUDA graph or eager launches."""
        if use_graph:
            graph = self._get_graph(n_substeps, zero_tick_spikes)
            graph.launch(stream=self._stream)
            return
        n = self.num_cells
        grid_n = ((n + NEURON_BLOCK - 1) // NEURON_BLOCK,)
        with self._stream:
            if zero_tick_spikes:
                self._k_zero_tick(grid_n, (NEURON_BLOCK,), (self.d_last_tick_spikes, np.int32(n)))
            for _ in range(n_substeps):
                self._launch_substep()

    def step_tick(self, *, use_graph: bool = True) -> None:
        self.step_substeps(
            self.params.substeps_per_tick, zero_tick_spikes=True, use_graph=use_graph
        )

    def synchronize(self) -> None:
        self._stream.synchronize()

    def device_bytes(self) -> int:
        arrays = (
            self.d_indptr,
            self.d_post,
            self.d_weight,
            self.d_v,
            self.d_g,
            self.d_refrac,
            self.d_delay_ring,
            self.d_stim_rate_hz,
            self.d_silenced,
            self.d_last_tick_spikes,
            self.d_total_spikes,
            self.d_fired_ids,
            self.d_n_fired,
            self.d_substep,
        )
        return int(sum(a.nbytes for a in arrays))


class CupyBatchedLIFRunner:
    """Run B independent brains in parallel on a shared CSR connectome (F1 benchmark / F3 base)."""

    def __init__(
        self,
        csr: CSRGraph,
        params: LIFParams,
        seeds: list[int],
        stim_rates_hz: np.ndarray[Any, Any],
    ) -> None:
        _ensure_cupy_cache_dir()
        import cupy as cp

        self.cp = cp
        self.params = params
        self.batch_size = len(seeds)
        self.num_cells = csr.num_cells
        self.ring_len = params.ring_len
        self.delay_steps = params.delay_steps
        self.refrac_steps = np.int16(params.refrac_steps)
        self.alpha_v, self.alpha_g, self.beta_vg = params.exact_coeffs_f32()
        self.v_0 = np.float32(params.v_0)
        self.v_rst = np.float32(params.v_rst)
        self.v_th = np.float32(params.v_th)
        self.w_syn = np.float32(params.w_syn_mv)
        self.stim_w = np.float32(params.stim_weight_mv)
        self.dt_s = np.float32(params.dt_s)
        self.stim_no_refrac = np.int32(1 if params.stim_no_refrac else 0)

        mod = _get_module()
        self._k_step = mod.get_function("lif_batched_step_neurons")
        self._k_scatter = mod.get_function("lif_batched_scatter_spikes")
        self._k_finish = mod.get_function("lif_batched_finish_substep")

        b = self.batch_size
        n = self.num_cells
        self.d_indptr = cp.asarray(np.ascontiguousarray(csr.indptr, dtype=np.int32))
        self.d_post = cp.asarray(np.ascontiguousarray(csr.post, dtype=np.int32))
        self.d_weight = cp.asarray(np.ascontiguousarray(csr.signed_weight, dtype=np.int32))
        self.d_seeds = cp.asarray(np.asarray(seeds, dtype=np.uint64))
        self.d_v = cp.full((b, n), self.v_0, dtype=cp.float32)
        self.d_g = cp.zeros((b, n), dtype=cp.float32)
        self.d_refrac = cp.zeros((b, n), dtype=cp.int16)
        self.d_delay_ring = cp.zeros((b, self.ring_len, n), dtype=cp.int32)
        self.d_stim_rate_hz = cp.asarray(np.ascontiguousarray(stim_rates_hz, dtype=np.float32))
        self.d_silenced = cp.zeros((b, n), dtype=cp.uint8)
        self.d_total_spikes = cp.zeros((b, n), dtype=cp.int32)
        self.d_fired_ids = cp.zeros((b, n), dtype=cp.int32)
        self.d_n_fired = cp.zeros(b, dtype=cp.int32)
        self.d_substep = cp.zeros(1, dtype=cp.int64)
        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graphs: dict[int, Any] = {}
        cp.cuda.get_current_stream().synchronize()

    def set_silenced(self, silenced: np.ndarray[Any, Any]) -> None:
        """Synchronously upload (B, N) uint8 silencing mask before graph launches."""
        self._stream.synchronize()
        self.d_silenced.set(np.ascontiguousarray(silenced, dtype=np.uint8))
        self.cp.cuda.get_current_stream().synchronize()

    def _launch_substep(self) -> None:
        b = self.batch_size
        n = self.num_cells
        grid_n = ((n + NEURON_BLOCK - 1) // NEURON_BLOCK, b)
        scatter_x = max(16, SCATTER_BLOCKS // max(1, min(b, 8)))
        self._k_step(
            grid_n,
            (NEURON_BLOCK,),
            (
                self.d_v,
                self.d_g,
                self.d_refrac,
                self.d_delay_ring,
                self.d_stim_rate_hz,
                self.d_silenced,
                self.d_total_spikes,
                self.d_fired_ids,
                self.d_n_fired,
                self.d_substep,
                self.d_seeds,
                np.int32(b),
                np.int32(n),
                np.int32(self.ring_len),
                self.v_0,
                self.v_rst,
                self.v_th,
                self.alpha_v,
                self.alpha_g,
                self.beta_vg,
                self.w_syn,
                self.stim_w,
                self.dt_s,
                self.refrac_steps,
                self.stim_no_refrac,
            ),
        )
        self._k_scatter(
            (scatter_x, b),
            (SCATTER_THREADS,),
            (
                self.d_indptr,
                self.d_post,
                self.d_weight,
                self.d_fired_ids,
                self.d_n_fired,
                self.d_substep,
                self.d_delay_ring,
                np.int32(b),
                np.int32(n),
                np.int32(self.delay_steps),
                np.int32(self.ring_len),
            ),
        )
        self._k_finish(
            ((b + 63) // 64,), (64,), (self.d_substep, self.d_n_fired, np.int32(b))
        )

    def step_substeps(self, n_substeps: int, *, use_graph: bool = True) -> None:
        if use_graph:
            graph = self._graphs.get(n_substeps)
            if graph is None:
                self.cp.cuda.Device().synchronize()
                with self._stream:
                    self._stream.begin_capture()
                    for _ in range(n_substeps):
                        self._launch_substep()
                    graph = self._stream.end_capture()
                self._stream.synchronize()
                self._graphs[n_substeps] = graph
            graph.launch(stream=self._stream)
            return
        with self._stream:
            for _ in range(n_substeps):
                self._launch_substep()

    def synchronize(self) -> None:
        self._stream.synchronize()

    def total_spikes_cpu(self) -> np.ndarray[Any, np.dtype[np.int32]]:
        self._stream.synchronize()
        out: np.ndarray[Any, np.dtype[np.int32]] = self.cp.asnumpy(self.d_total_spikes)
        return out

    def device_bytes(self) -> int:
        arrays = (
            self.d_indptr,
            self.d_post,
            self.d_weight,
            self.d_seeds,
            self.d_v,
            self.d_g,
            self.d_refrac,
            self.d_delay_ring,
            self.d_stim_rate_hz,
            self.d_silenced,
            self.d_total_spikes,
            self.d_fired_ids,
            self.d_n_fired,
            self.d_substep,
        )
        return int(sum(a.nbytes for a in arrays))
