"""GPU memory accounting inside a worker that uses CuPy.

Per-process GPU memory is not available from nvidia-smi under WSL2 (reported as [N/A]), so the
worker measures it itself: bytes held by CuPy's default memory pool (an upper bound of what the
simulation allocated through CuPy, tracked as a running peak) and device-wide usage from
``cudaMemGetInfo`` (includes other processes). Never imports CuPy itself.
"""

from __future__ import annotations

import sys
from typing import Any

MIB = 2**20


class GpuMemoryTracker:
    def __init__(self, backend: str = "cupy") -> None:
        self.backend = backend
        self.peak_pool_bytes = 0

    def _cupy(self) -> Any:
        return sys.modules.get("cupy") if self.backend == "cupy" else None

    def _jax_stats(self) -> dict[str, Any] | None:
        jax = sys.modules.get("jax") if self.backend == "jax_cuda" else None
        if jax is None:
            return None
        try:
            stats: dict[str, Any] = jax.devices("cuda")[0].memory_stats() or {}
        except RuntimeError:
            return None
        return stats

    def update(self) -> None:
        """Cheap; call after every step to keep the pool peak."""
        cp = self._cupy()
        if cp is not None:
            self.peak_pool_bytes = max(
                self.peak_pool_bytes, int(cp.get_default_memory_pool().total_bytes())
            )

    def _torch_snapshot(self) -> dict[str, Any] | None:
        torch = sys.modules.get("torch") if self.backend == "torch_cuda" else None
        if torch is None or not torch.cuda.is_initialized():
            return None
        used = int(torch.cuda.memory_allocated())
        held = int(torch.cuda.memory_reserved())
        self.peak_pool_bytes = max(self.peak_pool_bytes, int(torch.cuda.max_memory_reserved()))
        free, total = torch.cuda.mem_get_info()
        return {
            "pool_used_mib": round(used / MIB, 3),
            "pool_held_mib": round(held / MIB, 3),
            "peak_pool_held_mib": round(self.peak_pool_bytes / MIB, 3),
            "peak_pool_held_bytes": self.peak_pool_bytes,
            "device_used_mib": round((total - free) / MIB, 1),
            "device_total_mib": round(total / MIB, 1),
            "method": "torch caching allocator (reserved) + cudaMemGetInfo",
        }

    def snapshot(self) -> dict[str, Any] | None:
        if self.backend == "torch_cuda":
            return self._torch_snapshot()
        stats = self._jax_stats()
        if stats is not None:  # XLA allocator statistics (preallocation disabled)
            in_use = int(stats.get("bytes_in_use", 0))
            peak = int(stats.get("peak_bytes_in_use", in_use))
            limit = int(stats.get("bytes_limit", 0))
            self.peak_pool_bytes = max(self.peak_pool_bytes, peak)
            return {
                "pool_used_mib": round(in_use / MIB, 3),
                "pool_held_mib": round(in_use / MIB, 3),
                "peak_pool_held_mib": round(self.peak_pool_bytes / MIB, 3),
                "peak_pool_held_bytes": self.peak_pool_bytes,
                "device_used_mib": round(in_use / MIB, 1),
                "device_total_mib": round(limit / MIB, 1),
                "method": "jax memory_stats (XLA allocator; device_total = allocator limit)",
            }
        cp = self._cupy()
        if cp is None:
            return None
        self.update()
        pool = cp.get_default_memory_pool()
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "pool_used_mib": round(pool.used_bytes() / MIB, 3),
            "pool_held_mib": round(pool.total_bytes() / MIB, 3),
            "peak_pool_held_mib": round(self.peak_pool_bytes / MIB, 3),
            "peak_pool_held_bytes": self.peak_pool_bytes,
            "device_used_mib": round((total - free) / MIB, 1),
            "device_total_mib": round(total / MIB, 1),
            "method": "cupy_memory_pool+cudaMemGetInfo",
        }
