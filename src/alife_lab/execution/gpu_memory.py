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
    def __init__(self) -> None:
        self.peak_pool_bytes = 0

    def _cupy(self) -> Any:
        return sys.modules.get("cupy")

    def update(self) -> None:
        """Cheap; call after every step to keep the pool peak."""
        cp = self._cupy()
        if cp is not None:
            self.peak_pool_bytes = max(
                self.peak_pool_bytes, int(cp.get_default_memory_pool().total_bytes())
            )

    def snapshot(self) -> dict[str, Any] | None:
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
