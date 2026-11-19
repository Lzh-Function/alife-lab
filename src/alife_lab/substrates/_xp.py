"""Host/device array helpers for optional GPU backends (CuPy imported lazily, never here)."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

GPU_BACKENDS = frozenset({"cupy"})


def is_device(a: Any) -> bool:
    return type(a).__module__.startswith("cupy")


def xp_of(a: Any) -> Any:
    """Array namespace of ``a`` (numpy or cupy)."""
    if is_device(a):
        return sys.modules["cupy"]
    return np


def to_host(a: Any) -> Any:
    """NumPy copy of a device array; NumPy arrays are returned unchanged."""
    if is_device(a):
        return a.get()
    return a


def to_backend(a: Any, backend: str) -> Any:
    if backend in GPU_BACKENDS:
        import cupy

        return cupy.asarray(a)
    return to_host(a)
