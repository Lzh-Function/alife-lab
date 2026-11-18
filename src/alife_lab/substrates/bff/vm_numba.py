"""Optional Numba JIT backend for the BFF VM (``uv sync --extra fast``).

A direct port of :func:`alife_lab.substrates.bff.vm.evaluate_scalar`; pairs are independent,
so they run in parallel (``prange``) and the result does not depend on thread scheduling.
Equality with the scalar oracle, the NumPy backend and CuBFF fixtures is tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:  # pragma: no cover - availability depends on the installed extra
    import numba

    AVAILABLE = True
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]
    AVAILABLE = False

TAPE = 128


def _build() -> Any:
    assert numba is not None
    from numba import njit, prange

    @njit(cache=True, nogil=True)
    def eval_one(tape: Any, step_limit: int, heads: bool) -> int:
        if heads:
            head0 = tape[0] % TAPE
            head1 = tape[1] % TAPE
            pos = 2
        else:
            head0 = TAPE
            head1 = TAPE
            pos = 0
        nskip = 0
        i = 0
        while i < step_limit:
            head0 &= TAPE - 1
            head1 &= TAPE - 1
            cmd = tape[pos]
            if cmd == 60:  # <
                head0 -= 1
            elif cmd == 62:  # >
                head0 += 1
            elif cmd == 123:  # {
                head1 -= 1
            elif cmd == 125:  # }
                head1 += 1
            elif cmd == 43:  # +
                tape[head0] = (tape[head0] + 1) & 0xFF
            elif cmd == 45:  # -
                tape[head0] = (tape[head0] - 1) & 0xFF
            elif cmd == 46:  # .
                tape[head1] = tape[head0]
            elif cmd == 44:  # ,
                tape[head0] = tape[head1]
            elif cmd == 91:  # [
                if tape[head0] == 0:
                    depth = 1
                    pos += 1
                    while pos < TAPE and depth > 0:
                        if tape[pos] == 93:
                            depth -= 1
                        if tape[pos] == 91:
                            depth += 1
                        pos += 1
                    pos -= 1
                    if depth != 0:
                        pos = TAPE
            elif cmd == 93:  # ]
                if tape[head0] != 0:
                    depth = 1
                    pos -= 1
                    while pos >= 0 and depth > 0:
                        if tape[pos] == 93:
                            depth += 1
                        if tape[pos] == 91:
                            depth -= 1
                        pos -= 1
                    pos += 1
                    if depth != 0:
                        pos = -1
            else:
                nskip += 1
            if pos < 0:
                i += 1
                break
            pos += 1
            if pos >= TAPE:
                i += 1
                break
            i += 1
        return i - nskip

    @njit(cache=True, parallel=True, nogil=True)
    def eval_all(tapes: Any, step_limit: int, heads: bool) -> Any:
        n = tapes.shape[0]
        ops = np.zeros(n, dtype=np.int64)
        for p in prange(n):
            ops[p] = eval_one(tapes[p], step_limit, heads)
        return ops

    return eval_all


_KERNEL: Any = None


def evaluate_batch_numba(
    tapes: npt.NDArray[np.uint8], step_limit: int, variant: str
) -> npt.NDArray[np.int64]:
    """Evaluate tapes in place with the JIT kernel (compiled on first use, cached on disk)."""
    global _KERNEL
    if not AVAILABLE:
        raise RuntimeError("numba is not installed (uv sync --extra fast)")
    if tapes.ndim != 2 or tapes.shape[1] != TAPE or tapes.dtype != np.uint8:
        raise ValueError("tapes must be uint8 of shape (P, 128)")
    if _KERNEL is None:
        _KERNEL = _build()
    ops: npt.NDArray[np.int64] = _KERNEL(tapes, int(step_limit), variant == "bff")
    return ops
