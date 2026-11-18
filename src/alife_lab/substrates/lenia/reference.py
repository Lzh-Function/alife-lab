"""Core Lenia functions ported from Bert Chan's LeniaND.py.

Source: https://github.com/Chakazul/Lenia (Python/LeniaND.py, commit adfc5429, 2022-03-15).

    MIT License — Copyright (c) 2018 Bert Chan

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

Ported: kernel cores, growth functions, kernel shell, the default (reference) path of
``calc_kernel`` / ``calc_once`` for 2D on the CPU (no soft clip, multi-step, noise, mask or
quantisation), RLE decoding of ``animals.json`` cells, and the Orbium unicaudatus entry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

# Automaton.kernel_core (kn = index + 1)
KERNEL_CORE: dict[str, Callable[[F64], F64]] = {
    "polynomial": lambda r: (4 * r * (1 - r)) ** 4,  # quad4
    "exponential": lambda r: np.exp(4 - 1 / (r * (1 - r))),  # bump4
    "step": lambda r: ((r >= 0.25) * (r <= 0.75)).astype(np.float64),  # stpz1/4
    "staircase": lambda r: (r >= 0.25) * (r <= 0.75) + (r < 0.25) * 0.5,  # life
}

# Automaton.growth_func (gn = index + 1)
GROWTH: dict[str, Callable[[F64, float, float], F64]] = {
    "polynomial": lambda n, m, s: np.maximum(0, 1 - (n - m) ** 2 / (9 * s**2)) ** 4 * 2 - 1,
    "gaussian": lambda n, m, s: np.exp(-((n - m) ** 2) / (2 * s**2)) * 2 - 1,
    "step": lambda n, m, s: np.where(np.abs(n - m) <= s, 1.0, -1.0),
}


def kernel_shell(r: F64, b: list[float], core: str) -> F64:
    """Automaton.kernel_shell: multi-ring kernel from normalised distance r."""
    nb = len(b)
    br = nb * r
    bs = np.asarray(b, dtype=np.float64)
    weights = bs[np.minimum(np.floor(br).astype(int), nb - 1)]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shell = (r < 1) * KERNEL_CORE[core](np.minimum(br % 1, 1)) * weights
    return np.asarray(np.nan_to_num(shell, nan=0.0), dtype=np.float64)


def kernel_fft(height: int, width: int, R: int, b: list[float], core: str) -> F64:
    """Automaton.calc_kernel for 2D: normalised kernel centred at MID, then FFT."""
    mid = [height // 2, width // 2]
    i, j = np.mgrid[0:height, 0:width]  # rows (y), columns (x)
    x = [(j - mid[1]) / R, (i - mid[0]) / R]
    d = np.sqrt(x[0] ** 2 + x[1] ** 2)
    kernel = kernel_shell(d, b, core)
    total = kernel.sum()
    out: Any = np.fft.fftn(kernel / total)  # complex128
    return out  # type: ignore[no-any-return]


def potential(cells: F64, kfft: F64) -> F64:
    """calc_once: U = fftshift(real(ifftn(K_fft * fftn(A)))).

    ``world_fft`` must stay a named array: with a temporary operand NumPy elides the
    temporary and computes ``fftn(A) * K_fft`` in place, and complex multiplication is not
    bitwise commutative in NumPy's SIMD loops (1-ulp differences from LeniaND, which keeps
    ``world_FFT`` as an attribute and multiplies ``kernel_FFT * world_FFT``).
    """
    world_fft = np.fft.fftn(cells)
    potential_fft = kfft * world_fft
    out: F64 = np.fft.fftshift(np.real(np.fft.ifftn(potential_fft)))
    return out


def calc_once(cells: F64, kfft: F64, T: int, m: float, s: float, growth: str) -> F64:
    """One reference Lenia update (default path): A' = clip(A + G(U)/T, 0, 1)."""
    dt = 1 / T
    field = GROWTH[growth](potential(cells, kfft), m, s)
    out: F64 = np.clip(cells + dt * field, 0, 1)
    return out


# --------------------------------------------------------------------------- #
# animals.json RLE (Board.rle2arr / ch2val for 2D)
# --------------------------------------------------------------------------- #


def _ch2val(c: str) -> int:
    if c in ".b":
        return 0
    if c == "o":
        return 255
    if len(c) == 1:
        return ord(c) - ord("A") + 1
    return (ord(c[0]) - ord("p")) * 24 + (ord(c[1]) - ord("A") + 25)


def rle2arr(st: str) -> F64:
    """Decode a 2D Lenia RLE string ('$' separates rows, '!' ends) to values in [0, 1]."""
    rows: list[list[float]] = []
    row: list[float] = []
    last, count = "", ""
    for ch in st.rstrip("!") + "$":
        if ch.isdigit():
            count += ch
        elif ch in "pqrstuvwxy@":
            last = ch
        elif ch == "$" and last == "":
            rows.append(row)
            for _ in range(int(count) - 1 if count else 0):
                rows.append([])
            row, count = [], ""
        else:
            value = _ch2val(last + ch) / 255
            row.extend([value] * (int(count) if count else 1))
            last, count = "", ""
    width = max(len(r) for r in rows)
    return np.array([r + [0.0] * (width - len(r)) for r in rows], dtype=np.float64)


# Chan's animals.json, code "O2u" (Orbium unicaudatus).
ANIMALS: dict[str, dict[str, Any]] = {
    "O2u": {
        "name": "Orbium unicaudatus",
        "params": {
            "R": 13,
            "T": 10,
            "b": [1.0],
            "m": 0.15,
            "s": 0.015,
            "core": "polynomial",
            "growth": "polynomial",
        },
        "cells": (
            "7.MD6.qL$6.pKqEqFURpApBRAqQ$5.VqTrSsBrOpXpWpTpWpUpCrQ$4.CQrQsTsWsApITNPpGqGvL$3.IpIp"
            "WrOsGsBqXpJ4.LsFrL$A.DpKpSpJpDqOqUqSqE5.ExD$qL.pBpTT2.qCrGrVrWqM5.sTpP$.pGpWpD3.qUsMtI"
            "tQtJ6.tL$.uFqGH3.pXtOuR2vFsK5.sM$.tUqL4.GuNwAwVxBwNpC4.qXpA$2.uH5.vBxGyEyMyHtW4.qIpL$"
            "2.wV5.tIyG3yOxQqW2.FqHpJ$2.tUS4.rM2yOyJyOyHtVpPMpFqNV$2.HsR4.pUxAyOxLxDxEuVrMqBqGqKJ$"
            "3.sLpE3.pEuNxHwRwGvUuLsHrCqTpR$3.TrMS2.pFsLvDvPvEuPtNsGrGqIP$4.pRqRpNpFpTrNtGtVtStGsMr"
            "NqNpF$5.pMqKqLqRrIsCsLsIrTrFqJpHE$6.RpSqJqPqVqWqRqKpRXE$8.OpBpIpJpFTK!"
        ),
    },
}


def place_centered(world: F64, part: F64) -> None:
    """Board.add(is_centered=True) for 2D without shift: copy values > 1e-10, wrapping."""
    h1, w1 = world.shape
    h2, w2 = part.shape
    hs, ws = min(h1, h2), min(w1, w2)
    y1, x1 = (h1 - hs) // 2, (w1 - ws) // 2
    y2, x2 = (h2 - hs) // 2, (w2 - ws) // 2
    for y in range(hs):
        for x in range(ws):
            v = part[y2 + y, x2 + x]
            if v > 1e-10:
                world[(y1 + y) % h1, (x1 + x) % w1] = v
