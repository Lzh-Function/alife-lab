"""Multichannel Lenia numerics (NumPy, float64), semantics of ``lenia_impl.py`` in
SakanaAI/asal @ 677ba0e (Apache-2.0), itself from maxencefaldor/Leniabreeder (MIT), a version of
Bert Chan's multichannel Lenia (MIT):

* kernel k: D_k = |x| / R * len(b) / r;  K_k = [D_k < len(b)] * b[floor(D_k)] *
  bell(D_k mod 1, 0.5, 0.15), normalised to sum 1 (bell(x, m, s) = exp(-((x - m) / s)^2 / 2));
* U_k = K_k * A[c0] (periodic FFT convolution);  G_k = h_k (2 bell(U_k, m_k, s_k) - 1);
* A[c] <- clip(A[c] + (1 / T) * sum_{k: c1 = c} G_k, 0, 1).

Differences from Bert Chan's ``LeniaNDKC.py``: the kernel core is a Gaussian bump instead of
exp(4 - 1 / (r (1 - r))), and the growth is not divided by sum(h) per target channel. For the
catalogued Aquarium creatures sum(h) = 2 for every channel, so the update equals Chan's
normalised one with the time step doubled. Leniabreeder re-centres the world on the creature
before every step (a cyclic roll, exact on the torus); this implementation keeps the world
fixed and only tracks the shift, so both differ by floating-point rounding only.
Real FFTs (``rfft2``) are used; the convolution is the same up to rounding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import LeniaMCExperiment
from alife_lab.substrates.lenia_mc.patterns import load_pattern

F64 = npt.NDArray[np.float64]
CELLS_SIZE = 32  # Leniabreeder pads the pattern into a 32 x 32 embryo


def bell(x: Any, m: Any, s: Any) -> Any:
    return np.exp(-(((x - m) / s) ** 2) / 2)


def kernels_fft(exp: LeniaMCExperiment) -> Any:
    """rfft2 of the normalised, centred kernels: complex (k, H, W // 2 + 1)."""
    n = exp.world.size
    mid = n // 2
    yy, xx = np.mgrid[-mid : n - mid, -mid : n - mid]
    d = np.sqrt(yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2) / exp.dynamics.R
    out = []
    for k in exp.dynamics.kernels:
        nb = len(k.b)
        dk = d * nb / k.r
        b = np.asarray(k.b, dtype=np.float64)[np.minimum(dk.astype(int), nb - 1)]
        kern = (dk < nb) * b * bell(dk % 1, 0.5, 0.15)
        total = kern.sum()
        kern = kern / total if total > 0 else kern
        out.append(np.fft.rfft2(np.fft.ifftshift(kern)))
    return np.stack(out)


def initial_cells(exp: LeniaMCExperiment) -> F64:
    """World (C, H, W) at tick 0."""
    n = exp.world.size
    c = exp.dynamics.channels
    world = np.zeros((c, n, n))
    init = exp.initialization
    if init.preset == "empty":
        return world
    if init.preset == "pattern":
        cells = np.asarray(load_pattern(init.pattern)["cells"], dtype=np.float64)  # (C, y, x)
        cy, cx = cells.shape[1:]
        py, px = CELLS_SIZE - cy, CELLS_SIZE - cx
        block = np.pad(cells, ((0, 0), (py // 2, py - py // 2), (px // 2, px - px // 2)))
    else:
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        size = min(init.size, n)
        block = rng.random((c, size, size)) * (rng.random((c, size, size)) < init.density)
    by, bx = block.shape[1:]
    mid = n // 2
    world[:, mid - by // 2 : mid + by - by // 2, mid - bx // 2 : mid + bx - bx // 2] = block
    return world


def potentials(world: F64, kfft: Any, exp: LeniaMCExperiment) -> F64:
    """U_k for every kernel: (k, H, W)."""
    fa = np.fft.rfft2(world)
    src = np.array([k.c0 for k in exp.dynamics.kernels])
    out: F64 = np.fft.irfft2(kfft * fa[src], s=world.shape[1:])
    return out


def growth(u: F64, exp: LeniaMCExperiment) -> F64:
    """h_k (2 bell(U_k, m_k, s_k) - 1): (k, H, W)."""
    ks = exp.dynamics.kernels
    m = np.array([k.m for k in ks])[:, None, None]
    s = np.array([k.s for k in ks])[:, None, None]
    h = np.array([k.h for k in ks])[:, None, None]
    out: F64 = h * (2 * bell(u, m, s) - 1)
    return out


def step(world: F64, kfft: Any, exp: LeniaMCExperiment) -> F64:
    g_k = growth(potentials(world, kfft, exp), exp)
    g = np.zeros_like(world)
    for i, k in enumerate(exp.dynamics.kernels):
        g[k.c1] += g_k[i]
    out: F64 = np.clip(world + g / exp.dynamics.T, 0.0, 1.0)
    return out
