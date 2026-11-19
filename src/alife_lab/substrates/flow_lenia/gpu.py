"""CUDA reintegration tracking for Flow-Lenia (CuPy, float64).

A gather formulation of :func:`core.reintegrate`: every target cell loops over the
(2dd+1)^2 source window in the same order as the NumPy loop (dx outer, dy inner) and evaluates
the same expression ``src * (ox * oy) / norm``. FMA contraction is disabled, so for identical
inputs the result is bit-identical to the NumPy implementation (tested). CuPy is imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SOURCE = r"""
extern "C" __global__ void reintegrate(const double* cells, const double* mu, double* out,
                                       const int h, const int w, const int c, const int dd,
                                       const double sigma, const int torus) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)h * w * c;
    if (idx >= total) return;
    const int ch = (int)(idx % c);
    const int j = (int)((idx / c) % w);
    const int i = (int)(idx / ((long long)c * w));
    const double cap = sigma * 2.0 < 1.0 ? sigma * 2.0 : 1.0;
    const double norm = 4.0 * (sigma * sigma);
    const double py = i + 0.5, px = j + 0.5;
    double acc = 0.0;
    for (int dx = -dd; dx <= dd; ++dx) {
        int si = (i - dx) % h; if (si < 0) si += h;
        for (int dy = -dd; dy <= dd; ++dy) {
            int sj = (j - dy) % w; if (sj < 0) sj += w;
            const long long s = ((long long)si * w + sj);
            const double src = cells[s * c + ch];
            const double my = mu[(s * 2 + 0) * c + ch];
            const double mx = mu[(s * 2 + 1) * c + ch];
            double d0 = fabs(py - my), d1 = fabs(px - mx);
            if (torus) {
                const double e0 = fabs(d0 - (double)h), e1 = fabs(d1 - (double)w);
                d0 = d0 < e0 ? d0 : e0;
                d1 = d1 < e1 ? d1 : e1;
            }
            double o0 = 0.5 - d0 + sigma, o1 = 0.5 - d1 + sigma;
            o0 = o0 < 0.0 ? 0.0 : (o0 > cap ? cap : o0);
            o1 = o1 < 0.0 ? 0.0 : (o1 > cap ? cap : o1);
            acc = acc + src * (o0 * o1) / norm;
        }
    }
    out[idx] = acc;
}
"""

_MODULE: Any = None
BLOCK = 256


def _kernel() -> Any:
    global _MODULE
    if _MODULE is None:
        import cupy

        _MODULE = cupy.RawModule(code=_SOURCE, options=("--fmad=false",))
    return _MODULE.get_function("reintegrate")


def reintegrate_cupy(cells: Any, mu: Any, dd: int, sigma: float, boundary: str) -> Any:
    import cupy

    h, w, c = cells.shape
    cells = cupy.ascontiguousarray(cells, dtype=cupy.float64)
    mu = cupy.ascontiguousarray(mu, dtype=cupy.float64)
    out = cupy.empty_like(cells)
    total = h * w * c
    _kernel()(
        ((total + BLOCK - 1) // BLOCK,),
        (BLOCK,),
        (
            cells, mu, out, np.int32(h), np.int32(w), np.int32(c), np.int32(dd),
            np.float64(sigma), np.int32(boundary == "torus"),
        ),
    )  # fmt: skip
    return out
