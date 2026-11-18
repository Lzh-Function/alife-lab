"""Flow-Lenia numerics (NumPy, float64).

Reimplemented from the description in Plantec et al., "Flow-Lenia: Towards open-ended evolution
in cellular automata through mass conservation and parameter localization" (ALIFE 2023,
arXiv:2212.07906), following the semantics of the authors' JAX code
(github.com/erwanplantec/FlowLenia @ dce428c6, no licence; no code copied):

- kernels ``K = sigmoid(-10 (D - 1)) * sum_j b_j exp(-(D - a_j)^2 / w_j)`` with
  ``D = |x| / ((R + 15) r)``, normalised to sum 1, applied by periodic FFT convolution;
- growth ``G = (2 exp(-((U - m) / s)^2 / 2) - 1) * h``, summed per target channel (affinity U);
- flow ``F = (1 - alpha) grad(U) - alpha grad(sum_c A)``, ``alpha = clip((A / theta_A)^n, 0, 1)``
  with a zero-padded Sobel gradient (also on the torus);
- reintegration tracking: the mass of each cell moves to ``pos + clip(dt F, -(dd - sigma),
  dd - sigma)`` and is spread uniformly over a square of half-size ``sigma``; each target cell
  receives the overlap area / (2 sigma)^2. Total mass is conserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

# Official sampling ranges (FlowLenia.__init__).
RANGES = {
    "R": (2.0, 25.0),
    "r": (0.2, 1.0),
    "m": (0.05, 0.5),
    "s": (0.001, 0.18),
    "h": (0.01, 1.0),
    "a": (0.0, 1.0),
    "b": (0.001, 1.0),
    "w": (0.01, 0.5),
}


@dataclass(frozen=True)
class Rule:
    R: float
    source: I64  # (k,)
    target: I64  # (k,)
    r: F64
    m: F64
    s: F64
    h: F64
    a: F64  # (k, 3)
    b: F64
    w: F64

    @property
    def k(self) -> int:
        return int(self.source.shape[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "R": self.R,
            "kernels": [
                {
                    "source": int(self.source[i]),
                    "target": int(self.target[i]),
                    "r": float(self.r[i]),
                    "m": float(self.m[i]),
                    "s": float(self.s[i]),
                    "h": float(self.h[i]),
                    "a": [float(v) for v in self.a[i]],
                    "b": [float(v) for v in self.b[i]],
                    "w": [float(v) for v in self.w[i]],
                }
                for i in range(self.k)
            ],
        }


def rule_from_kernels(R: float, kernels: list[dict[str, Any]]) -> Rule:
    def col(name: str) -> F64:
        return np.array([k[name] for k in kernels], dtype=np.float64)

    return Rule(
        R=float(R),
        source=np.array([k["source"] for k in kernels], dtype=np.int64),
        target=np.array([k["target"] for k in kernels], dtype=np.int64),
        r=col("r"),
        m=col("m"),
        s=col("s"),
        h=col("h"),
        a=col("a").reshape(-1, 3),
        b=col("b").reshape(-1, 3),
        w=col("w").reshape(-1, 3),
    )


def conn_from_matrix(matrix: list[list[int]]) -> tuple[list[int], list[int]]:
    """Kernel (source, target) lists in the official ``conn_from_matrix`` order."""
    sources: list[int] = []
    targets: list[int] = []
    for s, row in enumerate(matrix):
        for t, n in enumerate(row):
            sources += [s] * n
            targets += [t] * n
    return sources, targets


def sample_rule(rng: np.random.Generator, connectivity: list[list[int]], R: float | None) -> Rule:
    """Draw kernel parameters from the official uniform ranges (NumPy PCG64, not JAX)."""
    sources, targets = conn_from_matrix(connectivity)
    k = len(sources)

    def u(name: str, shape: tuple[int, ...]) -> F64:
        lo, hi = RANGES[name]
        return rng.uniform(lo, hi, size=shape)

    r_draw = float(u("R", ())) if R is None else float(R)
    return Rule(
        R=r_draw,
        source=np.array(sources, dtype=np.int64),
        target=np.array(targets, dtype=np.int64),
        r=u("r", (k,)),
        m=u("m", (k,)),
        s=u("s", (k,)),
        h=u("h", (k,)),
        a=u("a", (k, 3)),
        b=u("b", (k, 3)),
        w=u("w", (k, 3)),
    )


def _signed_offsets(n: int) -> F64:
    """Periodic offsets from cell 0 (== fftshift(arange(-n//2, n//2)) for even n)."""
    idx = np.arange(n)
    return np.where(idx < (n + 1) // 2, idx, idx - n).astype(np.float64)


def sigmoid(x: F64) -> F64:
    out: F64 = 0.5 * (np.tanh(x / 2) + 1)
    return out


def kernels_spatial(height: int, width: int, rule: Rule) -> F64:
    """Normalised kernels (H, W, k) with the origin at cell (0, 0) (periodic)."""
    dy = _signed_offsets(height)[:, None]
    dx = _signed_offsets(width)[None, :]
    dist = np.sqrt(dy**2 + dx**2)
    out = np.empty((height, width, rule.k), dtype=np.float64)
    for i in range(rule.k):
        d = dist / ((rule.R + 15) * rule.r[i])
        rings = (rule.b[i] * np.exp(-((d[..., None] - rule.a[i]) ** 2) / rule.w[i])).sum(-1)
        kern = sigmoid(-(d - 1) * 10) * rings
        out[:, :, i] = kern / kern.sum()
    return out


def kernels_fft(height: int, width: int, rule: Rule) -> Any:
    return np.fft.fft2(kernels_spatial(height, width, rule), axes=(0, 1))


def growth(u: F64, m: F64, s: F64) -> F64:
    out: F64 = np.exp(-(((u - m) / s) ** 2) / 2) * 2 - 1
    return out


def affinity(cells: F64, kfft: Any, rule: Rule, channels: int) -> F64:
    """U (H, W, C): sum over kernels targeting each channel of h * G(K * A_source)."""
    fa = np.fft.fft2(cells, axes=(0, 1))
    potential = np.real(np.fft.ifft2(kfft * fa[:, :, rule.source], axes=(0, 1)))
    g = growth(potential, rule.m, rule.s) * rule.h
    out = np.zeros((*cells.shape[:2], channels), dtype=np.float64)
    for c in range(channels):
        sel = rule.target == c
        if sel.any():
            out[:, :, c] = g[:, :, sel].sum(axis=-1)
    return out


def sobel(field: F64) -> F64:
    """Zero-padded Sobel gradient of (H, W, C) -> (H, W, 2, C): [d/d row, d/d column].

    Equals ``scipy.signal.convolve2d(A, k, mode="same")`` with k = [[1,0,-1],[2,0,-2],[1,0,-1]]
    (column) and its transpose (row), i.e. the official ``sobel``.
    """
    p = np.pad(field, ((1, 1), (1, 1), (0, 0)))
    smooth_rows = p[:-2] + 2 * p[1:-1] + p[2:]  # (H, W+2, C)
    smooth_cols = p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]  # (H+2, W, C)
    d_col = smooth_rows[:, 2:] - smooth_rows[:, :-2]
    d_row = smooth_cols[2:] - smooth_cols[:-2]
    return np.stack([d_row, d_col], axis=2)


def flow(cells: F64, u: F64, theta_a: float, n: float) -> tuple[F64, F64]:
    grad_u = sobel(u)
    grad_a = sobel(cells.sum(axis=-1, keepdims=True))
    alpha = np.clip((cells[:, :, None, :] / theta_a) ** n, 0.0, 1.0)
    return grad_u * (1 - alpha) - grad_a * alpha, alpha


def target_positions(
    cells_shape: tuple[int, ...], f: F64, dt: float, dd: int, sigma: float, boundary: str
) -> F64:
    """Distribution centres mu (H, W, 2, C) in continuous coordinates (cell i spans [i, i+1))."""
    h, w = cells_shape[:2]
    pos = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), axis=-1) + 0.5
    ma = dd - sigma
    mu = pos[..., None] + np.clip(dt * f, -ma, ma)
    if boundary == "wall":
        size = np.array([h, w], dtype=np.float64)[None, None, :, None]
        mu = np.clip(mu, sigma, size - sigma)
    return mu


def reintegrate(cells: F64, mu: F64, dd: int, sigma: float, boundary: str) -> F64:
    """Move mass to the squares centred at ``mu``; vectorised over the (2dd+1)^2 window."""
    h, w, _ = cells.shape
    pos = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), axis=-1) + 0.5
    pos = pos[..., None]  # (H, W, 2, 1)
    size = np.array([h, w], dtype=np.float64)[None, None, :, None]
    cap = min(1.0, 2 * sigma)
    norm = 4 * sigma**2
    out = np.zeros_like(cells)
    for dx in range(-dd, dd + 1):
        for dy in range(-dd, dd + 1):
            src = np.roll(cells, (dx, dy), axis=(0, 1))
            mur = np.roll(mu, (dx, dy), axis=(0, 1))
            delta = np.abs(pos - mur)
            if boundary == "torus":
                delta = np.minimum(delta, np.abs(delta - size))
            overlap = np.clip(0.5 - delta + sigma, 0.0, cap)
            out += src * (overlap[:, :, 0] * overlap[:, :, 1]) / norm
    return out


def step(
    cells: F64,
    kfft: Any,
    rule: Rule,
    *,
    dt: float,
    dd: int,
    sigma: float,
    theta_a: float,
    n: float,
    boundary: str,
) -> F64:
    channels = cells.shape[2]
    u = affinity(cells, kfft, rule, channels)
    f, _ = flow(cells, u, theta_a, n)
    mu = target_positions(cells.shape, f, dt, dd, sigma, boundary)
    return reintegrate(cells, mu, dd, sigma, boundary)


def reintegrate_oracle(cells: F64, mu: F64, sigma: float, boundary: str) -> F64:
    """Scalar oracle: exact interval overlaps of each source square with the cells."""
    h, w, c_n = cells.shape
    out = np.zeros_like(cells)
    norm = (2 * sigma) ** 2
    for i in range(h):
        for j in range(w):
            for c in range(c_n):
                mass = cells[i, j, c]
                if mass == 0:
                    continue
                y0, x0 = mu[i, j, 0, c] - sigma, mu[i, j, 1, c] - sigma
                y1, x1 = mu[i, j, 0, c] + sigma, mu[i, j, 1, c] + sigma
                for row in range(int(np.floor(y0)), int(np.floor(y1)) + 1):
                    oy = min(row + 1, y1) - max(row, y0)
                    if oy <= 0:
                        continue
                    for col in range(int(np.floor(x0)), int(np.floor(x1)) + 1):
                        ox = min(col + 1, x1) - max(col, x0)
                        if ox <= 0:
                            continue
                        if boundary == "torus":
                            rr, cc = row % h, col % w
                        else:
                            if not (0 <= row < h and 0 <= col < w):
                                raise AssertionError("wall transport left the world")
                            rr, cc = row, col
                        out[rr, cc, c] += mass * oy * ox / norm
    return out
