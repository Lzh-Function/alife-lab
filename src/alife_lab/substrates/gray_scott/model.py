"""Gray-Scott numerics: 5-point periodic Laplacian and explicit Euler (Pearson 1993).

    du/dt = Du lap(u) - u v^2 + F (1 - u)
    dv/dt = Dv lap(v) + u v^2 - (F + k) v

J. E. Pearson, "Complex Patterns in a Simple System", Science 261:189-192 (1993): periodic
2.5 x 2.5 domain on 256 x 256 points, Du = 2e-5, Dv = 1e-5, forward Euler with dt = 1.
Works on NumPy or CuPy arrays (same expressions; the summation order is fixed).
"""

from __future__ import annotations

from typing import Any

from alife_lab.substrates._xp import xp_of


def laplacian(a: Any, dx: float) -> Any:
    xp = xp_of(a)
    up = xp.roll(a, 1, axis=0)
    down = xp.roll(a, -1, axis=0)
    left = xp.roll(a, 1, axis=1)
    right = xp.roll(a, -1, axis=1)
    return ((((up + down) + left) + right) - 4.0 * a) / (dx * dx)


def rates(
    u: Any, v: Any, *, Du: float, Dv: float, F: float, k: float, dx: float, reaction: bool
) -> tuple[Any, Any, Any]:
    """(du/dt, dv/dt, u v^2)."""
    uvv = u * v * v
    du = Du * laplacian(u, dx)
    dv = Dv * laplacian(v, dx)
    if reaction:
        du = (du - uvv) + F * (1.0 - u)
        dv = (dv + uvv) - (F + k) * v
    return du, dv, uvv


def step(
    u: Any, v: Any, *, Du: float, Dv: float, F: float, k: float, dt: float, dx: float,
    reaction: bool,
) -> tuple[Any, Any]:  # fmt: skip
    du, dv, _ = rates(u, v, Du=Du, Dv=Dv, F=F, k=k, dx=dx, reaction=reaction)
    return u + dt * du, v + dt * dv


def step_oracle(
    u: list[list[float]], v: list[list[float]], *, Du: float, Dv: float, F: float, k: float,
    dt: float, dx: float, reaction: bool,
) -> tuple[list[list[float]], list[list[float]]]:  # fmt: skip
    """Scalar reference loop (tests only): the same expression per cell, Python floats."""
    n, m = len(u), len(u[0])

    def lap(a: list[list[float]], i: int, j: int) -> float:
        s = ((a[(i - 1) % n][j] + a[(i + 1) % n][j]) + a[i][(j - 1) % m]) + a[i][(j + 1) % m]
        return (s - 4.0 * a[i][j]) / (dx * dx)

    nu = [[0.0] * m for _ in range(n)]
    nv = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            uvv = u[i][j] * v[i][j] * v[i][j]
            du = Du * lap(u, i, j)
            dv = Dv * lap(v, i, j)
            if reaction:
                du = (du - uvv) + F * (1.0 - u[i][j])
                dv = (dv + uvv) - (F + k) * v[i][j]
            nu[i][j] = u[i][j] + dt * du
            nv[i][j] = v[i][j] + dt * dv
    return nu, nv
