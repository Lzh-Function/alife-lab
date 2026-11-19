"""Jones (2010) Physarum-inspired transport network model — numerics.

J. Jones, "Characteristics of pattern formation and evolution in approximations of Physarum
transport networks", Artificial Life 16(2):127-153 (2010). Agents on a periodic lattice with a
trail (chemoattractant) map; implemented from the paper, no code reused.

Per tick (order fixed and documented):
1. **Motor stage**, agents in a random order: move ``step_size`` along the heading; with
   ``exclusion`` a move into a cell occupied by another agent fails and the agent takes a
   random new heading; a successful move deposits ``deposit`` into the new cell.
2. **Sensory stage** (all agents, reads only): sample the trail at three sensors at distance
   ``sensor_offset`` (front, front-left +SA, front-right -SA). F > L and F > R: keep heading;
   F < L and F < R: rotate by +-RA at random; L > R: rotate +RA; R > L: rotate -RA. Then, with
   probability ``p_random_turn``, a uniformly random heading.
3. Food sources add their amount to the trail in a disc.
4. Trail diffusion (3 x 3 mean filter, periodic), then decay ``trail *= 1 - decay``.

All random numbers of a tick are drawn up front in a fixed order, so the sequential motor loop
(Python oracle or Numba) is deterministic and backend-independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TickRandoms:
    order: I64
    blocked_heading: F64
    turn_left: npt.NDArray[np.bool_]
    random_turn: F64
    random_heading: F64


def draw_randoms(rng: np.random.Generator, n: int) -> TickRandoms:
    return TickRandoms(
        order=rng.permutation(n).astype(np.int64),
        blocked_heading=rng.random(n) * TWO_PI,
        turn_left=rng.random(n) < 0.5,
        random_turn=rng.random(n),
        random_heading=rng.random(n) * TWO_PI,
    )


def _motor_python(
    x: F64, y: F64, h: F64, dx: F64, dy: F64, occ: Any, trail: F64, order: I64,
    blocked_heading: F64, deposit: float, exclusion: bool,
) -> int:  # fmt: skip
    """Sequential motor stage (reference loop). Mutates x, y, h, occ, trail; returns blocked."""
    hgt, wid = trail.shape
    blocked = 0
    for a in order:
        nx = (x[a] + dx[a]) % wid
        ny = (y[a] + dy[a]) % hgt
        ci, cj = math.floor(y[a]) % hgt, math.floor(x[a]) % wid
        ni, nj = math.floor(ny) % hgt, math.floor(nx) % wid
        if exclusion and (ni != ci or nj != cj):
            if occ[ni, nj]:
                h[a] = blocked_heading[a]
                blocked += 1
                continue
            occ[ci, cj] = False
            occ[ni, nj] = True
        x[a] = nx
        y[a] = ny
        trail[ni, nj] += deposit
    return blocked


_MOTOR_JIT: Any = None


def _motor(*args: Any) -> int:
    """Numba-compiled copy of the reference loop when available (identical arithmetic)."""
    global _MOTOR_JIT
    if _MOTOR_JIT is None:
        try:
            from numba import njit

            _MOTOR_JIT = njit(cache=True)(_motor_python)
        except ImportError:  # pragma: no cover - numba extra not installed
            _MOTOR_JIT = _motor_python
    return int(_MOTOR_JIT(*args))


def motor_stage(
    x: F64, y: F64, h: F64, occ: Any, trail: F64, r: TickRandoms, *, step_size: float,
    deposit: float, exclusion: bool, jit: bool = True,
) -> int:  # fmt: skip
    dx = step_size * np.cos(h)
    dy = step_size * np.sin(h)
    if not exclusion:  # order-independent: vectorised
        hgt, wid = trail.shape
        x[:] = (x + dx) % wid
        y[:] = (y + dy) % hgt
        ni = np.floor(y).astype(np.int64) % hgt
        nj = np.floor(x).astype(np.int64) % wid
        np.add.at(trail, (ni, nj), deposit)
        return 0
    fn = _motor if jit else _motor_python
    return fn(x, y, h, dx, dy, occ, trail, r.order, r.blocked_heading, deposit, exclusion)


def sensor_values(
    x: F64, y: F64, h: F64, trail: F64, angle: float, offset: float
) -> tuple[F64, F64, F64]:
    hgt, wid = trail.shape

    def sample(theta: F64) -> F64:
        sx = np.floor(x + offset * np.cos(theta)).astype(np.int64) % wid
        sy = np.floor(y + offset * np.sin(theta)).astype(np.int64) % hgt
        out: F64 = trail[sy, sx]
        return out

    return sample(h), sample(h + angle), sample(h - angle)


def sensory_stage(
    x: F64, y: F64, h: F64, trail: F64, r: TickRandoms, *, sensor_angle: float,
    sensor_offset: float, rotation_angle: float, p_random_turn: float, sensing: bool,
) -> None:  # fmt: skip
    if sensing:
        f, left, right = sensor_values(x, y, h, trail, sensor_angle, sensor_offset)
        keep = (f > left) & (f > right)
        both = ~keep & (f < left) & (f < right)
        to_left = ~keep & ~both & (left > right)
        to_right = ~keep & ~both & (right > left)
        h[both] += np.where(r.turn_left[both], rotation_angle, -rotation_angle)
        h[to_left] += rotation_angle
        h[to_right] -= rotation_angle
    if p_random_turn > 0:
        hit = r.random_turn < p_random_turn
        h[hit] = r.random_heading[hit]
    h %= TWO_PI


def diffuse_decay(trail: F64, decay: float) -> F64:
    acc = np.zeros_like(trail)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            acc += np.roll(trail, (di, dj), axis=(0, 1))
    out: F64 = (acc / 9.0) * (1.0 - decay)
    return out


def disc_mask(height: int, width: int, x: int, y: int, radius: int) -> npt.NDArray[np.bool_]:
    yy, xx = np.ogrid[0:height, 0:width]
    dyy = (yy - y + height // 2) % height - height // 2
    dxx = (xx - x + width // 2) % width - width // 2
    out: npt.NDArray[np.bool_] = dxx * dxx + dyy * dyy <= radius * radius
    return out
