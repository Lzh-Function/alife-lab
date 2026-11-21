"""Pre-declared, versioned search objectives. They only *observe* a target world run headless
with its own adapter; nothing here feeds back into the target dynamics.

``lenia_moving_localized`` (version 1). The candidate world runs for ``eval_steps`` ticks
from its initial state and is checked every ``CHECK_EVERY`` ticks and at the end. A check passes
when the pattern is alive (some cell > 0.01), covers at most ``max_alive_fraction`` of the
world, and is *concentrated*: at least ``CONCENTRATION`` of its mass lies within ``2 R``
(periodic distance) of the mass-weighted circular centroid.

* viable = every check passed;
* speed  = norm of the unwrapped displacement of the circular centroid over the second half,
  divided by the number of ticks (cells per tick; only for viable candidates);
* score  = 0.1 + speed if viable, else 0.1 * (last passing check tick) / eval_steps.

The partial credit for survival time gives optimisers a slope towards persistence. The
concentration test rejects world-filling lattices of small blobs, whose shared centroid
jitters and would otherwise count as motion (observed during development: a first draft with
only an area limit was won by such a lattice). Separate blobs still share one centroid, so
speed is not a creature tracker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alife_lab.core.spec import ExperimentSpec, LeniaExperiment
from alife_lab.evolution import OBJECTIVE_VERSIONS as OBJECTIVE_VERSIONS
from alife_lab.substrates.lenia.lenia import ALIVE_THRESHOLD, LeniaAdapter, _circular_centroid

VIABLE_BONUS = 0.1
CONCENTRATION = 0.9
CHECK_EVERY = 10


def concentration(cells: Any, radius: float) -> float:
    """Fraction of the mass within ``radius`` (periodic distance) of the circular centroid."""
    total = float(cells.sum())
    centre = _circular_centroid(cells)
    if total <= 1e-12 or centre is None:
        return 0.0
    h, w = cells.shape
    dx = np.abs(np.arange(w) - centre[0])
    dy = np.abs(np.arange(h) - centre[1])
    dx = np.minimum(dx, w - dx)
    dy = np.minimum(dy, h - dy)
    inside = (dy[:, None] ** 2 + dx[None, :] ** 2) <= radius**2
    return float(cells[inside].sum()) / total


@dataclass
class Evaluation:
    score: float
    viable: bool
    alive_fraction: float
    concentration: float
    speed: float
    mass: float
    survived: int  # last tick at which every check had passed
    field: Any  # final state (H, W) float64, for display


def evaluate_lenia(spec: ExperimentSpec, steps: int, max_alive_fraction: float) -> Evaluation:
    exp = spec.experiment
    assert isinstance(exp, LeniaExperiment)
    radius = 2.0 * exp.dynamics.kernel.R
    adapter = LeniaAdapter()
    st = adapter.initialize(spec)
    half = steps // 2

    def check() -> tuple[bool, float, float]:
        frac = float((st.cells > ALIVE_THRESHOLD).mean())
        conc = concentration(st.cells, radius) if frac > 0 else 0.0
        return (0.0 < frac <= max_alive_fraction and conc >= CONCENTRATION), frac, conc

    survived = 0
    viable = True
    hx = hy = 0.0
    frac = conc = 0.0
    for t in range(1, steps + 1):
        st = adapter.step(st, spec)
        if t == half:
            hx, hy = st.dx, st.dy
        if t % CHECK_EVERY == 0 or t == steps:
            ok, frac, conc = check()
            if not ok:
                viable = False
                break
            survived = t
    speed = math.hypot(st.dx - hx, st.dy - hy) / max(1, steps - half) if viable else 0.0
    return Evaluation(
        score=VIABLE_BONUS + speed if viable else VIABLE_BONUS * survived / steps,
        viable=viable,
        alive_fraction=frac,
        concentration=conc,
        speed=speed,
        mass=float(st.cells.sum()),
        survived=survived,
        field=st.cells,
    )


@dataclass
class FrameRollout:
    frames: Any  # (K, H, W) float64 snapshots at the ends of K equal chunks
    alive: bool  # some cell > 0.01 at the end and alive fraction <= max_alive_fraction
    alive_fraction: float
    speed: float  # centroid displacement over the second half / ticks
    mass: float
    field: Any


def centred_crop(img: Any, weights: Any, side: int) -> Any:
    """Square of ``side`` around the circular centroid of ``weights`` (periodic world).
    ``img`` is (H, W) or (H, W, 3); an empty world is cropped around the world centre."""
    h, w = weights.shape
    centre = _circular_centroid(weights)
    cx, cy = (w / 2, h / 2) if centre is None else centre
    rolled = np.roll(img, (h // 2 - round(cy), w // 2 - round(cx)), axis=(0, 1))
    y0, x0 = h // 2 - side // 2, w // 2 - side // 2
    return rolled[y0 : y0 + side, x0 : x0 + side]


def rollout_lenia_frames(
    spec: ExperimentSpec,
    steps: int,
    frames: int,
    max_alive_fraction: float,
    crop: int | None = None,
) -> FrameRollout:
    """Run the target's unchanged adapter (Lenia or multichannel Lenia) and keep ``frames``
    snapshots at ticks ``steps * k / frames`` (k = 1..frames), as ASAL's
    ``time_sampling=(frames, True)``. Lenia frames are grey (H, W); multichannel frames are RGB
    (H, W, 3) with channels 0/1/2 as red/green/blue. ``crop`` cuts a square centred on the
    pattern (ASAL shows its multichannel Lenia as a 64 x 64 centred "phenotype")."""
    from alife_lab.core.spec import LeniaMCExperiment
    from alife_lab.substrates.lenia_mc.adapter import ALIVE_THRESHOLD as MC_ALIVE
    from alife_lab.substrates.lenia_mc.adapter import LeniaMCAdapter

    multi = isinstance(spec.experiment, LeniaMCExperiment)
    adapter: Any = LeniaMCAdapter() if multi else LeniaAdapter()
    st = adapter.initialize(spec)
    half = steps // 2
    marks = {steps * (k + 1) // frames for k in range(frames)}
    shots = []
    hx = hy = 0.0

    def snapshot() -> tuple[Any, Any]:
        if multi:
            return LeniaMCAdapter.rgb(st.cells), st.cells.sum(axis=0)
        cells = np.array(st.cells, dtype=np.float64)
        return cells, cells

    for t in range(1, steps + 1):
        st = adapter.step(st, spec)
        if t == half:
            hx, hy = st.dx, st.dy
        if t in marks:
            img, weights = snapshot()
            shots.append(centred_crop(img, weights, crop) if crop else img)
    img, weights = snapshot()
    frac = float((weights > (MC_ALIVE if multi else ALIVE_THRESHOLD)).mean())
    alive = 0.0 < frac <= max_alive_fraction
    return FrameRollout(
        frames=np.stack(shots),
        alive=alive,
        alive_fraction=frac,
        speed=math.hypot(st.dx - hx, st.dy - hy) / max(1, steps - half) if alive else 0.0,
        mass=float(st.cells.sum()),
        field=img,
    )


UPRIGHT_MIN = 0.5  # torso tilted less than 60 degrees from vertical


@dataclass
class CreatureEvaluation:
    score: float
    viable: bool
    upright_fraction: float
    speed: float  # displacement / simulated time (m/s)
    mass: float  # total body mass (kg): a morphology descriptor
    displacement: float
    energy: float


def evaluate_creature(spec: ExperimentSpec, ticks: int) -> CreatureEvaluation:
    """``creature_distance`` (version 1): planar displacement of the torso after ``ticks``
    ticks multiplied by the fraction of ticks with uprightness > 0.5. Flipping over is
    penalised instead of rewarded; viable = upright on at least 90 % of the ticks."""
    from alife_lab.core.spec import CreatureExperiment
    from alife_lab.substrates.creature import model as creature

    exp = spec.experiment
    assert isinstance(exp, CreatureExperiment)
    w = creature.make_world(exp)
    up = 0
    for _ in range(ticks):
        creature.step(w, exp)
        up += creature.upright(w) > UPRIGHT_MIN
    q = w.data.qpos
    disp = math.hypot(q[0] - w.start_xy[0], q[1] - w.start_xy[1])
    frac = up / max(1, ticks)
    return CreatureEvaluation(
        score=disp * frac,
        viable=frac >= 0.9,
        upright_fraction=frac,
        speed=disp / max(1e-9, float(w.data.time)),
        mass=float(w.model.body_subtreemass[1]),
        displacement=disp,
        energy=w.energy,
    )
