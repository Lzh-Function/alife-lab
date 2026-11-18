"""Conway's Game of Life and outer-totalistic Life-like CA (CPU / NumPy).

Scientific contract:
- State: 2D uint8 grid of {0, 1}; clock unit ``tick``; exact integer dynamics.
- Update: synchronous Moore-neighbourhood outer-totalistic rule ``B…/S…``.
- Boundary: ``periodic`` (torus) or ``dead`` (cells outside are always 0).
- Randomness is consumed only during initialization (preset ``random``).

Observers (centroid tracking) are passive and never change the grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ConwayExperiment, ExperimentSpec
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

Grid = npt.NDArray[np.uint8]

SERIALIZE_VERSION = 1
MAX_BRUSH_CELLS = 65_536
MAX_BRUSH_RADIUS = 64

PATTERNS: dict[str, tuple[tuple[int, int], ...]] = {
    # (x, y) offsets of live cells. The glider travels towards +x, +y.
    "glider": ((1, 0), (2, 1), (0, 2), (1, 2), (2, 2)),
    "blinker": ((0, 1), (1, 1), (2, 1)),
    "block": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "r_pentomino": ((1, 0), (2, 0), (0, 1), (1, 1), (1, 2)),
}


def parse_rule(rule: str) -> tuple[frozenset[int], frozenset[int]]:
    birth_part, survive_part = rule.split("/")
    return frozenset(int(c) for c in birth_part[1:]), frozenset(int(c) for c in survive_part[1:])


def _rule_tables(rule: str) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    birth, survive = parse_rule(rule)
    birth_table = np.zeros(9, dtype=np.bool_)
    survive_table = np.zeros(9, dtype=np.bool_)
    birth_table[list(birth)] = True
    survive_table[list(survive)] = True
    return birth_table, survive_table


def neighbour_counts(grid: Grid, boundary: str) -> npt.NDArray[np.uint8]:
    if boundary == "periodic":
        counts = np.zeros(grid.shape, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    counts += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        return counts
    padded = np.pad(grid, 1, mode="constant", constant_values=0)
    h, w = grid.shape
    counts = np.zeros(grid.shape, dtype=np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dx != 1 or dy != 1:
                counts += padded[dy : dy + h, dx : dx + w]
    return counts


def step_grid(grid: Grid, rule: str, boundary: str) -> Grid:
    birth_table, survive_table = _rule_tables(rule)
    counts = neighbour_counts(grid, boundary)
    alive = grid.astype(np.bool_)
    new = np.where(alive, survive_table[counts], birth_table[counts])
    return new.astype(np.uint8)


def step_grid_oracle(cells: list[list[int]], rule: str, boundary: str) -> list[list[int]]:
    """Scalar pure-Python reference implementation used only by tests."""
    birth, survive = parse_rule(rule)
    h = len(cells)
    w = len(cells[0])
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            n = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if boundary == "periodic":
                        n += cells[yy % h][xx % w]
                    elif 0 <= yy < h and 0 <= xx < w:
                        n += cells[yy][xx]
            alive = cells[y][x] == 1
            out[y][x] = 1 if (n in survive if alive else n in birth) else 0
    return out


# --------------------------------------------------------------------------- #
# Passive observer: centroid tracking
# --------------------------------------------------------------------------- #


def _axis_centroid(coords: npt.NDArray[np.int64], size: int, periodic: bool) -> float | None:
    if coords.size == 0:
        return None
    if not periodic:
        return float(coords.mean())
    angles = coords.astype(np.float64) * (2.0 * math.pi / size)
    s = float(np.sin(angles).mean())
    c = float(np.cos(angles).mean())
    if math.hypot(s, c) < 1e-9:
        return None  # circular mean undefined (e.g. uniform ring)
    return (math.atan2(s, c) * size / (2.0 * math.pi)) % size


def centroid(grid: Grid, periodic: bool) -> tuple[float, float] | None:
    ys, xs = np.nonzero(grid)
    h, w = grid.shape
    cx = _axis_centroid(xs.astype(np.int64), w, periodic)
    cy = _axis_centroid(ys.astype(np.int64), h, periodic)
    if cx is None or cy is None:
        return None
    return cx, cy


def _minimal_image(delta: float, size: int) -> float:
    return (delta + size / 2.0) % size - size / 2.0


@dataclass
class CentroidTracker:
    """Unwrapped live-cell centroid displacement since the last reset."""

    last: tuple[float, float] | None = None
    dx: float = 0.0
    dy: float = 0.0
    valid: bool = True

    def update(self, grid: Grid, periodic: bool) -> None:
        current = centroid(grid, periodic)
        if current is None:
            self.valid = False
            self.last = None
            return
        if self.last is not None:
            h, w = grid.shape
            ddx = current[0] - self.last[0]
            ddy = current[1] - self.last[1]
            if periodic:
                ddx = _minimal_image(ddx, w)
                ddy = _minimal_image(ddy, h)
            self.dx += ddx
            self.dy += ddy
        self.last = current

    def to_json(self) -> dict[str, Any]:
        return {
            "last": list(self.last) if self.last is not None else None,
            "dx": self.dx,
            "dy": self.dy,
            "valid": self.valid,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CentroidTracker:
        last = data.get("last")
        return cls(
            last=(float(last[0]), float(last[1])) if last is not None else None,
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            valid=bool(data["valid"]),
        )


@dataclass
class ConwayState:
    grid: Grid
    tick: int
    tracker: CentroidTracker = field(default_factory=CentroidTracker)
    rng_state: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #

_METRICS = (
    MetricDefinition(
        id="alive_cell_count",
        display_name_key="metric.alive_cell_count",
        unit="cells",
        clock="tick",
        definition="Number of cells with state 1 after the update of the given tick.",
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="never null",
    ),
    MetricDefinition(
        id="centroid_displacement",
        display_name_key="metric.centroid_displacement",
        unit="cells",
        clock="tick",
        definition=(
            "Euclidean norm of the unwrapped displacement of the live-cell centroid since tick 0 "
            "(or since the last intervention). On periodic boundaries the per-axis centroid is the "
            "circular mean and per-tick deltas use the minimal-image convention."
        ),
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="null when the grid is empty or the circular mean is undefined",
    ),
    MetricDefinition(
        id="glider_displacement",
        display_name_key="metric.glider_displacement",
        unit="cells",
        clock="tick",
        definition=(
            "Alias of centroid_displacement. Interpretable as glider displacement only when the "
            "world contains exactly one glider and nothing else; it is not a glider detector."
        ),
        aggregation="none",
        sampling_interval="every tick",
        implementation_version=1,
        null_policy="same as centroid_displacement",
    ),
)

CONWAY_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.ca.conway",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="conway",
    display_name_key="substrate.conway",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="grid",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=False,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("brush",),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Conway's Game of Life, rule B3/S23 (Gardner 1970).",
        "paper_urls": [],
        "implementation_commit": None,
    },
)


def _conway(spec: ExperimentSpec) -> ConwayExperiment:
    exp = spec.experiment
    if not isinstance(exp, ConwayExperiment):
        raise TypeError("conway adapter received a non-conway spec")
    return exp


def _place_pattern(grid: Grid, name: str, offset_x: int | None, offset_y: int | None) -> None:
    cells = PATTERNS[name]
    pw = max(x for x, _ in cells) + 1
    ph = max(y for _, y in cells) + 1
    h, w = grid.shape
    ox = (w - pw) // 2 if offset_x is None else offset_x
    oy = (h - ph) // 2 if offset_y is None else offset_y
    if ox + pw > w or oy + ph > h:
        raise ValueError(f"pattern {name!r} does not fit at offset ({ox}, {oy})")
    for x, y in cells:
        grid[oy + y, ox + x] = 1


class ConwayAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return CONWAY_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> ConwayState:
        exp = _conway(spec)
        h, w = exp.world.height, exp.world.width
        grid: Grid = np.zeros((h, w), dtype=np.uint8)
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        init = exp.initialization
        if init.preset == "random":
            grid = (rng.random((h, w)) < init.density).astype(np.uint8)
        elif init.preset != "empty":
            _place_pattern(grid, init.preset, init.offset_x, init.offset_y)
        state = ConwayState(grid=grid, tick=0, rng_state=dict(rng.bit_generator.state))
        state.tracker.update(grid, exp.world.boundary == "periodic")
        return state

    def step(self, state: ConwayState, spec: ExperimentSpec) -> ConwayState:
        exp = _conway(spec)
        grid = step_grid(state.grid, exp.dynamics.rule, exp.world.boundary)
        state.grid = grid
        state.tick += 1
        state.tracker.update(grid, exp.world.boundary == "periodic")
        return state

    def metrics(self, state: ConwayState, spec: ExperimentSpec) -> dict[str, float]:
        tracker = state.tracker
        displacement = math.hypot(tracker.dx, tracker.dy) if tracker.valid else math.nan
        out: dict[str, float] = {"alive_cell_count": float(int(state.grid.sum()))}
        for metric in _conway(spec).observations.metrics:
            if metric in ("centroid_displacement", "glider_displacement"):
                out[metric] = displacement
        return out

    def render_payload(self, state: ConwayState) -> bytes:
        return np.packbits(state.grid, axis=None).tobytes()

    def render(self, state: ConwayState) -> tuple[dict[str, Any], bytes]:
        h, w = state.grid.shape
        return {"codec": "bitpack1", "width": int(w), "height": int(h)}, self.render_payload(state)

    def inspect(
        self, state: ConwayState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        """Cell, live-neighbour count, next-tick outcome and a small patch (read-only)."""
        exp = _conway(spec)
        h, w = state.grid.shape
        x, y = int(query["x"]), int(query["y"])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        r = max(0, min(int(query.get("radius", 2)), 8))
        periodic = exp.world.boundary == "periodic"
        counts = neighbour_counts(state.grid, exp.world.boundary)
        birth, survive = parse_rule(exp.dynamics.rule)
        alive = bool(state.grid[y, x])
        n = int(counts[y, x])
        if alive:
            outcome = "survives" if n in survive else "dies"
        else:
            outcome = "born" if n in birth else "stays_dead"
        patch: list[list[int]] = []
        patch_counts: list[list[int | None]] = []
        for yy in range(y - r, y + r + 1):
            row: list[int] = []
            row_n: list[int | None] = []
            for xx in range(x - r, x + r + 1):
                if periodic:
                    row.append(int(state.grid[yy % h, xx % w]))
                    row_n.append(int(counts[yy % h, xx % w]))
                elif 0 <= yy < h and 0 <= xx < w:
                    row.append(int(state.grid[yy, xx]))
                    row_n.append(int(counts[yy, xx]))
                else:
                    row.append(-1)  # outside the world (dead boundary)
                    row_n.append(None)
            patch.append(row)
            patch_counts.append(row_n)
        return {
            "substrate": "conway",
            "tick": state.tick,
            "query": {"x": x, "y": y, "radius": r},
            "cell": {"alive": alive, "live_neighbors": n, "next_tick": outcome},
            "rule": exp.dynamics.rule,
            "patch": patch,
            "patch_live_neighbors": patch_counts,
            "patch_origin": {"x": x - r, "y": y - r},
        }

    def state_distance(self, a: ConwayState, b: ConwayState) -> dict[str, Any]:
        if a.grid.shape != b.grid.shape:
            return {"metric": "hamming", "value": None, "exact_equal": False}
        differing = int(np.count_nonzero(a.grid != b.grid))
        return {"metric": "hamming", "value": differing, "exact_equal": differing == 0}

    def apply_intervention(
        self,
        state: ConwayState,
        spec: ExperimentSpec,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[ConwayState, dict[str, Any]]:
        if kind != "brush":
            raise ValueError(f"unsupported intervention kind {kind!r}")
        h, w = state.grid.shape
        value = int(payload.get("value", 1))
        if value not in (0, 1):
            raise ValueError("brush value must be 0 or 1")
        cells = brush_cells(payload, w, h)
        before = int(state.grid.sum())
        if cells:
            ys = np.fromiter((y for _, y in cells), dtype=np.int64, count=len(cells))
            xs = np.fromiter((x for x, _ in cells), dtype=np.int64, count=len(cells))
            state.grid[ys, xs] = value
        after = int(state.grid.sum())
        # The tracker is reset: displacement is measured from the edited state onwards.
        state.tracker = CentroidTracker()
        state.tracker.update(state.grid, _conway(spec).world.boundary == "periodic")
        return state, {"cells_targeted": len(cells), "alive_before": before, "alive_after": after}

    def serialize_state(self, state: ConwayState) -> dict[str, Any]:
        return {
            "arrays": {"grid": state.grid},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "tracker": state.tracker.to_json(),
                "rng_state": state.rng_state,
                "dtype": "uint8",
                "shape": list(state.grid.shape),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> ConwayState:
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible conway checkpoint serialize_version")
        grid = np.asarray(payload["arrays"]["grid"], dtype=np.uint8)
        world = _conway(spec).world
        expected = (world.height, world.width)
        if grid.shape != expected:
            raise ValueError(f"checkpoint grid shape {grid.shape} != spec shape {expected}")
        if grid.size and int(grid.max()) > 1:
            raise ValueError("checkpoint grid contains values other than 0/1")
        return ConwayState(
            grid=grid.copy(),
            tick=int(meta["tick"]),
            tracker=CentroidTracker.from_json(meta["tracker"]),
            rng_state=dict(meta.get("rng_state", {})),
        )


def brush_cells(payload: dict[str, Any], width: int, height: int) -> list[tuple[int, int]]:
    """Resolve a brush payload into an explicit, bounded, de-duplicated cell list."""
    shape = payload.get("shape", "cells")
    result: set[tuple[int, int]] = set()
    if shape == "cells":
        raw = payload.get("cells", [])
        if not isinstance(raw, list) or len(raw) > MAX_BRUSH_CELLS:
            raise ValueError(f"cells must be a list of at most {MAX_BRUSH_CELLS} [x, y] pairs")
        for item in raw:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError("each cell must be an [x, y] pair")
            x, y = int(item[0]), int(item[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"cell ({x}, {y}) is outside the world")
            result.add((x, y))
    elif shape == "circle":
        cx, cy = int(payload["x"]), int(payload["y"])
        radius = int(payload.get("radius", 1))
        if not (0 <= cx < width and 0 <= cy < height):
            raise ValueError("circle center is outside the world")
        if not (0 <= radius <= MAX_BRUSH_RADIUS):
            raise ValueError(f"radius must be within 0..{MAX_BRUSH_RADIUS}")
        r2 = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= r2:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        result.add((x, y))
    else:
        raise ValueError(f"unsupported brush shape {shape!r}")
    return sorted(result)
