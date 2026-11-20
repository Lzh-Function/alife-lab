"""Display and inspection of a search run (read-only).

Frame layout (RGB, field_u8): left an S x S map of the first two search parameters (normalised
box; one dot per evaluated candidate, colour = score, grey = not viable (brighter =
survived longer), white ring = best,
red = CMA-ES mean), right the final state of the best candidate on its first search seed.
S is the target world's height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alife_lab.core.spec import SearchExperiment
from alife_lab.evolution.space import candidate_spec, decode, seeds_for

if TYPE_CHECKING:
    from alife_lab.substrates.search.adapter import SearchState

# A few viridis anchors (0 -> 1), linearly interpolated.
_VIRIDIS = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float64
)
_BG = np.array([16, 20, 28], dtype=np.float64)
_GRID = np.array([34, 40, 52], dtype=np.float64)
_DEAD = np.array([90, 96, 110], dtype=np.float64)


def _cmap(v: Any) -> Any:
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.minimum(np.floor(v).astype(int), len(_VIRIDIS) - 2)
    t = (v - i)[..., None]
    return _VIRIDIS[i] * (1 - t) + _VIRIDIS[i + 1] * t


def map_side(state: SearchState, field_shape: tuple[int, int] | None) -> int:
    return field_shape[0] if field_shape else 64


def to_pixel(u: Any, side: int) -> tuple[Any, Any]:
    """Normalised coordinates -> (column, row); parameter 1 to the right, parameter 2 upwards."""
    u = np.clip(np.atleast_2d(u), 0.0, 1.0)
    px = np.rint(u[:, 0] * (side - 1)).astype(int)
    py = (
        np.rint((1.0 - u[:, 1]) * (side - 1)).astype(int)
        if u.shape[1] > 1
        else np.full(len(u), side // 2)
    )
    return px, py


def lineage(state: SearchState, i: int, limit: int = 200) -> list[int]:
    """Ancestors of candidate i (parent first), following ``parent`` until -1."""
    out: list[int] = []
    p = int(state.parent[i]) if state.parent.size else -1
    while p >= 0 and len(out) < limit:
        out.append(p)
        p = int(state.parent[p])
    return out


def _panels(state: SearchState) -> tuple[int, int, int]:
    """(side, archive panel width, field width)."""
    fshape = state.best_field.shape if state.best_field is not None else None
    side = map_side(state, fshape)
    return side, side if state.grid is not None else 0, fshape[1] if fshape else side


def render(state: SearchState) -> tuple[dict[str, Any], bytes]:
    side, aw, fw = _panels(state)
    img = np.empty((side, side + aw + fw, 3), dtype=np.float64)
    img[:] = _BG
    for k in range(1, 4):  # quarter grid lines
        img[:, (side * k) // 4, :] = _GRID
        img[(side * k) // 4, :side, :] = _GRID
    if state.score.size:
        px, py = to_pixel(state.u, side)
        top = float(state.score.max()) or 1.0
        # Viable (on at least one search seed): colour by score. Otherwise grey, brighter the
        # longer the candidate survived (partial credit <= 0.1).
        grey = _DEAD[None, :] * (0.35 + 0.65 * np.clip(state.score / 0.1, 0, 1))[:, None]
        colors = np.where((state.viable > 0)[:, None], _cmap(state.score / top), grey)
        order = np.argsort(state.score, kind="stable")  # best drawn last
        img[py[order], px[order]] = colors[order]
        b = state.best
        assert b is not None
        anc = lineage(state, b)
        if anc:  # ancestry of the best candidate (MAP-Elites), cyan
            ax, ay = to_pixel(state.u[anc], side)
            img[ay, ax] = [80, 220, 240]
        bx, by = to_pixel(state.u[b], side)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -1), (2, 1), (-1, 2), (1, -2)):
            x, y = int(bx[0]) + dx, int(by[0]) + dy
            if 0 <= x < side and 0 <= y < side:
                img[y, x] = 255
    if state.cma is not None:
        cx, cy = to_pixel(state.cma.mean, side)
        img[int(cy[0]), int(cx[0])] = [240, 60, 60]
    if state.grid is not None and state.elites is not None:
        gx, gy = state.grid
        panel = np.empty((side, side, 3))
        panel[:] = _BG * 1.4
        top = float(state.score.max()) if state.score.size else 1.0
        cell_w, cell_h = side / gx, side / gy
        for c, e in enumerate(state.elites):
            col, row = c % gx, c // gx
            x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
            y1, y0 = side - int(row * cell_h), side - int((row + 1) * cell_h)  # mass upwards
            if e >= 0:
                panel[y0:y1, x0:x1] = _cmap(state.score[e] / (top or 1.0))
            panel[y0:y1, x0] = _GRID  # cell borders
            panel[y0, x0:x1] = _GRID
        img[:, side : side + aw] = panel
    if state.best_field is not None:
        img[:, side + aw :, :] = (np.clip(state.best_field, 0.0, 1.0) * 255.0)[..., None]
    data = np.rint(img).astype(np.uint8).tobytes()
    scale = "left: parameter map (colour = score)"
    if aw:
        scale += "; middle: QD archive (descriptor 0 to the right, 1 upwards)"
    return {
        "codec": "field_u8",
        "width": side + aw + fw,
        "height": side,
        "channels": 3,
        "scale": scale + "; right: best candidate final state",
    }, data


def _candidate(state: SearchState, exp: SearchExperiment, i: int) -> dict[str, Any]:
    params = decode(exp, state.u[i])
    seed0 = seeds_for(exp, 1, exp.dynamics.eval_seeds)[0]
    spec = candidate_spec(exp, params, seed0, name=f"{exp.name}-cand{i}")
    anc = lineage(state, i)
    return {
        "index": i,
        "generation": int(state.generation[i]),
        "params": params,
        "score": float(state.score[i]),
        "viable_fraction": float(state.viable[i]),
        "alive_fraction": float(state.alive_fraction[i]),
        "speed": float(state.speed[i]),
        "mass": float(state.mass[i]),
        "holdout_score": state.holdout.get(i),
        "out_of_box": bool(np.any((state.u[i] < 0) | (state.u[i] > 1))),
        "lineage": [
            {"index": a, "generation": int(state.generation[a]), "score": float(state.score[a])}
            for a in anc[:20]
        ],
        "lineage_depth": len(anc),
        "spec": spec.model_dump(mode="json") if spec is not None else None,
    }


def inspect(state: SearchState, exp: SearchExperiment, query: dict[str, Any]) -> dict[str, Any]:
    side, aw, fw = _panels(state)
    qx, qy = int(query.get("x", 0)), int(query.get("y", 0))
    out: dict[str, Any] = {
        "substrate": "search",
        "tick": state.tick,
        "query": {"x": qx, "y": qy},
        "algorithm": exp.dynamics.algorithm,
        "objective": exp.dynamics.objective,
        "axes": [p.path for p in exp.space[:2]],
        "evaluated": int(state.score.size),
        "search_seeds": seeds_for(exp, 1, exp.dynamics.eval_seeds),
        "holdout_seeds": seeds_for(exp, 2, exp.dynamics.holdout_seeds),
    }
    if exp.asal is not None:
        out["asal"] = exp.asal.model_dump(mode="json")
    if exp.qd is not None and state.elites is not None:
        filled = state.elites[state.elites >= 0]
        out["qd"] = {
            "descriptors": [d.model_dump(mode="json") for d in exp.qd.descriptors],
            "filled": int(filled.size),
            "cells": int(state.elites.size),
            "archive_version": exp.qd.archive_version,
        }
    best = state.best
    if best is None:
        return out
    if "candidate" in query:  # direct lookup by archive index (used by the discovery ledger)
        i = int(query["candidate"])
        if not 0 <= i < state.score.size:
            raise ValueError(f"candidate {i} does not exist at this generation")
        out["region"] = "index"
        out["candidate"] = _candidate(state, exp, i)
    elif qx < side and state.score.size:
        px, py = to_pixel(state.u, side)
        dist = (px - qx) ** 2 + (py - qy) ** 2
        near = int(np.argmin(dist - 1e-6 * state.score))  # ties: prefer the higher score
        out["region"] = "map"
        if dist[near] <= 16:
            out["candidate"] = _candidate(state, exp, near)
    elif aw and qx < side + aw and state.grid is not None and state.elites is not None:
        gx, gy = state.grid
        col = min(gx - 1, max(0, int((qx - side) * gx / side)))
        row = min(gy - 1, max(0, int((side - 1 - qy) * gy / side)))
        out["region"] = "archive"
        assert exp.qd is not None
        d0, d1 = exp.qd.descriptors
        w0, w1 = (d0.high - d0.low) / gx, (d1.high - d1.low) / gy
        out["archive_cell"] = {
            "column": col,
            "row": row,
            d0.name: [d0.low + col * w0, d0.low + (col + 1) * w0],
            d1.name: [d1.low + row * w1, d1.low + (row + 1) * w1],
        }
        e = int(state.elites[row * gx + col])
        if e >= 0:
            out["candidate"] = _candidate(state, exp, e)
    else:
        out["region"] = "best"
        x = qx - side - aw
        if state.best_field is not None and 0 <= qy < side and 0 <= x < fw:
            out["cell_value"] = float(state.best_field[qy, x])
    out["best"] = _candidate(state, exp, best)
    if state.cma is not None:
        out["cma"] = {"mean": decode(exp, state.cma.mean), "sigma": state.cma.sigma}
    return out
