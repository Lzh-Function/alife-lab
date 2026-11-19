"""``pd_nca``: Petri Dish NCA — competing NCAs that keep learning (one tick = one epoch).

Each epoch runs ``steps_before_update`` CA steps without gradients, ``steps_per_update`` steps
with per-entity gradient isolation, then one online update of all NCAs (unless ``learning`` is
``frozen``). The per-NCA objective (own territory) is part of the model's dynamics, not an
external evaluation. Deterministic: randomness is used only at initialisation.
"""

from __future__ import annotations

import colorsys
import contextlib
import importlib.util
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, PDNCAExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def _pd(spec: ExperimentSpec) -> PDNCAExperiment:
    exp = spec.experiment
    if not isinstance(exp, PDNCAExperiment):
        raise TypeError("pd_nca adapter received a different spec")
    return exp


def _model() -> Any:
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed (uv sync --extra nca)")
    from alife_lab.substrates.pd_nca import model

    return model


def _shape(exp: PDNCAExperiment) -> Any:
    m = _model()
    d = exp.dynamics
    return m.Shape(
        n=d.n_ncas, s=d.cell_state_dim, hd=d.cell_hidden_dim, hidden_dim=d.model.hidden_dim,
        n_hidden_layers=d.model.n_hidden_layers, k=d.model.kernel_size,
        per_hid_upd=d.model.per_hid_upd, alive_visible=d.model.alive_visible,
        threshold=d.alive_threshold, temp=d.softmax_temp, cosine=d.competition == "cosine",
    )  # fmt: skip


@contextlib.contextmanager
def _on(backend: str) -> Iterator[None]:
    m = _model()
    import jax

    with jax.default_device(m.device(backend)):
        yield


def colours(n: int) -> np.ndarray[Any, Any]:
    """Sun = dark grey; NCAs evenly spaced hues."""
    out = [(0.08, 0.08, 0.1)]
    out += [colorsys.hsv_to_rgb(i / max(1, n), 0.75, 0.95) for i in range(n)]
    return np.asarray(out, dtype=np.float32)


@dataclass
class PDState:
    x: np.ndarray[Any, Any]  # (H, W, C) float32
    tick: int
    params: dict[str, np.ndarray[Any, Any]]
    opt: dict[str, np.ndarray[Any, Any]]
    sun: np.ndarray[Any, Any]  # (OC,)
    seeds: np.ndarray[Any, Any]  # (N, OC) seed vectors
    last: dict[str, float] = field(default_factory=dict)
    epoch_fn: Any = field(default=None, repr=False)

    def territories(self, n: int) -> np.ndarray[Any, Any]:
        h, w = self.x.shape[:2]
        out: np.ndarray[Any, Any] = self.x[..., : n + 1].reshape(-1, n + 1).sum(0) / (h * w)
        return out


_M = MetricDefinition
_METRICS = (
    _M("territory_loss", "metric.territory_loss", "-sum asinh", "epoch",
       "Learning loss of the last epoch: -sum over entities of asinh(own territory).", "none",
       "every epoch", 1, "never null"),
    _M("grad_norm", "metric.grad_norm", "L2", "epoch", "Global gradient norm before clipping.",
       "none", "every epoch", 1, "never null"),
    _M("alive_ncas", "metric.alive_ncas", "NCAs", "epoch", "NCAs with territory > 0.", "none",
       "every epoch", 1, "never null"),
    _M("territory_entropy", "metric.territory_entropy", "effective NCAs", "epoch",
       "exp(Shannon entropy) of the NCAs' territory shares (sun excluded).", "none",
       "every epoch", 1, "null when no NCA is alive"),
    _M("sun_fraction", "metric.sun_fraction", "fraction", "epoch",
       "Share of total aliveness held by the sun (background).", "none", "every epoch", 1,
       "never null"),
    _M("reverted", "metric.reverted", "0/1", "epoch",
       "1 if the epoch was reverted because an NCA went extinct (revert_on_extinction).",
       "none", "every epoch", 1, "never null"),
)  # fmt: skip

PD_NCA_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.pd_nca",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="pd_nca",
    display_name_key="substrate.pd_nca",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="grid",
        clock_unit="epoch",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=True,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("erase", "seed"),
    ),
    backends=("jax", "jax_cuda"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    backend_numeric_policies={"jax_cuda": ("float32_jax",)},
    reference={
        "description": "Sakana AI, Petri Dish Neural Cellular Automata (2025): multiple NCAs "
        "compete for space and keep learning. Reimplemented in JAX (reference code unlicensed).",
        "paper_urls": ["https://pub.sakana.ai/pdnca/"],
        "implementation": "https://github.com/SakanaAI/petri-dish-nca",
        "implementation_commit": "875523f3228b92ed5669c88aa9a2012e25fbcc60",
    },
)


class PDNCAAdapter:
    backend: str = "jax"

    def set_backend(self, backend: str) -> None:
        if backend not in PD_NCA_MANIFEST.backends:
            raise ValueError(f"unsupported pd_nca backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return PD_NCA_MANIFEST

    def _prepare(self, exp: PDNCAExperiment, state: PDState) -> PDState:
        m = _model()
        d = exp.dynamics
        state.epoch_fn = m.make_epoch(
            _shape(exp), d.steps_before_update, d.steps_per_update, d.learning == "online",
            d.optimizer, d.learning_rate, d.grad_clip_norm,
        )  # fmt: skip
        return state

    def initialize(self, spec: ExperimentSpec) -> PDState:
        exp = _pd(spec)
        m = _model()
        import jax

        sh = _shape(exp)
        d = exp.dynamics
        h, w = exp.world.height, exp.world.width
        rng = np.random.Generator(np.random.PCG64(exp.seed))
        with _on(self.backend):
            key = jax.random.PRNGKey(exp.seed % (2**32))
            params = {k: np.asarray(v) for k, v in m.init_params(key, sh).items()}
            opt = {k: np.asarray(v) for k, v in m.init_opt(d.optimizer, params).items()}
        sun = rng.standard_normal(sh.oc).astype(np.float32)
        sun[sh.s :] = 0.0
        sun /= np.linalg.norm(sun)
        seeds = rng.standard_normal((sh.n, sh.oc)).astype(np.float32)
        seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
        x = np.zeros((h, w, sh.c), dtype=np.float32)
        x[..., 0] = 1.0
        cells = rng.permutation(h * w)[: sh.n * d.n_seeds].reshape(sh.n, d.n_seeds)
        for n in range(sh.n):
            for c in cells[n]:
                yy, xx = divmod(int(c), w)
                x[yy, xx, : sh.m] = 0.0
                x[yy, xx, n + 1] = 1.0
                x[yy, xx, sh.m :] = seeds[n]
        state = PDState(x=x, tick=0, params=params, opt=opt, sun=sun, seeds=seeds)
        return self._prepare(exp, state)

    def step(self, state: PDState, spec: ExperimentSpec) -> PDState:
        exp = _pd(spec)
        import jax.numpy as jnp

        with _on(self.backend):
            to_j = {k: jnp.asarray(v) for k, v in state.params.items()}
            opt_j = {k: jnp.asarray(v) for k, v in state.opt.items()}
            params, opt, final, loss, gnorm = state.epoch_fn(
                to_j, opt_j, jnp.float32(state.tick + 1), jnp.asarray(state.sun),
                jnp.asarray(state.x),
            )  # fmt: skip
            state.params = {k: np.asarray(v) for k, v in params.items()}
            state.opt = {k: np.asarray(v) for k, v in opt.items()}
            new = np.array(final)
        n = exp.dynamics.n_ncas
        extinct = bool((new[..., 1 : n + 1].reshape(-1, n).sum(0) <= 0).any())
        reverted = exp.dynamics.revert_on_extinction and extinct
        if not reverted:
            state.x = new
        state.tick += 1
        state.last = {"loss": float(loss), "grad_norm": float(gnorm), "reverted": float(reverted)}
        return state

    def metrics(self, state: PDState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _pd(spec)
        wanted = set(exp.observations.metrics)
        n = exp.dynamics.n_ncas
        terr = state.territories(n)
        ncas = terr[1:]
        out: dict[str, float] = {}
        if "territory_loss" in wanted:
            out["territory_loss"] = state.last.get("loss", math.nan)
        if "grad_norm" in wanted:
            out["grad_norm"] = state.last.get("grad_norm", math.nan)
        if "alive_ncas" in wanted:
            out["alive_ncas"] = float((ncas > 0).sum())
        if "territory_entropy" in wanted:
            total = float(ncas.sum())
            if total > 0:
                p = ncas[ncas > 0] / total
                out["territory_entropy"] = float(math.exp(-(p * np.log(p)).sum()))
            else:
                out["territory_entropy"] = math.nan
        if "sun_fraction" in wanted:
            out["sun_fraction"] = float(terr[0])
        if "reverted" in wanted:
            out["reverted"] = state.last.get("reverted", 0.0)
        if exp.observations.record_territories:
            out["territory.sun"] = float(terr[0])
            for k in range(n):
                out[f"territory.nca_{k}"] = float(terr[k + 1])
        return out

    def render(self, state: PDState) -> tuple[dict[str, Any], bytes]:
        h, w, c = state.x.shape
        m = c - (state.sun.shape[0])
        rgb = np.clip(state.x[..., :m] @ colours(m - 1), 0.0, 1.0)
        data = np.rint(rgb * 255).astype(np.uint8).tobytes()
        return {"codec": "field_u8", "width": w, "height": h, "channels": 3,
                "scale": "aliveness-weighted entity colours (sun dark)"}, data  # fmt: skip

    def apply_intervention(
        self, state: PDState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[PDState, dict[str, Any]]:
        exp = _pd(spec)
        n = exp.dynamics.n_ncas
        h, w = state.x.shape[:2]
        try:
            x, y = int(payload["x"]), int(payload["y"])
            radius = int(payload.get("radius", 3))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("payload needs integer x, y (optional radius, agent)") from exc
        if not (0 <= x < w and 0 <= y < h) or not 0 <= radius <= 64:
            raise ValueError("position or radius out of range")
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius * radius
        m = n + 1
        if kind == "erase":
            state.x[mask] = 0.0
            state.x[mask, 0] = 1.0  # the sun takes the cells back
            return state, {"cells": int(mask.sum())}
        if kind == "seed":
            agent = int(payload.get("agent", 0))
            if not 0 <= agent < n:
                raise ValueError(f"agent must be within 0..{n - 1}")
            state.x[mask, :m] = 0.0
            state.x[mask, agent + 1] = 1.0
            state.x[mask, m:] = state.seeds[agent]
            return state, {"cells": int(mask.sum()), "agent": agent}
        raise ValueError(f"unsupported intervention kind {kind!r}")

    def inspect(
        self, state: PDState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _pd(spec)
        m_ = _model()
        import jax.numpy as jnp

        n = exp.dynamics.n_ncas
        h, w = state.x.shape[:2]
        x, y = int(query["x"]), int(query["y"])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("(x, y) is outside the world")
        sh = _shape(exp)
        with _on(self.backend):
            p = {k: jnp.asarray(v) for k, v in state.params.items()}
            u = m_.entity_updates(p, jnp.asarray(state.sun), jnp.asarray(state.x), sh)
            alive = m_.alive_mask(jnp.moveaxis(jnp.asarray(state.x[..., : n + 1]), -1, 0),
                                  sh.threshold)  # fmt: skip
            u = u * alive[..., None]
            half = sh.s // 2
            cos = np.asarray(m_._cos(u[..., :half], u[..., half : sh.s]))
            alive_np = np.asarray(alive)
        both = alive_np[:, None] & alive_np[None, :]
        inter = np.where(both.any(axis=(2, 3)),
                         (cos * both).sum(axis=(2, 3)) / np.maximum(both.sum(axis=(2, 3)), 1),
                         np.nan)  # fmt: skip
        np.fill_diagonal(inter, np.nan)
        cell = state.x[y, x, : n + 1]
        return {
            "substrate": "pd_nca",
            "tick": state.tick,
            "query": {"x": x, "y": y},
            "aliveness": [float(v) for v in cell],
            "dominant": int(np.argmax(cell)),
            "territories": [float(v) for v in state.territories(n)],
            "interaction": [[None if math.isnan(v) else float(v) for v in row] for row in inter],
            "interaction_rule": "mean cos(attack_row, defence_col) over cells where both are alive",
            "last_epoch": state.last,
        }

    def state_distance(self, a: PDState, b: PDState) -> dict[str, Any]:
        if a.x.shape != b.x.shape:
            return {"metric": "max_abs_difference", "value": None, "exact_equal": False}
        exact = np.array_equal(a.x, b.x) and all(
            np.array_equal(a.params[k], b.params[k]) for k in a.params
        )
        return {"metric": "max_abs_difference", "value": float(np.abs(a.x - b.x).max()),
                "exact_equal": bool(exact)}  # fmt: skip

    def serialize_state(self, state: PDState) -> dict[str, Any]:
        arrays: dict[str, Any] = {"grid": state.x, "sun": state.sun, "seeds": state.seeds}
        arrays.update({f"w.{k}": v for k, v in state.params.items()})
        arrays.update({f"opt.{k}": v for k, v in state.opt.items()})
        return {
            "arrays": arrays,
            "meta": {"serialize_version": SERIALIZE_VERSION, "tick": state.tick,
                     "last": state.last},
        }  # fmt: skip

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> PDState:
        exp = _pd(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible pd_nca checkpoint serialize_version")
        arr = payload["arrays"]
        grid = np.asarray(arr["grid"], dtype=np.float32).copy()
        sh = _shape(exp)
        if grid.shape != (exp.world.height, exp.world.width, sh.c):
            raise ValueError("pd_nca checkpoint grid does not match the spec")
        params = {k[2:]: np.asarray(v, np.float32).copy() for k, v in arr.items()
                  if k.startswith("w.")}  # fmt: skip
        opt = {k[4:]: np.asarray(v, np.float32).copy() for k, v in arr.items()
               if k.startswith("opt.")}  # fmt: skip
        state = PDState(
            x=grid, tick=int(meta["tick"]), params=params, opt=opt,
            sun=np.asarray(arr["sun"], np.float32).copy(),
            seeds=np.asarray(arr["seeds"], np.float32).copy(), last=dict(meta.get("last") or {}),
        )  # fmt: skip
        return self._prepare(exp, state)
