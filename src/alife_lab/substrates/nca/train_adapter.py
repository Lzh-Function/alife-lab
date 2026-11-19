"""``nca_train``: an optimisation run that trains a Growing NCA (one tick = one optimizer step).

The objective (MSE to a target image) is explicit and external to the CA dynamics; the trained
weights are later run by the separate ``nca`` substrate. Weights, Adam moments, the sample pool,
the NumPy RNG (pool sampling, rollout lengths, damage) and the JAX key (stochastic updates)
are checkpointed separately. JAX is imported lazily (workers only).
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, NCATrainExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.nca.targets import padded_target

SERIALIZE_VERSION = 1
JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def _tr(spec: ExperimentSpec) -> NCATrainExperiment:
    exp = spec.experiment
    if not isinstance(exp, NCATrainExperiment):
        raise TypeError("nca_train adapter received a different spec")
    return exp


def _model() -> Any:
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed (uv sync --extra nca)")
    from alife_lab.substrates.nca import model

    return model


def seed_state(height: int, width: int, channels: int) -> np.ndarray[Any, Any]:
    """Single living cell in the centre: all channels except RGB set to 1 (as in the article)."""
    x = np.zeros((height, width, channels), dtype=np.float32)
    x[height // 2, width // 2, 3:] = 1.0
    return x


def _to_jax(tree: dict[str, np.ndarray[Any, Any]]) -> dict[str, Any]:
    import jax.numpy as jnp

    return {k: jnp.asarray(v) for k, v in tree.items()}


def learning_rate(exp: NCATrainExperiment, step: int) -> float:
    d = exp.dynamics
    return d.learning_rate * (d.lr_decay_factor if step >= d.lr_decay_step else 1.0)


@dataclass
class TrainState:
    params: dict[str, np.ndarray[Any, Any]]
    m: dict[str, np.ndarray[Any, Any]]
    v: dict[str, np.ndarray[Any, Any]]
    pool: np.ndarray[Any, Any]  # (P, H, W, C) float32; (1, H, W, C) for "growing"
    tick: int
    rng: np.random.Generator = field(repr=False)
    key: np.ndarray[Any, Any] = field(repr=False)  # JAX PRNG key data (uint32[2])
    last: dict[str, float] = field(default_factory=dict)
    preview: np.ndarray[Any, Any] | None = field(default=None, repr=False)
    target: np.ndarray[Any, Any] | None = field(default=None, repr=False)
    step_fn: Any = field(default=None, repr=False)


_M = MetricDefinition
_METRICS = (
    _M("loss", "metric.loss", "MSE", "optimizer_step", "Batch mean RGBA MSE to the target.",
       "none", "every step", 1, "never null"),
    _M("log10_loss", "metric.log10_loss", "log10 MSE", "optimizer_step", "log10 of the loss.",
       "none", "every step", 1, "never null"),
    _M("learning_rate", "metric.learning_rate", "lr", "optimizer_step", "Adam learning rate.",
       "none", "every step", 1, "never null"),
    _M("grad_norm", "metric.grad_norm", "L2", "optimizer_step",
       "Global gradient L2 norm before per-variable normalisation.", "none", "every step", 1,
       "never null"),
)  # fmt: skip

NCA_TRAIN_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.nca_train",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="nca_train",
    display_name_key="substrate.nca_train",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="grid",
        clock_unit="optimizer_step",
        supports_pause=True,
        supports_interventions=False,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=True,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=(),
    ),
    backends=("jax",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Mordvintsev, Randazzo, Niklasson, Levin (2020) Growing Neural Cellular "
        "Automata, Distill. Optimisation run with an explicit image-reconstruction objective.",
        "paper_urls": ["https://distill.pub/2020/growing-ca/"],
        "implementation": "this repository (JAX, CPU)",
    },
)


class NCATrainAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return NCA_TRAIN_MANIFEST

    def _prepare(self, exp: NCATrainExperiment, state: TrainState) -> TrainState:
        m = _model()
        d = exp.dynamics
        a = d.architecture
        state.target = padded_target(exp.target, d.pad)
        state.step_fn = m.make_train_step(d.steps_max, a.fire_rate, a.alive_threshold)
        return state

    def initialize(self, spec: ExperimentSpec) -> TrainState:
        exp = _tr(spec)
        m = _model()
        import jax

        d = exp.dynamics
        side = exp.target.size + 2 * d.pad
        key = jax.random.PRNGKey(exp.seed % (2**32))
        key, sub = jax.random.split(key)
        a = d.architecture
        params = {k: np.asarray(v) for k, v in m.init_params(sub, a.channels, a.hidden).items()}
        zeros = {k: np.zeros_like(v) for k, v in params.items()}
        size = 1 if d.experiment == "growing" else d.pool_size
        pool = np.repeat(seed_state(side, side, d.architecture.channels)[None], size, axis=0)
        state = TrainState(
            params=params, m=zeros, v={k: v.copy() for k, v in zeros.items()}, pool=pool,
            tick=0, rng=np.random.Generator(np.random.PCG64(exp.seed)),
            key=np.asarray(key, dtype=np.uint32),  # legacy uint32[2] PRNG key
        )  # fmt: skip
        state.preview = pool[0]
        return self._prepare(exp, state)

    def _damage(self, x: np.ndarray[Any, Any], rng: np.random.Generator, count: int) -> None:
        """Erase a random disc in each of the last ``count`` samples (the lowest losses)."""
        n, h, w, _ = x.shape
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(n - count, n):
            cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
            r = rng.uniform(0.1, 0.4) * min(h, w)
            x[i][(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 0.0

    def step(self, state: TrainState, spec: ExperimentSpec) -> TrainState:
        exp = _tr(spec)
        m = _model()
        import jax
        import jax.numpy as jnp

        d = exp.dynamics
        side = state.pool.shape[1]
        seed = seed_state(side, side, d.architecture.channels)
        if d.experiment == "growing":
            idx = None
            batch = np.repeat(seed[None], d.batch_size, axis=0)
        else:
            idx = state.rng.choice(state.pool.shape[0], size=d.batch_size, replace=False)
            batch = state.pool[idx].copy()
            order = np.argsort(-m.losses_of(batch, state.target), kind="stable")
            idx, batch = idx[order], batch[order]  # highest loss first
            batch[0] = seed
            if d.experiment == "regenerating" and d.damaged_per_batch:
                self._damage(batch, state.rng, d.damaged_per_batch)
        n = int(state.rng.integers(d.steps_min, d.steps_max + 1))
        key, sub = jax.random.split(jnp.asarray(state.key))
        lr = learning_rate(exp, state.tick)
        params, mm, vv, loss, _, out, gnorm = state.step_fn(
            _to_jax(state.params), _to_jax(state.m), _to_jax(state.v), jnp.float32(state.tick + 1),
            jnp.float32(lr), jnp.asarray(batch), sub, jnp.int32(n), jnp.asarray(state.target),
        )  # fmt: skip
        state.params = {k: np.asarray(v) for k, v in params.items()}
        state.m = {k: np.asarray(v) for k, v in mm.items()}
        state.v = {k: np.asarray(v) for k, v in vv.items()}
        out = np.asarray(out)
        if idx is not None:
            state.pool[idx] = out
        state.key = np.asarray(key, dtype=np.uint32)
        state.preview = out[0]
        state.tick += 1
        state.last = {"loss": float(loss), "lr": lr, "grad_norm": float(gnorm), "steps": n}
        return state

    def metrics(self, state: TrainState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_tr(spec).observations.metrics)
        if not state.last:
            return {}
        loss = state.last["loss"]
        out = {
            "loss": loss,
            "log10_loss": math.log10(loss) if loss > 0 else math.nan,
            "learning_rate": state.last["lr"],
            "grad_norm": state.last["grad_norm"],
        }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: TrainState) -> tuple[dict[str, Any], bytes]:
        from alife_lab.substrates.nca.render import rgb_bytes

        return rgb_bytes(state.preview)

    def apply_intervention(
        self, state: TrainState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[TrainState, dict[str, Any]]:
        raise ValueError("nca_train runs accept no interventions (optimisation run)")

    def inspect(
        self, state: TrainState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        from alife_lab.substrates.nca.render import cell_info

        info = cell_info(state.preview, int(query["x"]), int(query["y"]))
        return {
            "substrate": "nca_train",
            "tick": state.tick,
            "query": {"x": int(query["x"]), "y": int(query["y"])},
            "last_step": state.last,
            "parameter_count": int(sum(v.size for v in state.params.values())),
            **info,
        }

    def state_distance(self, a: TrainState, b: TrainState) -> dict[str, Any]:
        diff = max(float(np.abs(a.params[k] - b.params[k]).max()) for k in a.params)
        exact = all(np.array_equal(a.params[k], b.params[k]) for k in a.params) and np.array_equal(
            a.pool, b.pool
        )
        return {"metric": "weights_max_abs_difference", "value": diff, "exact_equal": exact}

    def serialize_state(self, state: TrainState) -> dict[str, Any]:
        arrays: dict[str, Any] = {"pool": state.pool, "key": state.key}
        for group, tree in (("w", state.params), ("adam_m", state.m), ("adam_v", state.v)):
            for k, v in tree.items():
                arrays[f"{group}.{k}"] = v
        return {
            "arrays": arrays,
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "rng_state": state.rng.bit_generator.state,
                "last": state.last,
                "groups": {"model_weights": "w.*", "optimizer_state": "adam_m.*, adam_v.*",
                           "cell_states": "pool", "rng": "key + rng_state"},
            },
        }  # fmt: skip

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> TrainState:
        exp = _tr(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible nca_train checkpoint serialize_version")
        arr = payload["arrays"]

        def group(prefix: str) -> dict[str, np.ndarray[Any, Any]]:
            return {k.split(".", 1)[1]: np.asarray(v, dtype=np.float32).copy()
                    for k, v in arr.items() if k.startswith(prefix + ".")}  # fmt: skip

        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        pool = np.asarray(arr["pool"], dtype=np.float32).copy()
        side = exp.target.size + 2 * exp.dynamics.pad
        if pool.shape[1:] != (side, side, exp.dynamics.architecture.channels):
            raise ValueError("nca_train checkpoint pool shape does not match the spec")
        state = TrainState(
            params=group("w"), m=group("adam_m"), v=group("adam_v"), pool=pool,
            tick=int(meta["tick"]), rng=rng, key=np.asarray(arr["key"], dtype=np.uint32).copy(),
            last=dict(meta.get("last") or {}),
        )  # fmt: skip
        state.preview = pool[0]
        return self._prepare(exp, state)
