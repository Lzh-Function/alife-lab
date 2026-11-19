"""Growing NCA in JAX (CPU, float32), from Mordvintsev et al., "Growing Neural Cellular
Automata", Distill (2020). Implemented from the article; no code reused.

- Perception: depthwise 3x3 identity, Sobel-x and Sobel-y (/8), zero padding.
- Update rule: dense(3C -> hidden, ReLU) -> dense(hidden -> C), final layer zero-initialised.
- Stochastic update: each cell updates with probability ``fire_rate``.
- Alive masking: a cell lives if the 3x3 max of alpha exceeds ``alive_threshold`` both before
  and after the update; dead cells are zeroed.
- Training: MSE of RGBA against the padded target after a random number of steps in
  [steps_min, steps_max]; per-variable gradient L2 normalisation; Adam (b1 0.9, b2 0.999,
  eps 1e-7) with a step decay of the learning rate. Pool / damage handled by ``training.py``.

JAX is an optional dependency (``uv sync --extra nca``) imported only by workers. The CPU
platform is forced so results do not depend on whether a CUDA build of JAX is installed.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]

_SOBEL = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], np.float32) / 8.0
_IDENT = np.zeros((3, 3), np.float32)
_IDENT[1, 1] = 1.0
_PERCEPTION = np.stack([_IDENT, _SOBEL, _SOBEL.T], axis=-1)  # (3, 3, 3)


def init_params(key: Any, channels: int, hidden: int) -> Params:
    """Glorot-uniform first layer, zero bias, zero output layer (as in the article)."""
    fan_in, fan_out = 3 * channels, hidden
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    w1 = jax.random.uniform(key, (fan_in, hidden), jnp.float32, -limit, limit)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden,), jnp.float32),
        "w2": jnp.zeros((hidden, channels), jnp.float32),
    }


def perceive(x: Any) -> Any:
    c = x.shape[-1]
    k = jnp.asarray(np.repeat(_PERCEPTION[:, :, None, :], c, axis=2).reshape(3, 3, 1, 3 * c))
    return jax.lax.conv_general_dilated(
        x, k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=c
    )


def alive_mask(x: Any, threshold: float) -> Any:
    pooled = jax.lax.reduce_window(
        x[..., 3:4], -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "SAME"
    )
    return pooled > threshold


def ca_step(params: Params, x: Any, key: Any, fire_rate: float, threshold: float) -> Any:
    pre = alive_mask(x, threshold)
    dx = jax.nn.relu(perceive(x) @ params["w1"] + params["b1"]) @ params["w2"]
    update = jax.random.uniform(key, (*x.shape[:3], 1)) <= fire_rate
    x = x + dx * update
    return x * (pre & alive_mask(x, threshold)).astype(x.dtype)


def rollout(
    params: Params, x: Any, key: Any, n: Any, n_max: int, fire_rate: float, threshold: float
) -> Any:
    """``n`` (traced) CA steps out of a static scan of ``n_max`` (one compilation)."""

    def body(carry: Any, inp: Any) -> Any:
        i, k = inp
        nxt = ca_step(params, carry, k, fire_rate, threshold)
        return jnp.where(i < n, nxt, carry), None

    keys = jax.random.split(key, n_max)
    out, _ = jax.lax.scan(body, x, (jnp.arange(n_max), keys))
    return out


def per_sample_loss(x: Any, target: Any) -> Any:
    return jnp.mean((x[..., :4] - target) ** 2, axis=(1, 2, 3))


def make_train_step(n_max: int, fire_rate: float, threshold: float) -> Any:
    def loss_fn(params: Params, x: Any, key: Any, n: Any, target: Any) -> Any:
        out = rollout(params, x, key, n, n_max, fire_rate, threshold)
        losses = per_sample_loss(out, target)
        return jnp.mean(losses), (out, losses)

    @jax.jit
    def train_step(
        params: Params, m: Params, v: Params, t: Any, lr: Any, x: Any, key: Any, n: Any,
        target: Any,
    ) -> Any:  # fmt: skip
        (loss, (out, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, key, n, target
        )
        grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-8), grads)
        b1, b2, eps = 0.9, 0.999, 1e-7
        m = jax.tree.map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
        v = jax.tree.map(lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
        mhat = jax.tree.map(lambda a: a / (1 - b1**t), m)
        vhat = jax.tree.map(lambda a: a / (1 - b2**t), v)
        params = jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mhat, vhat)
        return params, m, v, loss, losses, out, grad_norm

    return train_step


def make_ca_step(fire_rate: float, threshold: float) -> Any:
    @jax.jit
    def step(params: Params, x: Any, key: Any) -> Any:
        return ca_step(params, x, key, fire_rate, threshold)

    return step


@jax.jit
def _loss_only(x: Any, target: Any) -> Any:
    return per_sample_loss(x, target)


def losses_of(x: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(_loss_only(jnp.asarray(x), jnp.asarray(target)))


def pack_params(params: Params) -> bytes:
    return b"".join(np.asarray(params[k], dtype="<f4").tobytes() for k in ("w1", "b1", "w2"))


def unpack_params(raw: bytes, channels: int, hidden: int) -> Params:
    a = np.frombuffer(raw, dtype="<f4")
    n1, n2 = 3 * channels * hidden, hidden
    return {
        "w1": jnp.asarray(a[:n1].reshape(3 * channels, hidden)),
        "b1": jnp.asarray(a[n1 : n1 + n2]),
        "w2": jnp.asarray(a[n1 + n2 :].reshape(hidden, channels)),
    }
