"""Petri Dish NCA in JAX — competing neural CAs that keep learning while they run.

Reimplemented from Sakana AI's description and the semantics of their PyTorch code
(github.com/SakanaAI/petri-dish-nca @ 875523f, no licence; no code copied).

Cell layout (channels last): ``[alive_0 (sun), alive_1..alive_N | state (S) | hidden (Hd)]``.
Per step:
1. Every NCA proposes an update of state+hidden from a conv net (k x k conv -> GELU,
   optional grouped k x k residual layers, grouped 1 x 1 -> tanh); NCA ``n`` may write only the
   hidden channels ``(n + j) mod Hd`` for ``j < per_hid_upd * Hd``; the sun proposes a fixed
   random vector (hidden part zero). Updates of entities not alive in a cell are zeroed
   (alive = 3 x 3 max of the aliveness channel > threshold; the sun is alive everywhere).
2. Competition (``cosine``): the state update is split into attack | defence halves; for every
   ordered pair (a, d) that is alive together somewhere, a's strength gains
   cos(att_a, def_d) - cos(att_d, def_a) per cell; strengths of non-alive entities are -inf and a
   softmax over entities gives weights ``w``. ``uniform`` (control) uses equal strengths.
3. state+hidden += sum_m w_m * update_m; aliveness = softmax(w / temp) over alive entities, then
   entities whose new 3 x 3 max aliveness is <= threshold are removed and the rest renormalised;
   everything is clamped to [-1, 1].
Learning (one update per epoch): each entity's objective is ``asinh(total aliveness + 1e-3)`` in
its own *perspective*, a copy of the trajectory where only its own updates carry gradient (the
others' are stop-gradient); loss = -sum of these; global gradient-norm clipping; SGD / RMSprop /
Adam. The sun vector is not trained (the reference default).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]


@dataclass(frozen=True)
class Shape:
    n: int  # NCAs (without the sun)
    s: int  # state channels (attack | defence)
    hd: int  # hidden channels
    hidden_dim: int
    n_hidden_layers: int
    k: int
    per_hid_upd: float
    alive_visible: bool
    threshold: float
    temp: float
    cosine: bool

    @property
    def m(self) -> int:
        return self.n + 1

    @property
    def oc(self) -> int:
        return self.s + self.hd

    @property
    def c(self) -> int:
        return self.m + self.oc


def init_params(key: Any, sh: Shape) -> Params:
    """PyTorch-default-like uniform(+-1/sqrt(fan_in)) initialisation, no biases."""
    cin = sh.c if sh.alive_visible else sh.oc
    keys = jax.random.split(key, 2 + sh.n_hidden_layers)

    def u(k: Any, shape: tuple[int, ...], fan_in: int) -> Any:
        b = 1.0 / np.sqrt(fan_in)
        return jax.random.uniform(k, shape, jnp.float32, -b, b)

    p: Params = {"enc": u(keys[0], (sh.k, sh.k, cin, sh.n * sh.hidden_dim), cin * sh.k * sh.k)}
    for i in range(sh.n_hidden_layers):
        p[f"mid{i}"] = u(keys[1 + i], (sh.k, sh.k, sh.hidden_dim, sh.n * sh.hidden_dim),
                         sh.hidden_dim * sh.k * sh.k)  # fmt: skip
    p["comp"] = u(keys[-1], (sh.n, sh.hidden_dim, sh.oc), sh.hidden_dim)
    return p


def hidden_visibility(sh: Shape) -> np.ndarray[Any, Any]:
    vis = np.zeros((sh.n, sh.oc), np.float32)
    vis[:, : sh.s] = 1.0
    count = int(sh.per_hid_upd * sh.hd)
    for n in range(sh.n):
        for j in range(count):
            vis[n, sh.s + (n + j) % sh.hd] = 1.0
    return vis


def _conv(x: Any, w: Any, groups: int = 1) -> Any:
    return jax.lax.conv_general_dilated(
        x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=groups,
    )  # fmt: skip


def entity_updates(params: Params, sun: Any, x: Any, sh: Shape) -> Any:
    """Proposed state+hidden updates (M, H, W, OC); index 0 is the sun."""
    h_, w_ = x.shape[:2]
    inp = x if sh.alive_visible else x[..., sh.m :]
    h = jax.nn.gelu(_conv(inp[None], params["enc"]), approximate=False)
    for i in range(sh.n_hidden_layers):
        h = h + jax.nn.gelu(_conv(h, params[f"mid{i}"], groups=sh.n), approximate=False)
    h = h[0].reshape(h_, w_, sh.n, sh.hidden_dim)
    u = jnp.tanh(jnp.einsum("hwnd,ndo->nhwo", h, params["comp"]))
    u = u * jnp.asarray(hidden_visibility(sh))[:, None, None, :]
    sun_u = jnp.broadcast_to(sun, (1, h_, w_, sh.oc))
    return jnp.concatenate([sun_u, u], axis=0)


def alive_mask(alive: Any, threshold: float) -> Any:
    """(M, H, W) bool: 3x3 max of each aliveness channel > threshold; the sun always alive."""
    pooled = jax.lax.reduce_window(alive, -jnp.inf, jax.lax.max, (1, 3, 3), (1, 1, 1), "SAME")
    mask = pooled > threshold
    return mask.at[0].set(True)


def _cos(a: Any, d: Any) -> Any:
    """Pairwise cosine similarity (M, M, H, W) of a[i] with d[j] per cell (eps 1e-8)."""
    # Norms clamped at 1e-8 like torch.nn.functional.cosine_similarity, written so the gradient
    # stays finite for all-zero vectors (d sqrt(x)/dx is infinite at 0).
    na = jnp.sqrt(jnp.maximum(jnp.sum(a * a, axis=-1), 1e-16))
    nd = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-16))
    return jnp.einsum("ahwk,dhwk->adhw", a, d) / (na[:, None] * nd[None, :])


def strengths(u: Any, alive: Any, sh: Shape) -> Any:
    """Raw competition strengths (M, H, W) before masking."""
    if not sh.cosine:
        return jnp.zeros(alive.shape, jnp.float32)
    half = sh.s // 2
    att, dfn = u[..., :half], u[..., half : sh.s]
    c = _cos(att, dfn)  # c[a, d]: attack of a vs defence of d
    together = jnp.any(alive[:, None] & alive[None, :], axis=(2, 3))
    touch = together & ~jnp.eye(sh.m, dtype=bool)
    return jnp.einsum("ad,adhw->ahw", touch.astype(jnp.float32), c - jnp.swapaxes(c, 0, 1))


def ca_step(params: Params, sun: Any, x: Any, sh: Shape, own: Any = None) -> Any:
    """One step on (H, W, C). ``own``: (M,) bool — only these entities' updates carry grad."""
    u = entity_updates(params, sun, x, sh)
    if own is not None:
        u = jnp.where(own[:, None, None, None], u, jax.lax.stop_gradient(u))
    alive_prev = alive_mask(jnp.moveaxis(x[..., : sh.m], -1, 0), sh.threshold)
    u = u * alive_prev[..., None]
    st = jnp.where(alive_prev, strengths(u, alive_prev, sh), -jnp.inf)
    w = jax.nn.softmax(st, axis=0)  # (M, H, W)
    cells = x[..., sh.m :] + jnp.einsum("mhwc,mhw->hwc", u, w)
    al = jax.nn.softmax(jnp.where(alive_prev, w, -jnp.inf) / sh.temp, axis=0)
    keep = alive_mask(al, sh.threshold)
    al = al * keep
    al = al / jnp.sum(al, axis=0, keepdims=True)
    out = jnp.concatenate([jnp.moveaxis(al, 0, -1), cells], axis=-1)
    return jnp.clip(out, -1.0, 1.0)


def make_epoch(sh: Shape, steps_before: int, steps_per: int, learn: bool, optimizer: str,
               lr: float, clip: float) -> Any:  # fmt: skip
    def run(params: Params, sun: Any, x: Any, own: Any, n: int) -> Any:
        def body(c: Any, _: Any) -> Any:
            return ca_step(params, sun, c, sh, own), None

        out, _ = jax.lax.scan(body, x, None, length=n)
        return out

    def loss_fn(params: Params, sun: Any, x: Any) -> Any:
        ids = jnp.arange(sh.m)

        def perspective(p: Any) -> Any:
            final = run(params, sun, x, ids == p, steps_per)
            return jnp.arcsinh(jnp.sum(final[..., p]) + 1e-3), final

        logs, finals = jax.vmap(perspective)(ids)
        return -jnp.sum(logs), finals[0]

    @jax.jit
    def epoch(params: Params, opt: Params, t: Any, sun: Any, x: Any) -> Any:
        if steps_before:
            x = run(jax.lax.stop_gradient(params), sun, x, None, steps_before)
        (loss, final), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, sun, x)
        gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
        if learn:
            scale = jnp.minimum(1.0, clip / (gnorm + 1e-6))
            grads = jax.tree.map(lambda g: g * scale, grads)
            params, opt = _apply(optimizer, params, opt, grads, t, lr)
        return params, opt, final, loss, gnorm

    return epoch


def init_opt(optimizer: str, params: Params) -> Params:
    z = {k: jnp.zeros_like(v) for k, v in params.items()}
    if optimizer == "sgd":
        return {}
    if optimizer == "rmsprop":
        return {f"v.{k}": v for k, v in z.items()}
    return {**{f"m.{k}": v for k, v in z.items()}, **{f"v.{k}": v for k, v in z.items()}}


def _apply(optimizer: str, params: Params, opt: Params, grads: Params, t: Any, lr: float) -> Any:
    new_p, new_o = {}, dict(opt)
    for k, p in params.items():
        g = grads[k]
        if optimizer == "sgd":
            new_p[k] = p - lr * g
        elif optimizer == "rmsprop":  # PyTorch defaults: alpha 0.99, eps 1e-8
            v = 0.99 * opt[f"v.{k}"] + 0.01 * g * g
            new_o[f"v.{k}"] = v
            new_p[k] = p - lr * g / (jnp.sqrt(v) + 1e-8)
        else:  # adam, PyTorch defaults
            m = 0.9 * opt[f"m.{k}"] + 0.1 * g
            v = 0.999 * opt[f"v.{k}"] + 0.001 * g * g
            new_o[f"m.{k}"], new_o[f"v.{k}"] = m, v
            mh, vh = m / (1 - 0.9**t), v / (1 - 0.999**t)
            new_p[k] = p - lr * mh / (jnp.sqrt(vh) + 1e-8)
    return new_p, new_o


def device(backend: str) -> Any:
    return jax.devices("cuda")[0] if backend == "jax_cuda" else jax.devices("cpu")[0]
