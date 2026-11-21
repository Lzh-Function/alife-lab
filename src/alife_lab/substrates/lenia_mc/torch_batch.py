"""Batched multichannel Lenia rollouts in PyTorch (float64, CPU or CUDA) for searches.

Same update as ``model.step`` (kernels are built by ``model.kernels_fft`` on the host) for a
population of candidates that share the channel/kernel topology (c0, c1) and world size but may
differ in R, T, kernel parameters and initial cells. Results equal the NumPy reference up to
floating-point rounding, which grows chaotically (tests/reference/test_lenia_mc.py). Imported
only in workers that already use torch (search backends ``torch`` / ``torch_cuda``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alife_lab.core.spec import LeniaMCExperiment
from alife_lab.substrates.lenia_mc import model
from alife_lab.substrates.lenia_mc.adapter import ALIVE_THRESHOLD


@dataclass
class BatchRollout:
    frames: Any  # (P, F, h, w, 3) float64 numpy, RGB, optionally centred crops
    alive_fraction: Any  # (P,)
    mass: Any  # (P,)
    speed: Any  # (P,) circular-centroid displacement over the second half / ticks
    field: Any  # (P, H, W, 3) final RGB


def _centroid(weights: Any) -> tuple[Any, Any, Any]:
    """Circular centroid (x, y) of (P, H, W) weights and an 'empty' mask (torch)."""
    import torch

    h, w = weights.shape[1:]
    out = []
    for axis, size in ((1, w), (2, h)):  # x from columns, y from rows
        marg = weights.sum(dim=axis)  # (P, size)
        ang = torch.arange(size, device=weights.device, dtype=weights.dtype) * (2 * math.pi / size)
        s = (marg * torch.sin(ang)).sum(dim=1)
        c = (marg * torch.cos(ang)).sum(dim=1)
        out.append((torch.atan2(s, c) * size / (2 * math.pi)) % size)
    empty = weights.sum(dim=(1, 2)) <= 1e-12
    return out[0], out[1], empty


def _rgb(world: Any) -> Any:
    import torch

    p, c, h, w = world.shape
    rgb = torch.zeros((p, h, w, 3), dtype=world.dtype, device=world.device)
    rgb[..., : min(c, 3)] = world[:, :3].permute(0, 2, 3, 1)
    return rgb.clamp(0.0, 1.0)


def _crop(rgb: Any, weights: Any, side: int) -> Any:
    import torch

    p, h, w, _ = rgb.shape
    cx, cy, empty = _centroid(weights)
    cx = torch.where(empty, torch.full_like(cx, w / 2), cx)
    cy = torch.where(empty, torch.full_like(cy, h / 2), cy)
    out = []
    for i in range(p):
        sh = (h // 2 - round(float(cy[i])), w // 2 - round(float(cx[i])))
        rolled = torch.roll(rgb[i], sh, dims=(0, 1))
        y0, x0 = h // 2 - side // 2, w // 2 - side // 2
        out.append(rolled[y0 : y0 + side, x0 : x0 + side])
    return torch.stack(out)


def rollout(
    exps: list[LeniaMCExperiment],
    steps: int,
    frames: int,
    crop: int | None,
    device: str,
) -> BatchRollout:
    import torch

    first = exps[0]
    topo = [(k.c0, k.c1) for k in first.dynamics.kernels]
    for e in exps[1:]:
        if [(k.c0, k.c1) for k in e.dynamics.kernels] != topo or e.world.size != first.world.size:
            raise ValueError("batched candidates must share world size and kernel topology")
    dev = torch.device(device)
    f64 = torch.float64
    n, c = first.world.size, first.dynamics.channels
    kf = torch.as_tensor(np.stack([model.kernels_fft(e) for e in exps]), device=dev)  # (P,K,..)
    world = torch.as_tensor(np.stack([model.initial_cells(e) for e in exps]), dtype=f64, device=dev)
    par = {name: torch.tensor([[getattr(k, name) for k in e.dynamics.kernels] for e in exps],
                              dtype=f64, device=dev)[:, :, None, None]
           for name in ("m", "s", "h")}  # fmt: skip
    inv_t = torch.tensor([1.0 / e.dynamics.T for e in exps], dtype=f64, device=dev)[
        :, None, None, None
    ]
    src = torch.tensor([a for a, _ in topo], device=dev)
    dst = torch.tensor([b for _, b in topo], device=dev)
    scatter = torch.zeros((len(topo), c), dtype=f64, device=dev)
    scatter[torch.arange(len(topo), device=dev), dst] = 1.0
    marks = {steps * (k + 1) // frames for k in range(frames)}
    half = steps // 2
    shots = []
    track = None  # unwrapped centroid displacement over the second half
    last = None
    with torch.inference_mode():
        for t in range(1, steps + 1):
            fa = torch.fft.rfft2(world)  # (P, C, ...)
            u = torch.fft.irfft2(kf * fa[:, src], s=(n, n))  # (P, K, H, W)
            g = par["h"] * (2 * torch.exp(-(((u - par["m"]) / par["s"]) ** 2) / 2) - 1)
            world = (world + torch.einsum("pkhw,kc->pchw", g, scatter) * inv_t).clamp(0.0, 1.0)
            if t >= half:
                cx, cy, _ = _centroid(world.sum(dim=1))
                if last is None:
                    track = torch.zeros((len(exps), 2), dtype=f64, device=dev)
                else:
                    dx = (cx - last[0] + n / 2) % n - n / 2
                    dy = (cy - last[1] + n / 2) % n - n / 2
                    assert track is not None
                    track += torch.stack([dx, dy], dim=1)
                last = (cx, cy)
            if t in marks:
                rgb = _rgb(world)
                shots.append(_crop(rgb, world.sum(dim=1), crop) if crop else rgb)
        total = world.sum(dim=1)
        alive = (total > ALIVE_THRESHOLD).double().mean(dim=(1, 2))
        speed = (track.norm(dim=1) / max(1, steps - half)) if track is not None else alive * 0
        return BatchRollout(
            frames=torch.stack(shots, dim=1).cpu().numpy(),
            alive_fraction=alive.cpu().numpy(),
            mass=world.sum(dim=(1, 2, 3)).cpu().numpy(),
            speed=speed.cpu().numpy(),
            field=_rgb(world).cpu().numpy(),
        )
