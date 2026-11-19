"""NCA target images: procedurally drawn built-in shapes (no image assets) or custom RGBA."""

from __future__ import annotations

import base64

import numpy as np
import numpy.typing as npt

from alife_lab.core.spec import NCATarget

F32 = npt.NDArray[np.float32]
SUPERSAMPLE = 4


def _coverage(size: int, inside: object) -> F32:
    """Anti-aliased coverage of an implicit shape on [-1, 1]^2 (supersampled)."""
    n = size * SUPERSAMPLE
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    y, x = np.meshgrid(-c, c, indexing="ij")  # y up
    mask = inside(x, y).astype(np.float32)  # type: ignore[operator]
    out: F32 = mask.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE).mean(axis=(1, 3))
    return out


def _segment_distance(x: F32, y: F32, a: tuple[float, float], b: tuple[float, float]) -> F32:
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0)
    out: F32 = np.hypot(x - (ax + t * vx), y - (ay + t * vy))
    return out


def preset_rgba(name: str, size: int) -> F32:
    if name == "disc":
        cov = _coverage(size, lambda x, y: x * x + y * y <= 0.8**2)
        rgb = (0.95, 0.55, 0.15)
    elif name == "ring":
        cov = _coverage(size, lambda x, y: (x * x + y * y >= 0.45**2) & (x * x + y * y <= 0.85**2))
        rgb = (0.2, 0.55, 0.95)
    elif name == "heart":
        cov = _coverage(
            size,
            lambda x, y: (
                ((x / 0.85) ** 2 + ((y + 0.1) / 0.85) ** 2 - 1) ** 3
                - (x / 0.85) ** 2 * ((y + 0.1) / 0.85) ** 3
                <= 0
            ),
        )
        rgb = (0.9, 0.15, 0.25)
    elif name == "letter_a":
        cov = _coverage(
            size,
            lambda x, y: (
                (_segment_distance(x, y, (-0.6, -0.8), (0.0, 0.8)) < 0.16)
                | (_segment_distance(x, y, (0.6, -0.8), (0.0, 0.8)) < 0.16)
                | (_segment_distance(x, y, (-0.32, -0.1), (0.32, -0.1)) < 0.14)
            ),
        )
        rgb = (0.15, 0.6, 0.35)
    else:
        raise ValueError(f"unknown NCA target preset {name!r}")
    # Vertical shading so the target is not a single colour (makes the task non-trivial).
    shade = np.linspace(1.0, 0.65, size, dtype=np.float32)[:, None]
    out = np.zeros((size, size, 4), dtype=np.float32)
    for k in range(3):
        out[:, :, k] = np.clip(rgb[k] * shade, 0, 1) * cov  # premultiplied by alpha
    out[:, :, 3] = cov
    return out


def target_rgba(target: NCATarget) -> F32:
    """(size, size, 4) float32 with premultiplied RGB and alpha in [0, 1]."""
    if target.kind == "preset":
        return preset_rgba(target.preset, target.size)
    assert target.rgba_b64 is not None
    raw = np.frombuffer(base64.b64decode(target.rgba_b64), dtype=np.uint8)
    img = raw.reshape(target.size, target.size, 4).astype(np.float32) / 255.0
    img[:, :, :3] *= img[:, :, 3:4]  # straight -> premultiplied alpha
    return img


def padded_target(target: NCATarget, pad: int) -> F32:
    img = target_rgba(target)
    out: F32 = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    return out
