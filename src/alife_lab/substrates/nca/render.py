"""NumPy-only helpers shared by the NCA adapters (no JAX import)."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_rgb(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Composite premultiplied RGBA on white: rgb + (1 - alpha), clipped (as in the article)."""
    a = np.clip(x[..., 3:4], 0.0, 1.0)
    out: np.ndarray[Any, Any] = np.clip(1.0 - a + x[..., :3], 0.0, 1.0)
    return out


def rgb_bytes(x: np.ndarray[Any, Any] | None) -> tuple[dict[str, Any], bytes]:
    if x is None:
        raise ValueError("no state to render")
    h, w = x.shape[:2]
    data = np.rint(to_rgb(x) * 255).astype(np.uint8).tobytes()
    return {"codec": "field_u8", "width": w, "height": h, "channels": 3,
            "scale": "RGBA composited on white"}, data  # fmt: skip


def cell_info(x: np.ndarray[Any, Any] | None, cx: int, cy: int) -> dict[str, Any]:
    if x is None:
        return {}
    h, w, _ = x.shape
    if not (0 <= cx < w and 0 <= cy < h):
        raise ValueError("(x, y) is outside the world")
    cell = x[cy, cx]
    alive = bool(x[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2, 3].max() > 0.1)
    return {
        "rgba": [float(v) for v in cell[:4]],
        "hidden": [float(v) for v in cell[4:]],
        "alive": alive,
    }
