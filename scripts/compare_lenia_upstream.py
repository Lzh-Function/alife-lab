"""Compare the Lenia adapter with Chan's original LeniaND.py (CPU path) step by step.

Usage:
    uv run python scripts/compare_lenia_upstream.py PATH/LeniaND.py PATH/animals.json \
        [--steps 1000] [--size-pow 7] [--code O2u]

The upstream file is a trusted local copy (MIT, Chakazul/Lenia). Only the part before
``class Analyzer`` is executed; its GUI/GPU imports (reikna, PIL, scipy) are stubbed and the GPU
path is disabled so ``numpy.fft`` is used. Prints a JSON report; exit code 0 iff the final
states are bitwise equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

STUBS = (
    "reikna", "reikna.fft", "reikna.cluda", "PIL", "PIL.Image", "PIL.ImageTk", "PIL.ImageDraw",
    "PIL.ImageFont", "scipy", "scipy.ndimage", "scipy.spatial", "scipy.spatial.distance",
    "scipy.signal",
)  # fmt: skip


def load_upstream(path: Path, size_pow: int) -> dict[str, Any]:
    src = path.read_text(encoding="utf-8")
    src = src[: src.index("class Analyzer")]
    for name in STUBS:
        sys.modules.setdefault(name, types.ModuleType(name))
    saved = sys.argv
    sys.argv = ["LeniaND.py", "-d", "2", "-s", str(size_pow)]
    try:
        ns: dict[str, Any] = {"__name__": "lenia_upstream"}
        exec(compile(src, str(path), "exec"), ns)  # noqa: S102 - trusted local reference copy
    finally:
        sys.argv = saved

    def no_gpu(self: Any, a: Any) -> None:
        self.gpu_api = self.gpu_thr = self.gpu_fft1 = self.gpu_fftn = self.gpu_fftshift = None
        self.has_gpu = self.is_gpu = False

    ns["Automaton"].compile_gpu = no_gpu
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lenia_nd", type=Path)
    ap.add_argument("animals", type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--size-pow", type=int, default=7)
    ap.add_argument("--code", default="O2u")
    args = ap.parse_args()

    from alife_lab.core.spec import ExperimentSpec
    from alife_lab.substrates.lenia.lenia import LeniaAdapter

    ns = load_upstream(args.lenia_nd, args.size_pow)
    animals = json.loads(args.animals.read_text(encoding="utf-8"))
    data = next(a for a in animals if a.get("code") == args.code)
    size = 1 << args.size_pow
    world = ns["Board"]([size, size])
    animal = ns["Board"].from_data(data)
    world.params = dict(animal.params)
    world.add(animal)
    up = ns["Automaton"](world)

    p = animal.params
    kn = {1: "polynomial", 2: "exponential", 3: "step", 4: "staircase"}[p["kn"]]
    gn = {1: "polynomial", 2: "gaussian", 3: "step"}[p["gn"]]
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "upstream-compare",
                "substrate": "lenia",
                "seed": 0,
                "world": {"width": size, "height": size},
                "dynamics": {
                    "T": p["T"],
                    "kernel": {"R": p["R"], "b": [float(f) for f in p["b"]], "core": kn},
                    "growth": {"m": p["m"], "s": p["s"], "function": gn},
                },
                "initialization": {"preset": "pattern", "pattern": args.code},
            }
        }
    )
    adapter = LeniaAdapter()
    ours = adapter.initialize(spec)
    initial_equal = bool(np.array_equal(ours.cells, world.cells))
    kernel_equal = bool(np.array_equal(ours.kfft, up.kernel_FFT))
    worst, first_diff = 0.0, None
    for t in range(1, args.steps + 1):
        up.calc_once()
        ours = adapter.step(ours, spec)
        d = float(np.max(np.abs(ours.cells - up.world.cells)))
        if d > 0 and first_diff is None:
            first_diff = t
        worst = max(worst, d)
    up_cells = np.ascontiguousarray(up.world.cells, dtype=np.float64)
    report = {
        "upstream": args.lenia_nd.name,
        "code": args.code,
        "size": size,
        "steps": args.steps,
        "initial_equal": initial_equal,
        "kernel_fft_equal": kernel_equal,
        "max_abs_difference": worst,
        "first_step_with_difference": first_diff,
        "bitwise_equal_final": bool(np.array_equal(ours.cells, up_cells)),
        "final_sha256_ours": hashlib.sha256(ours.cells.tobytes()).hexdigest(),
        "final_sha256_upstream": hashlib.sha256(up_cells.tobytes()).hexdigest(),
        "final_mass": float(ours.cells.sum()),
        "numpy": np.__version__,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["bitwise_equal_final"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
