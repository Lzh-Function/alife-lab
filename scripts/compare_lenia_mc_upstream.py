"""Compare lenia_mc with SakanaAI/asal ``substrates/lenia_impl.py`` (JAX, CPU, float64) and
optionally write the test fixture ``tests/fixtures/lenia_mc_asal_reference.json``.

Running third-party code: download lenia_impl.py @ 677ba0e into its own directory and pass its
path; nothing else from that repository is executed.

    uv run --extra nca python scripts/compare_lenia_mc_upstream.py /path/to/lenia_impl.py \
        [--write-fixture]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/lenia_mc_asal_reference.json"
SNAP_TICKS = (50, 100, 200)
N_SAMPLES = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("impl")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--write-fixture", action="store_true")
    args = ap.parse_args()
    import jax

    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location("asal_lenia_impl", args.impl)
    assert spec and spec.loader
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)

    from alife_lab.core.spec import ExperimentSpec
    from alife_lab.substrates.lenia_mc import model

    rng = np.random.default_rng(0)
    out: dict[str, object] = {
        "source": "SakanaAI/asal substrates/lenia_impl.py @ "
        "677ba0ea4d3b3ca78273c9906c6e84d2b1481ce7",
        "jax": jax.__version__,
        "precision": "float64 (jax_enable_x64), CPU",
        "world_size": 128,
        "patterns": {},
    }
    for pattern in ("5N7KKM", "VT049W"):
        lenia = impl.Lenia(impl.ConfigLenia(pattern_id=pattern, world_size=128))
        carry, _, _ = lenia.load_pattern(lenia.pattern)
        exp = ExperimentSpec.model_validate({"experiment": {
            "name": "c", "substrate": "lenia_mc", "seed": 0,
            "initialization": {"preset": "pattern", "pattern": pattern}}}).experiment  # fmt: skip
        kf, ours = model.kernels_fft(exp), model.initial_cells(exp)
        assert np.array_equal(ours, np.transpose(np.asarray(carry.world), (2, 0, 1)))
        idx = rng.integers(0, 128, size=(N_SAMPLES, 3))
        idx[:, 0] %= 3
        mass, snaps, worst = {}, {}, {}
        for t in range(1, args.steps + 1):
            carry, _ = lenia.step(carry, None, phenotype_size=64, center_phenotype=True,
                                  record_phenotype=True)  # fmt: skip
            ours = model.step(ours, kf, exp)
            applied = np.asarray(carry.temp.total_shift) - np.asarray(carry.temp.last_shift)
            theirs = np.transpose(np.asarray(carry.world), (2, 0, 1))
            aligned = np.roll(ours, tuple(-applied), axis=(1, 2))
            if t % 16 == 0 or t == 1:
                mass[t] = float(theirs.sum())
                worst[t] = float(np.abs(aligned - theirs).max())
            if t in SNAP_TICKS:
                snaps[t] = {"shift": applied.tolist(),
                            "values": theirs[idx[:, 0], idx[:, 1], idx[:, 2]].tolist()}  # fmt: skip
        print(pattern, {t: f"{v:.1e}" for t, v in worst.items() if t in (1, 64, 128, 256)})
        out["patterns"][pattern] = {"indices": idx.tolist(), "mass": mass, "snapshots": snaps,  # type: ignore[index]
                                    "max_abs_diff_vs_alife_lab": worst}  # fmt: skip
    if args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(out, separators=(",", ":")))
        print("wrote", FIXTURE, FIXTURE.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
