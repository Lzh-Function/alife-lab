"""Compare the Flow-Lenia adapter with the authors' JAX implementation (erwanplantec/FlowLenia).

Two environments are needed because JAX is not a project dependency:

1. Project env — write the rule, the initial state and our trajectory::

       uv run python scripts/compare_flow_lenia_upstream.py ours OUT.npz [--steps 200] [--size 64]
           [--boundary wall|torus] [--seed 0]

2. A separate env with jax[cpu], equinox, jaxtyping — run the official code on the same rule and
   initial state (the upstream checkout is only added to sys.path, never modified)::

       JAXENV/bin/python -I scripts/compare_flow_lenia_upstream.py upstream OUT.npz UPSTREAM_DIR
           [--x64]

3. Project env — compare::

       uv run python scripts/compare_flow_lenia_upstream.py report OUT.npz

The upstream step is run in float32 (its default) and, with ``--x64``, in float64.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXAMPLE_M = [[3, 1, 0], [0, 3, 1], [1, 0, 3]]
RECORD_EVERY = 10


def cmd_ours(args: argparse.Namespace) -> int:
    from alife_lab.core.spec import ExperimentSpec
    from alife_lab.substrates.flow_lenia.flow_lenia import FlowLeniaAdapter

    spec = ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "upstream-compare",
                "substrate": "flow_lenia",
                "seed": args.seed,
                "world": {"width": args.size, "height": args.size, "boundary": args.boundary},
                "initialization": {"size": args.size * 5 // 16},
                "dynamics": {"channels": 3, "rule": {"connectivity": EXAMPLE_M}},
            }
        }
    )
    adapter = FlowLeniaAdapter()
    state = adapter.initialize(spec)
    traj = [state.cells.copy()]
    for t in range(1, args.steps + 1):
        state = adapter.step(state, spec)
        if t % RECORD_EVERY == 0:
            traj.append(state.cells.copy())
    d = spec.experiment.dynamics
    config = {
        "size": args.size, "channels": 3, "boundary": args.boundary, "dt": d.dt, "dd": d.dd,
        "sigma": d.sigma, "steps": args.steps, "record_every": RECORD_EVERY,
    }  # fmt: skip
    np.savez(
        args.out,
        ours=np.stack(traj),
        rule=np.array(json.dumps(state.rule.to_json())),
        config=np.array(json.dumps(config)),
    )
    print(f"wrote {args.out}: {len(traj)} frames")
    return 0


def cmd_upstream(args: argparse.Namespace) -> int:
    import jax

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    sys.path.insert(0, str(Path(args.upstream_dir).resolve()))
    from flowlenia.flowlenia import Config, FlowLenia

    data = np.load(args.out)
    rule = json.loads(str(data["rule"]))
    cfg_in = json.loads(str(data["config"]))
    kernels = rule["kernels"]
    k = len(kernels)
    c = cfg_in["channels"]
    c0 = [kk["source"] for kk in kernels]
    c1 = [[i for i, kk in enumerate(kernels) if kk["target"] == t] for t in range(c)]
    cfg = Config(
        X=cfg_in["size"], Y=cfg_in["size"], C=c, c0=c0, c1=c1, k=k, dt=cfg_in["dt"],
        dd=cfg_in["dd"], sigma=cfg_in["sigma"], border=cfg_in["boundary"],
    )  # fmt: skip
    dtype = jnp.float64 if args.x64 else jnp.float32
    fl = FlowLenia(cfg, key=jr.key(0))

    def arr(name: str) -> Any:
        return jnp.asarray([kk[name] for kk in kernels], dtype=dtype)

    fl = eqx.tree_at(
        lambda m: (m.R, m.r, m.m, m.s, m.h, m.a, m.b, m.w),
        fl,
        (jnp.asarray(rule["R"], dtype=dtype), arr("r"), arr("m"), arr("s"), arr("h"),
         arr("a"), arr("b"), arr("w")),
    )  # fmt: skip
    state = fl.initialize(jr.key(0))
    state = state._replace(A=jnp.asarray(data["ours"][0], dtype=dtype))
    step = jax.jit(fl.__call__)
    traj = [np.asarray(state.A, dtype=np.float64)]
    for t in range(1, cfg_in["steps"] + 1):
        state = step(state)
        if t % cfg_in["record_every"] == 0:
            traj.append(np.asarray(state.A, dtype=np.float64))
    key = "upstream_f64" if args.x64 else "upstream_f32"
    out = {name: data[name] for name in data.files}
    out[key] = np.stack(traj)
    out[key + "_versions"] = np.array(
        json.dumps({"jax": jax.__version__, "equinox": eqx.__version__, "numpy": np.__version__})
    )
    np.savez(args.out, **out)
    print(f"wrote {key} to {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    data = np.load(args.out)
    cfg = json.loads(str(data["config"]))
    ours = data["ours"]
    report: dict[str, Any] = {"config": cfg}
    for key in ("upstream_f32", "upstream_f64"):
        if key not in data.files:
            continue
        up = data[key]
        diff = np.abs(ours - up).reshape(len(ours), -1).max(axis=1)
        report[key] = {
            "versions": json.loads(str(data[key + "_versions"])),
            "max_abs_difference_by_tick": {
                str(i * cfg["record_every"]): float(v) for i, v in enumerate(diff)
            },
            "mass_ours_final": float(ours[-1].sum()),
            "mass_upstream_final": float(up[-1].sum()),
            "mass_upstream_initial": float(up[0].sum()),
            "max_value_final": float(ours[-1].max()),
        }
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ours")
    p.add_argument("out", type=Path)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--boundary", choices=["wall", "torus"], default="wall")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("upstream")
    p.add_argument("out", type=Path)
    p.add_argument("upstream_dir", type=Path)
    p.add_argument("--x64", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    args = ap.parse_args()
    return {"ours": cmd_ours, "upstream": cmd_upstream, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
