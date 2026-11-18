"""Command line interface. Shares the service layer and validation with the API."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from alife_lab import __version__

PLANNED_COMMANDS = (
    "runs reproduce / report / export / sweep / discoveries / assays / benchmark "
    "(planned, Phase 2+)"
)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    from alife_lab.execution.probe import probe_system
    from alife_lab.provenance import git_info, source_digest
    from alife_lab.service.catalog import load_presets, load_profiles
    from alife_lab.service.settings import load_settings

    settings = load_settings()
    info = probe_system(settings.runs_root, settings.frontend_dist)
    checks: list[tuple[str, str, str]] = []  # (status, name, detail)

    def add(ok: bool | None, name: str, detail: str, required: bool = True) -> None:
        status = "ok" if ok else ("FAIL" if required and ok is False else "warn")
        checks.append((status, name, detail))

    py = info["python"]["version"]
    add(tuple(int(x) for x in py.split(".")[:2]) >= (3, 12), "python", py)
    add(info["tools"]["uv"] is not None, "uv", str(info["tools"]["uv"]))
    add(info["disk"]["runs_root_writable"], "runs_dir_writable", str(settings.runs_root))
    add(True, "sqlite", info["sqlite"]["library_version"])
    try:
        presets = load_presets(settings.configs_dir)
        profiles = load_profiles(settings.configs_dir)
        add(True, "configs", f"{len(presets)} presets, {len(profiles)} execution profiles valid")
    except Exception as exc:
        add(False, "configs", f"invalid configuration: {exc}")
    add(info["tools"]["node"] is not None, "node", str(info["tools"]["node"]), required=False)
    add(info["tools"]["npm"] is not None, "npm", str(info["tools"]["npm"]), required=False)
    add(
        info["frontend"]["dist_present"],
        "frontend_build",
        "frontend/dist present"
        if info["frontend"]["dist_present"]
        else "missing: run `npm --prefix frontend ci && npm --prefix frontend run build`",
        required=False,
    )
    gpu = info["gpu"]
    if gpu["devices"]:
        for dev in gpu["devices"]:
            add(
                True,
                f"gpu[{dev['index']}]",
                f"{dev['name']} | VRAM total {dev['memory_total_mib']} MiB, free "
                f"{dev['memory_free_mib']} MiB | driver {dev['driver_version']} | "
                f"cc {dev['compute_capability']} (probed via nvidia-smi; no GPU runtime loaded)",
                required=False,
            )
    else:
        add(None, "gpu", f"no GPU detected ({gpu['status']}); CPU-only mode", required=False)
    optional = {k: v for k, v in info["optional_backends"].items() if v}
    add(
        None if not optional else True,
        "optional_backends",
        json.dumps(optional)
        if optional
        else "none installed (JAX/PyTorch/CuPy/MuJoCo are Phase 3+)",
        required=False,
    )
    git = git_info()
    add(
        git["git_commit"] is not None,
        "git",
        f"commit={git['git_commit']} dirty={git['git_dirty']}"
        if git["git_commit"]
        else "no commit yet: runs will be labelled non_release_reproducibility",
        required=False,
    )
    add(True, "source_digest", source_digest()[:16] + "…")
    server = _controller_running(settings.lock_path)
    add(True, "controller", "running (lock held)" if server else "not running")
    if settings.db_path.exists():
        try:
            conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
            version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            conn.close()
            add(True, "database", f"{settings.db_path.name}: schema v{version}, {runs} runs")
        except sqlite3.Error as exc:
            add(False, "database", str(exc))
    else:
        add(True, "database", "not created yet (created on first serve/run)")

    if args.json:
        _print_json({"version": __version__, "checks": checks, "probe": info})
    else:
        print(f"ALife Lab {__version__} — doctor")
        print(
            f"  OS: {info['os']['distribution']} ({info['os']['system']} {info['os']['release']}, "
            f"WSL={info['os']['wsl']})"
        )
        print(f"  CPU: {info['cpu']['model']} — {info['cpu']['logical_cores']} logical cores")
        print(
            f"  RAM: {info['memory']['available_mb']} / {info['memory']['total_mb']} MB available"
        )
        print(f"  Disk ({info['disk']['path_checked']}): {info['disk']['free_gb']} GB free")
        for status, name, detail in checks:
            print(f"  [{status:>4}] {name}: {detail}")
    return 1 if any(s == "FAIL" for s, _, _ in checks) else 0


def _controller_running(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from alife_lab.api.app import create_app
    from alife_lab.api.security import is_loopback, make_session_token
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.settings import load_settings

    if not is_loopback(args.host):
        print(
            f"refusing to bind {args.host!r}: only loopback hosts are allowed until "
            "authentication for remote access is implemented (spec 2.4).",
            file=sys.stderr,
        )
        return 2
    settings = load_settings()
    if _controller_running(settings.lock_path):
        print("another ALife controller is already running.", file=sys.stderr)
        return 2
    token = make_session_token(settings.token_path)
    app = create_app(settings, LocalExecutor(), token, args.port)
    host_display = f"[{args.host}]" if ":" in args.host else args.host
    print(f"ALife Lab {__version__}: http://{host_display}:{args.port}/  (Ctrl+C to stop)")
    if not (settings.frontend_dist / "index.html").is_file():
        print(
            "warning: frontend/dist missing; build with `npm --prefix frontend ci && "
            "npm --prefix frontend run build`",
            file=sys.stderr,
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, ws_max_size=1 << 20)
    return 0


# --------------------------------------------------------------------------- #
# catalog / targets
# --------------------------------------------------------------------------- #


def cmd_substrates(args: argparse.Namespace) -> int:
    from alife_lab.substrates import list_substrates

    for m in list_substrates():
        caps = m.capabilities
        print(
            f"{m.slug}\t{m.plugin_version}\tbackends={','.join(m.backends)}\t"
            f"clock={caps.clock_unit}\tinterventions={','.join(caps.intervention_types)}"
        )
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    from alife_lab.service.catalog import load_presets
    from alife_lab.service.settings import load_settings

    for preset in load_presets(load_settings().configs_dir).values():
        exp = preset.spec.experiment
        print(
            f"{preset.id}\t{exp.substrate}\t{exp.world.width}x{exp.world.height}\t"
            f"seed={exp.seed}\tspec={preset.spec.digest()[:12]}"
        )
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    from alife_lab.execution.executors import PLANNED_ROLES, LocalExecutor

    if args.targets_cmd == "probe" and args.target_id != "local":
        print(f"unknown target {args.target_id!r}; only 'local' is implemented", file=sys.stderr)
        return 2
    target = LocalExecutor().probe()
    _print_json(
        {"targets": [target.to_json()], "planned_roles_not_implemented": list(PLANNED_ROLES)}
    )
    return 0


# --------------------------------------------------------------------------- #
# headless run
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    from alife_lab.core.spec import load_experiment_spec
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.controller import Controller, ControllerError, ControllerLocked
    from alife_lab.service.settings import load_settings

    if args.target != "local":
        print(f"target {args.target!r} is not implemented; use --target local", file=sys.stderr)
        return 2
    spec = load_experiment_spec(Path(args.scenario))

    async def main() -> int:
        try:
            controller = Controller(load_settings(), LocalExecutor())
        except ControllerLocked as exc:
            print(f"{exc}. Use the web GUI/API of the running server instead.", file=sys.stderr)
            return 2
        await controller.start()
        try:
            summary = controller.create_run(spec, args.profile, ticks_per_second=0, author="cli")
            run_id = summary["id"]
            print(f"created {run_id} (spec {summary['spec_hash'][:12]}, profile {args.profile})")
            await controller.start_run(run_id)
            t0 = time.monotonic()
            while True:
                await asyncio.sleep(0.2)
                summary = controller.run_summary(run_id)
                if summary["state"] in ("completed", "failed", "cancelled", "interrupted"):
                    break
            elapsed = time.monotonic() - t0
            print(
                f"{run_id}: {summary['state']} at tick {summary['clock']['value']} "
                f"in {elapsed:.2f}s ({summary['finish_reason']})"
            )
            return 0 if summary["state"] == "completed" else 1
        except ControllerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            await controller.shutdown()

    return asyncio.run(main())


# --------------------------------------------------------------------------- #
# runs (read-only inspection)
# --------------------------------------------------------------------------- #


def _ro_db() -> sqlite3.Connection:
    from alife_lab.service.settings import load_settings

    path = load_settings().db_path
    if not path.exists():
        raise SystemExit("no database yet")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_runs(args: argparse.Namespace) -> int:
    from alife_lab.artifacts.store import LocalArtifactStore
    from alife_lab.service.settings import load_settings

    conn = _ro_db()
    if args.runs_cmd == "list":
        sql = (
            "SELECT r.id, r.state, r.tick, r.effective_mode, COALESCE(r.label, e.name) AS name "
            "FROM runs r JOIN experiments e ON e.id=r.experiment_id"
        )
        params: tuple[Any, ...] = ()
        if args.status:
            sql += " WHERE r.state=?"
            params = (args.status,)
        for row in conn.execute(sql + " ORDER BY r.created_at DESC", params):
            print(
                f"{row['id']}\t{row['state']}\ttick={row['tick']}\t{row['effective_mode']}\t"
                f"{row['name']}"
            )
        return 0
    run = conn.execute("SELECT * FROM runs WHERE id=?", (args.run_id,)).fetchone()
    if run is None:
        print("run not found", file=sys.stderr)
        return 1
    store = LocalArtifactStore(load_settings().runs_root)
    if args.runs_cmd == "inspect":
        ckpts = [
            dict(r)
            for r in conn.execute(
                "SELECT id, simulation_clock, reason, sha256 FROM checkpoints WHERE run_id=? "
                "ORDER BY created_at",
                (args.run_id,),
            )
        ]
        _print_json(
            {
                "run": dict(run),
                "checkpoints": ckpts,
                "manifest": json.loads(store.resolve(args.run_id, "manifest.json").read_text()),
            }
        )
        return 0
    if args.runs_cmd == "verify":
        bad = 0
        rows = list(
            conn.execute(
                "SELECT logical_path, sha256 FROM artifacts WHERE run_id=? "
                "UNION ALL SELECT logical_path, sha256 FROM checkpoints "
                "WHERE run_id=?",
                (args.run_id, args.run_id),
            )
        )
        for row in rows:
            ok = store.verify(args.run_id, row["logical_path"], row["sha256"])
            bad += 0 if ok else 1
            print(f"[{'ok' if ok else 'BAD'}] {row['logical_path']}")
        print(f"{len(rows) - bad}/{len(rows)} artifacts verified")
        return 0 if bad == 0 else 1
    return 2


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# fork (headless) / compare (read-only)
# --------------------------------------------------------------------------- #


def cmd_fork(args: argparse.Namespace) -> int:
    import yaml

    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.controller import Controller, ControllerError, ControllerLocked
    from alife_lab.service.settings import load_settings

    intervention = None
    if args.intervention:
        with Path(args.intervention).open("r", encoding="utf-8") as handle:
            intervention = yaml.safe_load(handle)
        if not isinstance(intervention, dict):
            print("intervention file must contain a mapping {kind, payload}", file=sys.stderr)
            return 2
    overrides = {"runner": {"max_steps": args.max_steps}} if args.max_steps else None

    async def main() -> int:
        try:
            controller = Controller(load_settings(), LocalExecutor())
        except ControllerLocked as exc:
            print(f"{exc}. Use the web GUI/API of the running server instead.", file=sys.stderr)
            return 2
        await controller.start()
        try:
            child = controller.fork_run(
                args.run_id,
                args.checkpoint,
                label=args.label,
                spec_overrides=overrides,
                intervention=intervention,
                ticks_per_second=0,
                author="cli",
            )
            rid = child["id"]
            print(f"forked {rid} from {args.run_id} at tick {child['clock']['value']}")
            await controller.start_run(rid)
            while controller.run_summary(rid)["state"] not in (
                "completed",
                "failed",
                "cancelled",
                "interrupted",
            ):
                await asyncio.sleep(0.2)
            summary = controller.run_summary(rid)
            print(f"{rid}: {summary['state']} at tick {summary['clock']['value']}")
            return 0 if summary["state"] == "completed" else 1
        except ControllerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            await controller.shutdown()

    return asyncio.run(main())


def cmd_replay_verify(args: argparse.Namespace) -> int:
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.controller import Controller, ControllerError, ControllerLocked
    from alife_lab.service.settings import load_settings

    async def main() -> int:
        try:
            controller = Controller(load_settings(), LocalExecutor())
        except ControllerLocked as exc:
            print(
                f"{exc}. Use POST /api/v1/experiments/<id>/replay/verify on the running server.",
                file=sys.stderr,
            )
            return 2
        await controller.start()
        try:
            report = await controller.verify_replay(args.run_id)
        except ControllerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            await controller.shutdown()
        for seg in report["segments"]:
            status = "ok" if seg["ok"] else "MISMATCH"
            print(
                f"[{status}] t={seg['from_tick']}->{seg['to_tick']} "
                f"differing={seg.get('differing_cells')} "
                f"replayed={seg.get('interventions_replayed')}"
            )
        print(
            f"{report['segments_checked']}/{report['segments_total']} segments checked: "
            f"{report['status']}"
        )
        return 0 if report["ok"] else 1

    return asyncio.run(main())


def cmd_compare(args: argparse.Namespace) -> int:
    from alife_lab.artifacts.store import LocalArtifactStore
    from alife_lab.service.compare import state_divergence
    from alife_lab.service.settings import load_settings

    conn = _ro_db()

    def query(sql: str, params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, params)]

    for rid in args.run_ids:
        if not query("SELECT id FROM runs WHERE id=?", (rid,)):
            print(f"run {rid} not found", file=sys.stderr)
            return 1
    report = state_divergence(query, LocalArtifactStore(load_settings().runs_root), args.run_ids)
    if args.json:
        _print_json(report)
        return 0
    print(f"common checkpoint ticks: {report['common_ticks']} (shown: {report['sampled_ticks']})")
    print("tick\t" + "\t".join(f"hamming[{i}]" for i in range(len(args.run_ids))) + "\talive")
    for point in report["points"]:
        values = "\t".join("-" if d is None else str(d) for d in point["hamming_to_first"])
        print(f"{point['tick']}\t{values}\t{point['alive']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alife",
        description="ALife Lab command line",
        epilog=f"Not yet available: {PLANNED_COMMANDS}",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="probe hardware/software and validate configuration")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="run the controller, API and web GUI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("substrates", help="list substrates")
    p.add_argument("substrates_cmd", choices=["list"])
    p.set_defaults(func=cmd_substrates)

    p = sub.add_parser("presets", help="list presets")
    p.add_argument("presets_cmd", choices=["list"])
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("targets", help="list or probe execution targets")
    tsub = p.add_subparsers(dest="targets_cmd", required=True)
    tsub.add_parser("list")
    tp = tsub.add_parser("probe")
    tp.add_argument("target_id")
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("run", help="run a scenario headless to completion (no server running)")
    p.add_argument("scenario")
    p.add_argument("--target", default="local")
    p.add_argument("--profile", default="cpu_dev")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("runs", help="inspect stored runs (read-only)")
    rsub = p.add_subparsers(dest="runs_cmd", required=True)
    rl = rsub.add_parser("list")
    rl.add_argument("--status")
    for name in ("inspect", "verify"):
        rp = rsub.add_parser(name)
        rp.add_argument("run_id")
    rf = rsub.add_parser("fork", help="fork a run from a checkpoint and run it headless")
    rf.add_argument("run_id")
    rf.add_argument("--checkpoint", default="latest")
    rf.add_argument("--intervention", help="YAML file with {kind: brush, payload: {...}}")
    rf.add_argument("--max-steps", type=int)
    rf.add_argument("--label")
    rf.set_defaults(func=cmd_fork)
    rv = rsub.add_parser("replay-verify", help="re-simulate between checkpoints and compare states")
    rv.add_argument("run_id")
    rv.set_defaults(func=cmd_replay_verify)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("compare", help="state divergence of runs at common checkpoint ticks")
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
