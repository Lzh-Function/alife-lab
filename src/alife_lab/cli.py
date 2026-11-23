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
    "runs reproduce / report / export / discoveries / assays / benchmark (planned, Phase 2+)"
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
        else "none installed (numba: uv sync --extra fast; CUDA via CuPy: uv sync --extra gpu)",
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
        world = getattr(exp, "world", None)
        size = (
            f"{world.width}x{world.height}"
            if world is not None
            else f"{exp.soup.num_programs} programs"  # type: ignore[union-attr]
        )
        print(
            f"{preset.id}\t{exp.substrate}\t{size}\tseed={exp.seed}\tspec={preset.spec.digest()[:12]}"
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
                f"{seg.get('distance_metric')}={seg.get('distance')} "
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


# --------------------------------------------------------------------------- #
# sweep (headless)
# --------------------------------------------------------------------------- #


def cmd_sweep(args: argparse.Namespace) -> int:
    import yaml

    from alife_lab.core.sweep import SweepSpec
    from alife_lab.execution.executors import LocalExecutor
    from alife_lab.service.controller import Controller, ControllerError, ControllerLocked
    from alife_lab.service.settings import load_settings

    with Path(args.sweep).open("r", encoding="utf-8") as handle:
        spec = SweepSpec.model_validate(yaml.safe_load(handle))

    async def main() -> int:
        try:
            controller = Controller(load_settings(), LocalExecutor())
        except ControllerLocked as exc:
            print(f"{exc}. Use POST /api/v1/sweeps on the running server.", file=sys.stderr)
            return 2
        await controller.start()
        try:
            plan = controller.sweeps.plan(spec)
            print(
                f"sweep {spec.sweep.name!r} ({spec.digest()[:12]}): {plan['runs']} runs, "
                f"{len(plan['conditions'])} conditions x {len(spec.sweep.seeds)} seeds, "
                f"{plan['total_ticks']} ticks in total"
            )
            for c in plan["conditions"]:
                print(f"  condition {c['index']}: {c['label']} {c['assignments']}")
            if args.dry_run:
                return 0
            summary = await controller.sweeps.create(spec, author="cli")
            sid = summary["id"]
            while controller.sweeps.summary(sid)["state"] == "running":
                await asyncio.sleep(0.5)
            report = controller.sweeps.report(sid)
            if args.json:
                _print_json(report)
                return 0
            print(f"{sid}: {report['sweep']['state']}  endpoint {report['endpoint']}")
            for cond in report["conditions"]:
                s = cond["summary"]
                ref = " (reference)" if cond["is_reference"] else ""
                mean = "-" if s["mean"] is None else f"{s['mean']:.4g}"
                sd = "-" if s["sd"] is None else f"{s['sd']:.3g}"
                print(f"  {cond['label']}{ref}: n={s['n']} mean={mean} sd={sd}")
            for comp in report["comparisons"]:
                ci = comp["mean_difference_ci95"]
                ci_s = f"[{ci[0]:.4g}, {ci[1]:.4g}]" if ci else "-"
                print(
                    f"  {comp['condition']} - {comp['reference']}: "
                    f"diff={comp['mean_difference']:.4g} "
                    f"CI95={ci_s} cliffs_delta={comp['cliffs_delta']:.3g} "
                    f"p={comp['permutation']['p_value']:.4g} ({comp['permutation']['method']}) "
                    f"p_holm={comp['p_value_holm']:.4g}"
                )
            for note in report["notes"]:
                print(f"  note: {note['text']}")
            return 0
        except ControllerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            await controller.shutdown()

    return asyncio.run(main())


# --------------------------------------------------------------------------- #
# backup
# --------------------------------------------------------------------------- #


def cmd_backup(args: argparse.Namespace) -> int:
    from alife_lab.service.settings import load_settings
    from alife_lab.storage.backup import (
        BackupError,
        create_backup,
        restore_backup,
        verify_backup,
    )

    settings = load_settings()
    try:
        if args.backup_cmd == "create":
            result = create_backup(settings.db_path, settings.runs_root, Path(args.dest))
            _print_json(result)
            return 0
        if args.backup_cmd == "verify":
            report = verify_backup(Path(args.path))
            _print_json(report)
            return 0 if report["ok"] else 1
        if args.backup_cmd == "restore":
            result = restore_backup(Path(args.path), Path(args.home), Path(args.runs_dir))
            _print_json(result)
            print(
                f"restored. Use it with: ALIFE_HOME={args.home} ALIFE_RUNS_DIR={args.runs_dir} "
                "uv run alife serve",
                file=sys.stderr,
            )
            return 0
    except BackupError as exc:
        print(f"backup error: {exc}", file=sys.stderr)
        return 1
    return 2


# --------------------------------------------------------------------------- #
# connectome datasets (F0+)
# --------------------------------------------------------------------------- #


def cmd_connectome(args: argparse.Namespace) -> int:
    from alife_lab.connectome import (
        DatasetError,
        fetch_dataset,
        list_dataset_specs,
        load_connectome_pack,
        summarize_connectome_pack,
        verify_dataset,
    )
    from alife_lab.service.settings import load_settings

    settings = load_settings()
    ds_dir = settings.resolved_datasets_dir
    try:
        if args.connectome_cmd == "list":
            rows = []
            for spec in list_dataset_specs():
                report = verify_dataset(spec.dataset_id, ds_dir, spec=spec)
                rows.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "species": spec.species,
                        "version": spec.version,
                        "files": len(spec.files),
                        "total_size_bytes": spec.total_size_bytes,
                        "verified_local": report["ok"],
                    }
                )
            if args.json:
                _print_json({"datasets_dir": str(ds_dir), "datasets": rows})
            else:
                for row in rows:
                    status = "verified" if row["verified_local"] else "not_present"
                    size_mb = row["total_size_bytes"] / (1024 * 1024)
                    print(
                        f"{row['dataset_id']}\t{row['species']}\t{row['version']}\t"
                        f"{row['files']} files ({size_mb:.1f} MiB)\t{status}"
                    )
            return 0

        if args.connectome_cmd == "fetch":
            report = fetch_dataset(
                args.dataset_id,
                ds_dir,
                max_datasets_mb=settings.max_datasets_mb,
            )
            _print_json(report)
            return 0 if report["ok"] else 1

        if args.connectome_cmd == "verify":
            report = verify_dataset(args.dataset_id, ds_dir)
            _print_json(report)
            return 0 if report["ok"] else 1

        if args.connectome_cmd == "summary":
            t0 = time.perf_counter()
            pack = load_connectome_pack(args.dataset_id, ds_dir, verify_hashes=not args.no_verify)
            load_s = round(time.perf_counter() - t0, 3)
            summary = summarize_connectome_pack(pack)
            summary["measured"] = {"load_seconds": load_s}
            if args.json:
                _print_json(summary)
            else:
                c = summary["counts"]
                d = summary["degree"]
                print(
                    f"{summary['dataset_id']} ({summary['species']})  "
                    f"pack_digest={summary['pack_digest'][:16]}…  loaded in {load_s:.2f}s"
                )
                print(
                    f"  cells={c['cells']:,}  edges={c['edges']:,}  "
                    f"synapses={c['total_synapses']:,}"
                )
                print(
                    f"  excitatory={c['excitatory_edges']:,} edges "
                    f"({100 * c['excitatory_edge_ratio']:.2f}%), "
                    f"{c['excitatory_synapses']:,} synapses "
                    f"({100 * c['excitatory_synapse_ratio']:.2f}%)"
                )
                print(
                    f"  inhibitory={c['inhibitory_edges']:,} edges "
                    f"({100 * c['inhibitory_edge_ratio']:.2f}%), "
                    f"{c['inhibitory_synapses']:,} synapses "
                    f"({100 * c['inhibitory_synapse_ratio']:.2f}%)"
                )
                print(
                    f"  degree out: median={d['out_median']:.0f} mean={d['out_mean']:.2f} "
                    f"max={d['out_max']} | in: median={d['in_median']:.0f} "
                    f"mean={d['in_mean']:.2f} max={d['in_max']}"
                )
                print(
                    f"  self_loops={c['self_loops']}  isolated={c['isolated_cells']}  "
                    f"mixed_sign_pre={c['mixed_sign_presynaptic_neurons']}  "
                    f"regions={c['unique_regions']}"
                )
                print(f"  positions: {summary['position_sources']}")
                for chk in summary["reference_checks"]:
                    tag = "ok" if chk["ok"] else "DIFF"
                    print(
                        f"  [{tag:>4}] {chk['check']}: actual={chk['actual']} "
                        f"expected={chk['expected']}"
                    )
            return 0

        if args.connectome_cmd == "screen":
            from alife_lab.connectome.ablation import run_silencing_screen
            from alife_lab.connectome.lif import (
                LIFParams,
                make_shuffled_csr,
                make_synthetic_tiny_pack,
            )

            stimuli: list[tuple[str, float]] = []
            for raw in args.stim:
                if ":" not in raw:
                    print(f"invalid --stim {raw!r}; expected PORT:RATE_HZ", file=sys.stderr)
                    return 2
                port_s, rate_s = raw.split(":", 1)
                stimuli.append((port_s.strip(), float(rate_s)))
            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            cell_types = (
                [c.strip() for c in args.cell_types.split(",") if c.strip()]
                if args.cell_types
                else None
            )
            if args.dataset_id == "synthetic_tiny":
                pack = make_synthetic_tiny_pack()
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
            csr = pack.edges.to_csr(pack.cells.num_cells)
            if args.wiring == "shuffled_control":
                csr = make_shuffled_csr(csr, int(seeds[0]))
            report = run_silencing_screen(
                pack,
                csr,
                LIFParams(),
                stimuli=stimuli,
                cell_types=cell_types,
                top_k=args.top_k,
                target_port=args.target,
                seeds=seeds,
                ticks=args.ticks,
                wiring=args.wiring,
                backend=args.backend,
                batch_size=args.batch_size,
            )
            if args.json:
                _print_json(report)
            else:
                b = report["baseline"]
                print(
                    f"screen {report['dataset_id']} wiring={report['wiring']} "
                    f"backend={report['backend']} (B={report['batch_size']}): "
                    f"{report['total_evaluations']} evals in {report['wall_seconds']:.2f}s "
                    f"({report['brain_seconds_per_wall_second']:.1f} brain-s/s)"
                )
                print(
                    f"  baseline {report['target_port']}: "
                    f"{b['mean_hz']:.2f} +/- {b['sd_hz']:.2f} Hz  seeds={report['seeds']}"
                )
                print("  cell_type\tcells\tmean_hz\tdelta_hz\tsupp%\tcliffs_d\tp_perm\tp_holm")
                for r in report["results"]:
                    p_str = "na" if r["p_value"] is None else f"{r['p_value']:.4f}"
                    ph_str = "na" if r["p_value_holm"] is None else f"{r['p_value_holm']:.4f}"
                    cd_str = "na" if r["cliffs_delta"] is None else f"{r['cliffs_delta']:+.2f}"
                    print(
                        f"  {r['cell_type']}\t{r['cell_count']}\t{r['mean_hz']:.2f}\t"
                        f"{r['delta_hz']:+.2f}\t{r['suppression_pct']:+.1f}%\t"
                        f"{cd_str}\t{p_str}\t{ph_str}"
                    )
            return 0

        if args.connectome_cmd == "pavlov":
            from alife_lab.connectome.learning import run_pavlovian_experiment
            from alife_lab.connectome.lif import make_synthetic_mb_pack
            from alife_lab.connectome.plasticity import MBPlasticityParams

            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            if args.dataset_id == "synthetic_mb":
                pack = make_synthetic_mb_pack()
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
            pp = MBPlasticityParams(
                enabled=True,
                tau_elig_ms=args.tau_elig,
                tau_dan_ms=args.tau_dan,
                dan_threshold=args.dan_threshold,
                lr_ltd=args.lr_ltd,
                lr_ext=args.lr_ext,
            )
            report = run_pavlovian_experiment(
                pack,
                seeds=seeds,
                p_params=pp,
                odor_hz=args.odor_hz,
                dan_hz=args.dan_hz,
                probe_ms=args.probe_ms,
                trial_ms=args.trial_ms,
                n_trials=args.trials,
                ext_ms=args.ext_ms,
                unpaired_gap_ms=args.unpaired_gap_ms,
                wiring=args.wiring,
                backend=args.backend,
            )
            if args.json:
                _print_json(report)
            else:
                sm = report["summary"]
                cp = report["comparisons"]
                print(
                    f"pavlov {report['dataset_id']} wiring={report['wiring']} "
                    f"backend={report['backend']} plastic_edges={report['plastic_edges_total']} "
                    f"wall={report['wall_seconds']:.2f}s"
                )
                print(
                    f"  OdorA MBON11: pre={sm['pre_odor_a_mbon11_hz']['mean']:.2f} "
                    f"-> 1-trial={sm['after1_odor_a_mbon11_hz']['mean']:.2f} "
                    f"-> post={sm['post_odor_a_mbon11_hz']['mean']:.2f} "
                    f"-> ext={sm['ext_odor_a_mbon11_hz']['mean']:.2f} Hz "
                    f"(p_post_vs_pre={cp['paired_post_vs_pre_odor_a']['permutation']['p_value']})"
                )
                print(
                    f"  OdorB MBON11: pre={sm['pre_odor_b_mbon11_hz']['mean']:.2f} "
                    f"-> post={sm['post_odor_b_mbon11_hz']['mean']:.2f} Hz"
                )
                print(
                    f"  Controls (OdorA MBON11): "
                    f"unpaired={sm['unpaired_odor_a_mbon11_hz']['mean']:.2f} "
                    f"no_dan={sm['no_dan_odor_a_mbon11_hz']['mean']:.2f} Hz "
                    f"(changed_edges: paired={sm['changed_edges_post']['mean']:.1f}, "
                    f"unpaired={sm['changed_edges_unpaired']['mean']:.1f}, "
                    f"no_dan={sm['changed_edges_no_dan']['mean']:.1f})"
                )
            return 0

        if args.connectome_cmd in ("sysid", "sysid-map"):
            from alife_lab.connectome.subcircuit import (
                extract_subcircuit,
                make_synthetic_olfactory_pack,
            )
            from alife_lab.connectome.sysid import (
                HiddenMechanismSpec,
                ObservationSpec,
                SysIDTrainConfig,
                run_hidden_truth_experiment,
                run_observability_study,
            )

            if args.dataset_id == "synthetic_olfactory":
                pack = make_synthetic_olfactory_pack()
                sc_id = "full"
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
                sc_id = args.subcircuit
            sub_ref = extract_subcircuit(pack, subcircuit_id=sc_id, wiring="reference")
            mechs = {m.strip().lower() for m in args.mechanisms.split(",") if m.strip()}
            all_on = "all" in mechs
            m_spec = HiddenMechanismSpec(
                tau_enabled=all_on or ("tau" in mechs),
                sign_flip_fraction=args.sign_flip_fraction if (all_on or "sign" in mechs) else 0.0,
                neuromod_enabled=all_on or ("neuromod" in mechs),
                gap_junctions_enabled=all_on or ("gap" in mechs),
                gap_candidates=args.gap_candidates,
            )
            t_cfg = SysIDTrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                n_restarts=args.restarts,
                device=args.device,
            )
            if args.connectome_cmd == "sysid":
                obs_classes = tuple(c.strip() for c in args.obs_classes.split(",") if c.strip())
                sub_shuf = (
                    None
                    if args.no_shuffled
                    else extract_subcircuit(
                        pack,
                        subcircuit_id=sc_id,
                        wiring="shuffled_control",
                        shuffle_seed=args.seed,
                    )
                )
                report = run_hidden_truth_experiment(
                    sub_ref,
                    sub_shuffled=sub_shuf,
                    mech=m_spec,
                    obs=ObservationSpec(
                        obs_fraction=args.obs_fraction,
                        obs_classes=obs_classes,
                        bin_ms=args.bin_ms,
                        noise_sd=args.noise_sd,
                    ),
                    cfg=t_cfg,
                    seed=args.seed,
                )
                if args.json:
                    _print_json(report)
                else:
                    c = report["circuit"]
                    u = report["untrained"]
                    rb = report["reference_best"]
                    sh = report["shuffled_control"]
                    idn = report["identifiability"]
                    print(
                        f"sysid {report['dataset_id']} ({report['subcircuit_id']}) "
                        f"cells={c['num_cells']:,} obs={c['observed_cells']:,} "
                        f"edges={c['num_edges']:,} device={report['device']} "
                        f"wall={report['wall_seconds']:.2f}s vram={report['peak_vram_mb']:.1f}MiB"
                    )
                    print(
                        f"  untrained : heldout_all_r2={u['heldout_all_r2']} "
                        f"unobs_r2={u['heldout_unobs_r2']} sign_auroc={u['sign_flip_auroc']}"
                    )
                    print(
                        f"  reference : heldout_all_r2={rb['heldout_all_r2']} "
                        f"unobs_r2={rb['heldout_unobs_r2']} tau_r={rb['tau_pearson_r']} "
                        f"sign_auroc={rb['sign_flip_auroc']} nm_r={rb['neuromod_pearson_r']} "
                        f"gap_auroc={rb['gap_auroc']}"
                    )
                    if sh is not None:
                        print(
                            f"  shuffled  : heldout_all_r2={sh['heldout_all_r2']} "
                            f"unobs_r2={sh['heldout_unobs_r2']} tau_r={sh['tau_pearson_r']} "
                            f"sign_auroc={sh['sign_flip_auroc']}"
                        )
                    print(
                        f"  identifiability: {idn['verdict']} "
                        f"(restarts={idn['n_restarts']}, fit_spread={idn['fit_r2_spread']}, "
                        f"tau_sd={idn['tau_disagreement_ms']}ms, "
                        f"unobs_spread={idn['heldout_unobs_r2_spread']})"
                    )
                return 0

            fractions = tuple(float(x.strip()) for x in args.fractions.split(",") if x.strip())
            classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
            map_rep = run_observability_study(
                sub_ref,
                fractions=fractions,
                class_targets=classes,
                mech=m_spec,
                bin_ms=args.bin_ms,
                noise_sd=args.noise_sd,
                cfg=t_cfg,
                seed=args.seed,
            )
            if args.json:
                _print_json(map_rep)
            else:
                print(
                    f"sysid-map {map_rep['dataset_id']} ({map_rep['subcircuit_id']}) "
                    f"wall={map_rep['wall_seconds']:.2f}s"
                )
                print(
                    "  Fraction curve "
                    "(obs_fraction -> obs_cells | unobs_r2 | tau_r | sign_auroc | verdict):"
                )
                for r in map_rep["fraction_curve"]:
                    rb = r["reference_best"]
                    print(
                        f"    frac={r['obs_fraction']:.2f} ({r['observed_cells']:5d} cells): "
                        f"unobs_r2={rb['heldout_unobs_r2']} tau_r={rb['tau_pearson_r']} "
                        f"sign_auroc={rb['sign_flip_auroc']} [{r['identifiability']['verdict']}]"
                    )
                print("  Cell-class observability map (sorted by heldout_unobs_r2):")
                for r in map_rep["cell_class_map"]:
                    rb = r["reference_best"]
                    print(
                        f"    class={r['obs_class']:14s} ({r['observed_cells']:5d} cells): "
                        f"unobs_r2={rb['heldout_unobs_r2']} tau_r={rb['tau_pearson_r']} "
                        f"sign_auroc={rb['sign_flip_auroc']} [{r['identifiability']['verdict']}]"
                    )
            return 0

        if args.connectome_cmd == "task":
            from alife_lab.connectome.subcircuit import (
                extract_subcircuit,
                make_synthetic_olfactory_pack,
            )
            from alife_lab.connectome.tasks import F6TaskConfig, run_topology_benchmark

            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            if args.dataset_id == "synthetic_olfactory":
                pack = make_synthetic_olfactory_pack()
                sc_id = "full"
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
                sc_id = args.subcircuit
            sub_ref = extract_subcircuit(pack, subcircuit_id=sc_id, wiring="reference")
            tcfg = F6TaskConfig(
                task=args.task,
                scope=args.scope,
                n_train=args.n_train,
                n_test=args.n_test,
                epochs=args.epochs,
                lr=args.lr,
                max_readout=args.max_readout,
                device=args.device,
            )
            rep = run_topology_benchmark(
                sub_ref,
                lambda s: extract_subcircuit(
                    pack,
                    subcircuit_id=sc_id,
                    wiring="shuffled_control",
                    shuffle_seed=int(s),
                ),
                tcfg,
                seeds=seeds,
            )
            if args.json:
                _print_json(rep)
            else:
                ev = rep["evaluation"]
                rs = ev["reference_stats"]
                ss = ev["shuffled_stats"]
                cm = ev["comparison"]
                c = rep["circuit"]
                print(
                    f"task {rep['dataset_id']} ({rep['subcircuit_id']}) "
                    f"task={rep['task']} scope={rep['scope']} "
                    f"cells={c['num_cells']:,} readout={c['num_readout_cells']} "
                    f"params={c['num_learnable_params']:,} "
                    f"device={rep['device']} wall={rep['wall_seconds']:.2f}s "
                    f"vram={rep['peak_vram_mb']:.1f}MiB"
                )
                print(f"  framing   : {rep['framing_note']}")
                print(f"  plan_hash : {rep['preregistration']['plan_digest'][:16]}…")
                print(
                    f"  reference : test_acc={rs['mean']:.3f} +/- {rs['sd']:.3f} "
                    f"(median={rs['median']:.3f})"
                )
                print(
                    f"  shuffled  : test_acc={ss['mean']:.3f} +/- {ss['sd']:.3f} "
                    f"(median={ss['median']:.3f})"
                )
                print(
                    f"  verdict   : {ev['verdict']} "
                    f"(diff={cm['mean_difference']:+.3f}, "
                    f"cliffs_d={cm['cliffs_delta']:+.2f}, "
                    f"p={cm['permutation']['p_value']:.4f})"
                )
            return 0

        if args.connectome_cmd == "fit-real":
            from alife_lab.connectome.real_fit import (
                run_aimon2023_fit,
                run_door2_fit,
                run_synthetic_f7_check,
            )

            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            if args.source == "synthetic":
                rep = run_synthetic_f7_check(
                    seed=int(seeds[0]), epochs=args.epochs, device=args.device
                )
                if args.json:
                    _print_json(rep)
                else:
                    for arm in ("reference", "shuffled", "unconnected"):
                        r = rep[arm]
                        print(
                            f"  {arm:11s}: temporal_r2={r['temporal_holdout_r2']:+.4f} "
                            f"cross_fly_r2={r['cross_fly_holdout_r2']} "
                            f"spatial_unobs_r2={r['spatial_unobs_r2']:+.4f}"
                        )
                return 0

            pack = load_connectome_pack(args.dataset_id, ds_dir, verify_hashes=not args.no_verify)
            if args.source == "aimon2023":
                rep = run_aimon2023_fit(
                    pack,
                    ds_dir,
                    max_flies=args.max_flies,
                    epochs=args.epochs,
                    lr=args.lr,
                    seeds=seeds,
                    device=args.device,
                )
                if args.json:
                    _print_json(rep)
                else:
                    sm = rep["summary"]
                    cp = rep["comparisons"]
                    print(
                        f"fit-real aimon2023 flies={rep['n_flies']} "
                        f"regions={rep['matched_flywire_regions']}/{rep['n_regions']} "
                        f"synapses={rep['aggregated_flywire_synapses']:,} "
                        f"device={rep['device']} wall={rep['wall_seconds']:.2f}s"
                    )
                    print(
                        f"  temporal holdout R2 : "
                        f"ref={sm['reference_temporal_r2']['mean']:+.4f} "
                        f"shuf={sm['shuffled_temporal_r2']['mean']:+.4f} "
                        f"unconn={sm['unconnected_temporal_r2']['mean']:+.4f}"
                    )
                    print(
                        f"  cross-fly holdout R2: "
                        f"ref={sm['reference_cross_fly_r2']['mean']:+.4f} "
                        f"shuf={sm['shuffled_cross_fly_r2']['mean']:+.4f} "
                        f"unconn={sm['unconnected_cross_fly_r2']['mean']:+.4f}"
                    )
                    sp_cmp = cp["spatial_unobs_r2_ref_vs_shuffled"]
                    print(
                        f"  spatial unobs R2    : "
                        f"ref={sm['reference_spatial_unobs_r2']['mean']:+.4f} "
                        f"(r={sm['reference_spatial_unobs_r']['mean']:+.4f}) "
                        f"shuf={sm['shuffled_spatial_unobs_r2']['mean']:+.4f} "
                        f"unconn={sm['unconnected_spatial_unobs_r2']['mean']:+.4f} "
                        f"(ref vs shuf: d={sp_cmp['cliffs_delta']:+.2f}, "
                        f"p={sp_cmp['permutation']['p_value']:.4f})"
                    )
                return 0

            rep = run_door2_fit(
                pack,
                ds_dir,
                epochs=args.epochs,
                lr=args.lr,
                seeds=seeds,
                device=args.device,
            )
            if args.json:
                _print_json(rep)
            else:
                sm = rep["summary"]
                cm = rep["comparison"]
                print(
                    f"fit-real door2 odors={rep['n_odors']} "
                    f"glomeruli={rep['n_glomeruli']} "
                    f"device={rep['device']} wall={rep['wall_seconds']:.2f}s"
                )
                print(
                    f"  held-out odor R2: "
                    f"ref={sm['reference_heldout_r2']['mean']:+.4f} "
                    f"(r={sm['reference_heldout_r']['mean']:+.4f}) "
                    f"shuf={sm['shuffled_heldout_r2']['mean']:+.4f} "
                    f"(r={sm['shuffled_heldout_r']['mean']:+.4f}) "
                    f"d={cm['cliffs_delta']:+.2f} p={cm['permutation']['p_value']:.4f}"
                )
            return 0

        if args.connectome_cmd == "embody":
            from alife_lab.connectome.embodied import (
                make_synthetic_fly_embodied_pack,
                run_embodied_verification,
            )

            if args.dataset_id == "synthetic_fly_embodied":
                pack = make_synthetic_fly_embodied_pack()
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
            backend = args.backend
            if backend == "auto":
                from alife_lab.connectome.lif_cupy import available as cupy_ok

                backend = "cupy" if (pack.cells.num_cells > 5000 and cupy_ok()) else "numpy"
            rep = run_embodied_verification(
                pack,
                backend=backend,
                seed=args.seed,
                ticks_per_scene=args.ticks,
                datasets_dir=ds_dir,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"embody {rep['dataset_id']} cells={rep['num_cells']:,} "
                    f"backend={rep['backend']} wall={rep['wall_seconds']:.2f}s "
                    f"all_checks_passed={rep['all_checks_passed']}"
                )
                print(f"  framing: {rep['framing_note']}")
                for ck, ok in rep["checks"].items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                for sname, sc in rep["scenes"].items():
                    tm = sc["timing_ms"]
                    print(
                        f"  {sname:20s}: prob={sc['max_proboscis_cmd']:.2f} "
                        f"feed={sc['feeding_ms']:3d}ms groom={sc['grooming_ms']:3d}ms "
                        f"rtf={tm['realtime_factor']:.2f}x "
                        f"(brain={tm['brain_ms']:.3f} body={tm['body_ms']:.3f} "
                        f"xfer={tm['transfer_ms']:.3f} ms)"
                    )
            return 0 if rep["all_checks_passed"] else 1

        if args.connectome_cmd == "embody-l2":
            from alife_lab.connectome.embodied import (
                make_synthetic_cns_embodied_pack,
                run_embodied_level2_verification,
            )

            if args.dataset_id == "synthetic_cns_embodied":
                pack = make_synthetic_cns_embodied_pack()
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
            backend = args.backend
            if backend == "auto":
                from alife_lab.connectome.lif_cupy import available as cupy_ok

                backend = "cupy" if (pack.cells.num_cells > 5000 and cupy_ok()) else "numpy"
            rep = run_embodied_level2_verification(
                pack,
                backend=backend,
                seed=args.seed,
                ticks_per_scene=args.ticks,
                control_mode=args.mode,
                vnc_synapse_scale=args.synapse_scale,
                vnc_proprio_gain_hz=args.proprio_gain,
                datasets_dir=ds_dir,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"embody-l2 {rep['dataset_id']} mode={rep['control_mode']} "
                    f"cells={rep['num_cells']:,} backend={rep['backend']} "
                    f"wall={rep['wall_seconds']:.2f}s all_checks_passed={rep['all_checks_passed']}"
                )
                print(f"  framing: {rep['framing_note']}")
                for ck, ok in rep["checks"].items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                for sname, sc in rep["scenes"].items():
                    tm = sc["timing_ms"]
                    print(
                        f"  {sname:18s}: dist={sc['distance_cm']:.3f}cm "
                        f"mn={sc['vnc_mn_mean_hz']:5.1f}Hz (peak {sc['vnc_mn_peak_hz']:5.1f}Hz) "
                        f"alt={sc['vnc_tripod_alternation_mean']:.3f} "
                        f"steps={sc['tarsus_contact_transitions']:3d} "
                        f"rtf={tm['realtime_factor']:.2f}x "
                        f"(brain={tm['brain_ms']:.3f} body={tm['body_ms']:.3f} "
                        f"xfer={tm['transfer_ms']:.3f} ms)"
                    )
            return 0 if rep["all_checks_passed"] else 1

        if args.connectome_cmd == "tune-vnc":
            from alife_lab.connectome.embodied import (
                make_synthetic_cns_embodied_pack,
                tune_vnc_locomotion,
            )

            if args.dataset_id == "synthetic_cns_embodied":
                pack = make_synthetic_cns_embodied_pack()
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
            backend = args.backend
            if backend == "auto":
                from alife_lab.connectome.lif_cupy import available as cupy_ok

                backend = "cupy" if (pack.cells.num_cells > 5000 and cupy_ok()) else "numpy"
            rep = tune_vnc_locomotion(
                pack,
                backend=backend,
                seed=args.seed,
                ticks_per_eval=args.ticks,
                datasets_dir=ds_dir,
            )
            if args.json:
                _print_json(rep)
            else:
                b0 = rep["baseline_untuned"]
                bi = rep["best_intact"]
                bs = rep["shuffled_at_best"]
                print(
                    f"tune-vnc {rep['dataset_id']} backend={rep['backend']} "
                    f"wall={rep['wall_seconds']:.2f}s"
                )
                print(
                    f"  untuned : syn={b0['vnc_synapse_scale']}x dist={b0['distance_cm']:.3f}cm "
                    f"mn={b0['vnc_mn_mean_hz']:.1f}Hz alt={b0['vnc_tripod_alternation']:.3f}"
                )
                print(
                    f"  best    : syn={bi['vnc_synapse_scale']}x "
                    f"prop={bi['vnc_proprio_gain_hz']}Hz "
                    f"dist={bi['distance_cm']:.3f}cm mn={bi['vnc_mn_mean_hz']:.1f}Hz "
                    f"alt={bi['vnc_tripod_alternation']:.3f} rtf={bi['realtime_factor']:.2f}x"
                )
                print(
                    f"  shuffled: dist={bs['distance_cm']:.3f}cm mn={bs['vnc_mn_mean_hz']:.1f}Hz "
                    f"alt={bs['vnc_tripod_alternation']:.3f}"
                )
            return 0
    except (DatasetError, ValueError, FileNotFoundError) as exc:
        print(f"connectome error: {exc}", file=sys.stderr)
        return 1
    return 2


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

    p = sub.add_parser("sweep", help="run conditions x seeds headless and report statistics")
    p.add_argument("sweep", help="sweep YAML (configs/sweeps/*.yaml)")
    p.add_argument("--dry-run", action="store_true", help="validate and show the plan only")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("backup", help="independent backup / verification / restore")
    bsub = p.add_subparsers(dest="backup_cmd", required=True)
    bc = bsub.add_parser("create", help="full backup of the DB and all run artifacts")
    bc.add_argument("--dest", required=True, help="directory outside ALIFE_HOME and runs dir")
    bv = bsub.add_parser("verify", help="re-hash every file and check DB references")
    bv.add_argument("path")
    br = bsub.add_parser("restore", help="restore into empty directories (verifies first)")
    br.add_argument("path")
    br.add_argument("--home", required=True)
    br.add_argument("--runs-dir", required=True)
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("compare", help="state divergence of runs at common checkpoint ticks")
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("connectome", help="manage and inspect biological connectome species packs")
    csub = p.add_subparsers(dest="connectome_cmd", required=True)
    cl = csub.add_parser("list", help="list registered connectome species packs")
    cl.add_argument("--json", action="store_true")
    cf = csub.add_parser("fetch", help="download and SHA-256 verify a connectome dataset")
    cf.add_argument("dataset_id")
    cv = csub.add_parser("verify", help="re-hash local connectome dataset files")
    cv.add_argument("dataset_id")
    cs = csub.add_parser("summary", help="load a species pack and print summary/reference checks")
    cs.add_argument("dataset_id")
    cs.add_argument("--json", action="store_true")
    cs.add_argument("--no-verify", action="store_true", help="skip file SHA-256 verification")
    csc = csub.add_parser(
        "screen", help="run systematic single-cell-type silencing screen (CPU or batched GPU)"
    )
    csc.add_argument("dataset_id")
    csc.add_argument(
        "--stim",
        action="append",
        default=[],
        required=True,
        help="stimulus PORT:RATE_HZ (repeatable, e.g. --stim sugar_grn:150)",
    )
    csc.add_argument(
        "--target", default="mn9_proboscis_extension", help="target IO port to measure"
    )
    csc.add_argument(
        "--cell-types", default="", help="comma-separated cell_types (omit for auto top-k active)"
    )
    csc.add_argument("--top-k", type=int, default=24, help="number of top active types to screen")
    csc.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated seeds")
    csc.add_argument("--ticks", type=int, default=1000, help="simulation ticks (1 tick = 1 ms)")
    csc.add_argument("--wiring", choices=["reference", "shuffled_control"], default="reference")
    csc.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    csc.add_argument("--batch-size", type=int, default=8, help="GPU batch size (default 8)")
    csc.add_argument("--no-verify", action="store_true", help="skip file SHA-256 verification")
    csc.add_argument("--json", action="store_true")
    cpav = csub.add_parser(
        "pavlov",
        help="run F4 mushroom body Pavlovian olfactory conditioning protocol and controls",
    )
    cpav.add_argument("dataset_id")
    cpav.add_argument("--seeds", default="1,2,3,4,5")
    cpav.add_argument("--odor-hz", type=float, default=150.0)
    cpav.add_argument("--dan-hz", type=float, default=220.0)
    cpav.add_argument("--probe-ms", type=int, default=300)
    cpav.add_argument("--trial-ms", type=int, default=200)
    cpav.add_argument("--trials", type=int, default=4)
    cpav.add_argument("--ext-ms", type=int, default=1200)
    cpav.add_argument("--unpaired-gap-ms", type=int, default=1500)
    cpav.add_argument("--tau-elig", type=float, default=200.0)
    cpav.add_argument("--tau-dan", type=float, default=150.0)
    cpav.add_argument("--dan-threshold", type=float, default=18.0)
    cpav.add_argument("--lr-ltd", type=float, default=0.00022)
    cpav.add_argument("--lr-ext", type=float, default=0.03)
    cpav.add_argument("--wiring", choices=["reference", "shuffled_control"], default="reference")
    cpav.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    cpav.add_argument("--no-verify", action="store_true")
    cpav.add_argument("--json", action="store_true")

    csid = csub.add_parser(
        "sysid",
        help="run F5 hidden-truth connectome system identification and identifiability check",
    )
    csid.add_argument("dataset_id")
    csid.add_argument(
        "--subcircuit", choices=["olfactory_mb_lh", "full"], default="olfactory_mb_lh"
    )
    csid.add_argument(
        "--mechanisms", default="all", help="all or comma list: tau,sign,neuromod,gap"
    )
    csid.add_argument("--sign-flip-fraction", type=float, default=0.15)
    csid.add_argument("--gap-candidates", type=int, default=128)
    csid.add_argument("--obs-fraction", type=float, default=0.25)
    csid.add_argument("--obs-classes", default="", help="comma list of cell classes to record")
    csid.add_argument("--bin-ms", type=float, default=10.0)
    csid.add_argument("--noise-sd", type=float, default=0.015)
    csid.add_argument("--epochs", type=int, default=100)
    csid.add_argument("--lr", type=float, default=0.06)
    csid.add_argument("--restarts", type=int, default=3)
    csid.add_argument("--seed", type=int, default=1)
    csid.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    csid.add_argument("--no-shuffled", action="store_true")
    csid.add_argument("--no-verify", action="store_true")
    csid.add_argument("--json", action="store_true")

    cmap = csub.add_parser(
        "sysid-map",
        help="run F5 observability curve (fraction sweep) and cell-class observability map",
    )
    cmap.add_argument("dataset_id")
    cmap.add_argument(
        "--subcircuit", choices=["olfactory_mb_lh", "full"], default="olfactory_mb_lh"
    )
    cmap.add_argument("--mechanisms", default="all")
    cmap.add_argument("--sign-flip-fraction", type=float, default=0.15)
    cmap.add_argument("--gap-candidates", type=int, default=128)
    cmap.add_argument("--fractions", default="0.05,0.20,0.50,1.0")
    cmap.add_argument("--classes", default="ALPN,ALLN,Kenyon_Cell,MBON,LHLN")
    cmap.add_argument("--bin-ms", type=float, default=10.0)
    cmap.add_argument("--noise-sd", type=float, default=0.015)
    cmap.add_argument("--epochs", type=int, default=80)
    cmap.add_argument("--lr", type=float, default=0.06)
    cmap.add_argument("--restarts", type=int, default=2)
    cmap.add_argument("--seed", type=int, default=1)
    cmap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cmap.add_argument("--no-verify", action="store_true")
    cmap.add_argument("--json", action="store_true")

    ctsk = csub.add_parser(
        "task",
        help="run F6 fixed-topology task learning (biological wiring vs shuffled wiring)",
    )
    ctsk.add_argument("dataset_id")
    ctsk.add_argument(
        "--subcircuit", choices=["olfactory_mb_lh", "full"], default="olfactory_mb_lh"
    )
    ctsk.add_argument(
        "--task",
        choices=["odor_classification", "short_memory", "nonlinear_xor"],
        default="odor_classification",
    )
    ctsk.add_argument(
        "--scope",
        choices=["readout_only", "io_only", "partial_weights"],
        default="readout_only",
    )
    ctsk.add_argument("--seeds", default="1,2,3,4,5")
    ctsk.add_argument("--n-train", type=int, default=64)
    ctsk.add_argument("--n-test", type=int, default=64)
    ctsk.add_argument("--epochs", type=int, default=80)
    ctsk.add_argument("--lr", type=float, default=0.08)
    ctsk.add_argument("--max-readout", type=int, default=256)
    ctsk.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ctsk.add_argument("--no-verify", action="store_true")
    ctsk.add_argument("--json", action="store_true")

    cfit = csub.add_parser(
        "fit-real",
        help="run F7 connectome fitting on real public Drosophila activity (Aimon 2023 / DoOR 2.0)",
    )
    cfit.add_argument("source", choices=["aimon2023", "door2", "synthetic"])
    cfit.add_argument("--dataset-id", default="flywire_783")
    cfit.add_argument("--max-flies", type=int, default=10)
    cfit.add_argument("--epochs", type=int, default=120)
    cfit.add_argument("--lr", type=float, default=0.04)
    cfit.add_argument("--seeds", default="1,2,3,4,5")
    cfit.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cfit.add_argument("--no-verify", action="store_true")
    cfit.add_argument("--json", action="store_true")

    cemb = csub.add_parser(
        "embody",
        help="run F8 Level 1 closed-loop embodied fly verification & real-time latency benchmark",
    )
    cemb.add_argument("dataset_id", choices=["flywire_783", "synthetic_fly_embodied"])
    cemb.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    cemb.add_argument("--seed", type=int, default=1)
    cemb.add_argument("--ticks", type=int, default=250)
    cemb.add_argument("--no-verify", action="store_true")
    cemb.add_argument("--json", action="store_true")

    cemb2 = csub.add_parser(
        "embody-l2",
        help="run F8 Level 2 VNC direct-drive verification & negative controls (no CPG)",
    )
    cemb2.add_argument("dataset_id", choices=["banc_888", "male_cns_v1", "synthetic_cns_embodied"])
    cemb2.add_argument("--mode", choices=["level2_vnc", "level2_vnc_tuned"], default="level2_vnc")
    cemb2.add_argument("--synapse-scale", type=float, default=1.0)
    cemb2.add_argument("--proprio-gain", type=float, default=95.0)
    cemb2.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    cemb2.add_argument("--seed", type=int, default=1)
    cemb2.add_argument("--ticks", type=int, default=400)
    cemb2.add_argument("--no-verify", action="store_true")
    cemb2.add_argument("--json", action="store_true")

    ctune = csub.add_parser(
        "tune-vnc",
        help="fixed-topology VNC synaptic/readout gain search vs shuffled-VNC control (F8 L2)",
    )
    ctune.add_argument("dataset_id", choices=["banc_888", "male_cns_v1", "synthetic_cns_embodied"])
    ctune.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    ctune.add_argument("--seed", type=int, default=1)
    ctune.add_argument("--ticks", type=int, default=500)
    ctune.add_argument("--no-verify", action="store_true")
    ctune.add_argument("--json", action="store_true")

    p.set_defaults(func=cmd_connectome)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
