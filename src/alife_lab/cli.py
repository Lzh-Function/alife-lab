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

        if args.connectome_cmd == "learn-vnc":
            from alife_lab.connectome.embodied import make_synthetic_cns_embodied_pack
            from alife_lab.connectome.vnc_learn import learn_vnc_locomotion

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
            save_path = Path(args.save_profile) if args.save_profile else None
            rep = learn_vnc_locomotion(
                pack,
                backend=backend,
                seed=args.seed,
                generations=args.generations,
                pop_size=args.pop,
                ticks_per_eval=args.ticks,
                verify_ticks=args.verify_ticks,
                include_shuffled_cmaes=not args.no_shuffled_cmaes,
                datasets_dir=ds_dir,
                save_profile_path=save_path,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"learn-vnc {rep['dataset_id']} full={rep['full_cells']:,} "
                    f"sub={rep['vnc_subcircuit_cells']:,} backend={rep['backend']} "
                    f"gens={rep['generations']}x{rep['pop_size']} wall={rep['wall_seconds']:.1f}s"
                )
                for k, v in rep["full_cns_verification"].items():
                    if v is None:
                        continue
                    print(
                        f"  {k:32s}: net_x={v['net_x_cm']:+.3f}cm "
                        f"disp={v['net_disp_cm']:.3f}cm path={v['path_cm']:.3f}cm "
                        f"mn={v['vnc_mn_mean_hz']:5.1f}Hz osc={v['tripod_osc_std_hz']:4.1f} "
                        f"corr={v['tripod_corr']:+.2f} steps={v['tarsus_transitions']:3d}"
                    )
            return 0

        if args.connectome_cmd == "embody-batch":
            from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
            from alife_lab.connectome.embodied_batch import (
                benchmark_batched_embodied,
                run_batched_embodied_screen,
                run_batched_embodied_search,
            )

            if args.dataset_id == "synthetic_fly_embodied":
                pack = make_synthetic_fly_embodied_pack()
            else:
                pack = load_connectome_pack(
                    args.dataset_id, ds_dir, verify_hashes=not args.no_verify
                )
            if args.mode == "benchmark":
                sizes = tuple(int(s.strip()) for s in args.batch_sizes.split(",") if s.strip())
                rep = benchmark_batched_embodied(
                    pack,
                    batch_sizes=sizes,
                    ticks=args.ticks,
                    seed=args.seed,
                    physics_backend=args.physics,
                    brain_backend=args.brain,
                )
                if args.json:
                    _print_json(rep)
                else:
                    print(
                        f"embody-batch benchmark {rep['dataset_id']} "
                        f"physics={rep['physics_backend']} brain={rep['brain_backend']}"
                    )
                    for row in rep["scaling"]:
                        print(
                            f"  B={row['batch_size']:2d}: "
                            f"{row['fly_seconds_per_wall_second']:6.2f} fly-s/s "
                            f"({row['ms_per_batched_tick']:.2f} ms/tick: "
                            f"brain={row['brain_ms_per_tick']:.2f} "
                            f"body={row['body_ms_per_tick']:.2f}) "
                            f"vram={row['brain_gpu_mib']:.1f}MiB feed={row['mean_feeding_ms']}ms"
                        )
                    ck = rep["cpu_vs_warp_check"]
                    print(
                        f"  cpu vs warp: feed={ck['cpu_feeding_ms']} vs {ck['warp_feeding_ms']}ms "
                        f"(diff={ck['feeding_ms_diff']}ms) dist_diff={ck['distance_cm_diff']}cm"
                    )
                return 0

            if args.mode == "search":
                rep = run_batched_embodied_search(
                    pack,
                    generations=args.generations,
                    pop_size=args.batch_size,
                    ticks=args.ticks,
                    seed=args.seed,
                    physics_backend=args.physics,
                    brain_backend=args.brain,
                )
                if args.json:
                    _print_json(rep)
                else:
                    print(
                        f"embody-batch search {rep['dataset_id']} "
                        f"physics={rep['physics_backend']} brain={rep['brain_backend']} "
                        f"{rep['fly_seconds_per_wall_second']:.2f} fly-s/s "
                        f"best_score={rep['best_score']} params={rep['best_params']}"
                    )
                return 0

            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            cts = [c.strip() for c in args.cell_types.split(",") if c.strip()]
            if not cts:
                cts = (
                    ["IN_sugar", "CB0701", "IN_groom"]
                    if args.dataset_id == "synthetic_fly_embodied"
                    else ["CB0701", "DNg84", "DNp09", "G2N_1"]
                )
            rep = run_batched_embodied_screen(
                pack,
                cell_types=cts,
                seeds=seeds,
                ticks=args.ticks,
                scene=args.scene,
                batch_size=args.batch_size,
                physics_backend=args.physics,
                brain_backend=args.brain,
                verify_top_cpu=not args.no_cpu_verify,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"embody-batch screen {rep['dataset_id']} scene={rep['scene']} "
                    f"physics={rep['physics_backend']} brain={rep['brain_backend']} "
                    f"evals={rep['total_evaluations']} wall={rep['wall_seconds']:.2f}s "
                    f"({rep['fly_seconds_per_wall_second']:.2f} fly-s/s)"
                )
                print(
                    f"  intact={rep['baseline']['mean_ms']:.1f}ms "
                    f"shuffled={rep['shuffled_control']['mean_ms']:.1f}ms"
                )
                for r in rep["results"]:
                    print(
                        f"  {r['cell_type']:12s} ({r['cell_count']:3d} cells): "
                        f"{r['mean_ms']:5.1f}ms (supp={r['suppression_pct']:+.1f}% "
                        f"p_holm={r.get('p_value_holm')})"
                    )
                if rep["cpu_confirmation"]:
                    print(f"  cpu_confirm: {rep['cpu_confirmation']}")
            return 0

        if args.connectome_cmd == "vision":
            from alife_lab.connectome.vision_flyvis import (
                flyvis_available,
                run_flyvis_visual_assay,
                run_synthetic_visual_assay,
            )

            if args.synthetic or not flyvis_available(ds_dir):
                rep = run_synthetic_visual_assay(n_frames=args.frames)
            else:
                rep = run_flyvis_visual_assay(ds_dir, model_id=args.model_id, n_frames=args.frames)
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"vision {rep['backend']} cells={rep['num_cells']:,} "
                    f"types={rep['unique_cell_types']} wall={rep['wall_seconds']:.2f}s "
                    f"all_checks_passed={rep['all_checks_passed']}"
                )
                for ck, ok in rep["checks"].items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                for sk, sv in rep["stimuli"].items():
                    print(f"  {sk:15s}: {sv}")
            return 0 if rep["all_checks_passed"] else 1

        if args.connectome_cmd == "compare-sex":
            from alife_lab.connectome.embodied import (
                make_synthetic_cns_embodied_pack,
                make_synthetic_fly_embodied_pack,
            )
            from alife_lab.connectome.sex_compare import compare_sex_connectomes

            ids = [s.strip() for s in args.datasets.split(",") if s.strip()]
            packs = {}
            for did in ids:
                if did == "synthetic_fly_embodied":
                    packs[did] = make_synthetic_fly_embodied_pack()
                elif did == "synthetic_cns_embodied":
                    packs[did] = make_synthetic_cns_embodied_pack()
                else:
                    packs[did] = load_connectome_pack(did, ds_dir, verify_hashes=not args.no_verify)
            rep = compare_sex_connectomes(
                packs, backend=args.backend, seed=args.seed, ticks=args.ticks
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"compare-sex datasets={rep['datasets']} "
                    f"shared_all={rep['shared_cell_types_all']:,} "
                    f"wall={rep['wall_seconds']:.2f}s"
                )
                print(f"  pairwise_shared: {rep['pairwise_shared_cell_types']}")
                for dname in rep["datasets"]:
                    an = rep["anatomy"][dname]
                    ph = rep["physiology"][dname]
                    ref = ph["reference"]
                    print(
                        f"  {dname:12s} ({an['num_cells']:>7,} cells, "
                        f"types={an['unique_cell_types']:>5,}): "
                        f"sugar_mn9={ref['sugar_150hz']['mn9_hz']:5.1f}Hz "
                        f"(bitter_supp={ph['bitter_suppression_pct']:5.1f}%) "
                        f"groom_dng84={ref['antennal_poke_180hz']['dng84_hz']:5.1f}Hz "
                        f"fwd_vnc_mn={ref['forward_dn_150hz']['vnc_leg_mn_hz']:5.2f}Hz"
                    )
            return 0

        if args.connectome_cmd == "social":
            from alife_lab.connectome.dyadic_fly import (
                make_synthetic_female_auditory_pack,
                make_synthetic_male_song_pack,
                run_f11_song_dialogue,
                run_f11_two_fly_arena,
            )

            m_pack = (
                make_synthetic_male_song_pack()
                if args.male == "synthetic_male_song"
                else load_connectome_pack(args.male, ds_dir, verify_hashes=not args.no_verify)
            )
            f_pack = (
                make_synthetic_female_auditory_pack()
                if args.female == "synthetic_female_auditory"
                else load_connectome_pack(args.female, ds_dir, verify_hashes=not args.no_verify)
            )
            if args.stage == "dialogue":
                rep = run_f11_song_dialogue(
                    m_pack,
                    f_pack,
                    backend=args.backend,
                    seed=args.seed,
                    ticks=args.ticks,
                    run_janelia_ablations=not args.no_ablations,
                )
            else:
                rep = run_f11_two_fly_arena(
                    m_pack,
                    f_pack,
                    backend=args.backend,
                    seed=args.seed,
                    ticks=args.ticks,
                )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"social {rep['stage']} male={rep['male_dataset']} "
                    f"female={rep['female_dataset']} total_cells={rep['total_cells']:,} "
                    f"backend={rep['backend']} wall={rep['wall_seconds']:.2f}s "
                    f"all_checks_passed={rep['all_checks_passed']}"
                )
                for ck, ok in rep["checks"].items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                for sn, sv in rep["scenes"].items():
                    print(f"  {sn:18s}: {sv}")
                if rep.get("janelia_song_ablations"):
                    print("  Janelia song-circuit silencing screen:")
                    for ab in rep["janelia_song_ablations"]:
                        print(
                            f"    -{ab['silenced_cell_type']:8s} ({ab['cell_count']:2d} cells): "
                            f"pulse={ab['male_pulse_ps1_hz']:5.1f}Hz "
                            f"({ab['pulse_suppression_pct']:+5.1f}%) "
                            f"sine={ab['male_sine_hg1_hz']:5.1f}Hz "
                            f"({ab['sine_suppression_pct']:+5.1f}%) "
                            f"female_aud={ab['female_auditory_relay_hz']:5.1f}Hz "
                            f"({ab['female_aud_suppression_pct']:+5.1f}%)"
                        )
            return 0 if rep["all_checks_passed"] else 1
        if args.connectome_cmd == "swarm":
            from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
            from alife_lab.connectome.fly_ecosystem import (
                make_synthetic_ecosystem_female_pack,
                make_synthetic_ecosystem_male_pack,
                run_dual_sex_ecosystem_assay,
            )
            from alife_lab.connectome.fly_swarm import (
                make_synthetic_male_aggression_pack,
                run_male_aggression_assay,
                run_ramdya_collective_assay,
            )

            def _load_swm(ds_id: str) -> Any:
                if ds_id == "synthetic_fly_embodied":
                    return make_synthetic_fly_embodied_pack()
                if ds_id == "synthetic_male_aggression":
                    return make_synthetic_male_aggression_pack()
                if ds_id == "synthetic_ecosystem_male":
                    return make_synthetic_ecosystem_male_pack()
                if ds_id == "synthetic_ecosystem_female":
                    return make_synthetic_ecosystem_female_pack()
                return load_connectome_pack(ds_id, ds_dir, verify_hashes=not args.no_verify)

            sizes = tuple(int(s.strip()) for s in args.sizes.split(",") if s.strip())
            if args.mode == "ecosystem":
                m_id = (
                    "synthetic_ecosystem_male"
                    if args.dataset_id == "synthetic_fly_embodied"
                    else args.dataset_id
                )
                f_id = args.female or (
                    "synthetic_ecosystem_female"
                    if m_id == "synthetic_ecosystem_male"
                    else "banc_888"
                )
                rep = run_dual_sex_ecosystem_assay(
                    _load_swm(m_id),
                    _load_swm(f_id),
                    num_flies=sizes[-1],
                    ticks=args.ticks,
                    seed=args.seed,
                    backend=args.backend,
                )
                if args.json:
                    _print_json(rep)
                else:
                    print(
                        f"ecosystem male={rep['male_dataset']} female={rep['female_dataset']} "
                        f"N={rep['num_flies']} total_neurons={rep['total_neurons']:,} "
                        f"ticks={rep['ticks']} wall={rep['wall_seconds']:.2f}s "
                        f"all_checks_passed={rep['all_checks_passed']}"
                    )
                    for ck, ok in rep["checks"].items():
                        print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                    for cname, row in rep["conditions"].items():
                        print(
                            f"    {cname:18s}: feed={row['feeding_fly_ms']:5d} "
                            f"song={row['song_fly_ms']:5d} lunges={row['total_lunges']:3d} "
                            f"threat={row['wing_threat_fly_ms']:5d} "
                            f"clean={row['mean_clean_fraction'] * 100:5.1f}% "
                            f"DA1={row['mean_da1_pn_hz']:5.1f}Hz "
                            f"hg34={row['mean_hg34_hz']:4.1f}Hz "
                            f"ps1={row['mean_male_ps1_hz']:4.1f}Hz "
                            f"fAud={row['mean_female_aud_hz']:4.1f}Hz "
                            f"RTF={row['realtime_factor']:.2f}x"
                        )
                return 0 if rep["all_checks_passed"] else 1
            pack = _load_swm(args.dataset_id)
            if args.mode == "aggression":
                rep = run_male_aggression_assay(
                    pack,
                    num_flies=sizes[-1],
                    ticks=args.ticks,
                    seed=args.seed,
                    backend=args.backend,
                )
                if args.json:
                    _print_json(rep)
                else:
                    print(
                        f"aggression dataset={rep['dataset_id']} N={rep['num_flies']} "
                        f"concurrent_neurons={rep['concurrent_neurons']:,} "
                        f"ticks={rep['ticks']} wall={rep['wall_seconds']:.2f}s "
                        f"all_checks_passed={rep['all_checks_passed']}"
                    )
                    for ck, ok in rep["checks"].items():
                        print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                    for cname, row in rep["conditions"].items():
                        print(
                            f"    {cname:18s}: lunges={row['total_lunges']:3d} "
                            f"threat={row['wing_threat_fly_ms']:5d} fly·ms "
                            f"DA1_PN={row['mean_da1_pn_hz']:5.1f}Hz "
                            f"DNg33={row['mean_dng33_hz']:5.1f}Hz "
                            f"hg3/4={row['mean_hg34_hz']:5.1f}Hz "
                            f"dom_spread={row['dominance_spread']:4.1f} "
                            f"RTF={row['realtime_factor']:.2f}x"
                        )
                return 0 if rep["all_checks_passed"] else 1
            rep = run_ramdya_collective_assay(
                pack,
                group_sizes=sizes,
                ticks=args.ticks,
                seed=args.seed,
                backend=args.backend,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"swarm dataset={rep['dataset_id']} cells/fly={rep['cells_per_fly']:,} "
                    f"max_concurrent_neurons={rep['max_concurrent_neurons']:,} "
                    f"ticks={rep['ticks']} wall={rep['wall_seconds']:.2f}s "
                    f"all_checks_passed={rep['all_checks_passed']}"
                )
                for ck, ok in rep["checks"].items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                print("  Ramdya (2015) group-size avoidance curve:")
                for row in rep["group_size_curve"]:
                    print(
                        f"    N={row['num_flies']:2d} (trials={row['trials']}): "
                        f"clean={row['final_clean_fraction'] * 100:5.1f}% "
                        f"(mean={row['mean_clean_fraction'] * 100:5.1f}%) "
                        f"encounters={row['encounters_per_fly_s']:5.2f}/fly·s "
                        f"RTF={row['realtime_factor']:.2f}x"
                    )
                print("  Negative controls (at max N):")
                for cname, row in rep["controls"].items():
                    print(
                        f"    {cname:18s}: clean={row['final_clean_fraction'] * 100:5.1f}% "
                        f"(mean={row['mean_clean_fraction'] * 100:5.1f}%) "
                        f"encounters={row['encounters_per_fly_s']:5.2f}/fly·s"
                    )
            return 0 if rep["all_checks_passed"] else 1
        if args.connectome_cmd == "evolve":
            from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
            from alife_lab.connectome.fly_world import run_f12_signal_emergence_assay

            pack = (
                make_synthetic_fly_embodied_pack()
                if args.dataset_id == "synthetic_fly_embodied"
                else load_connectome_pack(args.dataset_id, ds_dir, verify_hashes=not args.no_verify)
            )
            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            rep = run_f12_signal_emergence_assay(
                pack,
                seeds=seeds,
                evo_ticks=args.evo_ticks,
                eval_ticks=args.eval_ticks,
                backend=args.backend,
                replay_fine=args.replay_fine,
            )
            if args.json:
                _print_json(rep)
            else:
                fid = rep["fidelity"]
                sm = rep["summary"]
                print(
                    f"F12 fly_world dataset={rep['dataset_id']} "
                    f"cells={fid['source_cells']:,}->{fid['compressed_nodes']:,} "
                    f"({fid['compression_ratio']}x) wall={rep['wall_seconds']:.2f}s "
                    f"all_checks_passed={rep['all_checks_passed']}"
                )
                print(
                    f"  Fidelity: base={fid['baseline_mn9_hz']}Hz "
                    f"sugar={fid['sugar_mn9_hz']}Hz bitter={fid['sugar_bitter_mn9_hz']}Hz "
                    f"poke_DNg84={fid['antennal_poke_dng84_hz']}Hz"
                )
                for ck, ok in rep["checks"].items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {ck}")
                print(
                    f"  Summary: births={sm['mean_births']} fly_dist={sm['mean_fly_distance']} "
                    f"MI={sm['best_mi_bits']:.4f}b (shuf={sm['best_shuffled_mi_bits']:.4f}b) "
                    f"recv={sm['best_receiver_response']} "
                    f"fork_food={sm['mean_fork_intact_food']} vs muted={sm['mean_fork_muted_food']}"
                )
                fr = rep.get("fine_replay")
                if fr is not None:
                    d = fr["descendant"]
                    cr = fr["compressed_rates_hz"]
                    sr = fr["full_spiking_rates_hz"]
                    print(
                        f"  F12d Full-Brain Replay (fly #{d['fly_id']} Gen {d['generation']} "
                        f"dist={d['fly_distance']} · {fr['full_cells']:,} spiking cells, "
                        f"backend={fr['backend']}, wall={fr['wall_seconds']}s, "
                        f"agreement={fr['qualitative_agreement_count']}/4):"
                    )
                    print(
                        f"    compressed ({fr['compressed_nodes']:,} nodes): "
                        f"base={cr['baseline_mn9']}Hz sugar={cr['sugar_mn9']}Hz "
                        f"bitter={cr['sugar_bitter_mn9']}Hz "
                        f"poke_DNg84={cr['antennal_poke_dng84']}Hz"
                    )
                    print(
                        f"    full LIF   ({fr['full_cells']:,} cells): "
                        f"base={sr['baseline_mn9']}Hz sugar={sr['sugar_mn9']}Hz "
                        f"bitter={sr['sugar_bitter_mn9']}Hz "
                        f"poke_DNg84={sr['antennal_poke_dng84']}Hz"
                    )
            return 0 if rep["all_checks_passed"] else 1
        if args.connectome_cmd == "fit-worm":
            from alife_lab.connectome.worm_activity import (
                run_real_worm_activity_fit,
                run_worm_activity_benchmark,
            )

            seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
            if args.source == "synthetic":
                rep = run_worm_activity_benchmark(seed=int(seeds[0]))
                if args.json:
                    _print_json(rep)
                else:
                    print(
                        f"fit-worm synthetic cells={rep['evaluated_neurons']} steps={rep['steps']}"
                    )
                    for arm, r in rep["arms"].items():
                        print(
                            f"  {arm:18s}: heldout_r2={r['heldout_r2']:+.4f} "
                            f"r={r['heldout_pearson_r']:+.4f} rmse={r['rmse']:.4f}"
                        )
                return 0

            pack = load_connectome_pack(args.dataset_id, ds_dir, verify_hashes=not args.no_verify)
            rep = run_real_worm_activity_fit(
                pack,
                ds_dir,
                max_worms=args.max_worms,
                min_matched_neurons=args.min_matched,
                max_frames=args.max_frames,
                source_filter=args.source_filter or None,
                horizon=args.horizon,
                epochs=args.epochs,
                lr=args.lr,
                seeds=seeds,
                device=args.device,
                verify_hash=not args.no_verify,
            )
            if args.json:
                _print_json(rep)
            else:
                sm = rep["summary"]
                cp = rep["comparisons"]
                hz = rep["rollout_horizon_steps"]
                print(
                    f"fit-worm {rep['dataset']} connectome={rep['connectome_dataset']} "
                    f"worms={rep['n_worms']} mean_cells={rep['mean_matched_neurons']} "
                    f"horizon={hz} sources={rep['source_datasets']} device={rep['device']} "
                    f"vram={rep['peak_vram_mb']:.1f}MiB wall={rep['wall_seconds']:.2f}s"
                )
                for arm in (
                    "reference",
                    "no_gap_junctions",
                    "raw_em_gaba_only",
                    "shuffled_control",
                    "unconnected",
                ):
                    row = sm[arm]
                    print(
                        f"  {arm:18s}: spatial_r2={row['spatial_unobs_r2']['mean']:+.4f} "
                        f"(r={row['spatial_unobs_r']['mean']:+.4f}) | "
                        f"rollout{hz}_r2={row['temporal_rollout_r2']['mean']:+.4f} "
                        f"(r={row['temporal_rollout_r']['mean']:+.4f})"
                    )
                for metric in ("spatial_unobs_r2", "temporal_rollout_r2"):
                    sh = cp[metric]["reference_vs_shuffled_control"]
                    ng = cp[metric]["reference_vs_no_gap_junctions"]
                    rw = cp[metric]["reference_vs_raw_em_gaba_only"]
                    print(
                        f"  {metric:19s}: vs_shuf d={sh['cliffs_delta']:+.2f} "
                        f"(p={sh['permutation']['p_value']:.4f}) | "
                        f"vs_nogap d={ng['cliffs_delta']:+.2f} "
                        f"(p={ng['permutation']['p_value']:.4f}) | "
                        f"vs_rawGABA d={rw['cliffs_delta']:+.2f} "
                        f"(p={rw['permutation']['p_value']:.4f})"
                    )
            return 0

        if args.connectome_cmd == "compare-species":
            from alife_lab.connectome.cross_species import compare_species_connectomes

            dids = [d.strip() for d in args.datasets.split(",") if d.strip()]
            packs = {
                d: load_connectome_pack(
                    d,
                    ds_dir,
                    verify_hashes=(not args.no_verify and not d.startswith("synthetic_")),
                )
                for d in dids
            }
            rep = compare_species_connectomes(
                packs,
                seed=args.seed,
                ticks=args.ticks,
                run_physiology=not args.no_physiology,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"compare-species ({len(dids)} species, seed={args.seed}, "
                    f"ticks={args.ticks}, wall={rep['wall_seconds']:.2f}s)"
                )
                phys_map = {p["dataset_id"]: p for p in rep["sensorimotor_probe"]}
                for t in rep["topology"]:
                    did = t["dataset_id"]
                    p = phys_map.get(did)
                    p_txt = (
                        f" | {p['dynamics_mode']} ref={p['reference_motor_response']} "
                        f"vs shuf={p['shuffled_motor_response']} {p['unit']}"
                        if p
                        else ""
                    )
                    print(
                        f"  {did:24s}: N={t['num_cells']:6d} E={t['num_edges']:8d} "
                        f"dens={t['density']:.5f} recip={t['reciprocity']:.3f} "
                        f"top1={t['top1pct_hub_synapse_share']:.3f}{p_txt}"
                    )
            return 0

        if args.connectome_cmd == "chimera":
            from alife_lab.connectome.chimera import (
                run_f13a_intraspecies_hybrid_assay,
                run_f13b_brain_transplant_experiment,
                run_f13c_mb_graft_assay,
            )

            def _load(did: str) -> Any:
                return load_connectome_pack(
                    did,
                    ds_dir,
                    verify_hashes=(not args.no_verify and not did.startswith("synthetic_")),
                )

            if args.kind == "f13a":
                pa = _load(args.host)
                pb = _load(args.donor)
                rep = run_f13a_intraspecies_hybrid_assay(
                    pa, pb, blend_ratio=args.blend, seed=args.seed, ticks=args.ticks
                )
            elif args.kind == "f13b":
                donor = _load(args.donor)
                rep = run_f13b_brain_transplant_experiment(
                    donor, seed=args.seed, bridge_trials=args.trials, eval_steps=args.ticks
                )
            else:
                host = _load(args.host)
                donor = _load(args.donor)
                rep = run_f13c_mb_graft_assay(host, donor, seed=args.seed, train_trials=args.trials)

            if args.json:
                _print_json(rep)
            else:
                print(
                    f"chimera {rep['chimera_kind']} provenance={rep['provenance_breakdown']} "
                    f"wall={rep['wall_seconds']:.2f}s"
                )
                if args.kind == "f13a":
                    hp = rep["hybrid_probe"]
                    print(
                        f"  shared_types={rep['shared_cell_types']} "
                        f"crossed={rep['crossed_cell_types']} "
                        f"ref={hp['reference_motor_response']} "
                        f"shuf={hp['shuffled_motor_response']} {hp['unit']}"
                    )
                elif args.kind == "f13b":
                    for arm, a in rep["arms"].items():
                        print(
                            f"  {arm:16s}: dist={a['distance_mm']:.3f}mm "
                            f"approach={a['target_approach_mm']:+.3f}mm "
                            f"drive={a['mean_motor_drive']:.3f}"
                        )
                else:
                    print(f"  all_checks_passed={rep['all_checks_passed']} checks={rep['checks']}")
                    for arm, a in rep["arms"].items():
                        print(
                            f"  {arm:18s}: dA={a['delta_avoid_odor_a_mv']:+.3f}mV "
                            f"dB={a['delta_avoid_odor_b_mv']:+.3f}mV "
                            f"spec={a['odor_specificity_mv']:+.3f}mV"
                        )
            return 0
    except (DatasetError, ValueError, FileNotFoundError) as exc:
        print(f"connectome error: {exc}", file=sys.stderr)
        return 1
    return 2


def _load_cultural_spec(target: str) -> Any:
    import yaml

    from alife_lab.core.spec import CulturalWorldExperiment, ExperimentSpec
    from alife_lab.service.catalog import load_presets
    from alife_lab.service.settings import load_settings

    p = Path(target)
    if p.is_file():
        spec = ExperimentSpec.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    else:
        presets = load_presets(load_settings().configs_dir)
        if target not in presets:
            raise ValueError(f"unknown cultural preset or file {target!r}")
        spec = presets[target].spec
    if not isinstance(spec.experiment, CulturalWorldExperiment):
        raise ValueError(f"{target!r} is not a cultural_world experiment")
    return spec


def cmd_cultural(args: argparse.Namespace) -> int:
    from alife_lab.evolution.cultural_assays import (
        run_cultural_assay,
        run_cultural_phase_diagram,
    )

    try:
        spec = _load_cultural_spec(args.preset)
        if args.cultural_cmd == "assay":
            rep = run_cultural_assay(spec, n_seeds=args.seeds, steps=args.steps)
            if args.json:
                _print_json(rep)
            else:
                j = rep["judgement"]
                print(
                    f"cultural assay {rep['preset_name']}: status={j['status']} "
                    f"evidence={j['evidence_level']} seeds={len(rep['seeds'])} "
                    f"steps={rep['steps']} wall={rep['wall_seconds']:.2f}s"
                )
                for cid, verdict in j["verdicts"].items():
                    mark = "PASS" if verdict["passed"] else "FAIL"
                    print(f"  [{mark}] {cid:26s} {verdict['reason']}")
            return 0

        if args.cultural_cmd == "phase-diagram":
            periods = tuple(int(x) for x in args.periods.split(",") if x.strip())
            costs = tuple(float(x) for x in args.costs.split(",") if x.strip())
            seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
            rep = run_cultural_phase_diagram(
                spec,
                regime_periods=periods,
                tx_costs=costs,
                seeds=seeds,
                steps=args.steps,
            )
            if args.json:
                _print_json(rep)
            else:
                print(
                    f"cultural phase-diagram {rep['preset_name']} "
                    f"seeds={rep['seeds']} steps={rep['steps']} wall={rep['wall_seconds']:.2f}s"
                )
                for c in rep["cells"]:
                    print(
                        f"  period={c['regime_period']:4d} cost={c['transmission_cost']:.3f} | "
                        f"viable={c['viable_fraction']:.2f} depth={c['mean_depth']:.2f} "
                        f"ret={c['mean_retention']:.3f} util={c['mean_net_utility']:.3f} "
                        f"comp={c['mean_comp_fraction']:.3f} E={c['mean_energy']:.2f}"
                    )
            return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"cultural error: {exc}", file=sys.stderr)
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

    clearn = csub.add_parser(
        "learn-vnc",
        help="fixed-topology per-hemilineage ALIF CMA-ES on real VNC connectome + shuffled control",
    )
    clearn.add_argument("dataset_id", choices=["banc_888", "male_cns_v1", "synthetic_cns_embodied"])
    clearn.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    clearn.add_argument("--seed", type=int, default=1)
    clearn.add_argument("--generations", type=int, default=25)
    clearn.add_argument("--pop", type=int, default=12)
    clearn.add_argument("--ticks", type=int, default=600)
    clearn.add_argument("--verify-ticks", type=int, default=1500)
    clearn.add_argument("--save-profile", default="")
    clearn.add_argument("--no-shuffled-cmaes", action="store_true")
    clearn.add_argument("--no-verify", action="store_true")
    clearn.add_argument("--json", action="store_true")

    cbat = csub.add_parser(
        "embody-batch",
        help="run F9 parallel multi-fly GPU Warp + batched LIF benchmark, screen, or search",
    )
    cbat.add_argument("dataset_id", choices=["flywire_783", "synthetic_fly_embodied"])
    cbat.add_argument("--mode", choices=["screen", "benchmark", "search"], default="screen")
    cbat.add_argument(
        "--scene", choices=["sugar_feeding", "antennal_grooming"], default="sugar_feeding"
    )
    cbat.add_argument("--cell-types", default="")
    cbat.add_argument("--seeds", default="1,2,3,4")
    cbat.add_argument("--seed", type=int, default=1)
    cbat.add_argument("--ticks", type=int, default=250)
    cbat.add_argument("--batch-size", type=int, default=16)
    cbat.add_argument("--batch-sizes", default="1,8,16,32")
    cbat.add_argument("--generations", type=int, default=4)
    cbat.add_argument("--physics", choices=["auto", "warp", "cpu"], default="auto")
    cbat.add_argument("--brain", choices=["auto", "cupy", "numpy"], default="auto")
    cbat.add_argument("--no-cpu-verify", action="store_true")
    cbat.add_argument("--no-verify", action="store_true")
    cbat.add_argument("--json", action="store_true")

    cvis = csub.add_parser(
        "vision",
        help="run F10-A flyvis optic-lobe optomotor & looming assay (or offline synthetic HR)",
    )
    cvis.add_argument("--model-id", default="000")
    cvis.add_argument("--frames", type=int, default=30)
    cvis.add_argument("--synthetic", action="store_true")
    cvis.add_argument("--json", action="store_true")

    csex = csub.add_parser(
        "compare-sex",
        help="run F10-B cross-sex connectome anatomy & identical-stimulus physiology comparison",
    )
    csex.add_argument("--datasets", default="flywire_783,banc_888,male_cns_v1")
    csex.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    csex.add_argument("--seed", type=int, default=1)
    csex.add_argument("--ticks", type=int, default=300)
    csex.add_argument("--no-verify", action="store_true")
    csex.add_argument("--json", action="store_true")

    csoc = csub.add_parser(
        "social",
        help="run F11 two-fly communication (male courtship song <-> female auditory & pursuit)",
    )
    csoc.add_argument("--stage", choices=["dialogue", "arena"], default="dialogue")
    csoc.add_argument("--male", default="male_cns_v1")
    csoc.add_argument("--female", default="banc_888")
    csoc.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    csoc.add_argument("--seed", type=int, default=1)
    csoc.add_argument("--ticks", type=int, default=400)
    csoc.add_argument("--no-ablations", action="store_true")
    csoc.add_argument("--no-verify", action="store_true")
    csoc.add_argument("--json", action="store_true")

    cswm = csub.add_parser(
        "swarm",
        help="run F11 Stage 3 fly swarm (ramdya avoidance, male aggression, or dual-sex ecosystem)",
    )
    cswm.add_argument("dataset_id", nargs="?", default="synthetic_fly_embodied")
    cswm.add_argument("--female", default=None, help="female dataset_id for --mode ecosystem")
    cswm.add_argument("--mode", choices=["ramdya", "aggression", "ecosystem"], default="ramdya")
    cswm.add_argument("--sizes", default="1,8,16,24")
    cswm.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    cswm.add_argument("--seed", type=int, default=1)
    cswm.add_argument("--ticks", type=int, default=2000)
    cswm.add_argument("--no-verify", action="store_true")
    cswm.add_argument("--json", action="store_true")

    cevo = csub.add_parser(
        "evolve",
        help="run Phase F12 fly_world open-ended evolution & 3-part emergent signal assay",
    )
    cevo.add_argument("dataset_id", nargs="?", default="synthetic_fly_embodied")
    cevo.add_argument("--seeds", default="1,2,3")
    cevo.add_argument("--evo-ticks", type=int, default=1200)
    cevo.add_argument("--eval-ticks", type=int, default=400)
    cevo.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    cevo.add_argument(
        "--replay-fine",
        action="store_true",
        help="replay the top evolved descendant inside the full-resolution spiking brain (F12d)",
    )
    cevo.add_argument("--no-verify", action="store_true")
    cevo.add_argument("--json", action="store_true")

    cworm = csub.add_parser(
        "fit-worm",
        help="run Stage 2 / S1 C. elegans connectome fit on real GCaMP activity or synthetic check",
    )
    cworm.add_argument("--source", choices=["real", "synthetic"], default="real")
    cworm.add_argument("--dataset-id", default="celegans_302")
    cworm.add_argument("--max-worms", type=int, default=12)
    cworm.add_argument("--min-matched", type=int, default=40)
    cworm.add_argument("--max-frames", type=int, default=600)
    cworm.add_argument("--source-filter", default="")
    cworm.add_argument("--horizon", type=int, default=15)
    cworm.add_argument("--epochs", type=int, default=100)
    cworm.add_argument("--lr", type=float, default=0.05)
    cworm.add_argument("--seeds", default="1,2,3,4,5")
    cworm.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cworm.add_argument("--no-verify", action="store_true")
    cworm.add_argument("--json", action="store_true")

    csp = csub.add_parser(
        "compare-species",
        help="run Stage 4 S1-S4 cross-species topology and sensorimotor selectivity comparison",
    )
    csp.add_argument(
        "--datasets",
        default="ciona_larva,celegans_302,platynereis_3d,fly_larva_l1",
    )
    csp.add_argument("--seed", type=int, default=1)
    csp.add_argument("--ticks", type=int, default=120)
    csp.add_argument("--no-physiology", action="store_true")
    csp.add_argument("--no-verify", action="store_true")
    csp.add_argument("--json", action="store_true")

    cchi = csub.add_parser(
        "chimera",
        help="run Stage 4 Phase F13a-c cross-species chimera experiments with controls",
    )
    cchi.add_argument("--kind", choices=["f13a", "f13b", "f13c"], default="f13c")
    cchi.add_argument("--host", default="celegans_302")
    cchi.add_argument("--donor", default="fly_larva_l1")
    cchi.add_argument("--blend", type=float, default=0.5)
    cchi.add_argument("--seed", type=int, default=1)
    cchi.add_argument("--trials", type=int, default=4)
    cchi.add_argument("--ticks", type=int, default=80)
    cchi.add_argument("--no-verify", action="store_true")
    cchi.add_argument("--json", action="store_true")

    p.set_defaults(func=cmd_connectome)

    p = sub.add_parser("cultural", help="run P4/P5 cumulative-culture assays and phase diagrams")
    culsub = p.add_subparsers(dest="cultural_cmd", required=True)

    ca = culsub.add_parser("assay", help="run 5-criterion cumulative-culture assay (E1/E2/E3)")
    ca.add_argument("preset", nargs="?", default="cultural_world_compositional")
    ca.add_argument("--seeds", type=int, default=5)
    ca.add_argument("--steps", type=int, default=600)
    ca.add_argument("--json", action="store_true")

    cpd = culsub.add_parser("phase-diagram", help="sweep (regime_period x transmission_cost)")
    cpd.add_argument("preset", nargs="?", default="cultural_world_compositional")
    cpd.add_argument("--periods", default="80,200,500")
    cpd.add_argument("--costs", default="0.0,0.015,0.04")
    cpd.add_argument("--seeds", default="0,1,2")
    cpd.add_argument("--steps", type=int, default=500)
    cpd.add_argument("--json", action="store_true")

    p.set_defaults(func=cmd_cultural)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
