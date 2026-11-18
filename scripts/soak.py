"""Long-running stability (soak) test against an isolated `alife serve` instance.

Usage:
  uv run python scripts/soak.py --minutes 30 --out <result.json>

Starts its own server on a separate port with temporary ALIFE_HOME / ALIFE_RUNS_DIR,
runs a Conway world at unlimited speed and a Boids world paced at 60 ticks/s, keeps one
WebSocket client connected (forcibly reconnecting every 60 s), and samples resource usage
every 30 s. At the end it verifies artifacts, runs replay verification (within the
default compute budget) and times a full metrics read. Prints / writes JSON. All numbers
describe this machine at this time only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import psutil
import websockets

ROOT = Path(__file__).resolve().parents[1]


def dir_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6, 1)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = parser.parse_args()

    data = Path(tempfile.mkdtemp(prefix="alife-soak-"))
    env = {**os.environ, "ALIFE_HOME": str(data / "home"), "ALIFE_RUNS_DIR": str(data / "runs")}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "alife_lab.cli",
            "serve",
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    base = f"http://127.0.0.1:{args.port}"
    result: dict[str, Any] = {"minutes": args.minutes, "data_dir": str(data), "samples": []}
    try:
        async with httpx.AsyncClient(base_url=base, timeout=60) as client:
            for _ in range(200):
                try:
                    if (await client.get("/api/v1/health")).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.1)
            token = (data / "home" / "session_token").read_text().strip()
            client.headers["X-ALife-Token"] = token
            runs = {}
            for key, body in {
                "conway_512_max": {
                    "preset_id": "conway_random_soup",
                    "overrides": {
                        "world": {"width": 512, "height": 512},
                        "runner": {"max_steps": 10_000_000, "checkpoint_interval": 20_000},
                    },
                    "ticks_per_second": 0,
                },
                "boids_1000_paced": {
                    "preset_id": "boids_noisy",
                    "overrides": {
                        "initialization": {"count": 1000},
                        "world": {"width": 365, "height": 365},
                        "runner": {"max_steps": 10_000_000, "checkpoint_interval": 6_000},
                    },
                    "ticks_per_second": 60,
                },
            }.items():
                r = await client.post(
                    "/api/v1/experiments",
                    json={**body, "profile_id": "local_interactive", "auto_start": True},
                )
                r.raise_for_status()
                runs[key] = r.json()["data"]["id"]
            result["runs"] = runs
            server_proc = psutil.Process(server.pid)
            stats = {"ws_connections": 0, "ws_text": 0, "ws_binary": 0, "ws_resyncs": 0}
            deadline = time.monotonic() + args.minutes * 60

            async def watch(run_id: str) -> None:
                url = f"ws://127.0.0.1:{args.port}/api/v1/stream/experiments/{run_id}?token={token}"
                while time.monotonic() < deadline:
                    async with websockets.connect(url, origin=base, max_size=1 << 23) as ws:
                        stats["ws_connections"] += 1
                        end = min(deadline, time.monotonic() + 60)
                        while time.monotonic() < end:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), 1.0)
                            except TimeoutError:
                                continue
                            if isinstance(msg, bytes):
                                stats["ws_binary"] += 1
                            else:
                                stats["ws_text"] += 1
                                if '"resync_required"' in msg:
                                    stats["ws_resyncs"] += 1
                    # forced disconnect, then reconnect

            async def sample() -> None:
                t0 = time.monotonic()
                prev: dict[str, int] = {}
                while time.monotonic() < deadline:
                    await asyncio.sleep(30)
                    row: dict[str, Any] = {"t_s": round(time.monotonic() - t0)}
                    row["server_rss_mb"] = round(server_proc.memory_info().rss / 2**20, 1)
                    for key, rid in runs.items():
                        s = (await client.get(f"/api/v1/experiments/{rid}")).json()["data"]
                        tick = s["clock"]["value"]
                        live = s["live"] or {}
                        row[key] = {
                            "state": s["state"],
                            "tick": tick,
                            "ticks_per_s_window": round((tick - prev.get(key, 0)) / 30, 1),
                            "worker_rss_mb": live.get("worker_rss_mb"),
                        }
                        prev[key] = tick
                    row["runs_dir_mb"] = dir_size_mb(data / "runs")
                    row["ws"] = dict(stats)
                    result["samples"].append(row)
                    print(json.dumps(row), flush=True)

            await asyncio.gather(watch(runs["boids_1000_paced"]), sample())

            for rid in runs.values():
                await client.post(f"/api/v1/experiments/{rid}/pause", json={})
            final: dict[str, Any] = {}
            result["final"] = final
            # Verification can take minutes on one core (replay budget 200k ticks): no timeout.
            slow = httpx.Timeout(None)
            for key, rid in runs.items():
                print(f"final checks: {key}", flush=True)
                t = time.monotonic()
                metrics = (
                    await client.get(f"/api/v1/experiments/{rid}/metrics", timeout=slow)
                ).json()["data"]
                read_s = time.monotonic() - t
                verify = (
                    await client.post(f"/api/v1/experiments/{rid}/verify", timeout=slow)
                ).json()["data"]
                t = time.monotonic()
                replay = (
                    await client.post(f"/api/v1/experiments/{rid}/replay/verify", timeout=slow)
                ).json()
                replay_s = time.monotonic() - t
                print(f"  replay {replay['data']['status']} in {replay_s:.1f}s", flush=True)
                final[key] = {
                    "metric_points": metrics["total_points"],
                    "metrics_read_seconds": round(read_s, 2),
                    "artifacts_ok": verify["ok"],
                    "replay_status": replay["data"]["status"],
                    "replay_segments": [
                        replay["data"]["segments_checked"],
                        replay["data"]["segments_total"],
                    ],
                    "replay_seconds": round(replay_s, 1),
                }
            result["ws"] = stats
            result["runs_dir_mb"] = dir_size_mb(data / "runs")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.out:  # always keep what was measured, even if a final step failed
            args.out.write_text(json.dumps(result, indent=2))
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        if not args.keep:
            shutil.rmtree(data, ignore_errors=True)
    print(json.dumps(result.get("final"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
