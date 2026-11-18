"""Measure end-to-end Conway throughput and resource usage against a running server.

Usage: uv run python scripts/measure_throughput.py [--base http://127.0.0.1:8000] [--seconds 10]

Creates runs at unlimited speed (ticks_per_second=0) for a few world sizes, while one
WebSocket client consumes previews, and reports measured ticks/s, worker RSS and the
API server RSS. Results are printed as JSON; nothing here is a performance claim
beyond the measured machine and moment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path
from typing import Any

import httpx
import psutil
import websockets

ROOT = Path(__file__).resolve().parents[1]


def token() -> str:
    return (ROOT / ".alife" / "session_token").read_text(encoding="utf-8").strip()


async def measure(base: str, size: int, seconds: float, preset: str) -> dict[str, Any]:
    headers = {"X-ALife-Token": token()}
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=30) as client:
        body = {
            "preset_id": preset,
            "overrides": {
                "name": f"bench_{preset}_{size}",
                "world": {"width": size, "height": size},
                "runner": {"max_steps": 10_000_000, "checkpoint_interval": 1_000_000},
            },
            "profile_id": "cpu_dev",
            "ticks_per_second": 0,
            "auto_start": True,
            "start_paused": True,
        }
        run = (await client.post("/api/v1/experiments", json=body)).json()["data"]
        rid = run["id"]
        for _ in range(100):
            state = (await client.get(f"/api/v1/experiments/{rid}")).json()["data"]["state"]
            if state == "paused":
                break
            await asyncio.sleep(0.05)
        frames = 0
        ws_url = base.replace("http", "ws") + f"/api/v1/stream/experiments/{rid}?token={token()}"
        async with websockets.connect(ws_url, origin=base, max_size=1 << 22) as ws:  # type: ignore[arg-type]
            await client.post(f"/api/v1/experiments/{rid}/resume", json={})
            start_tick = (await client.get(f"/api/v1/experiments/{rid}")).json()["data"]["clock"][
                "value"
            ]
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                try:
                    msg = await asyncio.wait_for(ws.recv(), 0.5)
                except TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    frames += 1
            await client.post(f"/api/v1/experiments/{rid}/pause", json={})
            elapsed = time.monotonic() - t0
        summary = (await client.get(f"/api/v1/experiments/{rid}")).json()["data"]
        ticks = summary["clock"]["value"] - start_tick
        live = summary["live"] or {}
        await client.post(f"/api/v1/experiments/{rid}/cancel")
        return {
            "world": f"{size}x{size}",
            "preset": preset,
            "wall_seconds": round(elapsed, 2),
            "ticks": ticks,
            "ticks_per_second_end_to_end": round(ticks / elapsed, 1),
            "preview_frames_received": frames,
            "worker_rss_mb": live.get("worker_rss_mb"),
            "worker_peak_rss_mb": live.get("worker_peak_rss_mb"),
        }


def server_rss_mb(port: int) -> float | None:
    for proc in psutil.process_iter(["cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if "serve" in cmd and str(port) in cmd and any("alife" in c for c in cmd):
            return round(proc.memory_info().rss / 2**20, 1)
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    results = []
    for size, preset in (
        (128, "conway_glider"),
        (128, "conway_random_soup"),
        (512, "conway_random_soup"),
        (1024, "conway_random_soup"),
    ):
        results.append(await measure(args.base, size, args.seconds, preset))
    port = int(args.base.rsplit(":", 1)[1])
    print(
        json.dumps(
            {
                "machine": {
                    "cpu": platform.processor() or None,
                    "python": platform.python_version(),
                },
                "api_server_rss_mb": server_rss_mb(port),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
