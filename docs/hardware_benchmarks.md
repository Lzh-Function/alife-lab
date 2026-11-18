# Hardware acceptance & measurements

Values below are **measurements on one machine at one time**, not performance
claims. Re-measure with `uv run alife doctor` and
`uv run python scripts/measure_throughput.py` (server must be running).

## 2026-11-18 — local workstation (WSL2)

### Probe (`uv run alife doctor`)
| Item | Value |
|---|---|
| OS | Ubuntu 24.04.3 LTS on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2) |
| CPU | Intel Core Ultra 7 270K Plus, 24 logical cores |
| RAM | 48 171 MB total |
| GPU (nvidia-smi only; no GPU runtime loaded) | NVIDIA GeForce RTX 5070, 12 227 MiB, driver 610.88, compute capability 12.0 |
| Python / uv / Node / npm | 3.12.14 / 0.12.17 / 26.13.0 / 11.20.0 |
| SQLite library | 3.53.1 |
| Optional GPU backends | none installed (by design in Phase 1) |

### Resolved dependency versions
Python: fastapi 0.162.0, starlette 1.10.0, uvicorn 0.54.0, pydantic 2.14.2, numpy 2.5.4,
pyyaml 6.0.3, psutil 7.2.2, websockets 17.4; dev: pytest 9.2.0, hypothesis 6.170.14,
ruff 0.17.3, mypy 2.4.1. Frontend: react 19.3.1, @tanstack/react-query 5.105.3,
zustand 5.0.16, vite 8.3.5, typescript 5.9.3, vitest 5.1.0, @playwright/test 1.64.1
(E2E run with the pre-installed Chrome for Testing 153.0.8010.12 headless shell).

### Conway end-to-end throughput (CPU, NumPy backend)
Unlimited speed (`ticks_per_second=0`), one WebSocket client consuming previews,
`cpu_dev` profile (preview 2 fps), 8 s measurement window, via the HTTP API of
`uv run alife serve --host 127.0.0.1 --port 8000`.

| World | Initial state | ticks/s (end-to-end) | Worker RSS |
|---|---|---|---|
| 128×128 | glider | 5 920 | 52.8 MB |
| 128×128 | random 0.3 | 5 161 | 52.8 MB |
| 512×512 | random 0.3 | 505 | 56.6 MB |
| 1024×1024 | random 0.3 | 113 | 70.6 MB |

API server process RSS after these runs: 31.5 MB. Headless `alife run` of the
reference scenario (128², 1000 ticks, 11 checkpoints): 0.60 s wall including worker
spawn. Peak worker RSS in the browser E2E run: 51.7 MB.

Throughput includes per-tick metric computation, JSONL writes and periodic
checkpoints; it is single-threaded NumPy and is not a GPU number. No GPU code ran.

### Soak (5 min, 256×256 random soup, unlimited speed, `local_interactive`, 1 WS client)
| t (s) | tick | worker RSS (MB) | API server RSS (MB) | checkpoints |
|---|---|---|---|---|
| 30 | 57 413 | 53.2 | 31.5 | 3 |
| 120 | 222 814 | 54.0 | 31.5 | 12 |
| 271 | 501 825 | 54.3 | 31.5 | 26 |

~1 850 ticks/s sustained; worker RSS grew 1.1 MB over 4 min (no evidence of a leak at
this duration; the spec's 30–60 min soak has **not** been run yet). Disk: metrics
JSONL grows ~one row per tick (see limitation in ADR 0005).

### Boids step throughput (CPU, NumPy, constant density 300 / 200², 2026-11-18)
Adapter `step` only (cell-list neighbour search + steering), after 20 warm-up ticks:

| N | world | ticks/s |
|---|---|---|
| 300 | 200² | 2 011 |
| 1 000 | 365² | 1 202 |
| 3 000 | 632² | 343 |
| 10 000 | 1 155² | 121 |

All numbers above are **single-core**: measured CPU time / wall time = 1.01 for both Boids
(N=3000) and Conway (1024²) on this 24-logical-core machine; NumPy's thread pool exists but is
idle for these element-wise/sort/bincount operations. Parallelism today is only across runs
(one worker process per run, `ALIFE_MAX_ACTIVE_JOBS`, default 4). No GPU code runs.

Browser E2E (200 boids, cpu_dev, paced at 60 ticks/s): 57.9 ticks/s measured, worker RSS
53.3 MB.

### Soak tests (2026-11-18, `scripts/soak.py`, isolated server, single core per worker)
Two runs in parallel: Conway 512² random soup at unlimited speed and Boids 1000 (noise 0.15)
paced at 60 ticks/s, `local_interactive` profile, one WebSocket client force-reconnected every
60 s, resources sampled every 30 s. Other tests were running on the machine at times.

**30 minutes** (60 samples):

| t (s) | API server RSS | Conway ticks/s · worker RSS | Boids ticks/s · worker RSS | runs dir | WS reconnects |
|---|---|---|---|---|---|
| 30 | 69.8 MB | 512 · 56.9 MB | 59.6 · 53.5 MB | 2 MB | 1 |
| 390 | 77.8 MB | 526 · 56.9 MB | 60.0 · 53.5 MB | 27 MB | 7 |
| 570 | 110.3 MB | 531 · 56.9 MB | 60.3 · 53.5 MB | 39 MB | 10 |
| 1650 | 110.8 MB | 532 · 56.9 MB | 59.8 · 53.7 MB | 114 MB | 28 |

Workers flat; throughput flat; 30 forced reconnects with 0 `resync_required`. The API server
RSS stepped from ~78 to ~110 MB between 390 s and 570 s and was then flat (+0.5 MB over 20
min). **Cause not identified**; the step did not reproduce in the 10-minute run below (69.9 →
71.1 MB). The end-of-run verification of this run failed only because the script's HTTP
timeout (60 s) was shorter than replay verification (fixed in the script).

**10 minutes** (with end-of-run verification, data kept):

| Check | Conway 512² | Boids 1000 |
|---|---|---|
| Ticks / metric rows | 316 997 | 35 997 |
| Full metrics read (JSONL scan) | 0.72 s | 0.06 s |
| Artifact checksums | ok | ok |
| Replay verification | 10/15 segments exact (`partial`: 200k-tick budget), 361.5 s | 5/5 exact (`verified`, incl. RNG noise), 28.5 s |

Disk growth: ~0.21 MB per 1000 Conway ticks (dominated by metrics JSONL).

### Metric storage (Parquet chunks, 2026-11-18)
See ADR 0014: 1 M rows — JSONL 113 MB vs Parquet 16.9 MB; downsampled read 2.10 s vs 0.14 s.
Throughput re-measured with `scripts/measure_throughput.py` (8 s windows): 128² glider 5 719,
128² random 5 161, 512² 499, 1024² 112 ticks/s (unchanged within noise). Resident memory with
pyarrow loaded: worker ≈113 MB (was ≈53 MB), API process ≈103 MB (was ≈31.5 MB).
