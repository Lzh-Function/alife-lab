# ADR 0008 — Dependency pinning and supply-chain window

- Status: accepted
- Date: 2026-11-18

- Python: `uv` with `uv.lock`; `[tool.uv] exclude-newer = 2026-11-04` so resolution
  ignores packages published within ~2 weeks of locking.
- Frontend: exact versions in `package.json` + `package-lock.json`, installed with
  `npm install --before=2026-11-04`.
- Base environment is CPU-only (FastAPI, Pydantic, NumPy, PyYAML, psutil, uvicorn).
  GPU backends (JAX, PyTorch, CuPy, MuJoCo) are deliberately not installed; they will
  get separate optional environments after oracle validation (spec §11.4).
- Resolved versions on 2026-11-18: see `docs/hardware_benchmarks.md`.
