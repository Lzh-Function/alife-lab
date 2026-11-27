# ADR 0054: Complete GUI–CUI Parity, Research Tools Hub, and GUI-First Policy

- **Status:** Accepted (2026-11-27)
- **Context:** Following Stage 4 (`e729004`), an exhaustive audit revealed that while all 25 substrates supported interactive live execution in the browser, 21 CLI research assays/comparisons (`alife connectome <subcmd>`, `alife cultural phase-diagram`), 4 connectome dataset commands (`list/fetch/verify/summary`), and 3 backup commands (`backup create/verify/restore`) were CLI-only. In addition, 5 substrates lacked dedicated `InspectorPanel` renderers and `ConnectomePanel` hardcoded adult-fly `mn9`/`mbon` series.

## Decisions

### 1. Permanent GUI-First Parity Rule (`AGENTS.md` Boundary #11)
- No future feature may ship as CLI/CUI-only. Every dataset tool, comparative benchmark, chimera assay, phase diagram, and operational maintenance action must be runnable and inspectable end-to-end from the React Web GUI.

### 2. Isolated Subprocess `ResearchJobManager` (Zero GPU Contamination in Controller)
- [`src/alife_lab/service/research_tools.py`](../../src/alife_lab/service/research_tools.py) exposes a declarative allowlisted catalog (`RESEARCH_TOOLS`, 21 tools) covering every CLI research command (`compare-species`, `chimera`, `compare-sex`, `vision`, `screen`, `pavlov`, `sysid`, `sysid-map`, `task`, `fit-real`, `fit-worm`, `embody`, `embody-l2`, `tune-vnc`, `learn-vnc`, `embody-batch`, `social`, `swarm`, `evolve`, `cultural assay`, `cultural phase-diagram`) plus dataset `fetch`.
- Every submitted parameter is type-validated and regex-checked (`_SAFE_TOKEN_RE`), then executed via `subprocess.Popen([sys.executable, "-m", "alife_lab.cli", *argv], shell=False)` in an isolated worker subprocess. This strictly obeys `AGENTS.md` Technical Boundary #4 (GPU runtimes belong in worker processes, never the FastAPI API process) while eliminating code duplication.
- Job metadata and parsed JSON results are persisted atomically under `runs_root / "_research_jobs"` (capped at 2 concurrent jobs, with user cancellation and clean shutdown on server exit).

### 3. Browser Research Hub (`#/research`), Backup UI (`#/diagnostics`), and LiveLab Polish
- **[`ResearchHub.tsx`](../../frontend/src/components/ResearchHub.tsx)** provides three integrated tabs in Japanese and English:
  1. **Research Assays & Comparisons:** category filtering, auto-generated typed parameter forms, and dedicated visualizers (5-species topology/physiology tables, F13 chimera provenance color bar + arm tables + PASS/FAIL chips, 2D cultural phase-diagram heatmap, and downloadable JSON reports).
  2. **Connectome Datasets:** status table for all empirical + offline synthetic packs with inline Summary (including reference checks), SHA-256 verification, and async Fetch.
  3. **Job History:** live polling list with cancel and result replay.
- **[`Diagnostics.tsx`](../../frontend/src/components/Diagnostics.tsx)** gains a managed Backup & Restore panel (`GET/POST /api/v1/backups`, `verify`, non-destructive `test-restore` into a clean sibling directory).
- **[`ConnectomePanel.tsx`](../../frontend/src/components/ConnectomePanel.tsx)** dynamically adapts its time-series chart to non-adult-fly species (`mean_firing_rate_hz`, `mean_v_mv`, `active_neurons_tick`), **[`InspectorPanel.tsx`](../../frontend/src/components/InspectorPanel.tsx)** adds dedicated views for all 5 previously missing substrates (`connectome_sysid`, `connectome_task`, `fly_embodied`, `fly_swarm`, `fly_world`), and **[`substrates.ts`](../../frontend/src/lib/substrates.ts)** exposes `w_syn_mv` and `f_poi` on the connectome creation form.

## Consequences

- **Positive:** 100% of CLI research, dataset, phase-diagram, and backup workflows are now accessible from the browser with identical scientific semantics, CSRF/same-origin protection, and full JA/EN localization.
- **Reversibility:** All new endpoints live under `/api/v1/connectome/datasets`, `/api/v1/tools/research/*`, and `/api/v1/backups`, isolated in `research_tools.py` and `ResearchHub.tsx`, with zero impact on existing run schemas or preset hashes.
