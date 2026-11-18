# AGENTS.md — ALife Lab Development Contract

You are implementing a **local-first, browser-operated artificial-life research platform**. This is an implementation contract, not a statement that the described modules already exist.

Read `docs/ALIFE_LAB_SPEC.md` before architecture changes and `docs/GIT_AND_ARTIFACT_POLICY.md` before committing data. Both documents are the source of truth. Communicate with the user in Japanese; write source code, docstrings, source comments, structured log fields, API identifiers and commit messages in English. UI defaults to Japanese with English i18n-ready strings.

## Mission

- Build a browser laboratory to create, run, observe, pause, inspect, intervene, checkpoint, fork, resume, compare and export artificial worlds.
- Integrate diverse ALife paradigms through independent adapters: CA/particle systems, Lenia/Flow-Lenia, BFF, NCA, artificial chemistry/protocells, multi-agent systems, embodied evolution, digital evolution and related approaches.
- Keep substrate dynamics, external optimization, passive measurement, and GUI distinct. Never add a reward function to a reference substrate that deliberately has none.
- Separate **immutable ExperimentSpec** (scientific rules, seed, numerical policy, evaluation and stopping rules) from **ExecutionProfile** (compute placement, chunking, microbatching, preview, storage and resource budget).
- Support `cpu_dev` and `local_interactive` first; design `Executor` so optional `remote_compute`, `multi_gpu` and `cluster_batch` adapters can be added without revising the core scientific protocol. These are **capability roles**, not fixed VRAM tiers.
- Never silently change world size, agent population, numeric precision or evolutionary population size to make a run fit another GPU.

## Technical boundaries

1. FastAPI controller/SQLite metadata, independent workers, CLI and React Web GUI must share validated schemas and run semantics.
2. `LocalExecutor` and a fake executor are mandatory first implementations; remote/cluster adapters must be feature-gated until actually tested.
3. Only the controller writes the authoritative experiment DB. Do not let remote workers modify SQLite files across network shares.
4. GPU runtimes belong in worker processes, not the API process. Runtime/backend optional dependencies may require separate uv environments.
5. Interventions are applied at documented safe simulation boundaries and logged with requested/applied clock; remote batch jobs may lack live pause.
6. Preview frames can be dropped under load; scientific state, event evidence, checkpoints, metrics and run history cannot be silently dropped.
7. Hardware capability probe, effective limits, peak VRAM, throughput and backend versions must be measured and recorded. No presumed zero-copy or GPU performance claims.
8. Provide CPU oracle/reference/property tests for each physics/VM substrate and negative controls for proposed emergent phenomena.
9. Evidence levels E0–E4 distinguish visual curiosity from measured, reproduced, intervention-supported and externally checked findings.
10. CPU-only setup/CI must remain viable even when GPU backends are unavailable.

## Source vs artifacts

- Git: source code, frontend, API schema, tests, lockfiles, small licensed fixtures, CI, docs, reference-fidelity reports and experiment templates.
- ArtifactStore: run checkpoints, Zarr/Parquet, media, logs, lineage, optimizer states, large genomes and model weights.
- DVC/large-object data registry: optional, only for curated fixed datasets/models shared between environments. Git LFS is a reviewed exception.
- Never commit `runs/`, runtime SQLite, caches, secrets, `.env`, SSH keys, tokens or private URLs. Always inspect staged files before commits; `.gitignore` does not untrack previously tracked files.
- Every run must include source commit/digest, scientific config hash, execution profile hash, seed/RNG policy, backend, target capabilities, artifact hashes and checkpoint compatibility.
- Important runs require independent backup and restore verification. A Git commit is not a backup of runtime artifacts.

## Implementation sequence

1. **Phase 0:** inspect the repository, probe hardware, verify current official package/reference APIs, define contracts and ADRs, establish CPU tests and the Git/secret policy.
2. **Phase 1:** build FastAPI + React + SQLite + CLI + CPU Conway CA; from the browser support create/run/pause/step/brush/checkpoint/reopen.
3. **Phase 2:** implement replay/fork/compare, worker recovery, LocalExecutor, ArtifactStore, capability validation and one additional simple substrate.
4. **Phase 3:** BFF, Lenia and Flow-Lenia with scientific reference tests; add advanced GPU backends only after oracle validation.
5. **Phase 4:** NCA and independently selectable modules including multi-agent ecosystems, artificial chemistry, protocells, swarm and lightweight learning agents.
6. **Phase 5:** evolution, QD, ASAL, novelty, discovery evidence and assays.
7. **Phase 6+:** embodied evolution and optional remote/multi-GPU/batch adapters. Never present untested remote support as complete.

Complete one vertical slice (`engine → service/API → UI/CLI → persistence → tests`) before multiplying unfinished modules.

## Mandatory safety and reproducibility gates

- No arbitrary code execution through browser scenarios, no untrusted pickles, no untrusted shell construction, and no exposing GPU compute APIs to the public internet without appropriate authentication.
- Do not run paid/shared remote resources or transfer restricted data without explicit authorization.
- Do not erase prior user data, snapshots, checkpoints, Git history or unknown modified files.
- Document reference-mode deviations; log rejected candidates, negative controls, numerical tolerances and failure cases.
- Keep compute and disk budgets bounded. Failed uploads must never publish broken checkpoints.
- Before/after work: inspect `git status --short`; scan staged content for credentials and unexpectedly large binaries.
- Use official documentation at implementation time. Record exact versions and test results rather than relying on remembered compatibility.

## First assignment

Implement Phase 0 and Phase 1. Make the following commands work **after you have actually implemented them**:

```bash
uv run alife doctor
uv run alife serve --host 127.0.0.1 --port 8000
```

Verify in a browser that a Conway glider can be launched, observed, paused, advanced one tick, edited with a brush, checkpointed and restored after restarting the service. Show actual test commands, results, any measured resource usage and known limitations.

For every completed task, report: `Implemented`, `Verified`, `Measured`, `Scientific fidelity`, `Artifacts`, `Security`, `Known limitations`, `Next slice`. Distinguish implemented, partial and planned capabilities.
