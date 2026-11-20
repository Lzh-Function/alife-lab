# ADR 0025 — Discovery ledger with pre-declared assay plans

Status: accepted (2026-11-20)

## Context
Spec §3.10 and §9.7 ask for a ledger of discovery candidates with statuses
`new / triaged / assayed / validated / rejected / inconclusive`, evidence levels E0–E4, an
assay planner that does not presuppose results, and an exportable Discovery Record.

## Decision
- Candidates are registered from search runs by `(run_id, candidate index, tick)`; the
  controller re-derives the candidate (live worker or deterministic replay) instead of trusting
  a client-sent spec. The candidate's full target spec and hash, the source search spec hash,
  parameters and lineage are stored (SQLite migration 4: `discoveries`, `discovery_events`,
  `assays`).
- At registration an assay plan is generated and hashed (`alife_lab.evolution.assays`,
  plan version 1): `reproduce`, `longer` (≥ 3000 ticks), `larger_world` (2 × side) and, for
  the claim `faster_than_base`, `baseline` (the search target's own parameters). Pass rules are
  fixed numbers; seeds come from a stream disjoint from search and holdout seeds.
- Verdict statuses are set only by the plan's rules (author `assay_rule`); users can set `new`
  / `triaged` and add notes. Every change is an event. Assays interrupted by a controller stop
  are marked `interrupted` and never publish partial verdicts; results are also published as an
  artifact under the source run.
- Evidence: E1 at registration (measured by the search objective), E2 when every check passes.
  E3/E4 need intervention or independent-implementation checks and are never automatic.
- UI labels (Japanese): 未確認・確認予定・試験済み・裏付けあり・裏付けなし・結論保留.

## Consequences
- The ledger is per controller DB; Discovery Records (JSON) carry provenance for sharing.
- Planned: claim kinds for other substrates, intervention-based (E3) checks, human labels for
  detector evaluation, a gallery separate from the ledger.
