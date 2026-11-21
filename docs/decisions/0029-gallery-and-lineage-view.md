# ADR 0029 — Gallery and lineage view

Status: accepted (2026-11-21)

## Context
Spec §3.10 asks for a gallery of liked pictures kept apart from the discovery ledger, and
§7.4 for a lineage viewer. The project owner learns ALife while using the lab and asked for
both explicitly.

## Decision
- Gallery (SQLite migration 5): a PNG of what the browser canvas showed, with run, tick,
  title and note. The server validates the PNG (signature, IHDR size ≤ 4096², ≤ 2 MiB),
  stores it as an immutable artifact of the source run and records its SHA-256; removing an
  entry only hides it. Every card states that it is a visual record (E0) and links to the
  run, its guide and (for searches) its lineage.
- Lineage view for search runs: `GET /experiments/{id}/lineage` returns generation, parent,
  score, viability, normalised position of every candidate (live state or deterministic
  replay); the page draws dots (x = generation, y = behaviour, score or parameter) and
  parent → child links, highlights the selected candidate's ancestry and shows its details
  (via `inspect?candidate=`), from which it can be opened as a normal run.
- Runs of random search and CMA-ES have no parents; the page says so instead of drawing
  an empty tree.

## Consequences
- Lineage of agents (AgentWorld) and cells (protocells) is still shown only per individual
  in their inspectors; a population-level tree for them is a possible extension.
