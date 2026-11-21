# ADR 0034 — Fast replay: cursor and keyframe cache in the replay process

Status: accepted (2026-11-21)

## Context
Every replay request re-simulated from the nearest checkpoint: a one-tick step or a short
scrub back cost 0.3–1.4 s, and playback re-simulated from the checkpoint for every batch
(200 ticks of multichannel Lenia took ~10 s).

## Decision
- The replay process keeps, per base checkpoint (key: checkpoint sha256, tick, backend, spec),
  the state it last reached (cursor) and serialized keyframes every 25 ticks it passed. A
  request starts from the latest of these at or before its target, but only if it derives from
  the same base checkpoint and exactly the same logged interventions up to that tick;
  otherwise from the checkpoint as before. This memoises the computation that would be
  repeated, so answers are identical (tests compare cached and cold replays for 13 substrates,
  including interventions, frames, batches, inspection and state reconstruction).
- Keyframes are private copies with JSON-normalised meta (as a checkpoint file would be). The
  first two keyframes past the base are checked by a round trip (restored state must render
  identically); substrates with display-only state outside the checkpoint (the reaction
  network's count history) therefore use the cursor only. Budget: 512 MB of keyframes, 4
  sessions, least recently used first.
- `verify_segment` (replay verification) never uses the cache.
- The controller starts the replay process when a run's timeline is first loaded, and for a
  paused or stopped run pre-computes the last checkpoint segment in the background, so opening
  replay at the current tick and scrubbing back is immediate.
- The timeline seeks "latest wins" (no queue of stale targets) and offers 10/30/60 ticks/s
  playback.

## Measured (CPU, cpu_dev; before → after)
One-tick step: 0.3–1.4 s → 2–6 ms. Scrub back 10–40 ticks: → 2–25 ms. 200-tick playback:
lenia_orbium 1.9 → 0.09 s, physarum 4.9 → 0.31 s, lenia_mc 10.3 → 0.56 s. Opening replay on a
paused run: 2 ms. Remaining: the first jump into a segment not yet computed (0.4–1.0 s).
