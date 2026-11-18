# ADR 0010 — Timeline seek by deterministic replay

- Status: accepted (Phase 2, slice 2)
- Date: 2026-11-18

## Context
Spec §3.5 asks for seeking to saved states, guarantees only where checkpoints exist, and an
explicit "visual replay only" label otherwise. For substrates that are deterministic on CPU
(Conway, exact integers), every intermediate state can be reconstructed exactly.

## Decision
- The state "at tick T" = state after the update of tick T and after every intervention
  applied at T, in logged order.
- Reconstruction: latest checkpoint with tick ≤ T (ordered by tick, then creation time) +
  stepping + re-applying `applied` interventions with
  `applied_clock > base_tick` or (`applied_clock == base_tick` and `applied_at` later than the
  checkpoint). `rolled_back`/`aborted` interventions are excluded.
- Only offered when the substrate declares `supports_deterministic_cpu`; otherwise the
  timeline reports `visual_only` and replay endpoints refuse.
- Computation runs in a spawned single-worker process pool (`alife_lab.execution.replay`),
  never in the API process. Limits: 200 frames per request, 200 000 ticks per replay.
- Frames carry their provenance: `source` (`checkpoint_tick` | `deterministic_replay`),
  `base_checkpoint_id`, `base_tick`, `steps_from_base`, `interventions_replayed`.
- **Replay verification** re-simulates every consecutive checkpoint pair (with the
  interventions logged between them) and compares the exact grid with the stored
  checkpoint. Results are logged as `replay_verified` / `replay_mismatch` events.
- **Fork at any tick** materialises the replayed state as the child's fork-origin
  checkpoint; the child's manifest records `lineage.origin.method = deterministic_replay`,
  the base checkpoint and the number of re-applied interventions.

## Validation
- Ground truth: an identical run that checkpoints every tick with the same interventions;
  replayed frames match it at every tick 0–120 (integration test, includes two interventions
  at the same tick separated by a checkpoint).
- Negative control: altering one logged intervention payload makes verification fail on the
  affected segment.
- After a worker kill + restore, verification still passes (rolled-back intervention excluded).

## Limitations
- No replay for non-deterministic/GPU substrates yet (future: `statistically_consistent`).
- Replay frames are display data; the research state is the checkpoint (or the materialised
  fork origin).
