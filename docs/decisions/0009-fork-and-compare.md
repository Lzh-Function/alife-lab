# ADR 0009 — Fork, paired fork and state comparison

- Status: accepted (Phase 2, first slice)
- Date: 2026-11-18

## Decision
- **Fork** creates a new run (`parent_run_id`, `parent_checkpoint_id`, optional `label`).
  The verified parent checkpoint is copied into the child as
  `checkpoints/fork_origin_t<tick>.npz` (reason `fork_origin`, `epoch 0`) and the child's
  first job restores from it. The child's clock continues from the fork tick and its metric
  history starts there. `manifest.json` gains a `lineage` block.
- If the child's spec equals the parent's, the child shares the parent's `experiment_id`
  (paired design). Forks may change only `name`, `mode`, `runner`, `dynamics`,
  `observations`; world, substrate, seed and initialization are fixed by the parent state.
  A changed spec gets a new experiment and spec hash; the checkpoint table stores the
  `spec_hash` the state was produced under so the worker can validate the origin.
- A child of a parent whose effective mode is `exploratory` inherits `exploratory`.
- An intervention requested at fork time is queued as an *initial command* and applied by
  the worker before the first tick; it is logged with `applied_clock = fork tick`.
- **Paired fork** (`POST /experiments/{id}/paired-fork`) is the one-shot
  "Pause → Checkpoint → Fork → Intervention → Compare" of spec §3.4: checkpoint the live
  parent (or use its latest checkpoint), create `control` and `treatment` arms from the same
  checkpoint, record a `comparisons` row (`design = paired_fork`) and start both paused.
- **Comparison** metric `hamming_to_first`: number of differing cells relative to the first
  arm, computed only at ticks where *every* arm has a checksum-verified checkpoint
  (latest-created checkpoint per tick). Previews are never used.

## Negative control / validation
Two arms forked from one checkpoint without intervention, and a parent versus its fork, have
Hamming distance 0 at all common checkpoint ticks (integration test). A treatment brush gives
non-zero divergence (integration + browser E2E).

## Related fixes made in this slice
- After an intervention, a checkpoint at the same tick is no longer skipped as a duplicate,
  and checkpoint file names get a `_nK` suffix when the tick name is taken.
- Restoring from a checkpoint marks interventions applied after it as `rolled_back` (event
  `interventions_rolled_back`) and pending ones as `aborted`.
- Migrations now back up an existing database (`*.pre-v<N>-<ts>.bak`, SQLite backup API)
  before applying.

## Limitations
- Comparisons are pairwise against the first arm only; no statistics across seeds yet
  (single paired runs are not independent samples — spec §9.9).
- Restoring an *older* checkpoint in place is still refused; use fork instead.
