# ADR 0002 — ExperimentSpec vs ExecutionProfile and their hashes

- Status: accepted (Phase 1)
- Date: 2026-11-18

## Decision
- `ExperimentSpec` (pydantic, `extra="forbid"`, frozen) holds rules, world size,
  boundary, initialization, seed, numeric policy, max_steps, checkpoint interval and
  observed metrics. Its digest is SHA-256 of canonical JSON (sorted keys, no spaces).
- `ExecutionProfile` holds target selector, memory policy, scheduling, preview rate
  and limits; it is hashed independently.
- `checkpoint_interval` lives in the spec (it defines which states are recoverable and
  is recorded in the scenario template). `ticks_per_second` (pacing) and preview FPS
  are execution-only: changing them never changes dynamics and they are not hashed
  into the spec. Speed changes are logged as commands.
- Presets (`configs/presets/*.yaml`) reference a scenario plus overrides; the resolved
  spec is validated and hashed at creation and stored verbatim.
- The reference scenario `configs/scenarios/conway_reference.yaml` lists the metric
  `glider_displacement`; it is implemented as a documented alias of
  `centroid_displacement` (see `docs/reference_fidelity/conway.md`).
- Reference mode for Conway requires rule `B3/S23`. A manual intervention on a
  reference-mode run switches its *effective* mode to `exploratory` (declared mode and
  spec hash are unchanged) and is logged as a `mode_changed` event.
