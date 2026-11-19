# Substrate contracts — `nca_train` and `nca`

## `nca_train` (optimisation run)
- One tick = one optimizer step (clock unit `optimizer_step`). Explicit external objective:
  RGBA MSE to the padded target after a random rollout length in [steps_min, steps_max].
- Spec: target (`preset` shape or `custom` RGBA base64 drawn in the browser), task
  (`growing` / `persistent` pool / `regenerating` with damage), architecture (channels, hidden,
  fire rate, alive threshold), pad, batch, pool, rollout range, Adam learning-rate schedule.
- State, checkpointed as separate groups: `model_weights` (`w.*`), `optimizer_state`
  (`adam_m.*`, `adam_v.*`), `cell_states` (`pool`), `rng` (JAX key + NumPy PCG64 state).
- No interventions. Inspection: a cell of the displayed sample, last loss, parameter count.
- Metrics: `loss`, `log10_loss`, `learning_rate`, `grad_norm`.
- Export: `POST /experiments/{id}/nca/export[?checkpoint_id=]` → an `nca` spec with the weights
  embedded (base64 + sha256) and the training run/checkpoint as provenance.

## `nca` (dynamics with a fixed rule)
- One tick = one stochastic CA step. The weights are part of the scientific spec, so the rule
  is hashed with the experiment. Controls: `weights_kind: untrained` (zero output layer, no
  growth) and `noise` (random output layer).
- State: cells float32[H, W, C], JAX key. Interventions: `damage {x, y, radius}`,
  `seed {x, y}`. Inspection: RGBA, hidden channels, alive flag and target RGBA of a cell.
- Metrics: `alive_fraction`, `alpha_mass`, `target_mse` (passive; never used by the dynamics).

Both use the `jax` backend (CPU platform forced) from the optional `nca` extra; display codec
`field_u8` with 3 channels (RGBA composited on white).
