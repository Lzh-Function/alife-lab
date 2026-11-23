# ADR 0041: Connectome Fixed-Topology Task Learning (F6)

- Status: Accepted
- Date: 2026-11-23
- Related: ADR 0035 (species-pack), ADR 0038 (F3 screen & search), ADR 0039 (F4 plasticity), ADR 0040 (F5 hidden-truth sysid), `docs/plans/connectome_plan.md`

## Context

Phase F6 asks a concrete empirical question (`docs/plans/connectome_plan.md` §4 F6):

> **Does a learning machine borrowing the fixed biological fly connectome topology learn tasks better than one borrowing a degree- and weight-matched shuffled wiring diagram?**

To satisfy `AGENTS.md` and `connectome_plan.md` §4 F6:

1. **Strict scientific framing:** Never describe this as "a fly learning". Every module, CLI summary, UI tooltip, preset, and report explicitly frames it as a **learning machine borrowing the fly connectome topology**.
2. **External learning boundary:** Task loss and PyTorch optimization live strictly outside the `connectome` simulation substrate (separate `connectome_task` substrate and `alife connectome task` CLI, with lazy PyTorch imports so the FastAPI API process stays GPU-free and CPU-only CI stays green without `torch`).
3. **Pre-registered hypothesis & decision rule:** Before any training runs, `make_preregistration()` hashes the hypothesis, primary metric (`test_acc` on unseen holdout trials), control (`shuffled_control` across independent seeds), and verdict thresholds (`alpha = 0.05`, `min_delta_acc = 0.05`, `min_cliffs_delta = 0.50`) via `sha256_json`, assigning `"validated"`, `"rejected"`, or `"inconclusive"` automatically without post-hoc goalpost moving.

## Decision

### 1. Tasks and trainable scopes (`src/alife_lab/connectome/tasks.py`)

- **Three tasks** (deterministic train/test splits on `Subcircuit` glomerular input groups):
  1. `odor_classification`: 4-class multi-glomerular odor prototypes with 25% glomerular dropout and Gaussian input noise.
  2. `short_memory`: 4-class delayed cue recall (`16 ms` cue followed by `48 ms` noisy silent delay; readout only in the final `12 ms`).
  3. `nonlinear_xor`: linearly inseparable 2-group odor XOR (`a ^ b`).
- **Three trainable scopes** (wiring topology strictly fixed in all three):
  1. `readout_only`: linear readout from anatomical downstream neurons (`MBON` / `LHLN` / `LHCENT` / `MBIN`, capped at 256 deterministic cells; identical cell indices in both biological and shuffled wiring). Uses leak-free training-set z-score standardization so small-variance multi-hop readout features train cleanly without test leakage.
  2. `io_only`: learnable glomerular input projection `W_in` + linear readout; internal wiring 100% frozen.
  3. `partial_weights`: linear readout + learnable positive per-type synaptic scaling on `Kenyon_Cell` and `ALPN` presynaptic types (edge set and E/I signs strictly fixed).

### 2. Browser substrate, CLI & offline CPU CI

- Registers `connectome_task` (`src/alife_lab/substrates/connectome/task_adapter.py`, `mode="exploratory"` only, `backends=("torch", "torch_cuda")`), rendering a deterministic `160x120` RGB (`field_u8`) 4-quadrant diagnostic view (unseen test confusion matrix, train/test accuracy curves with chance line, stimulus + readout window timeline, readout activity heatmap) with full checkpoint serialization (`arrays` + `meta`).
- Adds CLI `alife connectome task <dataset_id>`, 4 catalog presets (`connectome_task_olfactory_tiny`, `connectome_task_fly_odor`, `connectome_task_fly_memory`, `connectome_task_fly_shuffled_control`), beginner guide, glossary entry, and full `ja`/`en` i18n.
- `tests/reference/test_connectome_f6.py` runs pre-registration, decision-rule, and dataset tests unconditionally in CPU-only CI and guards PyTorch tests with `@needs_torch`.

## Consequences

- On the real 11,276-neuron, 1,105,291-edge FlyWire v783 `olfactory_mb_lh` subcircuit (RTX 5070, 5 seeds per cell across all 3 tasks × 3 scopes), **0 of 9 pre-registered hypotheses were validated (all 9 were `inconclusive`)**, and on `odor_classification` (`readout_only` and `partial_weights`) degree-preserving shuffled wiring actually scored slightly higher (`0.991` vs `0.950`, `p = 0.0079`, below the `-0.05` rejection margin).
- This honest null/negative result reveals a real structural effect: degree-preserving whole-subcircuit shuffling creates direct 1-hop random projections from ORNs onto `MBON`/`LHLN` readout neurons (an ideal random Echo State feature map), whereas biological wiring routes inputs through 3–4 specialized feedforward/inhibitory layers tuned for biological invariance rather than generic supervised readout.
