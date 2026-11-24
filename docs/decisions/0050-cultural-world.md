# ADR 0050: `cultural_world` Architecture — Three-Layer Separation, Safe Skill DSL, Six Split RNG Streams, and Causal Culture Interventions (P0–P3)

- **Status:** Accepted
- **Date:** 2026-11-24
- **Related:** ADR 0025 (Discovery Ledger), `docs/plans/cultural_world_plan.md`, `src/alife_lab/substrates/agent_world/`

## Context

We want to add `cultural_world` (`mode = "exploratory"`) to ALife Lab in order to study when cumulative culture, social learning, and compositional techniques emerge above genetic evolution and individual lifetime learning—without breaking any existing behavior, snapshot compatibility, or config hashes of `agent_world`, and without pre-installing civilization stages or external "civilization fitness" rewards.

Following `docs/plans/cultural_world_plan.md` (§3 P0), seven architectural decisions must be locked in before code or sweeps are authored.

## Decisions

1. **Operational Definition of `skill` (Procedure vs Internal State):**
   - A transmissible skill (`SkillProcedure`) is an immutable, finite sequence of whitelisted non-code operations (`ALLOWED_OPS = ("locate", "gather", "process_a", "process_b", "extract", "store")`, max length `<= 12`, zero `eval/exec`), bound to a target material (`1 = material_a` or `2 = material_b`), a regime tuning phase `phase in [0, 1)`, an intrinsic efficiency, and explicit provenance (`skill_id`, `parent_skill_id`, `revision`, `inventor_id`, `origin_tick`, `depth`).
   - Private internal weights (`dw2`) and episodic patch coordinates (`private_best_*`) are **not** skills and are never serialized into `SkillProcedure` or socially copied.

2. **Explicit Designer-Provided Initial Primitives:**
   - The simulation designer supplies only low-level atomic primitives (move left/forward/right, gather resources with proportional contention sharing, attempt/execute a resource-extraction skill, demonstrate a skill within `observation_range`) and a parameterized yield law linking `(skill.efficiency, skill.phase, env_regime_phase)` to material-to-energy conversion.
   - No pre-baked recipes, technology trees, institutions, or external civilization scores are pre-installed; `forbid_preinstalled_recipes` holds by construction because all agents initialize with `skills = []`.

3. **Strict Three-Layer Separation Per Agent:**
   - **Genotype (inherited + mutated when `genetic_evolution=True`):** base policy weights `(w1, b1, w2, b2)`, `gene_lr`, `gene_social`.
   - **Lifetime state (never inherited by offspring):** `energy`, resource inventories `(inv_a, inv_b)`, lifetime learning delta `dw2`, reward baseline `rbase`, private episodic best `(private_best_x, private_best_y, private_best_val)`. Newborn agents always start with `dw2 = 0`, zero inventories, and zero private memory.
   - **Cultural state (independent social channel):** list of `SkillSlot` entries (`skill_id`, `teacher_id`, `mode`, `acquired_tick`, `uses`). Offspring acquire skills only through explicit `vertical` transmission when enabled, and unrelated agents only through explicit `horizontal` or `oblique` channels.

4. **Fixed Per-Tick Update Order and Six Independent RNG Streams:**
   - Every tick executes in exact fixed order:
     1. Environment ecology step (seasonal/shifting phase update, bounded resource inflow & decay booked into `ResourceLedger`).
     2. Sensory computation & controller action selection (with optional `scripted` baseline or `mlp` + `dw2`).
     3. Movement, contested gathering, skill execution / de-novo invention or refinement, and explicit `teach` triggers.
     4. Non-parent social transmission (`horizontal` / `oblique` within `observation_range`, applying energy costs & error probability, logging `TransmissionEvent`).
     5. Lifetime REINFORCE learning update to `dw2` (using only internal per-tick energy change, when `lifetime_learning=True`), metabolism, and age increment.
     6. Mortality (energy <= 0 or `age >= max_age`): unused `inv_a / inv_b` are returned to the local grid cell so total mass remains strictly conserved; skill inventors are marked deceased in `SkillRegistry` for `skill_retention_after_death`.
     7. Reproduction (energy >= `reproduce_threshold` and population headroom): energy splits evenly to child; genotype copies/mutates; `dw2` and private memory zeroed; optional `vertical` transmission runs strictly after birth.
   - RNG streams are split deterministically from `SeedSequence(seed)` into six independent `PCG64` generators (`env`, `birth`, `policy`, `learn`, `transmit`, `tech`), all saved in checkpoints, guaranteeing paired conditions share identical environmental/seasonal trajectories.

5. **Strict Per-Resource Mass Conservation Ledger:**
   - `food`, `material_a`, and `material_b` are non-negative everywhere (`>= 0`), and `ResourceLedger` tracks `initial`, `inflow`, `consumed`, and `decayed` per resource kind.
   - Every tick and every intervention (`add_resource`) must preserve `resource_ledger_residual <= 1e-9`.

6. **Causal Culture Interventions at Tick Boundaries:**
   - `erase_culture`: zeroes `agents.skills` cleanly across the entire population while keeping genotype arrays, `dw2`, `energy`, inventories, positions, population count, and resource fields bit-for-bit identical.
   - `set_transmission`: toggles runtime `vertical` / `horizontal` / `oblique` gates for paired-fork causal comparison without altering genomes or physics.
   - `shock_environment` and `add_resource`: shift regime phase and inject bounded tracked resource mass.

7. **Operational Definitions and E0–E4 Boundaries:**
   - Statistical comparisons always treat one independent seeded run as the single unit of analysis (`n = len(seeds)`), never individual agents or ticks within a run.
   - Copying a skill counts only as social transmission (L2); cumulative culture (L3) requires post-inventor retention, refinement depth, degradation under `erase_culture` / transmission ablation, and held-out generalization—documented honestly in `docs/reference_fidelity/cultural_world.md` even when effects are null or negative.
