# ADR 0049: Phase F12 — `fly_world` (Artificial Life Starting From the Biological Fly Connectome)

- **Status:** Accepted
- **Date:** 2026-11-23
- **Related:** ADR 0048, `docs/plans/connectome_plan.md §F12`, `docs/reference_fidelity/connectome_fly.md`

## Context

Phase F12 transitions from biological fly connectome emulation (F0–F11) to **artificial life starting from the biological fly connectome** (`fly_world`). Per `AGENTS.md` and `connectome_plan.md §F12`:
1. Descendants are explicitly framed as **artificial life**, no longer models of real *Drosophila*, and quantitative genetic distance from the ancestral biological fly (`fly_distance`) is tracked continuously.
2. Mode B (`fly_world`) uses **zero external fitness function** — natural selection arises solely from metabolism, foraging, overhead looming predator mortality, and autonomous sexual reproduction.
3. Brain resolution is chosen explicitly (`cell_type_compressed`), never silently downgraded, and its reflex fidelity against the full 138,639-cell spiking FlyWire brain is measured and recorded.

## Decisions

1. **Explicit Port-Preserved `(cell_type, side)` Ancestral Brain Compression (`fly_world.py`):**
   - Preserves key anatomical I/O ports (`sugar_grn`, `bitter_grn`, `groom_mech`, `MN9`, `DNg84`, `DNp09`, `DNa01/02`) as un-diluted clusters while compressing remaining cells by `(cell_type, side)` (or `(region, cell_class, side)` when unannotated).
   - Compresses `flywire_783` from `138,639` cells → **`17,641` nodes** (`7.86×`) and `3.48M` signed type-edges, running analytical Shiu LIF steady-state f-I updates via batched GPU CuPy SpMM (`~1 ms/step` across 32–48 evolving flies) or exact CPU SciPy SpMM.
   - **Verified fidelity vs full 138k spiking FlyWire:** `baseline = 0.0 Hz`, `sugar = 220.0 Hz`, `sugar + bitter = 0.0 Hz` (`100%` suppression), `antennal poke -> DNg84 = 97.4 Hz` (4/4 reflex checks PASS).

2. **Shared Ancestral Connectome + Compact Per-Individual Differential Genome (`GENOME_DIM = 24`):**
   - One immutable ancestral CSR matrix is shared across the population; each living individual carries a 24-float differential genome encoding pathway synaptic scales (with rare E/I sign flips), excitability deltas, continuous body morphology (`body_scale`, `leg_length`, `wing_span`, `antenna_length`), two evolvable communication signal channels (`sig0_food`, `sig1_alarm`), and sparse de-novo cross-pathway structural edges.
   - Quantitative `fly_distance` decomposes drift into `wiring_delta`, `morphology_delta`, `signal_novelty`, and `novel_edges`, driving live 3D coloration (amber ancestor → cyan → magenta descendant).

3. **Autonomous Artificial-Life Substrate (`fly_world`) & 3D Evolutionary UI (`FlyWorldView.tsx`):**
   - Supports dynamic regenerating food patches, periodic looming overhead predator shadows, metabolic & signaling costs, sexual crossover + mutation, lineage recording, live interventions (`toggle_signals`, `toggle_mutations`, `spawn_predator`, `add_food`), exact deterministic checkpoint/restore, and a live **Signal Dictionary** table (`emit_rate`, `MI bits`, `receiver_response`).

4. **Rigorous 3-Part Emergent Signal Verification (`alife connectome evolve`):**
   - Evaluates (1) Mutual Information `I(Signal; Context)` vs individual-shuffled control, (2) receiver behavioral shift vs muted playback, and (3) causal survival/foraging impact via an identical twin fork with `signal_channel_enabled = False`.

5. **Phase F12d: Full-Resolution Spiking Brain + MuJoCo 6-Legged Body Replay (`replay_evolved_genome_in_full_spiking_brain`, `POST /api/v1/experiments/{run_id}/fly-world/replay`):**
   - Any evolved `fly_world` descendant's 24-float differential genome can be replayed both in CLI (`alife connectome evolve --replay-fine`, comparing compressed vs full spiking LIF side-by-side across all 4 reflex scenes) and from the browser 3D `FlyWorldView` via a 1-click **"Replay in full brain"** table that checkpoints the run and launches a live `fly_embodied` 3D MuJoCo + 138,639-cell glowing brain lab (`evolved_genome` on `FlyEmbodiedDynamics`, with an explicit F12d artificial-life framing banner).
   - Removed top-level `scipy` hard dependency in `fly_world.py` (pure-NumPy `_NumpyCSR` fallback + optional `scipy.sparse` / `cupyx.scipy.sparse` via `importlib`) so SciPy-free minimal CPU CI stays 100% green.

## Measured Results (RTX 5070, Real FlyWire v783 Ancestor, 2026-11-23)

- **Command:** `alife connectome evolve flywire_783 --seeds 1,2,3 --evo-ticks 1200 --eval-ticks 400 --replay-fine --no-verify` (`25.80 s` wall, GPU CuPy SpMM + CUDA graph full-brain replay)
- **Compressed brain fidelity:** `138,639 -> 17,641` nodes (`7.86×`), `base=0.0 Hz`, `sugar=220.0 Hz`, `bitter=0.0 Hz`, `poke_DNg84=97.4 Hz` (**PASS**).
- **Evolution & drift (3 seeds):** `mean_births = 14.0`, `mean_fly_distance = 0.3789` (**PASS**).
- **Signal emergence Criterion #1 (Meaning vs Shuffled):** **`MI = 0.2697 bits`** vs shuffled **`0.0005 bits`** (**539×** above chance, **PASS**).
- **Signal emergence Criterion #2 (Receiver Response):** **`0.507`** (**PASS**).
- **Signal emergence Criterion #3 (Causal Channel-Blocking Fork — Honest Negative Result at 1,200 ticks):** `fork_intact_food = 7.984` vs `fork_muted_food = 10.576`. Over a short 1,200-tick horizon, context-correlated signals and receiver responses emerge reliably, but muting the channel does not yet reduce group foraging (early unoptimized receiver steering and metabolic signaling costs outweigh recruitment benefits). Demonstrating positive causal group-level signal utility requires longer multi-thousand-tick runs and remains an explicit open experiment for E3 validation.
- **Phase F12d Full-Brain Spiking Replay (Gen 10 descendant `#84`, `fly_distance = 0.4697`, `138,639` spiking FlyWire cells on GPU `cupy`, `1.79 s` wall):**
  - Compressed (`17,641` nodes): `base = 0.0 Hz`, `sugar = 99.9 Hz`, `sugar+bitter = 0.3 Hz`, `poke_DNg84 = 126.5 Hz`.
  - Full spiking LIF (`138,639` cells): `base = 0.0 Hz`, `sugar = 86.0 Hz` (`Δ = -13.9 Hz`), `sugar+bitter = 12.0 Hz` (`Δ = +11.7 Hz`, `86%` suppression), `poke_DNg84 = 72.0 Hz` (`Δ = -54.5 Hz`).
  - **Qualitative agreement: 4/4 PASS**, with quantitative rate differences (analytical steady-state f-I saturating higher than stochastic Poisson spiking LIF) transparently recorded.
