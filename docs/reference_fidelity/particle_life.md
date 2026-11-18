# Reference fidelity — Particle Life (`particle_life`, plugin 0.1.0)

## Status: exploratory only
A community model (J. Ventrella's "Clusters"; T. Mohr's "particle-life") without a canonical
paper or reference implementation. The spec accepts only `mode: exploratory`.

## Formulation (project-specific, documented)
Typed particles (types 0..K−1), interaction matrix `M[a][b] ∈ [−1, 1]` (row = own type).
For neighbours within `r_max` (minimal image), r = |d|/r_max, u = d/|d|, a = M[type_i][type_j]:

- `F(r, a) = r/β − 1` for r < β (universal short-range repulsion);
  `a · (1 − |2r − 1 − β| / (1 − β))` for β ≤ r < 1; zero beyond r_max.
- `acc_i = force_factor · r_max · Σ_j F(r_ij, a_ij) u_ij`
- `v ← v · 0.5^(dt / friction_half_life) + dt · acc`; `x ← x + dt · v`; periodic or reflect.
- Synchronous update; asymmetric M means momentum is not conserved (model property).
- `matrix: null` draws M uniformly from [−1, 1] with the run's RNG at initialization; the
  realised matrix is stored in the state, checkpoints and shown by the inspector.

## Verified (tests/reference/test_particle_life.py, tests/integration/test_particle_life_platform.py)
| Property | Result |
|---|---|
| Force profile values at r = 0, β/2, β, (1+β)/2, 1⁻ | pass |
| Cell-list accelerations == scalar O(N²) oracle (25 random cases, both boundaries) | pass (rtol 1e-9) |
| Finite state, particles stay in the world; checkpoint round trip incl. RNG ⇒ identical continuation | pass |
| Seeded matrix reproducible; spec rejects wrong matrix shape / out-of-range entries | pass |
| Worker restart + restore + exact replay verification with a typed spawn intervention | pass |

## Negative control (type-dependent attraction)
600 particles, 4 types, 200² torus, 800 ticks (sweep `configs/sweeps/particle_life_sorting.yaml`,
6 seeds per condition, measured 2026-11-18):

| matrix | segregation_index (mean ± sd) |
|---|---|
| all zeros (repulsion only) | 0.2536 ± 0.0069 (well-mixed baseline Σp_k² ≈ 0.25) |
| same-type attraction (diag 1, off −0.2) | 0.6417 ± 0.0315 |

Δ = 0.388 (bootstrap 95 % CI [0.363, 0.410]), Cliff's δ = 1, exact permutation p = 2/924.
A validation of the implementation against the expected qualitative effect, not a discovery.

## Comparison metric / interventions
`rms_position_distance` over matched particle ids (as for Boids). Interventions:
`spawn {x, y, radius, count, type?}` (types random from the run's RNG if omitted), `remove`.
