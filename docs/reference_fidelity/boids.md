# Reference fidelity — Boids (`boids`, plugin 0.1.0)

## Status: exploratory only
Reynolds (1987, doi:10.1145/37402.37406) describes the separation / alignment / cohesion
behaviours conceptually; there is no canonical numerical reference implementation to
reproduce. The spec therefore only accepts `mode: exploratory`, and no result from this
substrate is presented as a faithful reproduction of a published numerical model.

## Formulation (project-specific, documented)
Per tick, neighbours *j* of boid *i* are those within `perception_radius` (minimal image on
a torus). With `d_ij = x_j − x_i`:

- separation `s_i = Σ_{0 < |d_ij| < r_sep} −d_ij / |d_ij|²`
- alignment  `a_i = mean_j v_j − v_i` (only if *i* has neighbours)
- cohesion   `c_i = mean_j d_ij` (only if *i* has neighbours)
- `v_i ← v_i + dt (w_s s_i + w_a a_i + w_c c_i) [+ N(0, noise²)]`, then `|v_i|` clamped to
  `[min_speed, max_speed]` (a zero velocity becomes `(min_speed, 0)`), `x_i ← x_i + dt v_i`.
- Boundaries: periodic (wrap) or reflect (mirror position, flip the velocity component).
- Synchronous update: all accelerations use the state at the start of the tick.

The rules are scripted local behaviour. Flock-level order is the *emergent observable*; the
rules themselves are not evidence of emergence.

## Numerics
float64 NumPy on CPU. Neighbour search: uniform cell list with cell size ≥ perception radius
(world must be ≥ 3 radii wide/high). Results are bitwise deterministic for the same NumPy
build/platform; this is verified per run by replay verification, not assumed across builds.

## Verified (tests/reference/test_boids.py, tests/integration/test_boids_platform.py)
| Property | Result |
|---|---|
| Cell-list neighbour pairs == O(N²) oracle (60 random cases, both boundaries) | pass |
| Steering from cell-list pairs == oracle pairs (rtol 1e-12) | pass |
| Speeds stay within [min, max]; positions stay in the world (both boundaries, with noise) | pass |
| Same seed ⇒ identical trajectory; checkpoint mid-run (incl. RNG) ⇒ identical continuation | pass |
| Random-heading polarization ≈ sqrt(pi/(4N)) at initialization (20 seeds) | pass |
| Worker restart + restore + replay verification with noise and a spawn intervention | pass |

## Negative control (alignment)
300 boids, 200×200 torus, 600 ticks, 6 seeds per condition (measured 2026-11-18):

| alignment_weight | final polarization (6 seeds) |
|---|---|
| 0.05 | 0.848, 0.514, 0.786, 0.605, 0.226, 0.646 |
| 0.0 | 0.015, 0.073, 0.052, 0.053, 0.045, 0.008 |

Without alignment the polarization stays at the random baseline (√(π/4N) ≈ 0.051), and the
mean neighbour count stays near the uniform-density expectation (≈ 2.4 vs 32.8 with
alignment). Complete separation of 6 vs 6 seeds (one-sided exact Mann-Whitney p = 1/924).
This is a validation of the implementation against a well-known qualitative effect, not a
discovery claim. Several coexisting flocks with different headings explain values < 1.

## Comparison metric
`rms_position_distance`: RMS minimal-image position difference over boids present in both
runs (matched by stable id); `only_in_first` / `only_in_second` count unmatched ids.
`exact_equal` requires identical ids, positions and velocities.

## Interventions
`spawn {x, y, radius, count ≤ 500}` (positions uniform in the disc, random headings at the
mean speed, drawn from the run's RNG — replayable) and `remove {x, y, radius}`. Population is
capped at 10 000.
