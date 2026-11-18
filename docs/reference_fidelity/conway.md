# Reference fidelity — Conway's Game of Life (`conway`, plugin 0.1.0)

## Reference
Conway's Game of Life as popularised by M. Gardner (Scientific American, 1970):
synchronous update on a square lattice, Moore neighbourhood, rule B3/S23.

## Implementation
- `src/alife_lab/substrates/ca/conway.py`, NumPy, `uint8` grid, exact integer dynamics.
- Neighbour counts via `np.roll` (periodic) or zero padding (`dead` boundary).
- Scalar pure-Python oracle `step_grid_oracle` used only by tests.

## Verified (tests in `tests/reference`, `tests/property`)
| Property | Result |
|---|---|
| Glider translates (+1,+1) every 4 ticks, 5 cells, 8 periods | pass |
| Glider not equal to any small translate at ticks 1–3 | pass |
| Glider returns to start after 4·N ticks on an N-torus | pass |
| Blinker period 2, block still life (both boundaries) | pass |
| NumPy == scalar oracle: R-pentomino 60 ticks; 150 random grids × 5 rules × 2 boundaries (Hypothesis) | pass |
| Periodic step is translation-equivariant | pass |
| Observation (metrics, render) does not change the trajectory | pass |
| Checkpoint/restore across controller restart is bit-identical to a continuous run | pass |

## Deviations / extensions (explicit)
- Rules other than B3/S23 (Life-like outer-totalistic) are allowed **only** in
  `exploratory` mode.
- `dead` boundary is a finite-world approximation of the infinite plane (cells outside
  are permanently 0); behaviour near edges differs from the infinite-plane reference.
- Brush interventions near the edge are clipped, not wrapped, even on a torus.
- Metric `glider_displacement` (listed in the reference scenario) is an alias of
  `centroid_displacement`: the norm of the unwrapped live-cell centroid displacement
  (circular mean per axis on a torus; minimal-image per-tick deltas). It is **not a
  glider detector**; it equals glider displacement only when the world contains
  exactly one glider. It is reset after each intervention and can jump when cells
  appear/disappear (observed in E2E after a brush stroke).
- Randomness is consumed only at initialization (`numpy` PCG64 seeded with `seed`).

## Negative control
Preset `conway_block` (still life): centroid displacement stays 0
(`test_negative_control_block_has_zero_displacement`).

## Evidence level
No phenomenon claims are attached to this substrate (E0 visual only in the GUI).
