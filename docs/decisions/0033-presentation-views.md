# ADR 0033 — GPU presentation views for fields and particles

Status: accepted (2026-11-21)

## Decision
- The live lab gets a view switch. `2d` stays the default and is the measurement view: it
  shows the display quantisation of the state directly and is the only view that takes pointer
  tools (paint, erase, inspect, spawn).
- Fields (`field_u8`: lenia, flow_lenia, lenia_mc, gray_scott, physarum, nca, pd_nca) add two
  three.js/WebGL2 views, lazily loaded:
  - `terrain`: height and colour from the value (multi-channel: height between the brightest
    channel and the mean, colour = channels as RGB), lighting, a faint reference grid and
    bloom.
  - `spacetime`: the last 96 received frames, box-downsampled to ≤ 128 per side, stacked as a
    3D texture (newest on top) and ray-marched with front-to-back compositing. A selectable
    interval stacks 1 of every k frames. Slices are received preview frames, which may be
    dropped; the tick range is shown, and slice spacing is not claimed to be constant.
- Particles (boids, particle_life) add `glow`: additive sprites with fading trails.
- The view is sized from the available column width and the window height (≥ 720 px,
  ≤ 1400 px), with a fullscreen button. The chosen view is remembered per substrate in
  localStorage (a convenience; never part of a run).

## Consequences
- Presentation only: no state, spec, metric or artifact changes, and no server work.
  Exaggerations (height, glow, opacity) are labelled in the UI.
- Rendering cost is on the viewer's GPU. Headless SwiftShader renders at ~2 fps, enough for the
  E2E checks.
