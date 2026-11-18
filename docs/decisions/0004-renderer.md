# ADR 0004 — Canvas2D renderer for Phase 1 (deviation from PixiJS)

- Status: accepted for Phase 1
- Date: 2026-11-18

Binary grids up to 1024² are rendered by writing an `ImageData` into an offscreen
canvas and scaling with smoothing disabled, scheduled by `requestAnimationFrame`
outside React state. This avoids a WebGL dependency for the first slice. PixiJS /
WebGL2 will be introduced with multi-channel continuous fields (Lenia) where it
matters; the renderer is isolated in `frontend/src/components/WorldCanvas.tsx`.
