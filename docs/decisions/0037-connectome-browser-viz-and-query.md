# ADR 0037: Connectome Browser Observation, Cell Query and 3D Point-Cloud Visualization (F2)

- Status: Accepted
- Date: 2026-11-21
- Related: ADR 0035 (connectome species-pack contract), ADR 0036 (bit-identical CPU/CUDA LIF engine), `docs/plans/connectome_plan.md`

## Context

Phase F2 requires observing and operating whole-brain connectome simulations (up to 138,639 neurons in FlyWire v783, plus the offline 4-neuron `synthetic_tiny` circuit) from the browser without violating the core boundaries in `AGENTS.md`:

1. The FastAPI API process must remain CPU-only and fast to start (never import CuPy/CUDA and avoid loading the 100 MiB `Connectivity_783.parquet` edge table to serve cell search or 3D positions).
2. Live preview streaming uses a 1 MiB WebSocket frame limit and may drop previews under load, whereas deterministic replay (`execution/replay.py`) requires `adapter.render()` to be a pure, byte-for-byte deterministic function of `LIFState`.
3. Presentation views (including the 3D glowing brain) must never mutate scientific state.

## Decision

### 1. Lightweight CPU-only query & geometry index (`src/alife_lab/connectome/query.py`)

- Factor `load_flywire_783_cells()` out of `load_flywire_783()` so the API process loads only cell metadata, IO ports and 3D FAFB coordinates on demand and caches a `ConnectomeIndex` per `(dataset_id, datasets_dir)`.
- Expose three read-only endpoints under `/api/v1/connectome/{dataset_id}/`:
  - `GET /meta`: dataset metadata, role counts, brain region counts and named IO ports.
  - `GET /cells`: paginated filtering by text query (`cell_type`, `super_class`, `cell_class`, `region`, `source_id`), `region`, `role` and `port`.
  - `GET /geometry`: binary response (`connectome_geom_v1`) fetched once per dataset and cached in the browser. Coordinates are centered, oriented dorsal-up (`-y`), normalized to `[-1, 1]` and quantized to little-endian `int16` (`6 bytes/cell`), followed by `uint8[N]` role and `uint8[N]` region palette index (`1.11 MB` total for all 138,639 FlyWire neurons).

### 2. Compact deterministic preview codec (`connectome_v1`)

- Replace the F1 placeholder 64×64 `grid_f32` preview in `ConnectomeAdapter.render()` with `connectome_v1`:
  - JSON header: `dataset_id`, `num_cells`, `tick`, `time_ms`, `spikes_tick`, `active_neurons_tick`, `fired_count`, `stimulated_count`, `silenced_count`, per-port firing summaries (`ports`) and top-16 brain-region firing intensities (`regions`).
  - Binary body: exact sorted little-endian `int32` indices of all neurons that fired during the current tick (~1–8 KB/frame in typical regimes, well below the 1 MiB WebSocket cap).
- Because the output depends strictly on `LIFState` and immutable pack metadata, `serialize_state → deserialize_state → render()` stays byte-identical for deterministic replay keyframe verification.

### 3. Presentation views and interactive controls

- **2D measurement view (`ConnectomeView.tsx`)**: rolling spike raster plot (neurons × time), top-region firing-intensity bars and highlighted IO port rates (MN9, sugar, bitter, grooming).
- **3D glowing brain (`Brain3D.tsx`, lazy-loaded Three.js chunk)**: single `THREE.Points` GPU geometry placing all `N` neurons at their FAFB 3D positions with a custom GLSL shader, role/region coloring, OrbitControls and smooth live glow decay (exact per-tick spikes during replay scrubbing).
- **Controls & search (`ConnectomePanel.tsx`)**: named IO port stimulation/silencing/clear-all actions, paginated neuron search with per-cell inspect/stimulate/silence actions, highlighted firing-rate chart (`MultiLine`) and explicit framing notice ("hypothesis-testing wiring model, not a biological brain").
- **Catalog, guide & i18n**: five connectome presets (`connectome_tiny_circuit`, `connectome_fly_sugar`, `connectome_fly_sugar_bitter`, `connectome_fly_grooming`, `connectome_fly_shuffled_control`), beginner guide section, parameter/metric `?` help, glossary terms and full `ja`/`en` key parity.

## Consequences

- Browser observation scales cleanly from the 4-neuron offline circuit to all 138,639 FlyWire v783 neurons without saturating the WebSocket or loading GPU libraries in the API server.
- Future species packs (S1–S7) and chimeras (F13) can reuse the exact same `connectome_v1` codec, `connectome_geom_v1` geometry format and browser views as long as they populate `CellTable`, `IOPort` and 3D coordinates.
