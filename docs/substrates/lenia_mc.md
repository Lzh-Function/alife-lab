# Substrate contract — `lenia_mc`

- State: `cells` float64[C, H, W] in [0, 1] (C ≤ 4 channels, square periodic world), `tick`,
  circular-centroid tracker of the channel sum. Kernel FFTs are derived from the spec on load.
- Modes: reference (catalogued creatures with their own rule) and exploratory. Clock: tick.
  Backend: `numpy` (rFFT, float64). A batched PyTorch float64 implementation
  (`torch_batch.rollout`) is used only inside search runs (backends `torch` / `torch_cuda`).
- Spec: `world.size`; `dynamics` `channels`, `R`, `T`, `kernels` [{b, m, s, h, r, c0, c1}]
  (≤ 32). Omitted dynamics are filled from the pattern, so the canonical spec lists every value.
- Initialisation: `pattern` (5N7KKM, 5N7KKM_2, 5N7KKM_g, VT049W; padded to 32 × 32 and centred
  as Leniabreeder), `random` (seeded `size`² patch per channel with `density`), `empty`.
- Interventions: `paint {x, y, radius, value, channel?}`, `erase {x, y, radius, channel?}`.
- Inspection `{x, y}`: values and next values per channel, potential U and growth G per kernel.
- Metrics: `mass`, `alive_fraction` (channel sum > 0.1), `max_value`, `centroid_displacement`.
- Display `field_u8`, 3 channels: channels 0/1/2 as R/G/B.
- Comparison: `max_abs_difference` with `exact_equal`.
