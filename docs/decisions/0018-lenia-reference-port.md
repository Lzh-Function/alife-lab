# ADR 0018 — Lenia: port LeniaND's CPU update path; field display codec

Status: accepted (2026-11-18)

## Context
Phase 3 needs Lenia with a scientific reference. Chan's LeniaND.py is MIT licensed and the
project owner allowed code reuse with attribution. Its default path is short and its result
depends on details (kernel centred at `size // 2`, `fftshift` after the inverse FFT, clip).

## Decision
- Port the core functions (kernel cores, growth functions, kernel shell, 2D `calc_kernel` /
  `calc_once` default path, RLE decoder, Orbium entry) into
  `substrates/lenia/reference.py` with the licence text, and build the adapter on them.
- float64 + `numpy.fft` is the only backend (`numeric_policy: float64_fft_cpu`). GPU/float32
  backends must be added as separate numeric policies and compared against this oracle.
- Fidelity check: `scripts/compare_lenia_upstream.py` runs the original LeniaND.py CPU path
  side by side; bitwise equal for 1000 steps (128²) after fixing the complex multiply order.
- Independent check: an O(N·K) direct periodic convolution oracle in tests, not derived from
  the FFT code.
- New display codec `field_u8` (one byte per cell) rendered by `FieldCanvas` with a fixed
  colormap. Previews are lossy by design; checkpoints store the float64 field.
- API overrides are merged recursively (`deep_merge`, shared with sweeps) so that a form
  override such as `dynamics.kernel.R` does not drop sibling keys (`b`, `core`).

## Consequences
- Reference mode cannot use LeniaND options that are not ported (soft clip, multi-step,
  noise, masks, quantisation); the spec does not expose them.
- Flow-Lenia (mass-conserving) will be a separate substrate, not a mode of this adapter.
