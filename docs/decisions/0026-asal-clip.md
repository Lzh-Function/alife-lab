# ADR 0026 — ASAL objectives with CLIP (PyTorch, optional extra `asal`)

Status: accepted (2026-11-20)

## Context
Spec §8 asks for foundation-model-guided search (ASAL: supervised target, open-endedness,
illumination) as a search layer, with the model ID, weight hash, preprocessing and prompts
recorded, and with controls against reward hacking. The project owner approved PyTorch +
open_clip on the GPU.

## Decision
- Optional extra `asal` = torch 2.15 (CUDA 13.2 wheels from PyPI), open_clip_torch 3.3,
  safetensors. One lock with `nca-gpu`: cuDNN resolves to 9.26 for both torch and JAX
  (JAX GPU tests re-run and pass). The CPU-only base install and CI do not need it.
- Model: OpenAI CLIP ViT-B/32 (the ASAL paper's model) from
  `timm/vit_base_patch32_clip_224.openai`, revision a6f597a3, only the `.safetensors` file
  (SHA-256 e6d1bd77…, 605 MB) is downloaded to `~/.cache/alife_lab/models` (or
  `ALIFE_MODEL_DIR`) and verified before loading; `torch.load`/pickle is never used.
- Scores: NumPy ports of `asal_metrics.py` (SakanaAI/asal @ 677ba0e, Apache-2.0), negated to
  be maximised. Objectives `lenia_clip_target` (prompts) and `lenia_clip_open_endedness`
  (frames ≥ 2) with an optional `asal` section (digest-stable when absent) and numeric policy
  `float32_clip_torch`. Illumination is implemented as a score but not yet as an algorithm.
- Backends `torch` (CPU) and `torch_cuda` for the search plugin, restricted to that policy;
  `numpy` accepts only `float64_cpu`. The `auto` selector now skips backends that refuse the
  spec's numeric policy. The GUI keeps non-CLIP searches on the CPU when the GPU switch is on.
- Controls: `blank_score` (score of an empty world, a level dead candidates get for free) is
  reported every generation; equal-budget random search; holdout seeds for random initial
  states. Deterministic algorithms are requested; replay verification passes on the GPU.
- Deviations from ASAL are listed in the scenario and the model card (single-channel Lenia
  shown as grey RGB, 3 box parameters, full CMA-ES, open_clip instead of Flax CLIP).

## Consequences
- Evidence from CLIP scores is E0–E1 and model-relative; prompt-specificity checks (other
  prompts) are needed before any claim (planned as a ledger claim kind).
- Planned: illumination / novelty archive over the stored embeddings, Sep-CMA-ES, multichannel
  Lenia, DINO/SigLIP evaluators.
