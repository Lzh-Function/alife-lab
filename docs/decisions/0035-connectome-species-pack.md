# ADR 0035 — Connectome species-pack contract and verified dataset storage (F0)

Status: accepted (2026-11-21)

## Context
The connectome experiment plan (`docs/plans/connectome_plan.md`) introduces real biological
wiring diagrams starting with the adult female *Drosophila melanogaster* brain (FlyWire v783,
Shiu et al. 2024) and extending to *C. elegans*, fly larva, *Platynereis*, *Ciona*, mouse
visual cortex, human cortex, and cross-species chimeras. Each upstream dataset uses different
file formats, ID schemes, coordinate units and cell taxonomies.

## Decision
1. **Species-pack common schema (`alife_lab.connectome.schema`):** every species loader
   normalizes upstream tabular files into four structures:
   - `CellTable` (`0..N-1` dense indices, `source_id`, `species_names`/`species_idx` for future
     chimeras, taxonomy, primary region, `role`, `cell_model`, neurotransmitter, and 3D
     `position_nm` with explicit `position_source`).
   - `EdgeTable` (`pre`, `post`, strictly positive integer `count`, `sign` in `{-1, 0, +1}`,
     `kind` in `{chemical, gap_junction}`), convertible into a canonical CSR graph sorted
     deterministically by `(pre, post)` ascending.
   - `IOPort` (named sensory/motor/circuit index populations with explicit selection rules).
   - `DatasetProvenance` (version, citation, licence note, pinned file URLs/sizes/SHA-256) plus
     a deterministic `pack_digest` hashed over all normalized arrays and metadata.
2. **Tabular-only, pickle-free policy:** `validate_filename_and_url` only permits HTTPS URLs
   and tabular extensions (`.parquet`, `.csv`, `.tsv`, `.csv.gz`, `.tsv.gz`, `.json`). Pickle
   files (such as upstream `sez_neurons.pickle`) are explicitly rejected.
3. **Out-of-Git verified storage:** datasets live under `ALIFE_DATASETS_DIR` (default
   `.alife/datasets/<dataset_id>/raw/`), outside Git, bounded by `ALIFE_MAX_DATASETS_MB`
   (default 4096 MB). Downloads stage into `staging/`, enforce exact pinned byte sizes and
   SHA-256 checksums, and publish via atomic `os.replace`. Upstream GitHub raw URLs are pinned
   to immutable commit SHAs.
4. **FlyWire v783 coordinate and region normalization:** soma voxel coordinates
   (`Supplemental_file1_neuron_annotations.tsv`) are converted to FAFB nanometres via
   `(4, 4, 40) nm`, falling back to anchor voxel coordinates (`pos_x/y/z`) and then Codex
   `coordinates.csv.gz` (already in nm), reaching 100% 3D coverage across all 138,639 neurons.
   Primary neuropil is taken from Codex `neuropil_synapse_table.csv.gz` (`argmax(in + out)`
   excluding `UNASGD`), falling back to Codex `neurons.csv.gz` `group`, then `UNASGD`.
5. **CLI and offline CI:** `alife connectome {list,fetch,verify,summary}` shares the registry
   and loader. Unit tests run offline on a 5-neuron synthetic fixture; reference F0 tests run
   against local `flywire_783` when present and skip cleanly on CPU-only CI without network.

## Consequences
- Future species packs (S1 *C. elegans* with graded neurons and gap junctions, S2–S7) and
  chimeras (F13) plug into the same `ConnectomePack` contract without changing F1+ simulation
  engines or UI consumers.
- Experiment specs in F1+ will record `(dataset_id, pack_digest)` so any upstream revision
  changes the scientific hash.
