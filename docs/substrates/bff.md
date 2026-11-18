# Substrate contract — `bff`

- Clock: **epoch** (one pairing + execution pass over the whole soup). `runs.tick` stores it.
- State: `soup uint8[N, 64]`, epoch, last-epoch stats. RNG is stateless (SplitMix64 of seed and
  epoch), so checkpoints contain no RNG state and replay is exact.
- Modes: reference (CuBFF defaults for step limit and init) and exploratory.
- Interventions (explicit indices, replayable): `inject_program {indices, program, encoding:
  text|hex}` (≤ 64 bytes, NUL-padded), `zero_programs {indices}`.
- Inspect `{x, y}` = tile column/row in the GUI grid (`columns = ceil(sqrt(min(N, 4096)))`).
- Display codec `bff_soup`: header `{programs, total_programs, tape: 64, columns}`, payload the
  first min(N, 4096) programs.
- Metrics: `h0_bits_per_byte`, `brotli_bits_per_byte`, `higher_order_entropy`, `ops_per_pair`,
  `unique_programs`, `top_program_fraction`, `mutated_bytes`.
- Debugger: `GET /experiments/{id}/inspect?x&y&mode=trace_next_epoch|trace_self` adds a step
  trace to the program inspection (live or replayed tick); `POST /tools/bff/trace`
  `{program_a, program_b, encoding, variant, step_limit ≤ 8192}` traces a manual pair without a
  run. The trace is columnar (`pc`, `cmd`, `head0`, `head1`, `write_addr`, `write_old`,
  `write_new`, `next_pc`) plus `halt`, `ops`, initial/final tape and a summary.
