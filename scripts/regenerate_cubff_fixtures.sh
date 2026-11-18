#!/usr/bin/env bash
# Rebuild CuBFF (CPU) at the pinned commit in a scratch directory and print the soup digests
# used by tests/fixtures/cubff_soups.json. Third-party code: review before running.
set -euo pipefail
COMMIT=f212e849027c98fcf4b242eccfb5fed435223e23
WORK=${1:?usage: $0 <empty scratch dir>}
git clone -q https://github.com/paradigms-of-intelligence/cubff.git "$WORK"
cd "$WORK" && git checkout -q "$COMMIT"
make CUDA=0 -j8   # needs g++, OpenMP and libbrotli-dev
for cfg in "bff_noheads 1024 7 0.000244140625" "bff 512 3 0.000244140625" "bff_noheads 256 11 0.01"; do
  set -- $cfg
  out="out_$1_$2_$3"
  ./bin/main --lang "$1" --num "$2" --seed "$3" --mutation_prob "$4" --max_epochs 130 \
    --checkpoint_dir "$out" --save_interval 64 --print_interval 64 --disable_output
  for f in "$out"/*.dat; do
    printf '%s %s ' "$cfg" "$f"; tail -c +25 "$f" | sha256sum
  done
done
