#!/usr/bin/env bash
# Run all project tests.
set -euo pipefail

python3 -m _mctx._src.tests.policies_test
python3 -m _mctx._src.tests.qtransforms_test
python3 -m _mctx._src.tests.seq_halving_test
