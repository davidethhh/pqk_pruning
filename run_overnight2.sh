#!/bin/bash
# Four-kernel comparison (classical / fidelity / PQK full / PQK kta·W / PQK W-only) on REAL
# calibration chains, matched shot budget, 5 seeds.  Datasets with headroom only.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "=== START $(date)"
caffeinate -i python -u kernel_comparison.py --tag _realchains \
  --dataset sonar ionosphere synthetic \
  --profiles sim:0 results/chain_scan_ibm_kingston.json:median results/chain_scan_ibm_marrakesh.json:median \
             results/chain_scan_ibm_marrakesh.json:p90 results/chain_scan_ibm_marrakesh.json:worst \
  --seeds 42 43 44 45 46 --extra-seeds 2>&1 | grep -v -i warning
echo "=== END $(date)"
