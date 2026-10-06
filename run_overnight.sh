#!/bin/bash
# Overnight kernel comparison: two fidelity-shot settings, 6 doses, 3 seeds (+2 at 4x and 16x).
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "=== START $(date)"
# A. matched total shot budget (fidelity ~270 shots/pair vs PQK 54,000/point)
caffeinate -i python kernel_comparison.py --tag _matched 2>&1 | grep -v Warning
echo "=== matched done $(date)"
# B. generous fidelity (2000 shots/pair, 7x the PQK budget) - if fidelity still collapses, the claim is safe
caffeinate -i python kernel_comparison.py --tag _fid2000 --fid-shots 2000 2>&1 | grep -v Warning
echo "=== END $(date)"
