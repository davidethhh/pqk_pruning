# Noise-Aware Observable Pruning in PQKs — Phase 1

    pip install qiskit qiskit-aer scikit-learn
    python run_baseline.py        # noiseless K_exact: KTA + SVM accuracy, results/baseline.json

Modules
- `dataset_loader.py` — synthetic N=6 (linear topology, signal on (0,1),(1,2),(3,4),(4,5)) and PCA-reduced Breast Cancer. Scaled per dataset: synthetic to [-1,1], breast cancer to [-pi,pi] (see comment in file).
- `pqk.py` — feature map Ry(x)Rz(x) + adjacent CNOT chain (optional distant entanglers), 3N + 9·C(N,2) observables, `exact_features` (statevector) and `sampled_features` (27 orthogonal-array measurement settings, transpiled to a coupling map, run on AerSimulator with a noise model → the only path on which readout/crosstalk/SWAP noise acts).
- `kernel.py` — Gaussian projected kernel, median-heuristic gamma, centered KTA, per-observable KTA (train fold only), precomputed-kernel SVM, Frobenius distance, `prune`.

Next: `noise_experiments.py` (Exp A/B/C + hardware-weighted pruning) and the summary notebook.
