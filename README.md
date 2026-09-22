# Noise-Aware Observable Pruning in PQKs — Phase 1

    pip install qiskit qiskit-aer scikit-learn
    python run_baseline.py        # noiseless K_exact: KTA + SVM accuracy, results/baseline.json

Modules
- `dataset_loader.py` — synthetic N=6 (linear topology, signal on (0,1),(1,2),(3,4),(4,5)) and PCA-reduced Breast Cancer. Scaled per dataset: synthetic to [-1,1], breast cancer to [-pi,pi] (see comment in file).
- `pqk.py` — feature map Ry(x)Rz(x) + adjacent CNOT chain (optional distant entanglers), 3N + 9·C(N,2) observables, `exact_features` (statevector) and `sampled_features` (27 orthogonal-array measurement settings, transpiled to a coupling map, run on AerSimulator with a noise model → the only path on which readout/crosstalk/SWAP noise acts).
- `kernel.py` — Gaussian projected kernel, median-heuristic gamma, centered KTA, per-observable KTA (train fold only), precomputed-kernel SVM, Frobenius distance, `prune`.

- `noise_experiments.py` — `HardwareProfile` (readout errors, CX error, crosstalk; shaped to be filled from `backend.properties()` later) with `W_hardware`; Exp A/B/C; pruning ablation (random / KTA-only / W-only / KTA·W vs keep fraction); readout-noise dose–response. Readout noise is applied as a classical confusion process on sampled bitstrings (`pqk.ReadoutNoise`), which is what lets Exp C compare simultaneous vs staggered readout.

    python noise_experiments.py --quick                          # ~4 min smoke test, subsampled
    python noise_experiments.py                                  # full run, both datasets (~30-40 min)
    python noise_experiments.py --exp dose --seeds 42 43 44 45 46   # dose-response with 5 seeds
    python noise_experiments.py --exp budget --seeds 42 43 44       # fixed-shot-budget sweep

Filters in the ablation: `random`, `kta_only`, `w_only`, `kta_w` (workplan v1: raw product),
`kta_w_rank` (v2a: product of normalised ranks), `kta_w_beta` (v2b: KTA·W^β, β picked on a validation fold).
The shot-budget sweep holds total shots per data point fixed and uses a greedy set cover of measurement
settings (`pqk.measurement_jobs(min_settings=True)`), so pruning buys more shots per kept observable.

Outputs land in `results/*.json` and `figures/*.png`. Quick-mode numbers are on 60/30 samples and are only for checking the pipeline runs.

Next: summary notebook (Plots 1-3 of the workplan) once full runs exist.
