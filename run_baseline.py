"""Noiseless reference baseline (K_exact): KTA and SVM accuracy on both datasets.

Also reports the per-observable KTA ranking on the synthetic dataset so we can
see whether the label-relevant pairs (0,1),(1,2),(3,4),(4,5) surface on their own.
"""
import json
import sys

import numpy as np

from dataset_loader import load_dataset
from kernel import evaluate_kernel, median_gamma, per_observable_kta
from pqk import build_observables, exact_features


def run(dataset: str, n_qubits: int = 6, seed: int = 42) -> dict:
    X_tr, X_te, y_tr, y_te = load_dataset(dataset, n_qubits=n_qubits, seed=seed)
    obs = build_observables(n_qubits)
    F_tr, F_te = exact_features(X_tr, obs), exact_features(X_te, obs)
    gamma = median_gamma(F_tr)
    res = evaluate_kernel(F_tr, F_te, y_tr, y_te, gamma)
    scores = per_observable_kta(F_tr, y_tr, gamma)
    order = np.argsort(-scores)
    return {
        "dataset": dataset, "n_qubits": n_qubits, "n_observables": len(obs),
        "gamma": gamma, "kta": res["kta"], "svm_accuracy": res["acc"],
        "top10_observables": [(obs[i].label, round(float(scores[i]), 4)) for i in order[:10]],
        "bottom5_observables": [(obs[i].label, round(float(scores[i]), 4)) for i in order[-5:]],
    }


if __name__ == "__main__":
    out = {}
    for ds in ("synthetic", "breast_cancer"):
        r = run(ds)
        out[ds] = r
        print(f"\n=== {ds} ===  M={r['n_observables']}  gamma={r['gamma']:.3f}")
        print(f"KTA(K_exact) = {r['kta']:.4f}   SVM test accuracy = {r['svm_accuracy']:.4f}")
        print("top-10 observables by single-feature KTA:", r["top10_observables"])
        print("bottom-5:", r["bottom5_observables"])
    with open("results/baseline.json", "w") as f:
        json.dump(out, f, indent=2)
