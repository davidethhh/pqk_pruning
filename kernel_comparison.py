"""Why doesn't noise hurt accuracy?  Kernel comparison + kernel-matrix diagnostics.

    python kernel_comparison.py                       # both datasets, doses 0/1/4/16, 3 seeds
    python kernel_comparison.py --quick               # smoke test
    python kernel_comparison.py --dataset synthetic --doses 0 1 4 --seeds 42

Four kernels, same feature map U(x) = Ry(x)Rz(x) + adjacent CNOTs, same noise model:
  classical   RBF on the raw (angle-scaled) inputs — no quantum circuit, unaffected by noise by definition
  fidelity    K(x,y) = |<psi(x)|psi(y)>|^2 estimated from the compute-uncompute circuit U(y)^dag U(x),
              P(all zeros), one circuit per pair, shot-sampled with gate + readout noise
  pqk_full    projected quantum kernel on all 1- and 2-local Paulis
  pqk_pruned  PQK on the top 25% observables by rank(KTA)·rank(W)

Diagnostics on every noisy quantum kernel vs its noiseless counterpart:
  spearman / pearson   rank and linear correlation of the off-diagonal kernel entries
  subspace_overlap_k   ||U_k^T V_k||_F^2 / k for the top-k eigenvectors (k=10)
  decision_corr        correlation of the SVM decision values on test points (exact model vs noisy model)
  pred_flip_frac       fraction of test predictions that differ between the exact and the noisy model
  margin_frac_lt_0.5   fraction of test points inside |f(x)| < 0.5 under the exact model (how much slack there is)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.stats import pearsonr, spearmanr
from sklearn.svm import SVC

from dataset_loader import load_dataset
from kernel import gaussian_kernel, kta, median_gamma, per_observable_kta, prune
from noise_experiments import HardwareProfile, rank_product
from pqk import ReadoutNoise, _memory_to_bits, build_observables, exact_features, feature_map, linear_coupling_map, sampled_features

RESULTS, FIGURES = Path("results"), Path("figures")


# ============================================================================ fidelity kernel
def fidelity_kernel_exact(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    SA = np.array([Statevector(feature_map(x)).data for x in XA])
    SB = np.array([Statevector(feature_map(x)).data for x in XB])
    return np.abs(SA.conj() @ SB.T) ** 2


def _overlap_circuit(x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
    n = len(x)
    qc = QuantumCircuit(n, n)
    qc.compose(feature_map(x), inplace=True)
    qc.compose(feature_map(y).inverse(), inplace=True)
    qc.measure(range(n), range(n))
    return qc


def fidelity_kernel_sampled(XA: np.ndarray, XB: np.ndarray, *, noise_model=None, readout_noise: ReadoutNoise | None = None,
                            shots: int = 500, seed: int = 7, symmetric: bool = False, batch: int = 2000) -> np.ndarray:
    """P(all zeros) of U(y)^dag U(x), transpiled to the linear coupling map, with gate
    noise in Aer and readout noise applied to the sampled bitstrings."""
    n = XA.shape[1]
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(len(XA)) for j in range(len(XB)) if not symmetric or j >= i]
    K = np.zeros((len(XA), len(XB)))
    group = tuple(range(n))
    for b0 in range(0, len(pairs), batch):
        chunk = pairs[b0:b0 + batch]
        circs = [_overlap_circuit(XA[i], XB[j]) for i, j in chunk]
        tcircs = transpile(circs, coupling_map=linear_coupling_map(n), basis_gates=["rz", "sx", "x", "cx"],
                           initial_layout=list(range(n)), optimization_level=1, seed_transpiler=seed)
        res = sim.run(tcircs, shots=shots, memory=True).result()
        for c, (i, j) in enumerate(chunk):
            bits = _memory_to_bits(res.get_memory(c), n)
            if readout_noise is not None:
                bits = readout_noise.apply(bits, group, rng)
            K[i, j] = np.mean(bits.sum(axis=1) == 0)
            if symmetric:
                K[j, i] = K[i, j]
    return K


# ============================================================================ diagnostics
def kernel_diagnostics(K_noisy_tr, K_exact_tr, K_noisy_te, K_exact_te, y_tr, y_te, k: int = 10) -> dict:
    iu = np.triu_indices_from(K_exact_tr, k=1)
    sp = spearmanr(K_noisy_tr[iu], K_exact_tr[iu]).correlation
    pe = pearsonr(K_noisy_tr[iu], K_exact_tr[iu])[0]
    _, U = np.linalg.eigh(K_exact_tr); _, V = np.linalg.eigh(K_noisy_tr)
    Uk, Vk = U[:, -k:], V[:, -k:]
    overlap = np.linalg.norm(Uk.T @ Vk, "fro") ** 2 / k
    m_ex = SVC(kernel="precomputed", C=1.0).fit(K_exact_tr, y_tr)
    m_no = SVC(kernel="precomputed", C=1.0).fit(K_noisy_tr, y_tr)
    f_ex, f_no = m_ex.decision_function(K_exact_te), m_no.decision_function(K_noisy_te)
    return {"spearman": float(sp), "pearson": float(pe), f"subspace_overlap_{k}": float(overlap),
            "decision_corr": float(pearsonr(f_ex, f_no)[0]),
            "pred_flip_frac": float(np.mean(np.sign(f_ex) != np.sign(f_no))),
            "margin_frac_lt_0.5": float(np.mean(np.abs(f_ex) < 0.5)),
            "acc_exact_model": float(m_ex.score(K_exact_te, y_te)), "acc_noisy_model": float(m_no.score(K_noisy_te, y_te)),
            "kta_exact": kta(K_exact_tr, y_tr), "kta_noisy": kta(K_noisy_tr, y_tr),
            "frobenius_rel": float(np.linalg.norm(K_noisy_tr - K_exact_tr) / np.linalg.norm(K_exact_tr))}


def svm_acc(K_tr, K_te, y_tr, y_te) -> float:
    return float(SVC(kernel="precomputed", C=1.0).fit(K_tr, y_tr).score(K_te, y_te))


# ============================================================================ main comparison
def compare(dataset: str, hw: HardwareProfile, *, doses=(0.0, 1.0, 4.0, 16.0), seeds=(42,), quick=False,
            n_train_cap=200, n_test_cap=100, pqk_shots=2000, fid_shots=500, keep_fraction=0.25) -> dict:
    n = hw.n
    obs = build_observables(n)
    w = hw.weights(obs)
    out = {"dataset": dataset, "doses": list(doses), "seeds": list(seeds), "rows": []}
    for dose in doses:
        row = {"dose": dose, "acc": {}, "kta": {}, "diag": {}}
        accs = {k: [] for k in ("classical", "fidelity", "pqk_full", "pqk_pruned")}
        ktas = {k: [] for k in accs}
        diags = {k: [] for k in ("fidelity", "pqk_full", "pqk_pruned")}
        t = time.time()
        for sd in seeds:
            X_tr, X_te, y_tr, y_te = load_dataset(dataset, n_qubits=n, seed=sd)
            if quick:
                X_tr, y_tr, X_te, y_te = X_tr[:40], y_tr[:40], X_te[:20], y_te[:20]
            else:
                X_tr, y_tr, X_te, y_te = X_tr[:n_train_cap], y_tr[:n_train_cap], X_te[:n_test_cap], y_te[:n_test_cap]
            nm, ro = hw.gate_noise_model(), hw.readout_noise(dose)

            # classical
            g = median_gamma(X_tr)
            Kc_tr, Kc_te = gaussian_kernel(X_tr, X_tr, g), gaussian_kernel(X_te, X_tr, g)
            accs["classical"].append(svm_acc(Kc_tr, Kc_te, y_tr, y_te)); ktas["classical"].append(kta(Kc_tr, y_tr))

            # fidelity
            Kf_ex_tr, Kf_ex_te = fidelity_kernel_exact(X_tr, X_tr), fidelity_kernel_exact(X_te, X_tr)
            fs = 100 if quick else fid_shots
            Kf_tr = fidelity_kernel_sampled(X_tr, X_tr, noise_model=nm, readout_noise=ro, shots=fs, seed=sd, symmetric=True)
            Kf_te = fidelity_kernel_sampled(X_te, X_tr, noise_model=nm, readout_noise=ro, shots=fs, seed=sd + 1)
            accs["fidelity"].append(svm_acc(Kf_tr, Kf_te, y_tr, y_te)); ktas["fidelity"].append(kta(Kf_tr, y_tr))
            diags["fidelity"].append(kernel_diagnostics(Kf_tr, Kf_ex_tr, Kf_te, Kf_ex_te, y_tr, y_te))

            # PQK full and pruned
            ps = 500 if quick else pqk_shots
            Fx_tr, Fx_te = exact_features(X_tr, obs), exact_features(X_te, obs)
            kw = dict(noise_model=nm, readout_noise=ro, shots=ps, seed=sd)
            Fn_tr, Fn_te = sampled_features(X_tr, obs, **kw), sampled_features(X_te, obs, **kw)
            kta_scores = per_observable_kta(Fn_tr, y_tr, median_gamma(Fn_tr))
            keep = prune(rank_product(kta_scores, w), keep_fraction=keep_fraction)
            for name, idx in (("pqk_full", np.arange(len(obs))), ("pqk_pruned", keep)):
                g = median_gamma(Fx_tr[:, idx])
                Kx_tr, Kx_te = gaussian_kernel(Fx_tr[:, idx], Fx_tr[:, idx], g), gaussian_kernel(Fx_te[:, idx], Fx_tr[:, idx], g)
                Kn_tr, Kn_te = gaussian_kernel(Fn_tr[:, idx], Fn_tr[:, idx], g), gaussian_kernel(Fn_te[:, idx], Fn_tr[:, idx], g)
                accs[name].append(svm_acc(Kn_tr, Kn_te, y_tr, y_te)); ktas[name].append(kta(Kn_tr, y_tr))
                diags[name].append(kernel_diagnostics(Kn_tr, Kx_tr, Kn_te, Kx_te, y_tr, y_te))
            # noiseless references (recorded once per seed under dose row for convenience)
            row.setdefault("acc_exact", {}).setdefault("fidelity", []).append(svm_acc(Kf_ex_tr, Kf_ex_te, y_tr, y_te))
            row["acc_exact"].setdefault("pqk_full", []).append(diags["pqk_full"][-1]["acc_exact_model"])

        for k in accs:
            row["acc"][k] = {"mean": float(np.mean(accs[k])), "std": float(np.std(accs[k]))}
            row["kta"][k] = float(np.mean(ktas[k]))
        for k in diags:
            row["diag"][k] = {m: float(np.mean([d[m] for d in diags[k]])) for m in diags[k][0]}
        row["acc_exact"] = {k: float(np.mean(v)) for k, v in row["acc_exact"].items()}
        out["rows"].append(row)
        print(f"   dose {dose:>4}x  " + "  ".join(f"{k}={row['acc'][k]['mean']:.3f}" for k in accs)
              + f"   [exact: fid={row['acc_exact']['fidelity']:.3f} pqk={row['acc_exact']['pqk_full']:.3f}]  ({time.time() - t:.0f}s)")
        for k in diags:
            d = row["diag"][k]
            print(f"      {k:10s} spearman={d['spearman']:.3f} pearson={d['pearson']:.3f} subspace10={d['subspace_overlap_10']:.3f} "
                  f"decision_corr={d['decision_corr']:.3f} flips={d['pred_flip_frac']:.3f} frob={d['frobenius_rel']:.3f} kta={d['kta_noisy']:.3f}/{d['kta_exact']:.3f}")
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / f"compare_{dataset}.json", "w") as f:
            json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, st in (("classical", "k-"), ("fidelity", "s--"), ("pqk_full", "o-"), ("pqk_pruned", "^:")):
        axes[0].errorbar(doses, [r["acc"][k]["mean"] for r in out["rows"]], [r["acc"][k]["std"] for r in out["rows"]], fmt=st, label=k)
    axes[0].set_xlabel("readout-noise dose"); axes[0].set_ylabel("SVM test accuracy"); axes[0].legend(fontsize=7)
    for k, st in (("fidelity", "s--"), ("pqk_full", "o-"), ("pqk_pruned", "^:")):
        axes[1].plot(doses, [r["diag"][k]["spearman"] for r in out["rows"]], st, label=f"{k} spearman")
        axes[1].plot(doses, [1 - r["diag"][k]["pred_flip_frac"] for r in out["rows"]], st, alpha=0.4, label=f"{k} 1-flips")
    axes[1].set_xlabel("readout-noise dose"); axes[1].set_ylabel("rank corr. / prediction agreement"); axes[1].legend(fontsize=6)
    fig.suptitle(f"Kernel comparison under noise ({dataset})"); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"compare_{dataset}.png", dpi=150); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["synthetic", "breast_cancer"])
    ap.add_argument("--doses", type=float, nargs="+", default=[0.0, 1.0, 4.0, 16.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    hw = HardwareProfile()
    for ds in args.dataset:
        print(f"== compare / {ds}")
        compare(ds, hw, doses=tuple(args.doses), seeds=tuple(args.seeds), quick=args.quick)
