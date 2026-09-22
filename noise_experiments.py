"""Hardware noise experiments and the noise-aware pruning filter.

    python noise_experiments.py --exp A B C ablation dose --dataset synthetic breast_cancer
    python noise_experiments.py --quick          # small subsample, few shots: smoke test (~2 min)

Every noisy number here comes from `pqk.sampled_features` (explicit measurement
circuits, transpiled to the linear coupling map, shot-sampled).  Aer's Estimator
primitive is never used: it ignores readout error.

Experiments (workplan §3-4)
  A  SWAP overhead      gate noise + a distant (0,5) entangler -> |Δ_ij| vs topological distance
  B  non-uniform SPAM   readout 0.5% on qubits 0-2, 6.0% on 3-5 -> shot variance / bias per pair
  C  readout crosstalk  correlated neighbour flips -> simultaneous vs staggered readout
  ablation              random / KTA-only / W-only / KTA·W filters vs keep fraction
  dose                  full-vs-pruned gap as readout noise is scaled 0x .. 4x

Outputs: results/<exp>_<dataset>.json and figures/<exp>_<dataset>.png
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import RXGate, RZZGate
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error
from sklearn.kernel_ridge import KernelRidge

from dataset_loader import load_dataset
from kernel import (evaluate_kernel, frobenius_distance, gaussian_kernel, median_gamma,
                    per_observable_kta, prune)
from pqk import (Observable, ReadoutNoise, build_observables, exact_features,
                 measurement_jobs, sampled_features, staggered_groups)
from sklearn.model_selection import train_test_split

RESULTS = Path("results")
FIGURES = Path("figures")


# ============================================================================ hardware profile
@dataclass
class HardwareProfile:
    """The subset of a backend calibration that the pruning weight needs.

    Shaped so that it can be filled from qiskit `backend.properties()` in Phase 2:
    readout_error[q]  <- properties.readout_error(q)
    cx_error          <- mean properties.gate_error('cx', (i,j)) over the coupling map
    crosstalk[(i,j)]  <- from a readout-crosstalk characterisation (not in standard calibration)
    """
    n: int = 6
    readout_error: dict[int, float] = field(default_factory=lambda: {0: .005, 1: .005, 2: .005,
                                                                    3: .06, 4: .06, 5: .06})
    cx_error: float = 0.01
    sq_error: float = 0.001
    crosstalk: dict[tuple[int, int], float] = field(default_factory=lambda: {(0, 1): .10, (2, 3): .10, (4, 5): .10})

    # ---- simulation objects
    def readout_noise(self, factor: float = 1.0) -> ReadoutNoise:
        return ReadoutNoise.symmetric(self.readout_error, self.crosstalk).scaled(factor)

    def gate_noise_model(self, coherent: float = 0.0) -> NoiseModel:
        """Depolarizing gate noise; if `coherent` > 0, every sx/x is followed by an
        RX(coherent) over-rotation and every cx by an RZZ(coherent) — a systematic
        miscalibration that biases each Pauli feature differently, unlike
        depolarizing/readout noise which only attenuate."""
        nm = NoiseModel()
        if coherent > 0:
            nm.add_all_qubit_quantum_error(
                coherent_unitary_error(RXGate(coherent).to_matrix()).compose(depolarizing_error(self.sq_error, 1)), ["sx", "x"])
            nm.add_all_qubit_quantum_error(
                coherent_unitary_error(RZZGate(coherent).to_matrix()).compose(depolarizing_error(self.cx_error, 2)), ["cx"])
        else:
            nm.add_all_qubit_quantum_error(depolarizing_error(self.sq_error, 1), ["sx", "x"])
            nm.add_all_qubit_quantum_error(depolarizing_error(self.cx_error, 2), ["cx"])
        return nm

    # ---- pruning weight  W_hardware(O)
    def weight(self, o: Observable) -> float:
        """Expected signal retention of <O> under this profile.

        readout : each measured qubit attenuates a Pauli expectation by (1 - 2p_q)
        crosstalk: adjacent pair (i,j) with correlated flip rate c loses ~(1 - c)
        routing : a distance-d pair costs (d-1) SWAPs = 3(d-1) CX if the circuit has
                  to bring the qubits together; folded in as (1 - cx_error)^{3(d-1)}.
                  With an adjacent-only feature map this term is a proxy for the
                  *circuit* cost of using that pair, not a measurement cost.
        """
        w = float(np.prod([1 - 2 * self.readout_error.get(q, 0.0) for q in o.qubits]))
        if o.weight == 2:
            pair = tuple(sorted(o.qubits))
            w *= 1 - self.crosstalk.get(pair, 0.0)
            w *= (1 - self.cx_error) ** (3 * max(o.distance - 1, 0))
        return w

    def weights(self, observables: list[Observable]) -> np.ndarray:
        return np.array([self.weight(o) for o in observables])


# ============================================================================ helpers
def by_distance(observables: list[Observable], values: np.ndarray) -> dict[int, float]:
    """Mean of a per-observable quantity grouped by topological distance (0 = single-qubit)."""
    out = {}
    for d in sorted({o.distance for o in observables}):
        idx = [m for m, o in enumerate(observables) if o.distance == d]
        out[d] = float(np.mean(values[idx]))
    return out


def pair_label(o: Observable) -> str:
    return "single" if o.weight == 1 else f"({o.qubits[0]},{o.qubits[1]})"


def save(name: str, data: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2, default=float)


def load_split(dataset: str, n: int, seed: int, quick: bool):
    X_tr, X_te, y_tr, y_te = load_dataset(dataset, n_qubits=n, seed=seed)
    if quick:
        X_tr, y_tr, X_te, y_te = X_tr[:60], y_tr[:60], X_te[:30], y_te[:30]
    return X_tr, X_te, y_tr, y_te


# ============================================================================ Exp A
def exp_a(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seed: int = 42) -> dict:
    """SWAP overhead: with a (0,N-1) entangler the transpiler inserts N-2 SWAPs on the
    line; gate noise on those SWAPs corrupts every observable, and we ask whether
    the damage is ordered by topological distance.  The adjacent-only map (no
    SWAPs) under the same gate noise is the control."""
    n = hw.n
    X_tr, *_ = load_split(dataset, n, seed, quick)
    X = X_tr[: (20 if quick else 60)]
    obs = build_observables(n)
    nm = hw.gate_noise_model()
    out = {}
    for tag, extra in (("adjacent_only", None), ("with_0_5_entangler", [(0, n - 1)])):
        F_ex = exact_features(X, obs, extra_pairs=extra)
        F_no = sampled_features(X, obs, noise_model=nm, shots=shots, extra_pairs=extra, seed=seed)
        delta = np.abs(F_no - F_ex).mean(axis=0)
        out[tag] = {"delta_by_distance": by_distance(obs, delta),
                    "delta_per_observable": {o.label: float(d) for o, d in zip(obs, delta)}}
    out["cx_error"] = hw.cx_error
    save(f"expA_{dataset}", out)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for tag, style in (("adjacent_only", "o--"), ("with_0_5_entangler", "s-")):
        d = out[tag]["delta_by_distance"]
        ax.plot(list(d.keys()), list(d.values()), style, label=tag.replace("_", " "))
    ax.set_xlabel("topological distance |i - j|  (0 = single-qubit)")
    ax.set_ylabel("mean |<O>_noisy - <O>_exact|")
    ax.set_title(f"Exp A: SWAP overhead ({dataset})")
    ax.legend(); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"expA_{dataset}.png", dpi=150); plt.close(fig)
    return out


# ============================================================================ Exp B
def exp_b(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seed: int = 42) -> dict:
    """Non-uniform readout: bias and shot variance of every 2-qubit observable,
    grouped by which physical qubits it touches (low/low, low/high, high/high)."""
    n = hw.n
    X_tr, *_ = load_split(dataset, n, seed, quick)
    X = X_tr[: (20 if quick else 60)]
    obs = build_observables(n)
    ro = ReadoutNoise.symmetric(hw.readout_error)  # no crosstalk in B
    F_ex = exact_features(X, obs)
    F_no, V = sampled_features(X, obs, readout_noise=ro, shots=shots, seed=seed, return_variance=True)
    bias = np.abs(F_no - F_ex).mean(axis=0)
    var = V.mean(axis=0)
    per_pair = {}
    for m, o in enumerate(obs):
        per_pair.setdefault(pair_label(o), []).append((bias[m], var[m]))
    per_pair = {k: {"bias": float(np.mean([b for b, _ in v])), "shot_var": float(np.mean([s for _, s in v]))}
                for k, v in per_pair.items()}
    hi = {q for q, p in hw.readout_error.items() if p > 0.02}
    def cls(o):
        if o.weight == 1: return "single"
        k = sum(q in hi for q in o.qubits)
        return ["low-low", "low-high", "high-high"][k]
    by_class = {}
    for m, o in enumerate(obs):
        by_class.setdefault(cls(o), []).append((bias[m], var[m]))
    by_class = {k: {"bias": float(np.mean([b for b, _ in v])), "shot_var": float(np.mean([s for _, s in v]))}
                for k, v in by_class.items()}
    out = {"per_pair": per_pair, "by_class": by_class, "readout_error": hw.readout_error, "shots_per_setting": shots}
    save(f"expB_{dataset}", out)

    pairs = [k for k in per_pair if k != "single"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(pairs, [per_pair[k]["bias"] for k in pairs], color=["tab:blue" if cls_of(k, hi) == "low-low" else
                                                             "tab:orange" if cls_of(k, hi) == "low-high" else "tab:red" for k in pairs])
    ax.set_ylabel("mean |bias| of <P_i P_j>"); ax.set_xlabel("qubit pair")
    ax.set_title(f"Exp B: non-uniform readout ({dataset}); blue=low/low, orange=mixed, red=high/high")
    ax.tick_params(axis="x", rotation=60); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"expB_{dataset}.png", dpi=150); plt.close(fig)
    return out


def cls_of(pair_str: str, hi: set) -> str:
    i, j = (int(t) for t in pair_str.strip("()").split(","))
    return ["low-low", "low-high", "high-high"][(i in hi) + (j in hi)]


# ============================================================================ Exp C
def exp_c(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seed: int = 42) -> dict:
    """Readout crosstalk: simultaneous readout of all qubits vs a staggered schedule
    that never reads non-adjacent neighbours together.  Adjacent-pair observables
    still need simultaneous readout of their two qubits, so they are the floor."""
    n = hw.n
    X_tr, *_ = load_split(dataset, n, seed, quick)
    X = X_tr[: (12 if quick else 40)]
    obs = build_observables(n)
    ro = hw.readout_noise()
    F_ex = exact_features(X, obs)
    F_sim = sampled_features(X, obs, readout_noise=ro, shots=shots, seed=seed)
    F_stg = sampled_features(X, obs, readout_noise=ro, shots=shots, seed=seed, measure_groups=staggered_groups(n))
    d_sim, d_stg = np.abs(F_sim - F_ex).mean(axis=0), np.abs(F_stg - F_ex).mean(axis=0)
    out = {"simultaneous": {"by_distance": by_distance(obs, d_sim)},
           "staggered": {"by_distance": by_distance(obs, d_stg)},
           "per_pair": {pair_label(o): {"simultaneous": float(d_sim[m]), "staggered": float(d_stg[m])}
                        for m, o in enumerate(obs) if o.weight == 2 and o.paulis == "ZZ"},
           "crosstalk": {f"{k}": v for k, v in hw.crosstalk.items()},
           "n_circuits_per_sample": {"simultaneous": 27, "staggered": None}}
    save(f"expC_{dataset}", out)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for tag, style in (("simultaneous", "s-"), ("staggered", "o--")):
        d = out[tag]["by_distance"]
        ax.plot(list(d.keys()), list(d.values()), style, label=tag)
    ax.set_xlabel("topological distance"); ax.set_ylabel("mean |<O>_noisy - <O>_exact|")
    ax.set_title(f"Exp C: readout crosstalk ({dataset})"); ax.legend(); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"expC_{dataset}.png", dpi=150); plt.close(fig)
    return out


# ============================================================================ pruning
def noisy_split_features(dataset, hw, *, factor, shots, quick, seed, gate_noise=True,
                         n_qubits=None, n_train=None, total_shots=None, coherent=0.0):
    n = n_qubits or hw.n
    X_tr, X_te, y_tr, y_te = load_split(dataset, n, seed, quick)
    if n_train is not None:
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    obs = build_observables(n)
    kw = dict(readout_noise=hw.readout_noise(factor), seed=seed,
              noise_model=hw.gate_noise_model(coherent) if gate_noise else None)
    if total_shots is not None:
        kw.update(total_shots=total_shots, min_settings=True)
    else:
        kw.update(shots=shots)
    return dict(obs=obs, y_tr=y_tr, y_te=y_te,
                Fx_tr=exact_features(X_tr, obs), Fx_te=exact_features(X_te, obs),
                Fn_tr=sampled_features(X_tr, obs, **kw), Fn_te=sampled_features(X_te, obs, **kw))


def score_table(D: dict, hw: HardwareProfile, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The four filters of the ablation.  KTA is computed on the *noisy* training
    features (that is all one has on hardware); W from the calibration profile."""
    gamma = median_gamma(D["Fn_tr"])
    k = per_observable_kta(D["Fn_tr"], D["y_tr"], gamma)
    w = hw.weights(D["obs"])
    return {"random": rng.random(len(k)), "kta_only": k, "w_only": w, "kta_w": k * w,
            "kta_w_rank": rank_product(k, w), "kta_w_beta": k * w ** select_beta(D, k, w)}


def rank_product(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Filter v2a: product of normalised ranks, so neither term's scale dominates."""
    rk = np.argsort(np.argsort(k)) / (len(k) - 1)
    rw = np.argsort(np.argsort(w)) / (len(w) - 1)
    return rk * rw


def select_beta(D: dict, k: np.ndarray, w: np.ndarray, betas=(1, 2, 4, 8, 16), keep_fraction=0.5, seed=0) -> float:
    """Filter v2b: KTA * W^beta with beta chosen on a validation fold carved from train."""
    idx_tr, idx_va = train_test_split(np.arange(len(D["y_tr"])), test_size=0.25, random_state=seed, stratify=D["y_tr"])
    best, best_acc = betas[0], -1.0
    for b in betas:
        keep = prune(k * w ** b, keep_fraction=keep_fraction)
        gamma = median_gamma(D["Fn_tr"][idx_tr][:, keep])
        r = evaluate_kernel(D["Fn_tr"][idx_tr][:, keep], D["Fn_tr"][idx_va][:, keep],
                            D["y_tr"][idx_tr], D["y_tr"][idx_va], gamma)
        if r["acc"] > best_acc:
            best, best_acc = b, r["acc"]
    D["selected_beta"] = best
    return best


def _standardize(F_tr, F_te):
    mu, sd = F_tr.mean(axis=0), F_tr.std(axis=0) + 1e-9
    return (F_tr - mu) / sd, (F_te - mu) / sd


def _krr_accuracy(K_tr, K_te, y_tr, y_te, alpha=0.1):
    m = KernelRidge(kernel="precomputed", alpha=alpha).fit(K_tr, y_tr)
    return float(np.mean(np.sign(m.predict(K_te)) == y_te))


def evaluate_subset(D: dict, keep: np.ndarray, *, standardize=False, classifier="svm", gamma_scale=1.0) -> dict:
    """Noisy kernel on the kept observables, scored against K_exact restricted to
    the same observables with the same gamma (median heuristic on exact train features).

    standardize : z-score features on train stats (noisy stats for noisy, exact for exact)
    classifier  : "svm" (C=1) or "krr" (kernel ridge, alpha=0.1, sign of prediction)
    gamma_scale : multiply the median-heuristic gamma (sharper kernel > 1)
    """
    Fx_tr, Fx_te = D["Fx_tr"][:, keep], D["Fx_te"][:, keep]
    Fn_tr, Fn_te = D["Fn_tr"][:, keep], D["Fn_te"][:, keep]
    if standardize:
        Fx_tr, Fx_te = _standardize(Fx_tr, Fx_te)
        Fn_tr, Fn_te = _standardize(Fn_tr, Fn_te)
    gamma = median_gamma(Fx_tr) * gamma_scale
    K_ex = gaussian_kernel(Fx_tr, Fx_tr, gamma)
    res = evaluate_kernel(Fn_tr, Fn_te, D["y_tr"], D["y_te"], gamma)
    ideal = evaluate_kernel(Fx_tr, Fx_te, D["y_tr"], D["y_te"], gamma)
    if classifier == "krr":
        res["acc"] = _krr_accuracy(res["K_train"], res["K_test"], D["y_tr"], D["y_te"])
        ideal["acc"] = _krr_accuracy(ideal["K_train"], ideal["K_test"], D["y_tr"], D["y_te"])
    return {"frobenius": frobenius_distance(res["K_train"], K_ex) / np.linalg.norm(K_ex),
            "acc_noisy": res["acc"], "acc_ideal": ideal["acc"], "kta_noisy": res["kta"], "n_obs": int(len(keep))}


def ablation(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seed: int = 42,
             keep_fractions=(0.25, 0.5, 0.75, 1.0), n_random: int = 5) -> dict:
    D = noisy_split_features(dataset, hw, factor=1.0, shots=shots, quick=quick, seed=seed)
    rng = np.random.default_rng(seed)
    scores = score_table(D, hw, rng)
    out = {"keep_fractions": list(keep_fractions), "filters": {}}
    for name, sc in scores.items():
        rows = []
        for kf in keep_fractions:
            if name == "random":
                reps = [evaluate_subset(D, prune(rng.random(len(sc)), keep_fraction=kf)) for _ in range(n_random)]
                rows.append({k: float(np.mean([r[k] for r in reps])) for k in reps[0]})
            else:
                rows.append(evaluate_subset(D, prune(sc, keep_fraction=kf)))
        out["filters"][name] = rows
    # which pairs does KTA·W drop at 50%?
    keep = prune(scores["kta_w"], keep_fraction=0.5)
    dropped = [o.label for m, o in enumerate(D["obs"]) if m not in set(keep)]
    out["dropped_at_50pct_kta_w"] = dropped
    out["selected_beta"] = D.get("selected_beta")
    out["distant_pair_0_5_kept_at_50pct_by_filter"] = {
        name: int(sum(o.qubits == (0, hw.n - 1) for m, o in enumerate(D["obs"]) if m in set(prune(sc, keep_fraction=0.5))))
        for name, sc in scores.items()}
    out["distant_pair_0_5_kept_at_50pct"] = sum(o.qubits == (0, hw.n - 1) for m, o in enumerate(D["obs"]) if m in set(keep))
    save(f"ablation_{dataset}", out)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, rows in out["filters"].items():
        axes[0].plot(keep_fractions, [r["frobenius"] for r in rows], "o-", label=name)
        axes[1].plot(keep_fractions, [r["acc_noisy"] for r in rows], "o-", label=name)
    axes[1].axhline(out["filters"]["kta_w"][-1]["acc_ideal"], ls=":", c="k", label="ideal (all obs)")
    axes[0].set_ylabel("||K_pruned - K_exact|| / ||K_exact||"); axes[1].set_ylabel("SVM test accuracy (noisy)")
    for ax in axes: ax.set_xlabel("keep fraction"); ax.legend(fontsize=7)
    fig.suptitle(f"Pruning ablation ({dataset})"); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"ablation_{dataset}.png", dpi=150); plt.close(fig)
    return out


def dose_response(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seeds=(42,),
                  factors=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0), keep_fraction: float = 0.5) -> dict:
    """Full vs KTA·W-pruned accuracy as the readout profile is scaled.  At 0x the
    pruned kernel should be no better than the full one; the gap should open as
    noise grows.  Gate noise is held fixed."""
    out = {"factors": list(factors), "seeds": list(seeds), "rows": []}
    for f in factors:
        accs_full, accs_pr, frob_full, frob_pr = [], [], [], []
        for s in seeds:
            D = noisy_split_features(dataset, hw, factor=f, shots=shots, quick=quick, seed=s)
            sc = score_table(D, hw, np.random.default_rng(s))["kta_w"]
            full = evaluate_subset(D, np.arange(len(D["obs"])))
            pr = evaluate_subset(D, prune(sc, keep_fraction=keep_fraction))
            accs_full.append(full["acc_noisy"]); accs_pr.append(pr["acc_noisy"])
            frob_full.append(full["frobenius"]); frob_pr.append(pr["frobenius"])
        out["rows"].append({"factor": f,
                            "acc_full": float(np.mean(accs_full)), "acc_full_std": float(np.std(accs_full)),
                            "acc_pruned": float(np.mean(accs_pr)), "acc_pruned_std": float(np.std(accs_pr)),
                            "frob_full": float(np.mean(frob_full)), "frob_pruned": float(np.mean(frob_pr))})
        print(f"   dose {f:>3}x  full={np.mean(accs_full):.3f}  pruned={np.mean(accs_pr):.3f}")
    save(f"dose_{dataset}", out)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    r = out["rows"]
    ax.errorbar(factors, [x["acc_full"] for x in r], [x["acc_full_std"] for x in r], fmt="s-", label="full (all observables)")
    ax.errorbar(factors, [x["acc_pruned"] for x in r], [x["acc_pruned_std"] for x in r], fmt="o--", label=f"KTA·W pruned (keep {keep_fraction:.0%})")
    ax.set_xlabel("readout-noise scale factor"); ax.set_ylabel("SVM test accuracy")
    ax.set_title(f"Dose-response ({dataset})"); ax.legend(); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"dose_{dataset}.png", dpi=150); plt.close(fig)
    return out


# ============================================================================ shot budget
def shot_budget(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seeds=(42,),
                budgets=(540, 1350, 2700, 5400, 13500, 54000), keep_fraction: float = 0.25) -> dict:
    """Fixed total shots per data point.  Pruning reduces the number of measurement
    jobs (greedy set cover of the kept observables) so each job gets more shots.

      full        all observables, budget B
      w_only      calibration-only selection (no data needed), budget B on the kept set
      kta_w_rank  pilot run on all observables at B/2 to compute KTA, then B/2 on the kept set
    """
    n = hw.n
    obs = build_observables(n)
    w = hw.weights(obs)
    if quick:
        budgets = tuple(b for b in budgets if b <= 5400)
    out = {"budgets": list(budgets), "seeds": list(seeds), "keep_fraction": keep_fraction, "rows": []}
    for B in budgets:
        row = {"budget": B}
        for name in ("full", "w_only", "kta_w_rank"):
            accs, frobs, njobs = [], [], None
            for sd in seeds:
                X_tr, X_te, y_tr, y_te = load_split(dataset, n, sd, quick)
                kw = dict(readout_noise=hw.readout_noise(), noise_model=hw.gate_noise_model(),
                          min_settings=True, seed=sd)
                if name == "full":
                    keep, budget = np.arange(len(obs)), B
                elif name == "w_only":
                    keep, budget = prune(w, keep_fraction=keep_fraction), B
                else:
                    pilot = sampled_features(X_tr, obs, total_shots=B // 2, **kw)
                    k = per_observable_kta(pilot, y_tr, median_gamma(pilot))
                    keep, budget = prune(rank_product(k, w), keep_fraction=keep_fraction), B // 2
                sub = [obs[m] for m in keep]
                njobs = len(measurement_jobs(n, sub, min_settings=True)[0])
                Fn_tr = sampled_features(X_tr, sub, total_shots=budget, **kw)
                Fn_te = sampled_features(X_te, sub, total_shots=budget, **kw)
                Fx_tr, Fx_te = exact_features(X_tr, sub), exact_features(X_te, sub)
                gamma = median_gamma(Fx_tr)
                r = evaluate_kernel(Fn_tr, Fn_te, y_tr, y_te, gamma)
                K_ex = gaussian_kernel(Fx_tr, Fx_tr, gamma)
                accs.append(r["acc"]); frobs.append(frobenius_distance(r["K_train"], K_ex) / np.linalg.norm(K_ex))
            row[name] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                         "frob": float(np.mean(frobs)), "n_jobs": njobs, "shots_per_job": budget // njobs}
        out["rows"].append(row)
        print(f"   B={B:>6}  " + "  ".join(f"{k}={row[k]['acc']:.3f}/{row[k]['frob']:.3f}({row[k]['shots_per_job']}sh x {row[k]['n_jobs']}j)"
                                        for k in ("full", "w_only", "kta_w_rank")))
    save(f"budget_{dataset}", out)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, st in (("full", "s-"), ("w_only", "o--"), ("kta_w_rank", "^:")):
        axes[0].errorbar(budgets, [r[name]["acc"] for r in out["rows"]], [r[name]["acc_std"] for r in out["rows"]], fmt=st, label=name)
        axes[1].plot(budgets, [r[name]["frob"] for r in out["rows"]], st, label=name)
    for ax in axes: ax.set_xscale("log"); ax.set_xlabel("total shots per data point"); ax.legend(fontsize=7)
    axes[0].set_ylabel("SVM test accuracy"); axes[1].set_ylabel("||K_noisy - K_exact|| / ||K_exact||")
    fig.suptitle(f"Shot-budget sweep, keep {keep_fraction:.0%} ({dataset})"); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"budget_{dataset}.png", dpi=150); plt.close(fig)
    return out


# ============================================================================ regime scan
# Pre-registered regimes, each removing one link in the chain that makes the SVM
# noise-blind (multiplicative attenuation -> smooth Gaussian kernel -> wide margin).
REGIMES = {
    "baseline":     {},
    "standardize":  {"eval": {"standardize": True}},
    "krr":          {"eval": {"classifier": "krr"}},
    "coherent_3pct":{"feat": {"coherent": 0.03}},
    "coherent_8pct":{"feat": {"coherent": 0.08}},
    "sharp_gamma":  {"eval": {"gamma_scale": 8.0}},
    "small_train":  {"feat": {"n_train": 50}},
    "low_shot":     {"feat": {"total_shots": 150}},          # ~10 shots per measurement job
    "n8":           {"feat": {"n_qubits": 8}},
    "std_coherent": {"feat": {"coherent": 0.03}, "eval": {"standardize": True}},
}


def regime_scan(dataset: str, hw: HardwareProfile, *, shots: int, quick: bool, seeds=(42,),
                regimes=None, keep_fractions=(0.25, 0.5), filters=("random", "w_only", "kta_w_rank")) -> dict:
    regimes = regimes or list(REGIMES)
    out = {"seeds": list(seeds), "keep_fractions": list(keep_fractions), "regimes": {}}
    for name in regimes:
        cfg = REGIMES[name]
        feat_kw, eval_kw = cfg.get("feat", {}), cfg.get("eval", {})
        hw_r = hw
        if feat_kw.get("n_qubits"):
            n8 = feat_kw["n_qubits"]
            hw_r = HardwareProfile(n=n8,
                                   readout_error={q: (.005 if q < n8 // 2 else .06) for q in range(n8)},
                                   cx_error=hw.cx_error, sq_error=hw.sq_error,
                                   crosstalk={(q, q + 1): .10 for q in range(0, n8 - 1, 2)})
        acc = {"full": []}; frob = {"full": []}; kta = {"full": []}; ideal = []
        for f in filters:
            for kf in keep_fractions:
                acc[f"{f}@{kf}"], frob[f"{f}@{kf}"], kta[f"{f}@{kf}"] = [], [], []
        t = time.time()
        for sd in seeds:
            D = noisy_split_features(dataset, hw_r, factor=1.0, shots=shots, quick=quick, seed=sd, **feat_kw)
            rng = np.random.default_rng(sd)
            sc = score_table(D, hw_r, rng)
            r = evaluate_subset(D, np.arange(len(D["obs"])), **eval_kw)
            acc["full"].append(r["acc_noisy"]); frob["full"].append(r["frobenius"]); kta["full"].append(r["kta_noisy"]); ideal.append(r["acc_ideal"])
            for f in filters:
                for kf in keep_fractions:
                    if f == "random":
                        reps = [evaluate_subset(D, prune(rng.random(len(sc[f])), keep_fraction=kf), **eval_kw) for _ in range(3)]
                        r = {k: float(np.mean([x[k] for x in reps])) for k in reps[0]}
                    else:
                        r = evaluate_subset(D, prune(sc[f], keep_fraction=kf), **eval_kw)
                    key = f"{f}@{kf}"
                    acc[key].append(r["acc_noisy"]); frob[key].append(r["frobenius"]); kta[key].append(r["kta_noisy"])
        summ = {k: {"acc": float(np.mean(v)), "acc_std": float(np.std(v)),
                    "frob": float(np.mean(frob[k])), "kta": float(np.mean(kta[k]))} for k, v in acc.items()}
        summ["ideal_full"] = float(np.mean(ideal))
        best = max((k for k in summ if k != "full" and k != "ideal_full"), key=lambda k: summ[k]["acc"])
        summ["best_pruned"] = best
        summ["gap"] = summ[best]["acc"] - summ["full"]["acc"]
        out["regimes"][name] = summ
        print(f"   {name:14s} ideal={summ['ideal_full']:.3f} full={summ['full']['acc']:.3f}±{summ['full']['acc_std']:.3f} "
              f"| " + " ".join(f"{k}={summ[k]['acc']:.3f}" for k in summ if '@' in k)
              + f" | best={best} gap={summ['gap']:+.3f}  ({time.time() - t:.0f}s)")
        save(f"regime_{dataset}", out)  # save incrementally

    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = list(out["regimes"]); x = np.arange(len(names))
    ax.bar(x - 0.3, [out["regimes"][n]["ideal_full"] for n in names], 0.2, label="ideal (noiseless, all obs)", color="lightgray")
    ax.bar(x - 0.1, [out["regimes"][n]["full"]["acc"] for n in names], 0.2, yerr=[out["regimes"][n]["full"]["acc_std"] for n in names], label="full noisy")
    ax.bar(x + 0.1, [out["regimes"][n]["w_only@0.25"]["acc"] for n in names], 0.2, yerr=[out["regimes"][n]["w_only@0.25"]["acc_std"] for n in names], label="W-only @25%")
    ax.bar(x + 0.3, [out["regimes"][n]["kta_w_rank@0.25"]["acc"] for n in names], 0.2, yerr=[out["regimes"][n]["kta_w_rank@0.25"]["acc_std"] for n in names], label="KTA·W rank @25%")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Regime scan ({dataset})"); ax.legend(fontsize=7); fig.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig.savefig(FIGURES / f"regime_{dataset}.png", dpi=150); plt.close(fig)
    return out


# ============================================================================ CLI
EXPS = {"A": exp_a, "B": exp_b, "C": exp_c, "ablation": ablation, "dose": dose_response, "budget": shot_budget, "regime": regime_scan}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=list(EXPS), choices=list(EXPS))
    ap.add_argument("--dataset", nargs="+", default=["synthetic", "breast_cancer"])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="dose, budget and regime")
    ap.add_argument("--quick", action="store_true", help="subsample + 1000 shots; smoke test")
    ap.add_argument("--regimes", nargs="+", default=None, choices=list(REGIMES), help="regime scan only")
    args = ap.parse_args()
    shots = 1000 if args.quick else args.shots
    hw = HardwareProfile()
    for ds in args.dataset:
        for e in args.exp:
            t = time.time()
            print(f"== {e} / {ds}")
            kw = {"seeds": tuple(args.seeds)} if e in ("dose", "budget", "regime") else {}
            if e == "regime" and args.regimes:
                kw["regimes"] = args.regimes
            r = EXPS[e](ds, hw, shots=shots, quick=args.quick, **kw)
            if e == "A":
                for tag in ("adjacent_only", "with_0_5_entangler"):
                    print(f"   {tag:20s} |Δ| by distance:", {k: round(v, 4) for k, v in r[tag]["delta_by_distance"].items()})
            elif e == "B":
                print("   by class:", {k: {m: round(v, 4) for m, v in d.items()} for k, d in r["by_class"].items()})
            elif e == "C":
                print("   simultaneous:", {k: round(v, 4) for k, v in r["simultaneous"]["by_distance"].items()})
                print("   staggered:   ", {k: round(v, 4) for k, v in r["staggered"]["by_distance"].items()})
            elif e == "ablation":
                for name, rows in r["filters"].items():
                    print(f"   {name:9s}", [f"{x['acc_noisy']:.3f}/{x['frobenius']:.3f}" for x in rows], "(acc/frob at keep", r["keep_fractions"], ")")
                print(f"   (0,5) observables kept at 50%: {r['distant_pair_0_5_kept_at_50pct_by_filter']}  (beta={r['selected_beta']})")
            print(f"   {time.time() - t:.0f}s")
