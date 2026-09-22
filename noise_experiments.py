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
from qiskit_aer.noise import NoiseModel, depolarizing_error

from dataset_loader import load_dataset
from kernel import (evaluate_kernel, frobenius_distance, gaussian_kernel, median_gamma,
                    per_observable_kta, prune)
from pqk import (Observable, ReadoutNoise, build_observables, exact_features,
                 sampled_features, staggered_groups)

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

    def gate_noise_model(self) -> NoiseModel:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(self.sq_error, 1), ["sx", "x", "rz"])
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
def noisy_split_features(dataset, hw, *, factor, shots, quick, seed, gate_noise=True):
    n = hw.n
    X_tr, X_te, y_tr, y_te = load_split(dataset, n, seed, quick)
    obs = build_observables(n)
    kw = dict(readout_noise=hw.readout_noise(factor), shots=shots, seed=seed,
              noise_model=hw.gate_noise_model() if gate_noise else None)
    return dict(obs=obs, y_tr=y_tr, y_te=y_te,
                Fx_tr=exact_features(X_tr, obs), Fx_te=exact_features(X_te, obs),
                Fn_tr=sampled_features(X_tr, obs, **kw), Fn_te=sampled_features(X_te, obs, **kw))


def score_table(D: dict, hw: HardwareProfile, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The four filters of the ablation.  KTA is computed on the *noisy* training
    features (that is all one has on hardware); W from the calibration profile."""
    gamma = median_gamma(D["Fn_tr"])
    k = per_observable_kta(D["Fn_tr"], D["y_tr"], gamma)
    w = hw.weights(D["obs"])
    return {"random": rng.random(len(k)), "kta_only": k, "w_only": w, "kta_w": k * w}


def evaluate_subset(D: dict, keep: np.ndarray) -> dict:
    """Noisy kernel on the kept observables, scored against K_exact restricted to
    the same observables with the same gamma (median heuristic on exact train features)."""
    gamma = median_gamma(D["Fx_tr"][:, keep])
    K_ex = gaussian_kernel(D["Fx_tr"][:, keep], D["Fx_tr"][:, keep], gamma)
    res = evaluate_kernel(D["Fn_tr"][:, keep], D["Fn_te"][:, keep], D["y_tr"], D["y_te"], gamma)
    ideal = evaluate_kernel(D["Fx_tr"][:, keep], D["Fx_te"][:, keep], D["y_tr"], D["y_te"], gamma)
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
                  factors=(0.0, 0.5, 1.0, 2.0, 4.0), keep_fraction: float = 0.5) -> dict:
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


# ============================================================================ CLI
EXPS = {"A": exp_a, "B": exp_b, "C": exp_c, "ablation": ablation, "dose": dose_response}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=list(EXPS), choices=list(EXPS))
    ap.add_argument("--dataset", nargs="+", default=["synthetic", "breast_cancer"])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="dose-response only")
    ap.add_argument("--quick", action="store_true", help="subsample + 1000 shots; smoke test")
    args = ap.parse_args()
    shots = 1000 if args.quick else args.shots
    hw = HardwareProfile()
    for ds in args.dataset:
        for e in args.exp:
            t = time.time()
            print(f"== {e} / {ds}")
            kw = {"seeds": tuple(args.seeds)} if e == "dose" else {}
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
                print(f"   (0,5) observables kept at 50% by KTA·W: {r['distant_pair_0_5_kept_at_50pct']}/9")
            print(f"   {time.time() - t:.0f}s")
