"""Dataset panel x noise profiles x observable selectors.

Answers Unai's three questions in one run:
  Q1  real calibration profiles (from calibration_scan.py) instead of synthetic dose scaling
  Q2  a panel of real tabular datasets, not just Breast Cancer
  Q3  a classically computed, data-driven selector (mutual information of inputs with the
      label, mapped onto observables through the circuit's light cone) compared against the
      quantum-pilot KTA selector, each with and without the hardware weight W

    python panel_experiments.py --quick                                   # smoke test
    python panel_experiments.py                                           # full: all datasets, all profiles
    python panel_experiments.py --datasets pima heart --profiles sim:1 sim:4 results/chain_scan_ibm_marrakesh.json:p90

Profiles:  sim:<dose>                      our simulated profile scaled by <dose>
           <chain_scan.json>:<median|p90|worst>   a real 6-qubit chain from calibration_scan.py

Selectors (each ranks the 153 observables; we keep the top 25% / 50%):
  random
  w_only           hardware weight only (no data)
  kta_rank_w       rank(KTA on noisy quantum features) x rank(W)     -- needs a quantum pilot run
  cls0_rank_w      rank(classical MI, direct qubit<->input map) x rank(W)   -- free
  cls1_rank_w      rank(classical MI, light cone radius 1) x rank(W)        -- free
  cls1_only        classical MI, light cone radius 1, no hardware weight

On the synthetic dataset we also score each selector against the known interaction graph.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

from dataset_loader import ALL_DATASETS, SIGNAL_PAIRS, load_dataset
from kernel import gaussian_kernel, median_gamma, per_observable_kta, prune
from noise_experiments import HardwareProfile, evaluate_subset, rank_product
from pqk import Observable, build_observables, exact_features, sampled_features

RESULTS, FIGURES = Path("results"), Path("figures")


# ============================================================================ profiles
def load_profile(spec: str) -> tuple[HardwareProfile, float, str]:
    """'sim:2' -> (default profile, dose 2, 'sim2x');  'path.json:p90' -> (real chain, dose 1, 'marrakesh_p90')."""
    src, _, arg = spec.rpartition(":")
    if src == "sim":
        return HardwareProfile(), float(arg), f"sim{float(arg):g}x"
    scan = json.load(open(src))
    hp = scan["representative_chains"][arg]["hardware_profile"]
    hw = HardwareProfile(n=hp["n"], readout_error={int(k): v for k, v in hp["readout_error"].items()},
                         cx_error=hp["cx_error"], sq_error=hp["sq_error"], crosstalk={})
    tag = scan["backend"].replace("ibm_", "") + "_" + arg
    return hw, 1.0, tag


# ============================================================================ classical selector
def classical_relevance(X_tr: np.ndarray, y_tr: np.ndarray, obs: list[Observable], radius: int = 1, seed: int = 0) -> np.ndarray:
    """Mutual information of inputs (and pairwise products) with the label, mapped to observables.

    Single-qubit observable on q   <- MI(x_q; y) averaged over inputs within `radius` of q
    Two-qubit observable on (i,j)  <- same for both qubits, plus MI(x_i * x_j; y) for the pair
                                      (and its light-cone neighbours' products when radius > 0)
    radius = 0 is the naive qubit<->input identification; radius = 1 accounts for the
    adjacent-CNOT layer spreading each input to its neighbours.
    """
    n = X_tr.shape[1]
    mi1 = mutual_info_classif(X_tr, y_tr, random_state=seed)
    pairs = list(itertools.combinations(range(n), 2))
    P = np.column_stack([X_tr[:, i] * X_tr[:, j] for i, j in pairs])
    mi2 = dict(zip(pairs, mutual_info_classif(P, y_tr, random_state=seed)))

    def cone(q):
        return [k for k in range(n) if abs(k - q) <= radius]

    rel = np.empty(len(obs))
    for m, o in enumerate(obs):
        qs = o.qubits
        single = np.mean([mi1[k] for q in qs for k in cone(q)])
        if o.weight == 1:
            rel[m] = single
        else:
            i, j = qs
            pair_terms = [mi2[tuple(sorted((a, b)))] for a in cone(i) for b in cone(j) if a != b]
            rel[m] = single + np.mean(pair_terms)
    return rel


def ground_truth_score(obs: list[Observable], keep: np.ndarray) -> dict:
    """Synthetic only: of the kept 2-qubit observables, what fraction sit on a true signal pair,
    and what fraction of the 36 signal-pair observables were kept."""
    sig = {tuple(sorted(p)) for p in SIGNAL_PAIRS}
    kept2 = [obs[m] for m in keep if obs[m].weight == 2]
    on_sig = [o for o in kept2 if tuple(sorted(o.qubits)) in sig]
    total_sig = sum(1 for o in obs if o.weight == 2 and tuple(sorted(o.qubits)) in sig)
    return {"precision": len(on_sig) / max(len(kept2), 1), "recall": len(on_sig) / total_sig}


# ============================================================================ runner
def run_cell(dataset: str, hw: HardwareProfile, dose: float, seed: int, *, shots: int, quick: bool,
             keep_fractions=(0.25, 0.5)) -> dict:
    n = hw.n
    X_tr, X_te, y_tr, y_te = load_dataset(dataset, n_qubits=n, seed=seed)
    if quick:
        X_tr, y_tr, X_te, y_te = X_tr[:60], y_tr[:60], X_te[:40], y_te[:40]
    obs = build_observables(n)
    kw = dict(readout_noise=hw.readout_noise(dose), noise_model=hw.gate_noise_model(), shots=shots, seed=seed)
    D = dict(obs=obs, y_tr=y_tr, y_te=y_te,
             Fx_tr=exact_features(X_tr, obs), Fx_te=exact_features(X_te, obs),
             Fn_tr=sampled_features(X_tr, obs, **kw), Fn_te=sampled_features(X_te, obs, **kw))
    rng = np.random.default_rng(seed)
    w = hw.weights(obs)
    kta = per_observable_kta(D["Fn_tr"], y_tr, median_gamma(D["Fn_tr"]))
    c0 = classical_relevance(X_tr, y_tr, obs, radius=0, seed=seed)
    c1 = classical_relevance(X_tr, y_tr, obs, radius=1, seed=seed)
    selectors = {"random": rng.random(len(obs)), "w_only": w, "kta_rank_w": rank_product(kta, w),
                 "cls0_rank_w": rank_product(c0, w), "cls1_rank_w": rank_product(c1, w), "cls1_only": c1}

    def spearman_to_exact(keep):
        g = median_gamma(D["Fx_tr"][:, keep])
        Kx = gaussian_kernel(D["Fx_tr"][:, keep], D["Fx_tr"][:, keep], g)
        Kn = gaussian_kernel(D["Fn_tr"][:, keep], D["Fn_tr"][:, keep], g)
        iu = np.triu_indices_from(Kx, k=1)
        return float(spearmanr(Kn[iu], Kx[iu]).correlation)

    full = evaluate_subset(D, np.arange(len(obs)))
    out = {"full": {"acc": full["acc_noisy"], "acc_ideal": full["acc_ideal"], "spearman": spearman_to_exact(np.arange(len(obs)))},
           "selectors": {}}
    for name, sc in selectors.items():
        out["selectors"][name] = {}
        for kf in keep_fractions:
            keep = prune(sc, keep_fraction=kf)
            r = evaluate_subset(D, keep)
            cell = {"acc": r["acc_noisy"], "acc_ideal": r["acc_ideal"], "spearman": spearman_to_exact(keep)}
            if dataset == "synthetic":
                cell["ground_truth"] = ground_truth_score(obs, keep)
            out["selectors"][name][str(kf)] = cell
    # agreement between selectors at 25%: Jaccard of kept sets
    k25 = {nm: set(prune(sc, keep_fraction=0.25)) for nm, sc in selectors.items()}
    out["jaccard_25"] = {f"{a}|{b}": len(k25[a] & k25[b]) / len(k25[a] | k25[b])
                         for a, b in (("kta_rank_w", "cls1_rank_w"), ("kta_rank_w", "cls0_rank_w"), ("cls0_rank_w", "cls1_rank_w"))}
    return out


def aggregate(cells: list[dict]) -> dict:
    def ms(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    agg = {"full": {k: ms([c["full"][k] for c in cells]) for k in ("acc", "acc_ideal", "spearman")}, "selectors": {}}
    for name in cells[0]["selectors"]:
        agg["selectors"][name] = {}
        for kf in cells[0]["selectors"][name]:
            vals = [c["selectors"][name][kf] for c in cells]
            d = {k: ms([v[k] for v in vals]) for k in ("acc", "acc_ideal", "spearman")}
            if "ground_truth" in vals[0]:
                d["ground_truth"] = {k: float(np.mean([v["ground_truth"][k] for v in vals])) for k in ("precision", "recall")}
            agg["selectors"][name][kf] = d
    agg["jaccard_25"] = {k: float(np.mean([c["jaccard_25"][k] for c in cells])) for k in cells[0]["jaccard_25"]}
    return agg


def panel(datasets, profiles, seeds, *, shots, quick):
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    out_path = RESULTS / ("panel_quick.json" if quick else "panel.json")
    results = json.load(open(out_path)) if out_path.exists() else {}
    for ds in datasets:
        for spec in profiles:
            hw, dose, ptag = load_profile(spec)
            key = f"{ds}|{ptag}"
            if key in results and results[key].get("n_seeds") == len(seeds):
                print(f"   {key}: cached"); continue
            t = time.time()
            cells = [run_cell(ds, hw, dose, sd, shots=shots, quick=quick) for sd in seeds]
            agg = aggregate(cells); agg["n_seeds"] = len(seeds); agg["profile"] = spec
            agg["readout_profile"] = [hw.readout_error[q] * dose for q in range(hw.n)]
            results[key] = agg
            json.dump(results, open(out_path, "w"), indent=1)
            f = agg["full"]
            line = f"   {ds:13s} {ptag:16s} ideal={f['acc_ideal']['mean']:.3f} full={f['acc']['mean']:.3f}±{f['acc']['std']:.3f} sp={f['spearman']['mean']:.2f} |"
            for name in ("random", "w_only", "kta_rank_w", "cls0_rank_w", "cls1_rank_w", "cls1_only"):
                s = agg["selectors"][name]["0.25"]
                line += f" {name}={s['acc']['mean']:.3f}/{s['spearman']['mean']:.2f}"
            if ds == "synthetic":
                line += "  GT-recall@25%: " + " ".join(f"{nm}={agg['selectors'][nm]['0.25']['ground_truth']['recall']:.2f}" for nm in ("kta_rank_w", "cls0_rank_w", "cls1_rank_w"))
            line += f"  J(kta,cls1)={agg['jaccard_25']['kta_rank_w|cls1_rank_w']:.2f}  ({time.time() - t:.0f}s)"
            print(line)
    plot_panel(results, quick)
    return results


def plot_panel(results: dict, quick: bool):
    sel_names = ["random", "w_only", "kta_rank_w", "cls0_rank_w", "cls1_rank_w", "cls1_only"]
    keys = sorted(results)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(keys)), 7), sharex=True)
    x = np.arange(len(keys)); wdt = 0.11
    for ax, metric in zip(axes, ("acc", "spearman")):
        ax.bar(x - 3.5 * wdt, [results[k]["full"][metric]["mean"] for k in keys], wdt, label="full", color="k", alpha=0.6)
        for i, nm in enumerate(sel_names):
            ax.bar(x + (i - 2.5) * wdt, [results[k]["selectors"][nm]["0.25"][metric]["mean"] for k in keys], wdt, label=f"{nm}@25%")
        ax.set_ylabel("SVM accuracy" if metric == "acc" else "Spearman(K_noisy, K_exact)")
    axes[0].legend(fontsize=6, ncol=4); axes[1].set_xticks(x); axes[1].set_xticklabels(keys, rotation=60, ha="right", fontsize=7)
    fig.suptitle("Dataset x noise-profile panel, selectors at 25% keep"); fig.tight_layout()
    fig.savefig(FIGURES / ("panel_quick.png" if quick else "panel.png"), dpi=150); plt.close(fig)


DEFAULT_PROFILES = ["sim:1", "sim:4",
                    "results/chain_scan_ibm_kingston.json:median",
                    "results/chain_scan_ibm_fez.json:worst",
                    "results/chain_scan_ibm_marrakesh.json:median",
                    "results/chain_scan_ibm_marrakesh.json:p90",
                    "results/chain_scan_ibm_marrakesh.json:worst"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    panel(args.datasets, args.profiles, args.seeds, shots=1000 if args.quick else args.shots, quick=args.quick)
