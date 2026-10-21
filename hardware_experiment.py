"""Hardware validation of the readout-reweighting mechanism (Phase 2, ~10 min QPU).

Prediction made before the run, from same-day calibration:
    <O>_hardware  ~=  a_O * <O>_exact,      a_O = prod_{q in O} (1 - 2 p_q)
i.e. readout noise reweights each PQK feature by a calibration-determined factor, so an
uneven chain distorts PQK's kernel ordering, while the fidelity kernel is rescaled uniformly.

Design: the same circuits on the BEST and the WORST usable 6-qubit chain of one device.
  16 Sonar points -> PQK: 15 settings x 16 x 2 chains = 480 circuits
                     fidelity: 120 pairs x 2 chains  = 240 circuits
                     readout check (|0..0>, |1..1>) x 2 chains = 4 circuits
  ~720 circuits x 1000 shots.

    python hardware_experiment.py --dry-run                  # whole pipeline on FakeMarrakesh, no QPU
    python hardware_experiment.py --submit readout pqk_good  # live, stage 1 (cheap); check usage
    python hardware_experiment.py --submit pqk_bad fid       # live, stage 2
    python hardware_experiment.py --fetch                    # pull finished jobs into results/hw_live.json
    python hardware_experiment.py analyze --tag live         # the three analyses + figures

Outputs: results/hw_<tag>.json (counts, chains, calibration), results/hw_<tag>_analysis.json,
         figures/hw_<tag>_attenuation.png, figures/hw_<tag>_kernels.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.stats import spearmanr

from calibration_scan import MAX_2Q_ERROR, chain_profile, dump_calibration, linear_chains
from dataset_loader import load_dataset
from kernel import gaussian_kernel, median_gamma, prune
from kernel_comparison import _overlap_circuit, fidelity_kernel_exact
from pqk import _measurement_circuit, build_observables, exact_features, feature_map, measurement_jobs

RESULTS, FIGURES = Path("results"), Path("figures")
N = 6
N_POINTS = 16
SHOTS = 1000
DATASET = "sonar"  # overridable with --dataset
STAGES = ("readout", "pqk_good", "pqk_bad", "fid")


# ============================================================================ design
def pick_points(seed: int = 42, dataset: str = DATASET) -> tuple[np.ndarray, np.ndarray]:
    X_tr, _, y_tr, _ = load_dataset(dataset, n_qubits=N, seed=seed)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y_tr == c)[0], N_POINTS // 2, replace=False) for c in (-1, 1)])
    return X_tr[idx], y_tr[idx]


def pick_chains(cal: dict) -> dict:
    profs = [p for c in linear_chains(cal["edges"], N) if (p := chain_profile(cal, c)) is not None
             and max(p["two_qubit"]) <= MAX_2Q_ERROR]
    profs.sort(key=lambda p: p["dose_eq"])
    return {"good": profs[0], "bad": profs[-1]}


def build_circuits(X: np.ndarray) -> dict[str, list[QuantumCircuit]]:
    obs = build_observables(N)
    jobs, _ = measurement_jobs(N, obs, min_settings=True)
    pqk = [_measurement_circuit(feature_map(x), s, g) for x in X for s, g in jobs]
    fid = [_overlap_circuit(X[i], X[j]) for i, j in itertools.combinations(range(len(X)), 2)]
    zeros = QuantumCircuit(N, N); zeros.measure(range(N), range(N))
    ones = QuantumCircuit(N, N); ones.x(range(N)); ones.measure(range(N), range(N))
    return {"pqk": pqk, "fid": fid, "readout": [zeros, ones], "settings": [s for s, _ in jobs]}


def stage_circuits(circs: dict, chains: dict) -> dict[str, tuple[list[QuantumCircuit], list[int]]]:
    return {"readout": (circs["readout"] * 2, [chains["good"]["chain"]] * 2 + [chains["bad"]["chain"]] * 2),
            "pqk_good": (circs["pqk"], [chains["good"]["chain"]] * len(circs["pqk"])),
            "pqk_bad": (circs["pqk"], [chains["bad"]["chain"]] * len(circs["pqk"])),
            "fid": (circs["fid"] * 2, [chains["good"]["chain"]] * len(circs["fid"]) + [chains["bad"]["chain"]] * len(circs["fid"]))}


def transpile_to(backend, circs: list[QuantumCircuit], layouts: list[list[int]], seed: int = 7) -> list[QuantumCircuit]:
    out = []
    # group by layout so each group transpiles in one call
    for lay in sorted({tuple(l) for l in layouts}):
        idx = [i for i, l in enumerate(layouts) if tuple(l) == lay]
        t = transpile([circs[i] for i in idx], backend=backend, initial_layout=list(lay),
                      optimization_level=1, seed_transpiler=seed)
        out.extend(zip(idx, t))
    out.sort(key=lambda p: p[0])
    return [c for _, c in out]


def counts_of(pub_result) -> dict[str, int]:
    return pub_result.data.c.get_counts()


# ============================================================================ run
def run(backend, tag: str, stages=STAGES, submit_only=False, shots=SHOTS, dry=False, dataset=DATASET) -> dict:
    from qiskit_ibm_runtime import SamplerV2
    cal = dump_calibration(backend)
    chains = pick_chains(cal)
    X, y = pick_points(dataset=dataset)
    circs = build_circuits(X)
    staged = stage_circuits(circs, chains)
    path = RESULTS / f"hw_{tag}.json"
    state = json.load(open(path)) if path.exists() else {}
    state.update({"backend": backend.name, "dry_run": dry, "pulled_at": cal["pulled_at"], "shots": shots,
                  "dataset": dataset, "X": X.tolist(), "y": y.tolist(), "settings": circs["settings"],
                  "chains": chains, "calibration": {"readout_error": cal["readout_error"], "two_qubit_error": cal["two_qubit_error"]}})
    state.setdefault("jobs", {}); state.setdefault("counts", {})
    for name in ("good", "bad"):
        c = chains[name]
        print(f"   {name:4s} chain {c['chain']}  readout {[f'{v*100:.1f}' for v in c['readout']]}%  dose_eq={c['dose_eq']:.2f}")
    sampler = SamplerV2(mode=backend)
    RESULTS.mkdir(exist_ok=True)
    for st in stages:
        cl, lays = staged[st]
        t = time.time()
        tc = transpile_to(backend, cl, lays)
        job = sampler.run(tc, shots=shots)
        print(f"   submitted {st}: {len(tc)} circuits x {shots} shots, job {job.job_id()}  (transpile {time.time()-t:.0f}s)")
        state["jobs"][st] = job.job_id()
        json.dump(state, open(path, "w"))
        if not submit_only:
            res = job.result()
            state["counts"][st] = [counts_of(r) for r in res]
            try:
                state.setdefault("usage_s", {})[st] = job.usage()
            except Exception:
                pass
            json.dump(state, open(path, "w"))
            print(f"   done {st}  ({time.time()-t:.0f}s wall)")
    return state


def fetch(tag: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    path = RESULTS / f"hw_{tag}.json"
    state = json.load(open(path))
    s = QiskitRuntimeService()
    for st, jid in state["jobs"].items():
        if st in state["counts"]:
            continue
        job = s.job(jid)
        print(f"   {st}: {jid} status {job.status()}")
        if str(job.status()).upper().endswith("DONE") or job.status() == "DONE":
            state["counts"][st] = [counts_of(r) for r in job.result()]
            try:
                state.setdefault("usage_s", {})[st] = job.usage()
            except Exception:
                pass
    json.dump(state, open(path, "w"))
    print("   usage (s):", state.get("usage_s"))


# ============================================================================ analysis
def features_from_counts(counts: list[dict], settings: list[str], obs, n_points: int) -> np.ndarray:
    jobs, serves = measurement_jobs(N, obs, min_settings=True)
    assert [s for s, _ in jobs] == settings
    F = np.empty((n_points, len(obs)))
    for k in range(n_points):
        cs = counts[k * len(jobs):(k + 1) * len(jobs)]
        for m, o in enumerate(obs):
            acc, tot = 0.0, 0
            for j in serves[o]:
                for bitstr, c in cs[j].items():
                    bits = bitstr[::-1]
                    par = sum(int(bits[q]) for q in o.qubits) % 2
                    acc += (1 - 2 * par) * c
                    tot += c
            F[k, m] = acc / tot
    return F


def fidelity_from_counts(counts: list[dict], n_points: int) -> np.ndarray:
    K = np.eye(n_points)
    for (i, j), c in zip(itertools.combinations(range(n_points), 2), counts):
        tot = sum(c.values())
        K[i, j] = K[j, i] = c.get("0" * N, 0) / tot
    return K


def readout_from_counts(counts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """counts = [zeros, ones] -> (p01[q], p10[q]) measured."""
    def flip_rates(c, expect):
        tot = sum(c.values()); r = np.zeros(N)
        for bitstr, cnt in c.items():
            bits = bitstr[::-1]
            for q in range(N):
                r[q] += cnt * (bits[q] != expect)
        return r / tot
    return flip_rates(counts[0], "0"), flip_rates(counts[1], "1")


def analyze(tag: str) -> dict:
    st = json.load(open(RESULTS / f"hw_{tag}.json"))
    X, y = np.array(st["X"]), np.array(st["y"])
    obs = build_observables(N)
    F_ex = exact_features(X, obs)
    K_fid_ex = fidelity_kernel_exact(X, X)
    out = {"backend": st["backend"], "dry_run": st["dry_run"], "usage_s": st.get("usage_s"), "chains": {}}
    fig1, ax1 = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    fig2, ax2 = plt.subplots(figsize=(6, 3.8))
    bars = {}
    for ci, name in enumerate(("good", "bad")):
        ch = st["chains"][name]
        c_ro = st["counts"]["readout"][2 * ci:2 * ci + 2]
        p01, p10 = readout_from_counts(c_ro)
        p_meas = (p01 + p10) / 2
        p_cal = np.array(ch["readout"])
        F_hw = features_from_counts(st["counts"][f"pqk_{name}"], st["settings"], obs, len(X))
        nf = len(st["counts"]["fid"]) // 2
        K_fid = fidelity_from_counts(st["counts"]["fid"][ci * nf:(ci + 1) * nf], len(X))

        # --- 1. attenuation: measured slope per observable vs predicted from calibration
        slope = np.array([float(F_hw[:, m] @ F_ex[:, m] / max(F_ex[:, m] @ F_ex[:, m], 1e-9)) for m in range(len(obs))])
        pred_cal = np.array([np.prod([1 - 2 * p_cal[q] for q in o.qubits]) for o in obs])
        pred_meas = np.array([np.prod([1 - 2 * p_meas[q] for q in o.qubits]) for o in obs])
        strong = np.array([F_ex[:, m] @ F_ex[:, m] > 0.5 for m in range(len(obs))])  # observables with signal to regress on
        r_cal = float(np.corrcoef(pred_cal[strong], slope[strong])[0, 1])
        r_meas = float(np.corrcoef(pred_meas[strong], slope[strong])[0, 1])
        resid = float(np.sqrt(np.mean((slope[strong] - pred_meas[strong]) ** 2)))

        # --- 2. kernel ordering vs exact
        w = pred_cal  # hardware weight = predicted retention; prune by it
        keep = prune(w, keep_fraction=0.25)
        iu = np.triu_indices(len(X), k=1)
        def sp(Fa, Fb, idx):
            g = median_gamma(Fb[:, idx])
            return float(spearmanr(gaussian_kernel(Fa[:, idx], Fa[:, idx], g)[iu], gaussian_kernel(Fb[:, idx], Fb[:, idx], g)[iu]).correlation)
        allidx = np.arange(len(obs))
        # --- 3. mitigation: divide by predicted factor (clip tiny factors)
        F_mit = F_hw / np.clip(pred_meas, 0.05, None)
        res = {"chain": ch["chain"], "readout_calibration": p_cal.tolist(), "readout_measured": p_meas.tolist(),
               "dose_eq": ch["dose_eq"],
               "attenuation": {"corr_pred_cal_vs_measured": r_cal, "corr_pred_meas_vs_measured": r_meas,
                               "rms_residual": resid, "n_obs_used": int(strong.sum()),
                               "mean_slope_1q": float(slope[[o.weight == 1 for o in obs]].mean()),
                               "mean_slope_2q": float(slope[[o.weight == 2 for o in obs]].mean())},
               "spearman": {"pqk_full": sp(F_hw, F_ex, allidx), "pqk_pruned_W": sp(F_hw, F_ex, keep),
                            "pqk_mitigated": sp(F_mit, F_ex, allidx),
                            "fidelity": float(spearmanr(K_fid[iu], K_fid_ex[iu]).correlation)}}
        out["chains"][name] = res
        bars[name] = res["spearman"]
        ax1[ci].scatter(pred_meas[strong], slope[strong], s=12, alpha=0.7, c=["tab:blue" if o.weight == 1 else "tab:orange" for o, s_ in zip(obs, strong) if s_])
        ax1[ci].plot([0, 1], [0, 1], "k--", lw=1)
        ax1[ci].set_title(f"{name} chain {ch['chain']}\nr={r_meas:.2f}, rms={resid:.3f}")
        ax1[ci].set_xlabel("predicted attenuation  prod(1-2p_q)")
        print(f"   {name:4s} chain: readout cal {np.round(p_cal*100,1)}%  measured {np.round(p_meas*100,1)}%")
        spread = float(pred_meas[strong].max() - pred_meas[strong].min())
        note = "  (predictions span <0.1, so r is uninformative here; judge by rms)" if spread < 0.1 else ""
        print(f"         attenuation corr (measured p) = {r_meas:.3f}, (IBM cal p) = {r_cal:.3f}, rms resid = {resid:.3f}  [1q mean slope {res['attenuation']['mean_slope_1q']:.2f}, 2q {res['attenuation']['mean_slope_2q']:.2f}]{note}")
        print(f"         Spearman vs exact: " + "  ".join(f"{k}={v:.3f}" for k, v in res["spearman"].items()))
    ax1[0].set_ylabel("measured attenuation (slope hw vs exact)")
    fig1.suptitle(f"Readout reweighting: prediction vs hardware ({st['backend']}{' / dry run' if st['dry_run'] else ''})")
    fig1.tight_layout(); FIGURES.mkdir(exist_ok=True); fig1.savefig(FIGURES / f"hw_{tag}_attenuation.png", dpi=150); plt.close(fig1)
    keys = ["pqk_full", "pqk_pruned_W", "pqk_mitigated", "fidelity"]
    xk = np.arange(len(keys))
    ax2.bar(xk - 0.18, [bars["good"][k] for k in keys], 0.36, label=f"good chain (dose_eq {out['chains']['good']['dose_eq']:.1f})")
    ax2.bar(xk + 0.18, [bars["bad"][k] for k in keys], 0.36, label=f"bad chain (dose_eq {out['chains']['bad']['dose_eq']:.1f})")
    ax2.set_xticks(xk); ax2.set_xticklabels(keys); ax2.set_ylabel("Spearman(K_hardware, K_exact)"); ax2.set_ylim(0, 1.02); ax2.legend(fontsize=8)
    ax2.set_title("Kernel ordering preserved on hardware"); fig2.tight_layout()
    fig2.savefig(FIGURES / f"hw_{tag}_kernels.png", dpi=150); plt.close(fig2)
    json.dump(out, open(RESULTS / f"hw_{tag}_analysis.json", "w"), indent=1)
    return out


# ============================================================================ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "analyze"])
    ap.add_argument("--dry-run", action="store_true", help="FakeMarrakesh with its noise model; no QPU time")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--submit", nargs="*", default=None, choices=STAGES, help="live: submit these stages and exit (fetch later)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()
    tag = args.tag or ("dry" if args.dry_run else "live")
    if args.mode == "analyze":
        analyze(tag)
    elif args.fetch:
        fetch(tag)
    elif args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        print(f"== DRY RUN on FakeMarrakesh (local noisy simulation)")
        run(FakeMarrakesh(), tag, shots=args.shots, dry=True, dataset=args.dataset)
        analyze(tag)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        s = QiskitRuntimeService()
        b = s.backend(args.backend)
        stages = tuple(args.submit) if args.submit else STAGES
        print(f"== LIVE on {b.name}: stages {stages}")
        run(b, tag, stages=stages, submit_only=True, shots=args.shots, dataset=args.dataset)
        print("   jobs submitted; run `--fetch` once they finish, then `analyze --tag live`")
