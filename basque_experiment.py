"""PQK observable pruning on ibm_basquecountry: the hardware experiment with a real train/test split.

Same mechanism test as hardware_experiment.py (Oct 2026, 16 points, Marrakesh / Kingston), extended for
a device with a real QPU budget (50 min/month) so that we can finally measure ACCURACY on hardware:

    * n_train + n_test labelled points instead of 16
    * best and worst usable linear chain of the device, picked from same-day calibration
    * PQK features on both chains, fidelity kernel optional (it is the expensive part)
    * selectors: W only (calibration), rank(KTA) * rank(W) (train fold), readout mitigation
    * analyses: predicted vs measured attenuation, kernel ordering, SVM accuracy, decision correlation,
      flipped predictions, all against the noiseless kernel

Everything is driven by one JSON config (see basque_config.json) so the design can be changed in the
meeting without touching code.  Nothing is sent to the QPU without `estimate` having been printed and
the word YES typed.

    python basque_experiment.py estimate                         # circuits + QPU minutes per stage, no QPU
    python basque_experiment.py plan                             # today's best / worst chain on the device
    python basque_experiment.py dry-run                          # whole pipeline on FakeMarrakesh (local, noisy)
    python basque_experiment.py submit --stages readout pqk_good # live: submit stages in Batch mode, then exit
    python basque_experiment.py fetch                            # pull finished jobs into results/basque_<tag>.json
    python basque_experiment.py analyze                          # metrics + figures
    python basque_experiment.py verify                           # replicability: re-run the Oct 21 counts, compare numbers

Config file: --config basque_config.json (default).  Any key can be overridden on the command line,
e.g. --set n_train=60 n_test=30 n_qubits=12.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.stats import spearmanr
from sklearn.svm import SVC

from calibration_scan import MAX_2Q_ERROR, chain_profile, dump_calibration, linear_chains
from dataset_loader import load_dataset
from kernel import gaussian_kernel, median_gamma, per_observable_kta
from pqk import PAULIS, Observable, _measurement_circuit, build_observables, exact_features, feature_map

RESULTS, FIGURES = Path("results"), Path("figures")

# ---------------------------------------------------------------------------- config
DEFAULTS = {
    "service_name": "unai",            # QiskitRuntimeService(name=...) -> credentials saved locally by Unai
    "backend": "ibm_basquecountry",
    "tag": "basque",                   # results/basque_<tag>.json
    "dataset": "sonar",
    "n_qubits": 6,                     # 6 -> 153 observables, 12 -> 630, 20 -> 1770
    "n_train": 100,                    # labelled points measured on hardware (balanced classes)
    "n_test": 50,
    "shots": 1000,
    "stages": ["readout", "pqk_good", "pqk_bad"],   # add "fid_good", "fid_bad" for the fidelity kernel
    "max_circuits_per_job": 1000,      # Unai's current per-job cap
    "budget_minutes": 50.0,            # monthly QPU allowance on ibm_basquecountry
    "keep_fraction": 0.25,             # fraction of observables kept by the pruning selectors
    "seed": 42,
    "chains": None,                    # {"good": [..], "bad": [..]} to fix qubits by hand; null = pick from calibration
    "include_october_points": True,    # put the 16 Sonar points of the Oct 21 runs (results/hw_live.json) inside the test set
}
STAGES = ("readout", "pqk_good", "pqk_bad", "fid_good", "fid_bad")

# QPU-time model, fitted on our Oct 21 runs (Marrakesh 6q: 240 circuits x 1000 shots = 66 s;
# Marrakesh / Kingston 12q: 304 circuits x 1000 shots = 83-85 s; readout 4 circuits = 3 s).
QPU_S_PER_JOB = 2.5
QPU_S_PER_CIRCUIT_SHOT = 2.75e-4


def load_config(path: str | None, overrides: list[str] | None) -> dict:
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        cfg.update(json.load(open(path)))
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown config key {k}")
        cfg[k] = json.loads(v) if v[:1] in "[{\"" or v in ("null", "true", "false") else type(DEFAULTS[k])(v) if DEFAULTS[k] is not None else v
    cfg["stages"] = [s for s in cfg["stages"] if s in STAGES]
    return cfg


# ---------------------------------------------------------------------------- measurement settings
def pauli_covering(n: int) -> list[str]:
    """Measurement bases such that every pair of qubits sees all 9 Pauli combinations.

    Qubit q gets a direction v_q in GF(3)^k, pairwise linearly independent; the setting indexed by
    s in GF(3)^k measures qubit q in basis PAULIS[v_q . s mod 3].  k=3 covers 13 qubits with 27
    settings (what pqk.measurement_settings does); k=4 covers 40 qubits with 81 settings.  The greedy
    set cover below then keeps only the settings actually needed (15 for 6 qubits, 19 for 12).
    """
    for k in (3, 4):
        vectors: list[tuple[int, ...]] = []
        for v in itertools.product(range(3), repeat=k):
            if not any(v):
                continue
            if any(all((a * m) % 3 == b for a, b in zip(v, w)) for w in vectors for m in (1, 2)):
                continue
            vectors.append(v)
        if n <= len(vectors):
            break
    else:
        raise ValueError("covering supports at most 40 qubits")
    settings = {"".join(PAULIS[sum(a * b for a, b in zip(vectors[q], s)) % 3] for q in range(n))
                for s in itertools.product(range(3), repeat=k)}
    return sorted(settings)


def measurement_plan(n: int, obs: list[Observable]) -> tuple[list[str], dict[Observable, list[int]]]:
    """Greedy set cover of the covering: (settings, serves[o] = indices of settings that estimate o)."""
    allset = pauli_covering(n)
    serves_all = {o: [i for i, s in enumerate(allset) if all(s[q] == p for q, p in zip(o.qubits, o.paulis))] for o in obs}
    uncovered, chosen = set(range(len(obs))), []
    while uncovered:
        best = max(range(len(allset)), key=lambda j: sum(1 for m in uncovered if j in serves_all[obs[m]]))
        chosen.append(best)
        uncovered -= {m for m in uncovered if best in serves_all[obs[m]]}
    keep = sorted(set(chosen))
    remap = {old: new for new, old in enumerate(keep)}
    return [allset[j] for j in keep], {o: [remap[j] for j in js if j in remap] for o, js in serves_all.items()}


# ---------------------------------------------------------------------------- design
def pick_points(cfg: dict) -> dict[str, np.ndarray]:
    """Balanced train / test subsets of the dataset's own split, scaled to rotation angles."""
    X_tr, X_te, y_tr, y_te = load_dataset(cfg["dataset"], n_qubits=cfg["n_qubits"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    def balanced(X, y, n):
        per = n // 2
        idx = np.concatenate([rng.choice(np.where(y == c)[0], min(per, int((y == c).sum())), replace=False) for c in (-1, 1)])
        return X[idx], y[idx]

    Xa, ya = balanced(X_tr, y_tr, cfg["n_train"])
    Xb, yb = balanced(X_te, y_te, cfg["n_test"])
    oct_path = RESULTS / "hw_live.json"
    if cfg.get("include_october_points") and cfg["dataset"] == "sonar" and cfg["n_qubits"] == 6 and oct_path.exists():
        # continuity with the Oct 21 hardware run: its 16 points go first in the test set (they are a subset of
        # the dataset's own test split with seed 42, so no leakage into training)
        oc = json.load(open(oct_path)); Xo, yo = np.array(oc["X"]), np.array(oc["y"])
        keep = [i for i in range(len(Xb)) if not any(np.allclose(Xb[i], x) for x in Xo)]
        Xb, yb = np.vstack([Xo, Xb[keep]])[:max(cfg["n_test"], len(Xo))], np.concatenate([yo, yb[keep]])[:max(cfg["n_test"], len(Xo))]
    return {"X_train": Xa, "y_train": ya, "X_test": Xb, "y_test": yb}


def pick_chains(cal: dict, n: int, fixed: dict | None = None) -> dict:
    """Best and worst usable linear chain of length n (2q gate error <= MAX_2Q_ERROR), by readout dose."""
    if fixed:
        return {k: chain_profile(cal, tuple(v)) for k, v in fixed.items()}
    profs = [p for c in linear_chains(cal["edges"], n) if (p := chain_profile(cal, c)) is not None
             and max(p["two_qubit"]) <= MAX_2Q_ERROR]
    if not profs:
        raise SystemExit(f"no usable {n}-qubit chain on this device today")
    profs.sort(key=lambda p: p["dose_eq"])
    return {"good": profs[0], "bad": profs[-1]}


def build_circuits(cfg: dict, pts: dict) -> dict:
    n = cfg["n_qubits"]
    obs = build_observables(n)
    settings, _ = measurement_plan(n, obs)
    X_all = np.vstack([pts["X_train"], pts["X_test"]])
    pqk = [_measurement_circuit(feature_map(x), s, tuple(range(n))) for x in X_all for s in settings]
    # fidelity: train x train (upper triangle) + test x train
    pairs = [(i, j) for i, j in itertools.combinations(range(len(pts["X_train"])), 2)]
    pairs += [(len(pts["X_train"]) + a, b) for a in range(len(pts["X_test"])) for b in range(len(pts["X_train"]))]
    fid = []
    for i, j in pairs:
        qc = QuantumCircuit(n, n)
        qc.compose(feature_map(X_all[i]), inplace=True)
        qc.compose(feature_map(X_all[j]).inverse(), inplace=True)
        qc.measure(range(n), range(n))
        fid.append(qc)
    zeros = QuantumCircuit(n, n); zeros.measure(range(n), range(n))
    ones = QuantumCircuit(n, n); ones.x(range(n)); ones.measure(range(n), range(n))
    return {"pqk": pqk, "fid": fid, "fid_pairs": pairs, "readout": [zeros, ones], "settings": settings, "n_obs": len(obs)}


def stage_circuits(circs: dict, chains: dict) -> dict[str, tuple[list[QuantumCircuit], list[int]]]:
    g, b = chains["good"]["chain"], chains["bad"]["chain"]
    return {"readout": (circs["readout"] * 2, [g, g, b, b]),
            "pqk_good": (circs["pqk"], g), "pqk_bad": (circs["pqk"], b),
            "fid_good": (circs["fid"], g), "fid_bad": (circs["fid"], b)}


# ---------------------------------------------------------------------------- budget
def n_circuits(cfg: dict) -> dict[str, int]:
    n, ntr, nte = cfg["n_qubits"], cfg["n_train"], cfg["n_test"]
    n_set = len(measurement_plan(n, build_observables(n))[0])
    n_pqk = (ntr + nte) * n_set
    n_fid = ntr * (ntr - 1) // 2 + nte * ntr
    return {"readout": 4, "pqk_good": n_pqk, "pqk_bad": n_pqk, "fid_good": n_fid, "fid_bad": n_fid, "_settings": n_set}


def estimate_seconds(n_circ: int, shots: int, per_job: int) -> float:
    jobs = max(1, math.ceil(n_circ / per_job))
    return jobs * QPU_S_PER_JOB + n_circ * shots * QPU_S_PER_CIRCUIT_SHOT


def estimate(cfg: dict) -> dict:
    """Circuits, jobs and estimated QPU minutes per stage; printed against the monthly budget."""
    nc = n_circuits(cfg)
    n_obs = len(build_observables(cfg["n_qubits"]))
    print(f"== estimate: {cfg['n_qubits']} qubits ({n_obs} observables, {nc['_settings']} measurement settings), "
          f"{cfg['dataset']} {cfg['n_train']} train + {cfg['n_test']} test points, {cfg['shots']} shots")
    print(f"   {'stage':10s} {'circuits':>9s} {'jobs':>5s} {'QPU min':>8s}  selected")
    tot_sel = tot_all = 0.0
    out = {}
    for st in STAGES:
        c = nc[st]
        s = estimate_seconds(c, cfg["shots"], cfg["max_circuits_per_job"])
        sel = st in cfg["stages"]
        out[st] = {"circuits": c, "jobs": math.ceil(c / cfg["max_circuits_per_job"]), "qpu_min": s / 60, "selected": sel}
        tot_all += s; tot_sel += s if sel else 0
        print(f"   {st:10s} {c:9d} {out[st]['jobs']:5d} {s/60:8.1f}  {'yes' if sel else '-'}")
    print(f"   selected stages: {tot_sel/60:.1f} min of {cfg['budget_minutes']:.0f} ({100*tot_sel/60/cfg['budget_minutes']:.0f}% of the budget)"
          f"   | all five stages: {tot_all/60:.1f} min")
    if tot_sel / 60 > cfg["budget_minutes"]:
        print("   !! over budget: reduce n_train / n_test, drop the fidelity stages, or split across months")
    out["total_selected_min"] = tot_sel / 60
    return out


# ---------------------------------------------------------------------------- run
def connect(cfg: dict):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(name=cfg["service_name"])   # Unai's saved credentials
    return service, service.backend(cfg["backend"])


def transpile_to(backend, circs, layout, seed=7):
    return transpile(circs, backend=backend, initial_layout=list(layout), optimization_level=1, seed_transpiler=seed)


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def run(cfg: dict, backend, dry: bool = False, submit_only: bool = True) -> dict:
    """Submit the configured stages in Batch mode.  State (design, chains, job ids, counts) is saved after
    every job so a crash or a closed laptop loses nothing: `fetch` completes it later."""
    from qiskit_ibm_runtime import Batch, SamplerV2
    cal = dump_calibration(backend)
    chains = pick_chains(cal, cfg["n_qubits"], cfg["chains"])
    pts = pick_points(cfg)
    circs = build_circuits(cfg, pts)
    staged = stage_circuits(circs, chains)
    path = RESULTS / f"basque_{cfg['tag']}.json"
    state = json.load(open(path)) if path.exists() else {}
    state.update({"config": cfg, "backend": backend.name, "dry_run": dry, "pulled_at": cal["pulled_at"],
                  "X_train": pts["X_train"].tolist(), "y_train": pts["y_train"].tolist(),
                  "X_test": pts["X_test"].tolist(), "y_test": pts["y_test"].tolist(),
                  "settings": circs["settings"], "fid_pairs": circs["fid_pairs"], "chains": chains,
                  "calibration": {"readout_error": cal["readout_error"], "two_qubit_error": cal["two_qubit_error"]}})
    state.setdefault("jobs", {}); state.setdefault("counts", {}); state.setdefault("usage_s", {})
    for name in ("good", "bad"):
        c = chains[name]
        print(f"   {name:4s} chain {c['chain']}  readout {[f'{v*100:.1f}' for v in c['readout']]}%  dose_eq={c['dose_eq']:.2f}  2q={c['cx_mean']*100:.2f}%")
    est = estimate(cfg)
    if not dry:
        print(f"   about to submit to {backend.name}: ~{est['total_selected_min']:.1f} QPU minutes")
        if input("   type YES to submit: ").strip() != "YES":
            print("   aborted, nothing submitted"); return state
    RESULTS.mkdir(exist_ok=True)
    ctx = Batch(backend=backend) if not dry else None
    sampler = SamplerV2(mode=ctx if ctx else backend)
    try:
        for st in cfg["stages"]:
            if st in state["counts"]:
                print(f"   {st}: already have counts, skipped"); continue
            cl, lay = staged[st]
            if st == "readout":
                tc = transpile_to(backend, cl[:2], lay[0]) + transpile_to(backend, cl[2:], lay[2])
            else:
                t0 = time.time(); tc = transpile_to(backend, cl, lay)
                print(f"   {st}: transpiled {len(tc)} circuits in {time.time()-t0:.0f}s")
            ids, cnts = [], []
            for k, ch in enumerate(chunks(tc, cfg["max_circuits_per_job"])):
                job = sampler.run(ch, shots=cfg["shots"])
                ids.append(job.job_id())
                print(f"   {st}: job {k+1}/{math.ceil(len(tc)/cfg['max_circuits_per_job'])} = {job.job_id()} ({len(ch)} circuits)")
                state["jobs"][st] = ids; json.dump(state, open(path, "w"))
                if dry or not submit_only:
                    res = job.result()
                    cnts += [r.data.c.get_counts() for r in res]
                    try:
                        state["usage_s"][st] = state["usage_s"].get(st, 0) + float(job.usage() or 0)
                    except Exception:
                        pass
            if cnts:
                state["counts"][st] = cnts
            json.dump(state, open(path, "w"))
    finally:
        if ctx:
            ctx.close()
    print(f"   saved {path}")
    return state


def fetch(cfg: dict) -> None:
    service, _ = connect(cfg)
    path = RESULTS / f"basque_{cfg['tag']}.json"
    state = json.load(open(path))
    for st, ids in state["jobs"].items():
        if st in state["counts"]:
            continue
        cnts, done = [], True
        for jid in ids:
            job = service.job(jid)
            status = str(job.status()).upper()
            print(f"   {st}: {jid} {status}")
            if not status.endswith("DONE"):
                done = False; break
            cnts += [r.data.c.get_counts() for r in job.result()]
            try:
                state["usage_s"][st] = state["usage_s"].get(st, 0) + float(job.usage() or 0)
            except Exception:
                pass
        if done:
            state["counts"][st] = cnts
    json.dump(state, open(path, "w"))
    print("   usage (s):", state.get("usage_s"))


# ---------------------------------------------------------------------------- analysis helpers
def features_from_counts(counts, settings, n, obs, n_points) -> np.ndarray:
    _, serves = measurement_plan(n, obs)
    F = np.empty((n_points, len(obs)))
    for k in range(n_points):
        cs = counts[k * len(settings):(k + 1) * len(settings)]
        for m, o in enumerate(obs):
            acc, tot = 0.0, 0
            for j in serves[o]:
                for bitstr, c in cs[j].items():
                    bits = bitstr[::-1]
                    acc += (1 - 2 * (sum(int(bits[q]) for q in o.qubits) % 2)) * c
                    tot += c
            F[k, m] = acc / tot
    return F


def fidelity_from_counts(counts, pairs, n, ntr, nte):
    K_tr, K_te = np.eye(ntr), np.zeros((nte, ntr))
    for (i, j), c in zip(pairs, counts):
        p0 = c.get("0" * n, 0) / sum(c.values())
        if i < ntr:
            K_tr[i, j] = K_tr[j, i] = p0
        else:
            K_te[i - ntr, j] = p0
    return K_tr, K_te


def readout_from_counts(counts, n):
    def rate(c, expect):
        tot = sum(c.values()); r = np.zeros(n)
        for bitstr, cnt in c.items():
            bits = bitstr[::-1]
            for q in range(n):
                r[q] += cnt * (bits[q] != expect)
        return r / tot
    return (rate(counts[0], "0") + rate(counts[1], "1")) / 2


def svm_eval(K_tr, K_te, y_tr, y_te, ref_decision=None, ref_pred=None) -> dict:
    """Accuracy of a precomputed-kernel SVM, plus agreement with a reference (noiseless) classifier."""
    clf = SVC(kernel="precomputed", C=1.0).fit(K_tr, y_tr)
    dec, pred = clf.decision_function(K_te), clf.predict(K_te)
    out = {"acc": float((pred == y_te).mean())}
    if ref_decision is not None:
        out["decision_corr"] = float(np.corrcoef(dec, ref_decision)[0, 1])
        out["flips"] = float((pred != ref_pred).mean())
    return out | {"_dec": dec, "_pred": pred}


def kernels(F_tr, F_te, idx, gamma_ref=None):
    g = gamma_ref if gamma_ref is not None else median_gamma(F_tr[:, idx])
    return gaussian_kernel(F_tr[:, idx], F_tr[:, idx], g), gaussian_kernel(F_te[:, idx], F_tr[:, idx], g)


def prune(scores: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Top-scoring observables, never splitting a tie group at the cutoff.

    W = prod(1-2p) is identical for the 9 Pauli products of the same qubit pair, so a 25 % cutoff usually falls
    inside a group of 9 equal scores.  kernel.prune breaks such ties with np.argsort's default quicksort, whose
    order depends on the numpy build; `verify` showed the October Spearman values move in the 3rd decimal with
    it (Marrakesh 6q bad chain: 0.980-0.993 over tie orders, reported 0.988).  Keeping every observable tied
    with the last one kept is deterministic on all machines and physically sensible (equally clean observables
    are treated alike).  The kept fraction is then slightly above keep_fraction.
    """
    top_k = max(1, int(round(keep_fraction * len(scores))))
    cutoff = np.sort(scores)[::-1][top_k - 1]
    return np.where(scores >= cutoff - 1e-12)[0]


def chain_metrics(F_hw, F_ex, p_cal, p_meas, obs, keep_fraction=0.25):
    """The October analysis, unchanged: attenuation slopes vs prod(1-2p), and Spearman of the hardware kernel
    against the exact one for full / pruned-by-W / mitigated features.  Shared by `analyze` and `verify`."""
    n_pts = len(F_ex)
    slope = np.array([float(F_hw[:, m] @ F_ex[:, m] / max(F_ex[:, m] @ F_ex[:, m], 1e-9)) for m in range(len(obs))])
    pred_cal = np.array([np.prod([1 - 2 * p_cal[q] for q in o.qubits]) for o in obs])
    pred_meas = np.array([np.prod([1 - 2 * p_meas[q] for q in o.qubits]) for o in obs])
    strong = np.array([F_ex[:, m] @ F_ex[:, m] > 0.5 * n_pts / 16 for m in range(len(obs))])  # = 0.5 for the 16 Oct points
    att = {"corr_pred_cal": float(np.corrcoef(pred_cal[strong], slope[strong])[0, 1]),
           "corr_pred_meas": float(np.corrcoef(pred_meas[strong], slope[strong])[0, 1]),
           "rms_residual": float(np.sqrt(np.mean((slope[strong] - pred_meas[strong]) ** 2))), "n_obs_used": int(strong.sum()),
           "mean_slope_1q": float(slope[[o.weight == 1 for o in obs]].mean()), "mean_slope_2q": float(slope[[o.weight == 2 for o in obs]].mean())}
    keep_W = prune(pred_cal, keep_fraction=keep_fraction)
    iu = np.triu_indices(n_pts, k=1)
    def sp(Fa, Fb, idx):
        g = median_gamma(Fb[:, idx])
        return float(spearmanr(gaussian_kernel(Fa[:, idx], Fa[:, idx], g)[iu], gaussian_kernel(Fb[:, idx], Fb[:, idx], g)[iu]).correlation)
    allidx = np.arange(len(obs)); F_mit = F_hw / np.clip(pred_meas, 0.05, None)
    spear = {"pqk_full": sp(F_hw, F_ex, allidx), "pqk_pruned_W": sp(F_hw, F_ex, keep_W), "pqk_mitigated": sp(F_mit, F_ex, allidx)}
    return {"slope": slope, "pred_cal": pred_cal, "pred_meas": pred_meas, "strong": strong, "keep_W": keep_W, "attenuation": att, "spearman": spear}


def verify_october(path: str = "results/hw_live.json", ref: str = "results/hw_live_analysis.json", tol: float = 1e-6) -> bool:
    """Replicability check: run the Oct 21 Marrakesh counts through this file's analysis and compare with the
    numbers hardware_experiment.py produced then.  Also checks the measurement settings are identical."""
    st, old = json.load(open(path)), json.load(open(ref))
    n = len(st["X"][0]); obs = build_observables(n)
    new_settings, _ = measurement_plan(n, obs)
    same = list(st["settings"]) == list(new_settings)
    print(f"== verify against {path}: {n} qubits, {len(st['X'])} points, settings identical: {same}")
    X = np.array(st["X"]); F_ex = exact_features(X, obs); ok = same
    for ci, name in enumerate(("good", "bad")):
        p_meas = readout_from_counts(st["counts"]["readout"][2 * ci:2 * ci + 2], n)
        F_hw = features_from_counts(st["counts"][f"pqk_{name}"], st["settings"], n, obs, len(X))
        m = chain_metrics(F_hw, F_ex, np.array(st["chains"][name]["readout"]), p_meas, obs)
        o = old["chains"][name]
        pairs = [("attenuation corr (cal p)", m["attenuation"]["corr_pred_cal"], o["attenuation"]["corr_pred_cal_vs_measured"]),
                 ("attenuation corr (meas p)", m["attenuation"]["corr_pred_meas"], o["attenuation"]["corr_pred_meas_vs_measured"]),
                 ("rms residual", m["attenuation"]["rms_residual"], o["attenuation"]["rms_residual"])]
        pairs += [(f"spearman {k}", m["spearman"][k], o["spearman"][k]) for k in m["spearman"]]
        tie_tol = {"spearman pqk_pruned_W": 0.01}   # October split a tie group of 9 with an unstable sort; see prune()
        print(f"   {name} chain {st['chains'][name]['chain']}")
        for label, a, b in pairs:
            t = tie_tol.get(label, tol); good = abs(a - b) < t; ok &= good
            print(f"      {label:28s} new {a:.6f}   oct {b:.6f}   {'ok' if good else 'MISMATCH'}{'  (tie group kept whole, tol 0.01)' if label in tie_tol else ''}")
    print("   RESULT:", "PASS: identical settings, attenuation and full/mitigated Spearman to 1e-6; pruned_W within tie-break tolerance" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------- analysis
def analyze(cfg: dict) -> dict:
    st = json.load(open(RESULTS / f"basque_{cfg['tag']}.json"))
    c = st["config"]; n = c["n_qubits"]
    Xtr, ytr, Xte, yte = (np.array(st[k]) for k in ("X_train", "y_train", "X_test", "y_test"))
    ntr, nte = len(Xtr), len(Xte)
    obs = build_observables(n)
    F_ex_tr, F_ex_te = exact_features(Xtr, obs), exact_features(Xte, obs)
    allidx = np.arange(len(obs))
    g_ex = median_gamma(F_ex_tr)
    K_ex_tr, K_ex_te = kernels(F_ex_tr, F_ex_te, allidx, g_ex)
    ref = svm_eval(K_ex_tr, K_ex_te, ytr, yte)
    out = {"backend": st["backend"], "dry_run": st["dry_run"], "n_qubits": n, "n_train": ntr, "n_test": nte,
           "usage_s": st.get("usage_s"), "noiseless": {"acc": ref["acc"]}, "chains": {}}
    print(f"== analyze {st['backend']} ({'dry run' if st['dry_run'] else 'live'}): {n} qubits, {ntr} train / {nte} test, noiseless PQK acc {ref['acc']:.3f}")
    iu = np.triu_indices(ntr, k=1)
    fig1, ax1 = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    bars = {}
    for ci, name in enumerate(("good", "bad")):
        if f"pqk_{name}" not in st["counts"]:
            print(f"   {name:4s} chain: no PQK counts yet, skipped"); continue
        ch = st["chains"][name]
        p_cal = np.array(ch["readout"])
        p_meas = readout_from_counts(st["counts"]["readout"][2 * ci:2 * ci + 2], n) if "readout" in st["counts"] else p_cal
        F_hw = features_from_counts(st["counts"][f"pqk_{name}"], st["settings"], n, obs, ntr + nte)
        F_tr, F_te = F_hw[:ntr], F_hw[ntr:]

        # 1. attenuation + kernel ordering, exactly as in October (shared with `verify`)
        F_ex_all = np.vstack([F_ex_tr, F_ex_te])
        cm = chain_metrics(F_hw, F_ex_all, p_cal, p_meas, obs, c["keep_fraction"])
        slope, pred_cal, pred_meas, strong = cm["slope"], cm["pred_cal"], cm["pred_meas"], cm["strong"]
        r_cal, r_meas = cm["attenuation"]["corr_pred_cal"], cm["attenuation"]["corr_pred_meas"]

        # 2. selectors (train fold only, no test labels used)
        W = pred_cal
        keep_W = cm["keep_W"]
        kta_o = per_observable_kta(F_tr, ytr, median_gamma(F_tr))
        rank = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
        keep_KW = prune(rank(kta_o) * rank(W), keep_fraction=c["keep_fraction"])
        F_mit_tr, F_mit_te = F_tr / np.clip(pred_meas, 0.05, None), F_te / np.clip(pred_meas, 0.05, None)

        variants = {"pqk_full": (F_tr, F_te, allidx), "pqk_pruned_W": (F_tr, F_te, keep_W),
                    "pqk_pruned_KTAxW": (F_tr, F_te, keep_KW), "pqk_mitigated": (F_mit_tr, F_mit_te, allidx)}
        res = {"chain": ch["chain"], "dose_eq": ch["dose_eq"], "readout_calibration": p_cal.tolist(), "readout_measured": p_meas.tolist(),
               "attenuation": cm["attenuation"], "spearman_october_style": cm["spearman"], "kernels": {}}
        for k, (a, b, idx) in variants.items():
            # two effects, reported separately: (i) what pruning itself costs, measured on NOISELESS features
            # restricted to the same observables; (ii) what noise costs, measured against that same-subset
            # noiseless classifier (decision correlation and flips).  flips_vs_full keeps the old all-in number.
            K_tr, K_te = kernels(a, b, idx)
            g_sub = median_gamma(F_ex_tr[:, idx])
            K_ref_tr, K_ref_te = kernels(F_ex_tr, F_ex_te, idx, g_sub)
            ref_sub = svm_eval(K_ref_tr, K_ref_te, ytr, yte)
            ev = svm_eval(K_tr, K_te, ytr, yte, ref_sub["_dec"], ref_sub["_pred"])
            ev_full = svm_eval(K_tr, K_te, ytr, yte, ref["_dec"], ref["_pred"])
            res["kernels"][k] = {"spearman": float(spearmanr(K_tr[iu], K_ref_tr[iu]).correlation), "n_obs": int(len(idx)),
                                 "acc": ev["acc"], "acc_noiseless_same_obs": ref_sub["acc"],
                                 "decision_corr": ev["decision_corr"], "flips": ev["flips"], "flips_vs_full": ev_full["flips"]}
        # keep-fraction sweep: how much to prune is a free post-hoc question (the counts hold all observables)
        res["keep_sweep"] = []
        for kf in (0.10, 0.25, 0.50, 0.75, 1.0):
            row = {"keep_fraction": kf}
            for sel, score in (("W", W), ("KTAxW", rank(kta_o) * rank(W))):
                idx = prune(score, kf) if kf < 1.0 else allidx
                K_tr, K_te = kernels(F_tr, F_te, idx)
                g_sub = median_gamma(F_ex_tr[:, idx]); K_ref_tr, K_ref_te = kernels(F_ex_tr, F_ex_te, idx, g_sub)
                rs = svm_eval(K_ref_tr, K_ref_te, ytr, yte); ev = svm_eval(K_tr, K_te, ytr, yte, rs["_dec"], rs["_pred"])
                row[sel] = {"n_obs": int(len(idx)), "spearman": float(spearmanr(K_tr[iu], K_ref_tr[iu]).correlation),
                            "acc_hw": ev["acc"], "acc_exact": rs["acc"], "noise_flips": ev["flips"]}
            res["keep_sweep"].append(row)
        print(f"        keep-fraction sweep (selector W | KTAxW):  n_obs  spearman  acc hw / acc exact  noise flips")
        for row in res["keep_sweep"]:
            w_, k_ = row["W"], row["KTAxW"]
            print(f"          keep {int(row['keep_fraction']*100):3d}%   W: {w_['n_obs']:3d} {w_['spearman']:.3f} {w_['acc_hw']:.2f}/{w_['acc_exact']:.2f} {100*w_['noise_flips']:4.0f}%   |  KTAxW: {k_['n_obs']:3d} {k_['spearman']:.3f} {k_['acc_hw']:.2f}/{k_['acc_exact']:.2f} {100*k_['noise_flips']:4.0f}%")
        if f"fid_{name}" in st["counts"]:
            K_tr, K_te = fidelity_from_counts(st["counts"][f"fid_{name}"], st["fid_pairs"], n, ntr, nte)
            from kernel_comparison import fidelity_kernel_exact
            K_fx_tr = fidelity_kernel_exact(Xtr, Xtr)
            ev = svm_eval(K_tr, K_te, ytr, yte, ref["_dec"], ref["_pred"])
            K_fx_te = fidelity_kernel_exact(Xte, Xtr); ref_f = svm_eval(K_fx_tr, K_fx_te, ytr, yte)
            ev = svm_eval(K_tr, K_te, ytr, yte, ref_f["_dec"], ref_f["_pred"])
            res["kernels"]["fidelity"] = {"spearman": float(spearmanr(K_tr[iu], K_fx_tr[iu]).correlation), "n_obs": None,
                                          "acc": ev["acc"], "acc_noiseless_same_obs": ref_f["acc"],
                                          "decision_corr": ev["decision_corr"], "flips": ev["flips"], "flips_vs_full": ev["flips"]}
        out["chains"][name] = res; bars[name] = res["kernels"]
        print(f"   {name:4s} chain {ch['chain']} dose_eq {ch['dose_eq']:.2f}: attenuation corr {r_meas:.2f} (measured p) / {r_cal:.2f} (calibration p)")
        print(f"        {'kernel':18s} {'spearman':>8s} {'acc hw':>7s} {'acc exact':>9s} {'dec corr':>8s} {'noise flips':>11s} {'flips vs full':>13s}")
        for k, v in res["kernels"].items():
            print(f"        {k:18s} {v['spearman']:8.3f} {v['acc']:7.3f} {v['acc_noiseless_same_obs']:9.3f} {v['decision_corr']:8.3f} {100*v['flips']:10.1f}% {100*v['flips_vs_full']:12.1f}%")
        print("        (acc exact = noiseless accuracy with the SAME observables -> cost of pruning; noise flips / dec corr = hardware vs that same-subset classifier -> cost of noise)")
        ax1[ci].scatter(pred_meas[strong], slope[strong], s=10, alpha=.7, c=["tab:blue" if o.weight == 1 else "tab:orange" for o, s_ in zip(obs, strong) if s_])
        ax1[ci].plot([0, 1], [0, 1], "k--", lw=1); ax1[ci].set_title(f"{name} chain {ch['chain']}\nr={r_meas:.2f}"); ax1[ci].set_xlabel("predicted  prod(1-2p)")
    ax1[0].set_ylabel("measured attenuation"); fig1.suptitle(f"Readout reweighting on {st['backend']} ({n} qubits)"); fig1.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig1.savefig(FIGURES / f"basque_{cfg['tag']}_attenuation.png", dpi=150); plt.close(fig1)
    if bars:
        keys = list(next(iter(bars.values())).keys())
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.8))
        xk = np.arange(len(keys))
        for off, name in ((-0.18, "good"), (0.18, "bad")):
            if name in bars:
                ax2[0].bar(xk + off, [bars[name][k]["spearman"] for k in keys], .36, label=f"{name} chain")
                ax2[1].bar(xk + off, [bars[name][k]["acc"] for k in keys], .36, label=f"{name} chain")
        ax2[1].axhline(ref["acc"], color="k", ls="--", lw=1, label="noiseless PQK")
        for a, t in zip(ax2, ("Spearman(K_hw, K_exact)", "test accuracy")):
            a.set_xticks(xk); a.set_xticklabels(keys, rotation=20, fontsize=8); a.set_ylim(0, 1.02); a.set_title(t); a.legend(fontsize=7)
        fig2.tight_layout(); fig2.savefig(FIGURES / f"basque_{cfg['tag']}_kernels.png", dpi=150); plt.close(fig2)
    json.dump(out, open(RESULTS / f"basque_{cfg['tag']}_analysis.json", "w"), indent=1)
    return out


# ---------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["estimate", "plan", "dry-run", "submit", "fetch", "analyze", "verify"])
    ap.add_argument("--config", default="basque_config.json")
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE", help="override config keys")
    ap.add_argument("--stages", nargs="*", choices=STAGES, help="override the stages to run")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.stages:
        cfg["stages"] = list(args.stages)
    if args.mode == "verify":
        raise SystemExit(0 if verify_october() else 1)
    if args.mode == "estimate":
        estimate(cfg)
    elif args.mode == "plan":
        _, b = connect(cfg)
        cal = dump_calibration(b); ch = pick_chains(cal, cfg["n_qubits"], cfg["chains"])
        print(f"== {b.name} ({cal['pulled_at'][:16]} UTC), {cfg['n_qubits']}-qubit chains")
        for k, p in ch.items():
            print(f"   {k:4s} {p['chain']} readout {[f'{v*100:.1f}' for v in p['readout']]}% dose_eq={p['dose_eq']:.2f} 2q={p['cx_mean']*100:.2f}%")
        estimate(cfg)
    elif args.mode == "dry-run":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        print(f"== DRY RUN on FakeMarrakesh (local noisy simulation), tag {cfg['tag']}")
        run(cfg, FakeMarrakesh(), dry=True)
        analyze(cfg)
    elif args.mode == "submit":
        _, b = connect(cfg)
        print(f"== LIVE on {b.name}, stages {cfg['stages']}")
        run(cfg, b, dry=False, submit_only=True)
        print("   jobs submitted in Batch mode; run `fetch` once they finish, then `analyze`")
    elif args.mode == "fetch":
        fetch(cfg)
    else:
        analyze(cfg)
