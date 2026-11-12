"""PQK observable pruning on ibm_basquecountry: the hardware experiment with a real train/test split.

Same mechanism test as hardware_experiment.py (Oct 2026, 16 points, Marrakesh / Kingston), extended for
a device with a real QPU budget (50 min/month) so that we can finally measure ACCURACY on hardware:

    * n_train + n_test labelled points instead of 16
    * best and worst usable linear chain of the device, picked from same-day calibration
    * PQK features on both chains; fidelity kernel optional, by default at the SAME TOTAL shot budget as PQK
      ("fid_shots": "matched"), the equal-budget baseline of the simulations
    * selectors: W only (calibration), rank(KTA) * rank(W) (train fold), readout mitigation
    * analyses: predicted vs measured attenuation, kernel ordering, SVM accuracy, decision correlation,
      flipped predictions, all against the noiseless kernel

Everything is driven by one JSON config (see basque_config.json) so the design can be changed in the
meeting without touching code.  Safety rules for real QPU time:
    * nothing is sent without the ACTUAL transpiled circuit count and QPU estimate being printed and YES typed;
    * the estimate carries a safety factor and a run that would exceed the monthly budget is refused;
    * the design (points, chains, calibration) is frozen in the state file on the first submit; later stages
      reuse it, so pqk_bad run a month after pqk_good measures the same qubits and the same points;
    * every job is tagged (tag, stage, chunk) and its id saved atomically before anything else; chunks already
      submitted are never resent, ids lost to a timeout are recovered from the server by tag, and only chunks
      that `fetch` marked as failed are resubmitted;
    * each stage carries its own readout check and records its own calibration snapshot, so a stage run on
      another day is analysed against that day;
    * dry runs always write to results/basque_<tag>_dry.json and `submit` refuses to touch a dry-run file.

    python basque_experiment.py estimate                         # circuits + QPU minutes per stage, no QPU
    python basque_experiment.py plan                             # today's best / worst chain on the device
    python basque_experiment.py dry-run                          # whole pipeline on FakeMarrakesh (local, noisy) -> <tag>_dry
    python basque_experiment.py submit --stages pqk_good         # live: submit one stage in Batch mode, then exit
    python basque_experiment.py fetch                            # pull finished jobs into results/basque_<tag>.json
    python basque_experiment.py analyze                          # metrics + figures (add --set tag=<tag>_dry for a dry run)
    python basque_experiment.py verify                           # replicability: re-run the Oct 21 counts, compare numbers

Config file: --config basque_config.json (default).  Any key can be overridden on the command line,
e.g. --set n_train=60 n_test=30 n_qubits=12.
"""
from __future__ import annotations

import argparse
import itertools
import json
import hashlib
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.stats import rankdata, spearmanr
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
    "fid_shots": "matched",            # shots per fidelity circuit: "matched" = same TOTAL shots as the PQK stage
                                       # (shots * n_pqk_circuits / n_fid_circuits), an integer, or null = same as shots
    "stages": ["pqk_good", "pqk_bad"],   # add "fid_good", "fid_bad" for the fidelity kernel
    "max_circuits_per_job": 1000,      # Unai's current per-job cap
    "budget_minutes": 50.0,            # monthly QPU allowance on ibm_basquecountry
    "keep_fraction": 0.25,             # fraction of observables kept by the pruning selectors
    "seed": 42,
    "chains": None,                    # {"good": [..], "bad": [..]} to fix qubits by hand; null = pick from calibration
    "qpu_safety_factor": 1.5,          # the time model comes from other devices; estimates are multiplied by this before the budget check
}
STAGES = ("pqk_good", "pqk_bad", "fid_good", "fid_bad")   # every CHUNK of a stage carries its own 2 readout circuits

# QPU-time model, fitted on three points from our Oct 21 runs (Marrakesh 6q: 240 circuits x 1000 shots = 66 s;
# Marrakesh / Kingston 12q: 304 circuits x 1000 shots = 83-85 s; readout 4 circuits = 3 s).  It is an
# extrapolation to a different device: hence qpu_safety_factor, and the rule that the first live stage is
# small and its measured usage_s (from `fetch`) is used to refit these two constants before the next stage.
QPU_S_PER_JOB = 2.5
QPU_S_PER_CIRCUIT_SHOT = 2.75e-4


def load_config(path: str | None, overrides: list[str] | None) -> dict:
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        file_cfg = json.load(open(path))
        unknown = [k for k in file_cfg if k not in DEFAULTS]
        if unknown:
            raise SystemExit(f"unknown key(s) in {path}: {unknown}; valid keys: {list(DEFAULTS)}")
        cfg.update(file_cfg)
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown config key {k}")
        if v[:1] in "[{\"" or v.lower() in ("null", "true", "false"):
            cfg[k] = json.loads(v.lower() if v.lower() in ("null", "true", "false") else v)
        elif isinstance(DEFAULTS[k], bool):
            raise SystemExit(f"{k} must be true or false")
        elif isinstance(DEFAULTS[k], (int, float)) and not isinstance(DEFAULTS[k], bool):
            try:
                cfg[k] = type(DEFAULTS[k])(v)
            except ValueError:
                raise SystemExit(f"{k} must be a {type(DEFAULTS[k]).__name__}, got {v!r}")
        else:
            cfg[k] = int(v) if v.isdigit() else v
    fs = cfg["fid_shots"]
    if not (fs is None or fs == "matched" or (isinstance(fs, int) and not isinstance(fs, bool) and fs > 0)):
        raise SystemExit(f"fid_shots must be \"matched\", a positive integer or null, got {fs!r}")
    bad = [x for x in cfg["stages"] if x not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; valid: {list(STAGES)}")
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
    if len(Xa) != cfg["n_train"] or len(Xb) != cfg["n_test"]:
        print(f"   note: dataset split too small for a balanced {cfg['n_train']}/{cfg['n_test']}; using {len(Xa)}/{len(Xb)}")
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
    """Payload circuits per stage.  The 2 readout-check circuits (|0..0>, |1..1>) are added to EVERY chunk at
    submission time, so each chunk, whatever day it runs, carries its own readout measurement."""
    g, b = chains["good"]["chain"], chains["bad"]["chain"]
    return {"pqk_good": (circs["pqk"], g), "pqk_bad": (circs["pqk"], b),
            "fid_good": (circs["fid"], g), "fid_bad": (circs["fid"], b)}


RO_PER_CHUNK = 2


def payload_per_chunk(cfg: dict) -> int:
    return max(1, cfg["max_circuits_per_job"] - RO_PER_CHUNK)


# ---------------------------------------------------------------------------- budget
def n_circuits(cfg: dict) -> dict[str, int]:
    n, ntr, nte = cfg["n_qubits"], cfg["n_train"], cfg["n_test"]
    n_set = len(measurement_plan(n, build_observables(n))[0])
    per = payload_per_chunk(cfg)
    def total(payload):                                      # payload + 2 readout circuits per chunk
        return payload + RO_PER_CHUNK * math.ceil(payload / per) if payload else 0
    n_pqk = (ntr + nte) * n_set
    n_fid = ntr * (ntr - 1) // 2 + nte * ntr
    return {"pqk_good": total(n_pqk), "pqk_bad": total(n_pqk), "fid_good": total(n_fid), "fid_bad": total(n_fid), "_settings": n_set}


def payload_counts(cfg: dict) -> tuple[int, int]:
    """(PQK circuits, fidelity circuits) per chain, without the readout circuits."""
    n, ntr, nte = cfg["n_qubits"], cfg["n_train"], cfg["n_test"]
    n_set = len(measurement_plan(n, build_observables(n))[0])
    return (ntr + nte) * n_set, ntr * (ntr - 1) // 2 + nte * ntr


def stage_shots(cfg: dict, st: str) -> int:
    """Shots per circuit for a stage.  PQK stages use `shots`.  Fidelity stages use `fid_shots`: an integer,
    null (= `shots`), or "matched" = the same TOTAL number of shots as the PQK stage on that chain, spread over
    the n^2 fidelity circuits -- the equal-budget comparison used in the simulations (Oct 5)."""
    if not st.startswith("fid"):
        return cfg["shots"]
    fs = cfg["fid_shots"]
    if fs is None:
        return cfg["shots"]
    if fs == "matched":
        n_pqk, n_fid = payload_counts(cfg)
        return max(1, int(round(cfg["shots"] * n_pqk / max(n_fid, 1))))
    return int(fs)


def estimate_seconds(n_circ: int, shots: int, per_job: int, safety: float = 1.0) -> float:
    jobs = max(1, math.ceil(n_circ / per_job))
    return safety * (jobs * QPU_S_PER_JOB + n_circ * shots * QPU_S_PER_CIRCUIT_SHOT)


def estimate(cfg: dict, actual: dict | None = None, spent_min: float = 0.0, pending_min: float = 0.0) -> dict:
    """Circuits, jobs and estimated QPU minutes per stage against the monthly budget.

    `actual` = {stage: circuits that will really be sent} overrides the arithmetic from n_train/n_test (used by
    `submit`, so the YES prompt shows what goes out).  `spent_min` = QPU minutes already charged (from `fetch`);
    `pending_min` = estimate for chunks already submitted whose usage is not known yet."""
    nc = n_circuits(cfg)
    if actual:
        nc.update(actual)
    n_obs = len(build_observables(cfg["n_qubits"])); sf = cfg["qpu_safety_factor"]
    fsh = stage_shots(cfg, "fid_good")
    fid_txt = f" (fidelity {fsh}/circuit, fid_shots={cfg['fid_shots']!r})" if fsh != cfg["shots"] else ""
    print(f"== estimate: {cfg['n_qubits']} qubits ({n_obs} observables, {nc['_settings']} measurement settings), "
          f"{cfg['dataset']} {cfg['n_train']} train + {cfg['n_test']} test points, {cfg['shots']} shots{fid_txt}, safety factor x{sf}")
    print(f"   {'stage':10s} {'circuits':>9s} {'jobs':>5s} {'shots':>6s} {'QPU min':>8s}  selected")
    tot_sel = tot_all = 0.0
    out = {}
    for st in STAGES:
        c = nc[st]
        sh = stage_shots(cfg, st)
        s = estimate_seconds(c, sh, cfg["max_circuits_per_job"], sf) if c else 0.0
        sel = st in cfg["stages"]
        out[st] = {"circuits": c, "jobs": math.ceil(c / cfg["max_circuits_per_job"]), "shots": sh, "qpu_min": s / 60, "selected": sel}
        tot_all += s; tot_sel += s if sel else 0
        print(f"   {st:10s} {c:9d} {out[st]['jobs']:5d} {sh:6d} {s/60:8.1f}  {'yes' if sel else '-'}")
    left = cfg["budget_minutes"] - spent_min - pending_min
    extra = (f", {spent_min:.1f} already charged" if spent_min else "") + (f", {pending_min:.1f} submitted and not yet charged" if pending_min else "")
    print(f"   selected stages: {tot_sel/60:.1f} min of {left:.1f} left ({100*tot_sel/60/max(left, 1e-9):.0f}%){extra}   | all four stages: {tot_all/60:.1f} min")
    out["total_selected_min"] = tot_sel / 60; out["over_budget"] = tot_sel / 60 > left
    if out["over_budget"]:
        print("   !! over budget: reduce n_train / n_test, drop the fidelity stages, or split across months")
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


def state_path(cfg: dict) -> Path:
    return RESULTS / f"basque_{cfg['tag']}.json"


def save_state(state: dict, path: Path) -> None:
    """Atomic write: the state file is the only record of the frozen design and every job id."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def design_hash(state: dict) -> str:
    """Short fingerprint of the frozen design (points, chains, settings, shots, backend); goes into every job's
    tags so that jobs from another design under the same tag are never adopted."""
    blob = json.dumps({k: state[k] for k in ("X_train", "X_test", "settings", "backend")} |
                      {"chains": {n: state["chains"][n]["chain"] for n in ("good", "bad")}, "shots": state["config"]["shots"],
                       "per_chunk": payload_per_chunk(state["config"])}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def job_tags(cfg: dict, state: dict, st: str, k: int) -> list[str]:
    return [f"basque_{cfg['tag']}", f"design={state['design_hash']}", f"stage={st}", f"chunk={k}"]


def chunk_plan(st_jobs: dict, n_chunks: int, failed: list[int]) -> list[int]:
    """Chunk indices that still have to be submitted: never submitted, or marked failed by `fetch`."""
    return [k for k in range(n_chunks) if str(k) not in st_jobs.get("chunks", {}) or k in failed]


def reconcile_with_server(service, cfg: dict, state: dict) -> int:
    """Recover job ids that were created on the server but never saved (e.g. a timeout after submission),
    using the tags every job carries.  Returns the number of ids recovered."""
    found = 0
    try:   # newest first; only jobs carrying THIS design's fingerprint
        jobs = service.jobs(limit=500, job_tags=[f"basque_{cfg['tag']}", f"design={state['design_hash']}"], descending=True)
    except Exception as e:
        # without the server list we cannot know whether an unrecorded job exists; sending now could double-spend
        raise SystemExit(f"   refused: could not list server jobs to reconcile ({e}); retry when the connection is back")
    superseded = set(state.get("superseded", []))
    for job in jobs:
        raw = job.tags
        raw = raw() if callable(raw) else raw                 # property in 0.43, method in older releases
        tags = {t.split("=", 1)[0]: t.split("=", 1)[1] for t in (raw or []) if "=" in t}
        st, k = tags.get("stage"), tags.get("chunk")
        if st in STAGES and k is not None and job.job_id() not in superseded:
            entry = state["jobs"].setdefault(st, {"n_chunks": None, "chunks": {}})
            if k not in entry["chunks"]:                      # newest job for that chunk wins
                entry["chunks"][k] = job.job_id(); found += 1
                print(f"   recovered unrecorded job {job.job_id()} ({st}, chunk {k}) from the server")
    return found


def run(cfg: dict, backend, service=None, dry: bool = False, submit_only: bool = True) -> dict:
    """Submit the configured stages in Batch mode.

    First call: picks points and chains from today's calibration and FREEZES them (and the backend) in the state
    file.  Later calls reuse the frozen design and refuse config changes that would make stages incomparable.
    Each stage is sent in chunks of <= max_circuits_per_job; every job is tagged (tag, stage, chunk), the id is
    saved atomically before anything else happens, and only chunks that were never submitted or that `fetch`
    marked as failed are (re)sent.  Each stage records its own calibration snapshot and carries its own readout
    check, so stages run on different days are each analysed against their own day."""
    from qiskit_ibm_runtime import Batch, SamplerV2
    path = state_path(cfg); RESULTS.mkdir(exist_ok=True)
    state = json.load(open(path)) if path.exists() else {}
    if state and bool(state.get("dry_run")) != dry:
        raise SystemExit(f"{path} holds a {'dry-run' if state.get('dry_run') else 'LIVE'} design; use another tag")
    cal = dump_calibration(backend)
    if "X_train" in state:                                   # frozen design from an earlier call
        for key in ("n_qubits", "n_train", "n_test", "dataset", "seed", "shots", "backend", "max_circuits_per_job"):
            if state["config"][key] != cfg[key]:
                raise SystemExit(f"config {key}={cfg[key]} differs from the frozen design ({state['config'][key]}); use a new tag")
        if any(k.startswith("fid") for k in state.get("jobs", {})) and state["config"].get("fid_shots") != cfg["fid_shots"]:
            raise SystemExit(f"fid_shots={cfg['fid_shots']!r} differs from the frozen design ({state['config'].get('fid_shots')!r}) and fidelity jobs exist; use a new tag")
        state["config"]["fid_shots"] = cfg["fid_shots"]        # may still be chosen while no fidelity job exists
        if cfg["chains"] and {k: list(v) for k, v in cfg["chains"].items()} != {k: list(state["chains"][k]["chain"]) for k in cfg["chains"]}:
            raise SystemExit("chains differ from the frozen design; use a new tag")
        pts = {k: np.array(state[k]) for k in ("X_train", "y_train", "X_test", "y_test")}
        chains = state["chains"]
        print(f"   reusing frozen design from {path} (first calibration pulled {state['pulled_at'][:16]} UTC)")
    else:
        chains = pick_chains(cal, cfg["n_qubits"], cfg["chains"])
        pts = pick_points(cfg)
        state.update({"config": dict(cfg), "backend": backend.name, "dry_run": dry, "pulled_at": cal["pulled_at"],
                      "X_train": pts["X_train"].tolist(), "y_train": pts["y_train"].tolist(),
                      "X_test": pts["X_test"].tolist(), "y_test": pts["y_test"].tolist(), "chains": chains,
                      "calibration": {"readout_error": cal["readout_error"], "two_qubit_error": cal["two_qubit_error"]}})
    circs = build_circuits(cfg, pts)
    state.update({"settings": circs["settings"], "fid_pairs": circs["fid_pairs"]})
    state.setdefault("design_hash", design_hash(state))
    for key, default in (("jobs", {}), ("counts", {}), ("chunk_counts", {}), ("readout_counts", {}), ("usage_by_job", {}),
                         ("failed", {}), ("stage_cal", {}), ("superseded", [])):
        state.setdefault(key, default)
    save_state(state, path)                                    # design on disk before talking to the server
    staged = stage_circuits(circs, chains)
    for name in ("good", "bad"):
        c = chains[name]; today = chain_profile(cal, tuple(c["chain"]))
        warn = "" if today is None or max(today["two_qubit"]) <= MAX_2Q_ERROR else "  !! a 2q gate on this chain is above the usability limit today"
        print(f"   {name:4s} chain {c['chain']}  readout today {[f'{v*100:.1f}' for v in (today or c)['readout']]}%  dose_eq today={(today or c)['dose_eq']:.2f}{warn}")
    if service is not None and not dry:
        reconcile_with_server(service, cfg, state)
    # which chunks of which stage still have to go out
    todo, skipped, actual = {}, [], {}
    per = payload_per_chunk(cfg)
    for st in cfg["stages"]:
        n_chunks = math.ceil(len(staged[st][0]) / per)
        if st in state["jobs"]:
            state["jobs"][st]["n_chunks"] = n_chunks            # also repairs entries recovered from the server
        if st in state["counts"]:
            skipped.append((st, "complete")); continue
        missing = chunk_plan(state["jobs"].get(st, {}), n_chunks, state["failed"].get(st, []))
        if not missing:
            if assemble_stage(state, st, len(staged[st][0])):
                skipped.append((st, "complete (assembled from fetched chunks)"))
            else:
                skipped.append((st, f"all {n_chunks} chunk(s) submitted, run `fetch`"))
            continue
        todo[st] = (n_chunks, missing)
    for st, why in skipped:
        print(f"   {st}: skipped, {why}")
    if not todo:
        save_state(state, path); print("   nothing to submit"); return state
    transpiled = {}
    for st, (n_chunks, missing) in todo.items():
        cl, lay = staged[st]; t0 = time.time()
        ro_t = transpile_to(backend, circs["readout"], lay)
        tc = transpile_to(backend, cl, lay)
        parts = list(chunks(tc, per))
        transpiled[st] = {k: ro_t + parts[k] for k in missing}   # readout check first in every chunk
        actual[st] = sum(len(transpiled[st][k]) for k in missing)
        print(f"   {st}: transpiled {len(tc)} circuits in {time.time()-t0:.0f}s; chunks to send {missing} of {n_chunks} ({actual[st]} circuits incl. readout)")
    spent = sum(state["usage_by_job"].values()) / 60
    pending = sum(estimate_seconds(cfg["max_circuits_per_job"], stage_shots(cfg, st), cfg["max_circuits_per_job"], cfg["qpu_safety_factor"])
                  for st, e in state["jobs"].items() for k, jid in e.get("chunks", {}).items()
                  if jid not in state["usage_by_job"] and int(k) not in state["failed"].get(st, [])) / 60
    est = estimate({**cfg, "stages": list(todo)}, actual={**{st: 0 for st in STAGES}, **actual}, spent_min=spent, pending_min=pending)
    if not dry:
        if est["over_budget"]:
            raise SystemExit("   refused: estimated QPU time exceeds the remaining budget (lower n_train/n_test or run fewer stages)")
        shots_txt = ", ".join(f"{st}: {actual[st]} x {stage_shots(cfg, st)} shots" for st in todo)
        print(f"   about to submit to {backend.name}: {shots_txt}, "
              f"~{est['total_selected_min']:.1f} QPU minutes (with x{cfg['qpu_safety_factor']} safety), stages {list(todo)}")
        if input("   type YES to submit: ").strip() != "YES":
            print("   aborted, nothing submitted"); return state
    save_state(state, path)
    ctx = Batch(backend=backend) if not dry else None
    sampler = SamplerV2(mode=ctx if ctx else backend)
    try:
        for st, (n_chunks, missing) in todo.items():
            entry = state["jobs"].setdefault(st, {"n_chunks": n_chunks, "chunks": {}}); entry["n_chunks"] = n_chunks
            state["stage_cal"].setdefault(st, {})
            name = "good" if st.endswith("good") else "bad"
            for k in missing:
                old = entry["chunks"].pop(str(k), None)                   # a failed chunk's old id is retired FIRST
                if old:
                    state["superseded"].append(old)
                state["failed"][st] = [x for x in state["failed"].get(st, []) if x != k]
                state["stage_cal"][st][str(k)] = {"pulled_at": cal["pulled_at"], "chain": chains[name]["chain"],
                                                   "profile": chain_profile(cal, tuple(chains[name]["chain"]))}
                save_state(state, path)                                   # so a crash now leaves no stale id
                sampler.options.environment.job_tags = job_tags(cfg, state, st, k)
                job = sampler.run(transpiled[st][k], shots=stage_shots(cfg, st))
                entry["chunks"][str(k)] = job.job_id()
                save_state(state, path)                                   # id on disk before anything else
                print(f"   {st}: chunk {k+1}/{n_chunks} = job {job.job_id()} ({len(transpiled[st][k])} circuits)")
                if dry or not submit_only:
                    res = job.result()
                    cnts = [r.data.c.get_counts() for r in res]
                    state["readout_counts"].setdefault(st, {})[str(k)] = cnts[:RO_PER_CHUNK]
                    state["chunk_counts"].setdefault(st, {})[str(k)] = cnts[RO_PER_CHUNK:]
                    try:
                        u = job.usage()
                        if u is not None:
                            state["usage_by_job"][job.job_id()] = float(u)
                    except Exception as e:
                        print(f"   warning: usage unavailable for {job.job_id()} ({e})")
            assemble_stage(state, st, len(staged[st][0]))
            save_state(state, path)
    finally:
        if ctx:
            ctx.close()
    print(f"   saved {path}")
    return state


def assemble_stage(state: dict, st: str, expected_circuits: int) -> bool:
    """Concatenate chunk counts in chunk order once every chunk is in; verify the total length."""
    entry = state["jobs"].get(st); cc = state["chunk_counts"].get(st, {})
    if not entry or entry.get("n_chunks") is None or len(cc) < entry["n_chunks"]:
        return False
    counts = [c for k in range(entry["n_chunks"]) for c in cc[str(k)]]
    if len(counts) != expected_circuits:
        print(f"   !! {st}: {len(counts)} count dicts but {expected_circuits} circuits expected; stage NOT assembled")
        return False
    state["counts"][st] = counts
    return True


def fetch(cfg: dict) -> None:
    """Pull finished chunks into the state file.  DONE chunks are stored once (usage recorded per job id, so
    repeated calls never double count); ERROR/CANCELLED chunks are marked failed and only those are resubmitted
    by `submit`; a stage is assembled only when every chunk is in and the count length matches."""
    path = state_path(cfg)
    state = json.load(open(path))
    if state.get("dry_run"):
        raise SystemExit("this is a dry-run file: its jobs ran locally and are already fetched")
    service, _ = connect(cfg)
    for key, default in (("jobs", {}), ("counts", {}), ("chunk_counts", {}), ("readout_counts", {}), ("usage_by_job", {}), ("failed", {}), ("superseded", [])):
        state.setdefault(key, default)
    reconcile_with_server(service, cfg, state)
    n_set = len(state["settings"]); n_pts = len(state["X_train"]) + len(state["X_test"]); per = payload_per_chunk(state["config"])
    expected = {"pqk_good": n_pts * n_set, "pqk_bad": n_pts * n_set, "fid_good": len(state["fid_pairs"]), "fid_bad": len(state["fid_pairs"])}

    def record_usage(jid, job):
        try:
            u = job.usage()
            if u is not None:
                state["usage_by_job"][jid] = float(u)
            else:
                print(f"   warning: no usage reported yet for {jid}; will retry next fetch")
        except Exception as e:
            print(f"   warning: usage unavailable for {jid} ({e})")

    for st, entry in state["jobs"].items():
        entry["n_chunks"] = math.ceil(expected[st] / per)      # known from the frozen design, even for recovered ids
        if st in state["counts"]:
            continue
        cc = state["chunk_counts"].setdefault(st, {}); failed = []
        for k, jid in sorted(entry["chunks"].items(), key=lambda kv: int(kv[0])):
            if k in cc:
                if jid not in state["usage_by_job"]:
                    try:
                        record_usage(jid, service.job(jid))
                    except Exception as e:
                        print(f"   warning: {jid} could not be queried for usage ({e})")
                continue
            try:
                job = service.job(jid); status = str(job.status()).upper()
            except Exception as e:
                print(f"   {st} chunk {k}: {jid} could not be queried ({e})"); continue
            print(f"   {st} chunk {k}: {jid} {status}")
            if any(bad in status for bad in ("ERROR", "CANCEL", "FAIL")):
                failed.append(int(k)); record_usage(jid, job); continue   # a job that died mid-run still cost time
            if not status.endswith("DONE"):
                continue
            try:
                cnts = [r.data.c.get_counts() for r in job.result()]
            except Exception as e:
                print(f"   {st} chunk {k}: result not readable ({e}); will retry next fetch"); continue
            want = RO_PER_CHUNK + min(per, expected[st] - int(k) * per)
            if len(cnts) != want:
                print(f"   !! {st} chunk {k}: {len(cnts)} results, expected {want}; not stored (resubmit with `submit` after marking it failed)")
                state["failed"][st] = sorted(set(state["failed"].get(st, [])) | {int(k)}); continue
            state["readout_counts"].setdefault(st, {})[k] = cnts[:RO_PER_CHUNK]; cc[k] = cnts[RO_PER_CHUNK:]
            record_usage(jid, job)
            save_state(state, path)
        if failed:
            state["failed"][st] = sorted(set(state["failed"].get(st, [])) | set(failed))
            print(f"   {st}: chunk(s) {failed} failed -> `submit --stages {st}` resubmits only those")
        if assemble_stage(state, st, expected[st]):
            print(f"   {st}: complete ({expected[st]} circuits)")
        else:
            print(f"   {st}: {len(cc)}/{entry['n_chunks']} chunks in")
    for jid in state["superseded"]:                             # failed-then-replaced jobs still cost time
        if jid not in state["usage_by_job"]:
            try:
                record_usage(jid, service.job(jid))
            except Exception as e:
                print(f"   warning: superseded job {jid} could not be queried for usage ({e})")
    save_state(state, path)
    tot = sum(state["usage_by_job"].values())
    print(f"   QPU charged so far: {tot:.0f} s = {tot/60:.1f} min over {len(state['usage_by_job'])} job(s)")



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
        out["decision_corr"] = float(np.corrcoef(dec, ref_decision)[0, 1]) if np.std(dec) > 0 and np.std(ref_decision) > 0 else float("nan")
        out["flips"] = float((pred != ref_pred).mean())
    return out | {"_dec": dec, "_pred": pred}


def kernels(F_tr, F_te, idx, gamma_ref=None):
    g = gamma_ref if gamma_ref is not None else median_gamma(F_tr[:, idx])
    return gaussian_kernel(F_tr[:, idx], F_tr[:, idx], g), gaussian_kernel(F_te[:, idx], F_tr[:, idx], g)


def prune(scores: np.ndarray, keep_fraction: float | None = None, top_k: int | None = None) -> np.ndarray:
    """Top-scoring observables, never splitting a tie group at the cutoff.

    W = prod(1-2p) is identical for the 9 Pauli products of the same qubit pair, so a 25 % cutoff usually falls
    inside a group of 9 equal scores.  kernel.prune breaks such ties with np.argsort's default quicksort, whose
    order depends on the numpy build; `verify` showed the October Spearman values move in the 3rd decimal with
    it (Marrakesh 6q bad chain: 0.980-0.993 over tie orders, reported 0.988).  Keeping every observable tied
    with the last one kept is deterministic on all machines and physically sensible (equally clean observables
    are treated alike).  The kept fraction is then slightly above keep_fraction.
    """
    if top_k is None:
        top_k = max(1, int(round(keep_fraction * len(scores))))
    cutoff = np.sort(scores)[::-1][top_k - 1]
    return np.where(scores >= cutoff - 1e-12)[0]


def rank01(v: np.ndarray) -> np.ndarray:
    """Ranks in [0, 1] with ties given their average rank (deterministic; np.argsort would break ties by build)."""
    r = rankdata(v, method="average")
    return (r - 1) / max(len(v) - 1, 1)


def chain_metrics(F_hw, F_ex, p_cal, p_meas, obs, keep_fraction=0.25):
    """The October analysis, unchanged: attenuation slopes vs prod(1-2p), and Spearman of the hardware kernel
    against the exact one for full / pruned-by-W / mitigated features.  Shared by `analyze` and `verify`."""
    n_pts = len(F_ex)
    slope = np.array([float(F_hw[:, m] @ F_ex[:, m] / max(F_ex[:, m] @ F_ex[:, m], 1e-9)) for m in range(len(obs))])
    pred_cal = np.array([np.prod([1 - 2 * p_cal[q] for q in o.qubits]) for o in obs])
    pred_meas = np.array([np.prod([1 - 2 * p_meas[q] for q in o.qubits]) for o in obs])
    # observables with enough signal to regress on: mean squared exact value > 0.5/16 per point (the October
    # threshold was 0.5 on the sum over its 16 points; this is the same per-point threshold for any n)
    strong = np.array([F_ex[:, m] @ F_ex[:, m] > 0.5 * n_pts / 16 for m in range(len(obs))])
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


def tie_order_values(F_hw, F_ex, pred_cal, obs, keep_fraction, n_pts, max_orders=500):
    """Spearman of the pruned-by-W kernel for every way of breaking the tie group at the cutoff (what
    kernel.prune does implicitly with an unstable sort).  Returns the list of values."""
    top_k = max(1, int(round(keep_fraction * len(obs))))
    srt = np.sort(pred_cal)[::-1]; cutoff = srt[top_k - 1]
    above = np.where(pred_cal > cutoff + 1e-12)[0]; tied = np.where(np.isclose(pred_cal, cutoff))[0]
    need = top_k - len(above)
    combos = list(itertools.combinations(tied, need))
    if len(combos) > max_orders:
        rng = np.random.default_rng(0); combos = [tuple(rng.choice(tied, need, replace=False)) for _ in range(max_orders)]
    iu = np.triu_indices(n_pts, k=1); vals = []
    for cmb in combos:
        idx = np.sort(np.concatenate([above, np.array(cmb, dtype=int)]))
        g = median_gamma(F_ex[:, idx])
        vals.append(float(spearmanr(gaussian_kernel(F_hw[:, idx], F_hw[:, idx], g)[iu], gaussian_kernel(F_ex[:, idx], F_ex[:, idx], g)[iu]).correlation))
    return vals, len(tied), need


def verify_october(path: str = "results/hw_live.json", ref: str = "results/hw_live_analysis.json", tol: float = 1e-6) -> bool:
    """Replicability check against the Oct 21 Marrakesh run (hardware_experiment.py).

    Everything deterministic must match to `tol`: measurement settings, attenuation statistics, Spearman of the
    full and mitigated kernels.  For the pruned-by-W kernel October's number depends on how numpy broke a tie
    group at the 25 % cutoff, so the check is: October's value must equal (to `tol`) the value of SOME tie
    order, enumerated exactly.  The whole-tie-group number this script uses is reported next to it."""
    st, old = json.load(open(path)), json.load(open(ref))
    n = len(st["X"][0]); obs = build_observables(n)
    new_settings, _ = measurement_plan(n, obs)
    same = list(st["settings"]) == list(new_settings)
    print(f"== verify against {path}: {n} qubits, {len(st['X'])} points, settings identical: {same}")
    X = np.array(st["X"]); F_ex = exact_features(X, obs); ok = same
    for ci, name in enumerate(("good", "bad")):
        p_meas = readout_from_counts(st["counts"]["readout"][2 * ci:2 * ci + 2], n)
        F_hw = features_from_counts(st["counts"][f"pqk_{name}"], st["settings"], n, obs, len(X))
        p_cal = np.array(st["chains"][name]["readout"])
        m = chain_metrics(F_hw, F_ex, p_cal, p_meas, obs)
        o = old["chains"][name]
        pairs = [("attenuation corr (cal p)", m["attenuation"]["corr_pred_cal"], o["attenuation"]["corr_pred_cal_vs_measured"]),
                 ("attenuation corr (meas p)", m["attenuation"]["corr_pred_meas"], o["attenuation"]["corr_pred_meas_vs_measured"]),
                 ("rms residual", m["attenuation"]["rms_residual"], o["attenuation"]["rms_residual"]),
                 ("spearman pqk_full", m["spearman"]["pqk_full"], o["spearman"]["pqk_full"]),
                 ("spearman pqk_mitigated", m["spearman"]["pqk_mitigated"], o["spearman"]["pqk_mitigated"])]
        print(f"   {name} chain {st['chains'][name]['chain']}")
        for label, a, b in pairs:
            good = abs(a - b) < tol; ok &= good
            print(f"      {label:28s} new {a:.6f}   oct {b:.6f}   {'ok' if good else 'MISMATCH'}")
        vals, n_tied, need = tie_order_values(F_hw, F_ex, m["pred_cal"], obs, 0.25, len(X))
        oct_w = o["spearman"]["pqk_pruned_W"]; hit = min(abs(v - oct_w) for v in vals) < tol; ok &= hit
        print(f"      {'spearman pqk_pruned_W':28s} oct {oct_w:.6f} equals one of the {len(vals)} tie orders ({need} of {n_tied} tied kept): "
              f"{'ok' if hit else 'MISMATCH'}   range over tie orders {min(vals):.6f}-{max(vals):.6f}   this script (whole tie group, {len(m['keep_W'])} obs) {m['spearman']['pqk_pruned_W']:.6f}")
    print("   RESULT:", "PASS: October reproduced to 1e-6 (pruned_W under its tie order)" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------- analysis
def analyze(cfg: dict) -> dict:
    st = json.load(open(state_path(cfg)))
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
           "qpu_seconds_charged": sum(st.get("usage_by_job", {}).values()), "noiseless": {"acc": ref["acc"]}, "chains": {}}
    print(f"== analyze {st['backend']} ({'dry run' if st['dry_run'] else 'live'}): {n} qubits, {ntr} train / {nte} test, noiseless PQK acc {ref['acc']:.3f}")
    iu = np.triu_indices(ntr, k=1)
    fig1, ax1 = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    bars = {}
    for ci, name in enumerate(("good", "bad")):
        stage = f"pqk_{name}"
        if stage not in st["counts"]:
            print(f"   {name:4s} chain: no PQK counts yet, skipped"); continue
        ch = st["chains"][name]
        # every chunk carried its own readout check and calibration snapshot; p_cal and p_meas are the means over
        # the chunks of this stage (per-chunk values are kept in the JSON), so multi-day stages are self-described
        snaps = [v for v in st.get("stage_cal", {}).get(stage, {}).values() if v.get("profile")]
        p_cal = np.mean([v["profile"]["readout"] for v in snaps], axis=0) if snaps else np.array(ch["readout"])
        dose = float(np.mean([v["profile"]["dose_eq"] for v in snaps])) if snaps else ch["dose_eq"]
        ro_chunks = st.get("readout_counts", {}).get(stage, {})
        p_meas_chunks = [readout_from_counts(ro_chunks[k], n) for k in sorted(ro_chunks, key=int)] or [p_cal]
        p_meas = np.mean(p_meas_chunks, axis=0)
        F_hw = features_from_counts(st["counts"][stage], st["settings"], n, obs, ntr + nte)
        F_tr, F_te = F_hw[:ntr], F_hw[ntr:]

        # 1. attenuation + kernel ordering, exactly as in October (shared with `verify`)
        F_ex_all = np.vstack([F_ex_tr, F_ex_te])
        cm = chain_metrics(F_hw, F_ex_all, p_cal, p_meas, obs, cfg["keep_fraction"])
        slope, pred_cal, pred_meas, strong = cm["slope"], cm["pred_cal"], cm["pred_meas"], cm["strong"]
        r_cal, r_meas = cm["attenuation"]["corr_pred_cal"], cm["attenuation"]["corr_pred_meas"]

        # 2. selectors (train fold only, no test labels used)
        W = pred_cal
        keep_W = cm["keep_W"]
        kta_o = per_observable_kta(F_tr, ytr, median_gamma(F_tr))
        if cfg["keep_fraction"] != c["keep_fraction"]:
            print(f"        (analysing with keep_fraction={cfg['keep_fraction']} instead of the frozen {c['keep_fraction']})")
        # KTA x W keeps the SAME number of observables as the W selector, so the two are compared at equal n_obs
        keep_KW = prune(rank01(kta_o) * rank01(W), top_k=len(keep_W))
        F_mit_tr, F_mit_te = F_tr / np.clip(pred_meas, 0.05, None), F_te / np.clip(pred_meas, 0.05, None)

        variants = {"pqk_full": (F_tr, F_te, allidx), "pqk_pruned_W": (F_tr, F_te, keep_W),
                    "pqk_pruned_KTAxW": (F_tr, F_te, keep_KW), "pqk_mitigated": (F_mit_tr, F_mit_te, allidx)}
        res = {"chain": ch["chain"], "dose_eq": dose, "calibration_pulled_at": sorted({v["pulled_at"] for v in snaps}) or [st["pulled_at"]],
               "readout_calibration": p_cal.tolist(), "readout_measured": p_meas.tolist(),
               "readout_measured_per_chunk": [p.tolist() for p in p_meas_chunks],
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
        # keep-fraction sweep.  EXPLORATORY: it evaluates several fractions on the test labels, so choosing a
        # fraction from it is test-set tuning; report it as a sensitivity analysis, never pick keep_fraction from it.
        res["keep_sweep"] = []
        for kf in (0.10, 0.25, 0.50, 0.75, 1.0):
            row = {"keep_fraction": kf}
            idx_W = prune(W, kf) if kf < 1.0 else allidx
            for sel, score in (("W", W), ("KTAxW", rank01(kta_o) * rank01(W))):
                idx = idx_W if sel == "W" else (prune(score, top_k=len(idx_W)) if kf < 1.0 else allidx)   # equal n_obs
                K_tr, K_te = kernels(F_tr, F_te, idx)
                g_sub = median_gamma(F_ex_tr[:, idx]); K_ref_tr, K_ref_te = kernels(F_ex_tr, F_ex_te, idx, g_sub)
                rs = svm_eval(K_ref_tr, K_ref_te, ytr, yte); ev = svm_eval(K_tr, K_te, ytr, yte, rs["_dec"], rs["_pred"])
                row[sel] = {"n_obs": int(len(idx)), "spearman": float(spearmanr(K_tr[iu], K_ref_tr[iu]).correlation),
                            "acc_hw": ev["acc"], "acc_exact": rs["acc"], "noise_flips": ev["flips"]}
            res["keep_sweep"].append(row)
        print(f"        keep-fraction sweep, EXPLORATORY (uses test labels; sensitivity only) (W | KTAxW):  n_obs  spearman  acc hw / acc exact  noise flips")
        for row in res["keep_sweep"]:
            w_, k_ = row["W"], row["KTAxW"]
            print(f"          keep {int(row['keep_fraction']*100):3d}%   W: {w_['n_obs']:3d} {w_['spearman']:.3f} {w_['acc_hw']:.2f}/{w_['acc_exact']:.2f} {100*w_['noise_flips']:4.0f}%   |  KTAxW: {k_['n_obs']:3d} {k_['spearman']:.3f} {k_['acc_hw']:.2f}/{k_['acc_exact']:.2f} {100*k_['noise_flips']:4.0f}%")
        # repeated stratified splits over ALL measured points (features are measured once; the split is free):
        # mean accuracy and noise flips per selector, KTA recomputed on each train fold.  Splits overlap, so the
        # naive std/sqrt(R) understates the error; we report the Nadeau-Bengio corrected s.e.
        # sqrt(var * (1/R + n_test/n_train)) and state it in the paper.
        F_all, y_all = F_hw, np.concatenate([ytr, yte]); F_exa = F_ex_all
        n_all, n_te = len(y_all), max(1, round(len(y_all) * nte / max(ntr + nte, 1))) if nte else max(1, n_all // 3)
        rng = np.random.default_rng(c["seed"]); R = 50; acc_rs = {k: [] for k in ("full", "W", "KTAxW", "mitigated")}; fl_rs = {k: [] for k in acc_rs}
        for r in range(R):
            te_idx = np.concatenate([rng.choice(np.where(y_all == cl)[0], max(1, int(round(n_te * (y_all == cl).mean()))), replace=False) for cl in (-1, 1)])
            tr_idx = np.setdiff1d(np.arange(n_all), te_idx)
            kta_r = per_observable_kta(F_all[tr_idx], y_all[tr_idx], median_gamma(F_all[tr_idx]))
            sels = {"full": allidx, "W": cm["keep_W"], "KTAxW": prune(rank01(kta_r) * rank01(W), top_k=len(cm["keep_W"])), "mitigated": allidx}
            for k, idx in sels.items():
                A = F_all / np.clip(pred_meas, 0.05, None) if k == "mitigated" else F_all
                K_tr, K_te = kernels(A[tr_idx], A[te_idx], idx)
                g_sub = median_gamma(F_exa[tr_idx][:, idx]); R_tr, R_te = kernels(F_exa[tr_idx], F_exa[te_idx], idx, g_sub)
                rs = svm_eval(R_tr, R_te, y_all[tr_idx], y_all[te_idx]); ev = svm_eval(K_tr, K_te, y_all[tr_idx], y_all[te_idx], rs["_dec"], rs["_pred"])
                acc_rs[k].append(ev["acc"]); fl_rs[k].append(ev["flips"])
        nb = lambda v: float(np.sqrt(np.var(v, ddof=1) * (1 / R + n_te / (n_all - n_te))))
        res["repeated_splits"] = {k: {"acc_mean": float(np.mean(acc_rs[k])), "acc_se_nadeau_bengio": nb(acc_rs[k]),
                                      "noise_flips_mean": float(np.mean(fl_rs[k])), "n_splits": R, "n_test_per_split": int(n_te)} for k in acc_rs}
        print(f"        repeated splits ({R} x {n_all - n_te}/{n_te}, KTA refit per fold, s.e. Nadeau-Bengio):  " + "  ".join(
            f"{k} {100*v['acc_mean']:.1f}+-{100*v['acc_se_nadeau_bengio']:.1f}% (flips {100*v['noise_flips_mean']:.1f}%)" for k, v in res["repeated_splits"].items()))
        if f"fid_{name}" in st["counts"]:
            K_tr, K_te = fidelity_from_counts(st["counts"][f"fid_{name}"], st["fid_pairs"], n, ntr, nte)
            from kernel_comparison import fidelity_kernel_exact
            K_fx_tr = fidelity_kernel_exact(Xtr, Xtr)
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
        ax1[ci].set_xlim(0, 1.05); ax1[ci].set_ylim(0, 1.05)
    ax1[0].set_ylabel("measured attenuation"); fig1.suptitle(f"Readout reweighting on {st['backend']} ({n} qubits)"); fig1.tight_layout()
    FIGURES.mkdir(exist_ok=True); fig1.savefig(FIGURES / f"basque_{cfg['tag']}_attenuation.png", dpi=150); plt.close(fig1)
    if bars:
        keys = [k for k in next(iter(bars.values())) if all(k in b for b in bars.values())]
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
        spent = pending = 0.0
        if state_path(cfg).exists():
            s_ = json.load(open(state_path(cfg))); spent = sum(s_.get("usage_by_job", {}).values()) / 60
            pending = sum(estimate_seconds(cfg["max_circuits_per_job"], stage_shots(cfg, st), cfg["max_circuits_per_job"], cfg["qpu_safety_factor"])
                          for st, e in s_.get("jobs", {}).items() for k, jid in e.get("chunks", {}).items()
                          if jid not in s_.get("usage_by_job", {}) and int(k) not in s_.get("failed", {}).get(st, [])) / 60
        estimate(cfg, spent_min=spent, pending_min=pending)
    elif args.mode == "plan":
        _, b = connect(cfg)
        cal = dump_calibration(b); ch = pick_chains(cal, cfg["n_qubits"], cfg["chains"])
        print(f"== {b.name} ({cal['pulled_at'][:16]} UTC), {cfg['n_qubits']}-qubit chains")
        for k, p in ch.items():
            print(f"   {k:4s} {p['chain']} readout {[f'{v*100:.1f}' for v in p['readout']]}% dose_eq={p['dose_eq']:.2f} 2q={p['cx_mean']*100:.2f}%")
        estimate(cfg)
    elif args.mode == "dry-run":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        if not cfg["tag"].endswith("_dry"):
            cfg["tag"] += "_dry"
        print(f"== DRY RUN on FakeMarrakesh (local noisy simulation), tag {cfg['tag']} -> {state_path(cfg)}")
        run(cfg, FakeMarrakesh(), service=None, dry=True)
        analyze(cfg)
    elif args.mode == "submit":
        if cfg["tag"].endswith("_dry"):
            raise SystemExit("tags ending in _dry are reserved for dry runs")
        svc, b = connect(cfg)
        print(f"== LIVE on {b.name}, stages {cfg['stages']}")
        run(cfg, b, service=svc, dry=False, submit_only=True)
        print("   jobs submitted in Batch mode; run `fetch` once they finish, then `analyze`")
    elif args.mode == "fetch":
        fetch(cfg)
    else:
        analyze(cfg)
