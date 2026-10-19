"""Pull real calibration data and ask: does any 6-qubit linear chain on this device
look like the noise profile where observable selection starts to help?

    python calibration_scan.py                       # live: all backends visible to the saved IBM account
    python calibration_scan.py --backends ibm_fez     # live: one backend
    python calibration_scan.py --fake                 # offline: FakeFez / FakeMarrakesh snapshots (old data)
    python calibration_scan.py --from-json results/calibration_ibm_pittsburgh.json   # re-scan a saved dump

Step 1 saves results/calibration_<backend>.json: per-qubit readout error, per-edge 2q gate
error, T1/T2, coupling map.  Send this script to collaborators with access to other
machines; the JSON they send back is all we need.

Step 2 enumerates every linear 6-qubit path on the coupling map and expresses its readout
profile as an equivalent dose relative to our simulation profile
    1x = {0.5%, 0.5%, 0.5%, 6%, 6%, 6%}
    dose_eq(chain) = mean(worst 3 readout errors) / 6%
so dose_eq = 2 means the chain's three worst qubits average 12% readout error, which is
where selection started to pay off on the synthetic dataset (crossover between 2x and 4x).

Outputs results/chain_scan_<backend>.json with the dose distribution and three representative
chains (median / 90th percentile / worst) as HardwareProfile-ready dicts.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path("results")
SIM_HIGH = 0.06     # readout error on the "bad" half of our 1x profile
SIM_LOW = 0.005     # readout error on the "good" half
CHAIN_LEN = 6
MAX_2Q_ERROR = 0.05  # chains with a worse coupler are unusable, not "noisy"


# ============================================================================ step 1: pull
def dump_calibration(backend) -> dict:
    t = backend.target
    n = backend.num_qubits
    readout = {}
    for q in range(n):
        p = t["measure"].get((q,)) if "measure" in t else None
        readout[q] = float(p.error) if p is not None and p.error is not None else None
    two_q = {}
    for gate in ("cz", "ecr", "cx"):
        if gate in t:
            for qs, p in t[gate].items():
                if p is not None and p.error is not None:
                    two_q[f"{qs[0]}-{qs[1]}"] = float(p.error)
            gate_name = gate
            break
    else:
        gate_name = None
    qp = t.qubit_properties or []
    t1 = {q: (float(qp[q].t1) if q < len(qp) and qp[q].t1 is not None else None) for q in range(n)}
    t2 = {q: (float(qp[q].t2) if q < len(qp) and qp[q].t2 is not None else None) for q in range(n)}
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    return {"backend": backend.name, "pulled_at": datetime.now(timezone.utc).isoformat(),
            "num_qubits": n, "two_qubit_gate": gate_name, "readout_error": readout,
            "two_qubit_error": two_q, "t1_s": t1, "t2_s": t2, "edges": [list(e) for e in edges]}


def pull_live(names: list[str] | None) -> list[dict]:
    from qiskit_ibm_runtime import QiskitRuntimeService
    s = QiskitRuntimeService()
    backends = s.backends() if not names else [s.backend(n) for n in names]
    out = []
    for b in backends:
        if b.num_qubits < CHAIN_LEN:
            continue
        cal = dump_calibration(b)
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / f"calibration_{b.name}.json", "w") as f:
            json.dump(cal, f, indent=1)
        print(f"saved calibration_{b.name}.json  ({b.num_qubits} qubits, 2q gate {cal['two_qubit_gate']})")
        out.append(cal)
    return out


def pull_fake() -> list[dict]:
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh
    out = []
    for cls in (FakeFez, FakeMarrakesh):
        b = cls()
        cal = dump_calibration(b)
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / f"calibration_{b.name}.json", "w") as f:
            json.dump(cal, f, indent=1)
        print(f"saved calibration_{b.name}.json  (fake snapshot)")
        out.append(cal)
    return out


# ============================================================================ step 2: scan
def linear_chains(edges: list[list[int]], length: int) -> list[tuple[int, ...]]:
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    chains = set()

    def dfs(path):
        if len(path) == length:
            c = tuple(path)
            chains.add(min(c, c[::-1]))
            return
        for nxt in adj.get(path[-1], ()):
            if nxt not in path:
                dfs(path + [nxt])

    for q in adj:
        dfs([q])
    return sorted(chains)


def chain_profile(cal: dict, chain: tuple[int, ...]) -> dict | None:
    ro = cal["readout_error"]
    r = [ro.get(str(q), ro.get(q)) for q in chain]
    if any(v is None for v in r):
        return None
    tq = cal["two_qubit_error"]
    e2 = []
    for a, b in zip(chain[:-1], chain[1:]):
        v = tq.get(f"{a}-{b}", tq.get(f"{b}-{a}"))
        if v is None:
            return None
        e2.append(v)
    rs = sorted(r)
    worst3, best3 = rs[-3:], rs[:3]
    return {"chain": list(chain), "readout": r, "two_qubit": e2,
            "readout_median": float(np.median(r)), "readout_max": float(max(r)),
            "worst3_mean": float(np.mean(worst3)), "best3_mean": float(np.mean(best3)),
            "dose_eq": float(np.mean(worst3) / SIM_HIGH),
            "dose_eq_lowhalf": float(np.mean(best3) / SIM_LOW),
            "heterogeneity": float(np.mean(worst3) / max(np.mean(best3), 1e-4)),
            "cx_mean": float(np.mean(e2))}


def to_hardware_profile(p: dict) -> dict:
    """Dict that HardwareProfile(**d) accepts: chain re-indexed to 0..5 in chain order."""
    return {"n": CHAIN_LEN,
            "readout_error": {i: float(v) for i, v in enumerate(p["readout"])},
            "cx_error": p["cx_mean"], "sq_error": 0.001,
            "crosstalk": {},  # not in standard calibration; set separately if characterised
            "physical_qubits": p["chain"]}


def scan(cal: dict) -> dict:
    chains = linear_chains(cal["edges"], CHAIN_LEN)
    profiles = [p for c in chains if (p := chain_profile(cal, c)) is not None]
    n_total = len(profiles)
    profiles = [p for p in profiles if max(p["two_qubit"]) <= MAX_2Q_ERROR]
    if not profiles:
        raise ValueError("no complete chains; calibration has missing readout or gate errors")
    doses = np.array([p["dose_eq"] for p in profiles])
    ro_all = np.array([v for v in cal["readout_error"].values() if v is not None])
    order = np.argsort(doses)
    pick = {"median": profiles[order[len(order) // 2]],
            "p90": profiles[order[int(0.9 * (len(order) - 1))]],
            "worst": profiles[order[-1]]}
    out = {"backend": cal["backend"], "pulled_at": cal["pulled_at"], "n_chains": len(profiles), "n_chains_total": n_total,
           "max_2q_error_filter": MAX_2Q_ERROR,
           "device_readout": {"median": float(np.median(ro_all)), "p90": float(np.percentile(ro_all, 90)),
                              "max": float(ro_all.max()), "n_above_5pct": int((ro_all > 0.05).sum()),
                              "n_above_10pct": int((ro_all > 0.10).sum())},
           "dose_eq_percentiles": {str(q): float(np.percentile(doses, q)) for q in (10, 50, 90, 99, 100)},
           "frac_chains_dose_ge": {str(d): float((doses >= d).mean()) for d in (0.5, 1, 2, 4)},
           "representative_chains": {k: {**v, "hardware_profile": to_hardware_profile(v)} for k, v in pick.items()}}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"chain_scan_{cal['backend']}.json", "w") as f:
        json.dump(out, f, indent=1)
    d = out["device_readout"]
    print(f"\n== {cal['backend']}  ({cal['pulled_at'][:10]})  {len(profiles)} usable linear 6-chains of {n_total} (2q error <= {MAX_2Q_ERROR:.0%})")
    print(f"   device readout error: median {d['median']*100:.2f}%  p90 {d['p90']*100:.2f}%  max {d['max']*100:.1f}%  "
          f"qubits >5%: {d['n_above_5pct']}  >10%: {d['n_above_10pct']}")
    print("   dose_eq percentiles (worst-3 mean / 6%): " + "  ".join(f"p{q}={v:.2f}" for q, v in out["dose_eq_percentiles"].items()))
    print("   fraction of chains with dose_eq >= : " + "  ".join(f"{k}x: {v*100:.1f}%" for k, v in out["frac_chains_dose_ge"].items()))
    for k, p in pick.items():
        print(f"   {k:6s} chain {p['chain']}  readout {[f'{v*100:.1f}' for v in p['readout']]}%  "
              f"dose_eq={p['dose_eq']:.2f}  hetero={p['heterogeneity']:.1f}x  cx={p['cx_mean']*100:.2f}%")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--from-json", nargs="*", default=None)
    args = ap.parse_args()
    if args.from_json:
        cals = [json.load(open(p)) for p in args.from_json]
    elif args.fake:
        cals = pull_fake()
    else:
        cals = pull_live(args.backends)
    for cal in cals:
        scan(cal)
