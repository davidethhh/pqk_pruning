"""Projected quantum kernel feature extraction.

Feature map  U(x) = [1D adjacent CNOT chain] . prod_i Ry(x_i) Rz(x_i)
Observables  3N single-qubit Paulis  +  9*C(N,2) two-qubit Pauli products.

Two extraction paths:
  * exact_features   -- statevector expectation values (noiseless reference)
  * sampled_features -- explicit measurement circuits transpiled to a coupling map
                        and run on AerSimulator with a noise model.  This is the
                        only path on which readout error / crosstalk / SWAP routing
                        actually act, so it is the one used for every noise experiment.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

PAULIS = "XYZ"


# ----------------------------------------------------------------------------- observables
@dataclass(frozen=True)
class Observable:
    qubits: tuple[int, ...]
    paulis: str  # e.g. "Z" or "XY" (paulis[k] acts on qubits[k])

    @property
    def label(self) -> str:
        return "".join(f"{p}{q}" for p, q in zip(self.paulis, self.qubits))

    @property
    def weight(self) -> int:
        return len(self.qubits)

    @property
    def distance(self) -> int:
        """Topological distance on a linear chain (0 for single-qubit)."""
        return 0 if self.weight == 1 else abs(self.qubits[0] - self.qubits[1])

    def sparse_pauli(self, n: int) -> SparsePauliOp:
        s = ["I"] * n
        for p, q in zip(self.paulis, self.qubits):
            s[n - 1 - q] = p  # qiskit little-endian labels
        return SparsePauliOp("".join(s))


def build_observables(n: int) -> list[Observable]:
    obs = [Observable((q,), p) for q in range(n) for p in PAULIS]
    for i, j in itertools.combinations(range(n), 2):
        for pi, pj in itertools.product(PAULIS, repeat=2):
            obs.append(Observable((i, j), pi + pj))
    return obs


# ----------------------------------------------------------------------------- feature map
def linear_coupling_map(n: int) -> CouplingMap:
    return CouplingMap.from_line(n, bidirectional=True)


def feature_map(x: np.ndarray, extra_pairs: list[tuple[int, int]] | None = None) -> QuantumCircuit:
    n = len(x)
    qc = QuantumCircuit(n)
    for i, xi in enumerate(x):
        qc.ry(float(xi), i)
        qc.rz(float(xi), i)
    qc.barrier()
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i, j in extra_pairs or []:
        qc.cx(i, j)  # distant entangler -> forces SWAP routing on a linear coupling map
    return qc


# ----------------------------------------------------------------------------- exact path
def exact_features(X: np.ndarray, observables: list[Observable],
                   extra_pairs=None) -> np.ndarray:
    n = X.shape[1]
    ops = [o.sparse_pauli(n) for o in observables]
    F = np.empty((len(X), len(observables)))
    for k, x in enumerate(X):
        sv = Statevector(feature_map(x, extra_pairs))
        F[k] = [sv.expectation_value(op).real for op in ops]
    return F


# ----------------------------------------------------------------------------- sampled path
def measurement_settings(n: int) -> list[str]:
    """Orthogonal-array covering: every qubit pair sees all 9 basis combinations.

    Qubit q gets a direction vector v_q in GF(3)^3 with pairwise linearly
    independent vectors; setting (s,t,u) measures qubit q in basis (v_q . (s,t,u)) mod 3.
    27 settings cover up to 13 qubits.
    """
    # representatives of the 13 lines through the origin of GF(3)^3
    vectors = []
    for v in itertools.product(range(3), repeat=3):
        if v == (0, 0, 0):
            continue
        if any(all((a * m) % 3 == b for a, b in zip(v, w)) for w in vectors for m in (1, 2)):
            continue
        vectors.append(v)
    if n > len(vectors):
        raise ValueError("covering supports at most 13 qubits")
    settings = []
    for s in itertools.product(range(3), repeat=3):
        basis = "".join(PAULIS[sum(a * b for a, b in zip(vectors[q], s)) % 3] for q in range(n))
        settings.append(basis)
    return sorted(set(settings))


def _measurement_circuit(qc: QuantumCircuit, basis: str, group: tuple[int, ...]) -> QuantumCircuit:
    """Basis change on every qubit, then measure only the qubits in `group`
    (classical bit k <- qubit group[k])."""
    meas = QuantumCircuit(qc.num_qubits, len(group))
    meas.compose(qc, inplace=True)
    meas.barrier()
    for q, b in enumerate(basis):
        if b == "X":
            meas.h(q)
        elif b == "Y":
            meas.sdg(q)
            meas.h(q)
    for k, q in enumerate(group):
        meas.measure(q, k)
    return meas


def _memory_to_bits(memory: list[str], n_meas: int) -> np.ndarray:
    """Per-shot outcomes as an int8 array (shots x n_meas); column k is qubit group[k]."""
    arr = np.frombuffer("".join(m[::-1] for m in memory).encode(), dtype=np.uint8) - ord("0")
    return arr.reshape(len(memory), n_meas).astype(np.int8)


class ReadoutNoise:
    """Classical assignment noise applied to sampled bitstrings.

    Readout error on hardware is a classical confusion process acting after the
    projective measurement, so applying it to ideal outcomes is exact.  Doing it
    here (rather than inside Aer) gives full control over two things Aer's
    primitives do not expose: which qubits are read out *together* in a circuit,
    and correlated (crosstalk) flips that only fire when neighbouring qubits are
    measured simultaneously.

      p01[q] : P(read 1 | true 0)      p10[q] : P(read 0 | true 1)
      crosstalk[(i, j)] = c : with probability c, when i and j are measured in the
                           same circuit and their true outcomes differ, both are
                           reported as the outcome of i (a correlated assignment
                           error that pulls neighbours toward agreement).
    """

    def __init__(self, p01: dict[int, float] | None = None, p10: dict[int, float] | None = None,
                 crosstalk: dict[tuple[int, int], float] | None = None):
        self.p01 = dict(p01 or {})
        self.p10 = dict(p10 or {})
        self.crosstalk = {tuple(sorted(k)): v for k, v in (crosstalk or {}).items()}

    @classmethod
    def symmetric(cls, p: dict[int, float], crosstalk=None):
        return cls(p01=dict(p), p10=dict(p), crosstalk=crosstalk)

    def scaled(self, factor: float) -> "ReadoutNoise":
        return ReadoutNoise({q: min(0.5, p * factor) for q, p in self.p01.items()},
                            {q: min(0.5, p * factor) for q, p in self.p10.items()},
                            {k: min(1.0, v * factor) for k, v in self.crosstalk.items()})

    def apply(self, bits: np.ndarray, group: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        out = bits.copy()
        col = {q: k for k, q in enumerate(group)}
        # correlated crosstalk first (acts on true outcomes), then independent flips
        for (i, j), c in self.crosstalk.items():
            if i in col and j in col and c > 0:
                ci, cj = col[i], col[j]
                fire = (bits[:, ci] != bits[:, cj]) & (rng.random(len(bits)) < c)
                out[fire, cj] = bits[fire, ci]
        for q, k in col.items():
            p01, p10 = self.p01.get(q, 0.0), self.p10.get(q, 0.0)
            u = rng.random(len(bits))
            flip = np.where(bits[:, k] == 0, u < p01, u < p10)
            out[flip, k] ^= 1
        return out


def measurement_jobs(n: int, observables: list[Observable], measure_groups=None, min_settings: bool = False):
    """Return (jobs, serves): jobs = [(basis, group)], serves[o] = job indices estimating o."""
    if measure_groups is None:
        measure_groups = [tuple(range(n))]
    measure_groups = [tuple(g) for g in measure_groups]
    # (basis restricted to group, group) -> distinct measurement jobs
    jobs: list[tuple[str, tuple[int, ...]]] = []
    seen = set()
    for g in measure_groups:
        for s in measurement_settings(n):
            key = ("".join(s[q] for q in g), g)
            if key not in seen:
                seen.add(key)
                jobs.append((s, g))
    serves = {o: [idx for idx, (s, g) in enumerate(jobs)
                  if set(o.qubits) <= set(g) and all(s[q] == p for q, p in zip(o.qubits, o.paulis))]
              for o in observables}
    for o, js in serves.items():
        if not js:
            raise ValueError(f"observable {o.label} is not served by any measurement group")
    if min_settings:
        # greedy set cover: keep adding the job that serves the most still-uncovered observables
        uncovered = set(range(len(observables)))
        chosen: list[int] = []
        while uncovered:
            best = max(range(len(jobs)), key=lambda j: sum(1 for m in uncovered if j in serves[observables[m]]))
            chosen.append(best)
            uncovered -= {m for m in uncovered if best in serves[observables[m]]}
        keep = sorted(set(chosen))
        remap = {old: new for new, old in enumerate(keep)}
        jobs = [jobs[j] for j in keep]
        serves = {o: [remap[j] for j in js if j in remap] for o, js in serves.items()}
    return jobs, serves


def sampled_features(X: np.ndarray, observables: list[Observable], *,
                     noise_model=None, readout_noise: ReadoutNoise | None = None,
                     coupling_map: CouplingMap | None = None,
                     measure_groups: list[tuple[int, ...]] | None = None,
                     shots: int = 2000, total_shots: int | None = None, min_settings: bool = False,
                     extra_pairs=None, seed: int = 7,
                     optimization_level: int = 1, return_variance: bool = False):
    """Shot-based expectation values under gate noise (Aer noise_model), readout
    noise (ReadoutNoise) and a coupling map (SWAP routing).

    Shot budget: `shots` is per measurement job.  If `total_shots` is given it is
    the budget per data point and is split evenly over the jobs actually run, so
    a smaller observable set gets more shots per job.  With `min_settings=True`
    the jobs are a greedy set cover of the requested observables (instead of the
    full 27-setting orthogonal array), which is what makes pruning pay in shots.

    measure_groups: which qubits are read out together.  Default: all at once.
    A staggered schedule (e.g. [(0,2,4),(1,3,5),(0,1),(1,2),...]) reads out
    non-neighbouring qubits in separate circuits so crosstalk cannot fire; every
    observable is estimated from the groups that contain all of its qubits.

    Returns F (n_samples x n_obs) and, optionally, the estimator variance
    Var[<O>] = (1 - <O>^2) / n_shots_used.
    """
    n = X.shape[1]
    rng = np.random.default_rng(seed)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    if coupling_map is None:
        coupling_map = linear_coupling_map(n)
    jobs, serves = measurement_jobs(n, observables, measure_groups, min_settings)
    if total_shots is not None:
        shots = max(1, total_shots // len(jobs))

    F = np.empty((len(X), len(observables)))
    V = np.empty_like(F)
    for k, x in enumerate(X):
        base = feature_map(x, extra_pairs)
        circs = [_measurement_circuit(base, s, g) for s, g in jobs]
        tcircs = transpile(circs, coupling_map=coupling_map,
                           basis_gates=["rz", "sx", "x", "cx"],
                           initial_layout=list(range(n)), optimization_level=optimization_level,
                           seed_transpiler=seed)
        result = sim.run(tcircs, shots=shots, memory=True).result()
        bits = []
        for idx, (s, g) in enumerate(jobs):
            b = _memory_to_bits(result.get_memory(idx), len(g))
            if readout_noise is not None:
                b = readout_noise.apply(b, g, rng)
            bits.append(b)
        for m, o in enumerate(observables):
            acc, tot = 0.0, 0
            for idx in serves[o]:
                g = jobs[idx][1]
                cols = [g.index(q) for q in o.qubits]
                parity = bits[idx][:, cols].sum(axis=1) % 2
                acc += float((1 - 2 * parity).sum())
                tot += len(parity)
            F[k, m] = acc / tot
            V[k, m] = (1 - F[k, m] ** 2) / tot
    return (F, V) if return_variance else F


def staggered_groups(n: int) -> list[tuple[int, ...]]:
    """Read-out schedule in which no two *neighbouring* qubits are measured in
    the same circuit except where an adjacent-pair observable makes it
    unavoidable: even qubits together, odd qubits together, one job per
    non-adjacent odd-distance pair, and one job per adjacent pair."""
    groups = [tuple(range(0, n, 2)), tuple(range(1, n, 2))]
    groups += [(i, j) for i in range(n) for j in range(i + 2, n) if (j - i) % 2 == 1]
    groups += [(i, i + 1) for i in range(n - 1)]
    return groups
