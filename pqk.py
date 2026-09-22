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


def _add_basis_change_and_measure(qc: QuantumCircuit, basis: str) -> QuantumCircuit:
    meas = qc.copy()
    meas.barrier()
    for q, b in enumerate(basis):
        if b == "X":
            meas.h(q)
        elif b == "Y":
            meas.sdg(q)
            meas.h(q)
    meas.measure_all()
    return meas


def _parity_expectation(counts: dict[str, int], qubits: tuple[int, ...], n: int) -> tuple[float, int]:
    total = sum(counts.values())
    acc = 0.0
    for bitstr, c in counts.items():
        bits = bitstr[::-1]  # bits[q] is the outcome of qubit q
        parity = sum(int(bits[q]) for q in qubits) % 2
        acc += (1 - 2 * parity) * c
    return acc / total, total


def sampled_features(X: np.ndarray, observables: list[Observable], *,
                     noise_model=None, coupling_map: CouplingMap | None = None,
                     shots: int = 2000, extra_pairs=None, seed: int = 7,
                     optimization_level: int = 1, return_variance: bool = False):
    """Shot-based expectation values under a noise model and coupling map.

    Returns F (n_samples x n_obs) and, optionally, the per-entry estimator
    variance Var[<O>] = (1 - <O>^2) / n_shots_used.
    """
    n = X.shape[1]
    settings = measurement_settings(n)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    if coupling_map is None:
        coupling_map = linear_coupling_map(n)

    # which settings serve which observable
    serves = {o: [s for s in settings if all(s[q] == p for q, p in zip(o.qubits, o.paulis))]
              for o in observables}

    F = np.empty((len(X), len(observables)))
    V = np.empty_like(F)
    for k, x in enumerate(X):
        base = feature_map(x, extra_pairs)
        circs = [_add_basis_change_and_measure(base, s) for s in settings]
        tcircs = transpile(circs, coupling_map=coupling_map,
                           basis_gates=["rz", "sx", "x", "cx"],
                           initial_layout=list(range(n)), optimization_level=optimization_level,
                           seed_transpiler=seed)
        result = sim.run(tcircs, shots=shots).result()
        counts = {s: result.get_counts(i) for i, s in enumerate(settings)}
        for m, o in enumerate(observables):
            est, tot = [], 0
            for s in serves[o]:
                e, t = _parity_expectation(counts[s], o.qubits, n)
                est.append(e * t)
                tot += t
            F[k, m] = sum(est) / tot
            V[k, m] = (1 - F[k, m] ** 2) / tot
    return (F, V) if return_variance else F


def swap_count(n: int, extra_pairs=None, coupling_map=None, seed: int = 7) -> int:
    """Number of SWAPs the transpiler inserts for this feature map on the coupling map."""
    qc = feature_map(np.zeros(n), extra_pairs)
    t = transpile(qc, coupling_map=coupling_map or linear_coupling_map(n),
                  initial_layout=list(range(n)), basis_gates=["rz", "sx", "x", "cx", "swap"],
                  optimization_level=1, seed_transpiler=seed)
    return t.count_ops().get("swap", 0)
