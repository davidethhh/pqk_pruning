"""Dataset generation and preprocessing for the noise-aware observable pruning study.

Two datasets:
  * synthetic  -- N=6 controlled ground-truth interaction structure (linear topology)
  * breast_cancer -- PCA-reduced Breast Cancer Wisconsin (Diagnostic)

Both are returned scaled to [-pi, pi] so they can be fed directly as rotation angles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Linear physical topology 0-1-2-3-4-5.  Signal only on these adjacent pairs.
SIGNAL_PAIRS = [(0, 1), (1, 2), (3, 4), (4, 5)]
SIGNAL_STRENGTH = 1.5
LINEAR_WEIGHT = 0.5


def synthetic_interaction_matrix(num_features: int = 6) -> np.ndarray:
    J = np.zeros((num_features, num_features))
    for i, j in SIGNAL_PAIRS:
        J[i, j] = SIGNAL_STRENGTH
    return J


def generate_synthetic_pqk_data(num_samples: int = 300, num_features: int = 6, seed: int = 42):
    """Score(x) = sum_i w_i x_i + sum_{i<j} J_ij x_i x_j ;  y = sign(Score)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(num_samples, num_features))
    w = np.full(num_features, LINEAR_WEIGHT)
    J = synthetic_interaction_matrix(num_features)
    linear = X @ w
    interaction = np.einsum("ki,ij,kj->k", X, np.triu(J, 1), X)
    y = np.where(linear + interaction >= 0.0, 1.0, -1.0)
    return X, y


def load_preprocessed_breast_cancer(n_components: int = 6, seed: int = 42):
    data = load_breast_cancer()
    X, y = data.data, np.where(data.target == 0, -1.0, 1.0)
    X = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return X, y


# OpenML / sklearn tabular datasets, PCA-reduced to n_qubits.  Chosen so the noiseless PQK is
# NOT saturated (Breast Cancer at 0.94 leaves noise nowhere to bite).  Approximate classical
# RBF accuracies: pima ~0.76, heart ~0.82, sonar ~0.85, ionosphere ~0.93, banknote ~0.99 (saturated
# control), wine (class 0 vs rest) ~0.98.
OPENML = {"pima": ("diabetes", 1), "heart": ("heart-statlog", 1), "sonar": ("sonar", 1),
          "ionosphere": ("ionosphere", 1), "banknote": ("banknote-authentication", 1)}
# banknote has only 4 features (< 6 qubits) and is excluded from the default panel
REAL_DATASETS = ["breast_cancer", "wine", "pima", "heart", "sonar", "ionosphere"]
ALL_DATASETS = ["synthetic", *REAL_DATASETS]


def load_tabular(name: str, n_components: int = 6, seed: int = 42):
    """Any real dataset -> (X_pca, y in {-1,+1}).  Binary classes: the first class label
    (as given by the source) is -1, everything else +1."""
    if name == "breast_cancer":
        return load_preprocessed_breast_cancer(n_components, seed)
    if name == "wine":
        d = load_wine(); X, t = d.data, (d.target != 0).astype(int)
    else:
        oml_name, version = OPENML[name]
        d = fetch_openml(name=oml_name, version=version, as_frame=True, parser="auto")
        X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        t = pd.factorize(d.target)[0]
    y = np.where(t == 0, -1.0, 1.0)
    if X.shape[1] < n_components:
        raise ValueError(f"{name} has {X.shape[1]} features, fewer than {n_components} qubits")
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return X, y


# Angle range per dataset.  The synthetic label is a low-degree polynomial in x;
# stretching x over a full period makes the PQK features oscillate too fast and the
# exact kernel drops to chance (KTA 0.04, acc 0.57).  Keeping the workplan's [-1, 1]
# gives acc ~0.77.  Breast cancer is insensitive to the range.
DEFAULT_ANGLE_RANGE = {"synthetic": (-1.0, 1.0), "breast_cancer": (-np.pi, np.pi)}


def scale_to_angles(X_train: np.ndarray, X_test: np.ndarray, angle_range=(-np.pi, np.pi)):
    """MinMax to angle_range, fit on train only."""
    scaler = MinMaxScaler(feature_range=angle_range).fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def load_dataset(name: str, n_qubits: int = 6, test_size: float = 0.3, seed: int = 42,
                 angle_range=None):
    """Return (X_train, X_test, y_train, y_test) with features scaled to angle_range."""
    angle_range = angle_range or DEFAULT_ANGLE_RANGE.get(name, (-np.pi, np.pi))
    if name == "synthetic":
        X, y = generate_synthetic_pqk_data(num_features=n_qubits, seed=seed)
    elif name in REAL_DATASETS:
        X, y = load_tabular(name, n_components=n_qubits, seed=seed)
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {ALL_DATASETS}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, X_te = scale_to_angles(X_tr, X_te, angle_range)
    return X_tr, X_te, y_tr, y_te


if __name__ == "__main__":
    for name in ALL_DATASETS:
        X_tr, X_te, y_tr, y_te = load_dataset(name)
        print(f"{name:14s} train={X_tr.shape} test={X_te.shape} "
              f"pos-frac={np.mean(y_tr == 1):.2f} range=[{X_tr.min():.2f},{X_tr.max():.2f}]")
