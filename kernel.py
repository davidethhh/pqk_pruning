"""Gaussian projected kernel, kernel-target alignment, SVM evaluation, pruning filter."""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def median_gamma(F: np.ndarray) -> float:
    """Median heuristic: gamma = 1 / median squared distance between feature vectors."""
    sq = np.sum(F ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2 * F @ F.T
    d = D2[np.triu_indices_from(D2, k=1)]
    d = d[d > 0]
    return 1.0 / np.median(d)


def gaussian_kernel(FA: np.ndarray, FB: np.ndarray, gamma: float) -> np.ndarray:
    sa, sb = np.sum(FA ** 2, axis=1), np.sum(FB ** 2, axis=1)
    D2 = np.maximum(sa[:, None] + sb[None, :] - 2 * FA @ FB.T, 0.0)
    return np.exp(-gamma * D2)


def kta(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment <K_c, yy^T> / (||K_c|| ||yy^T||)."""
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Y = np.outer(y, y)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def per_observable_kta(F_train: np.ndarray, y_train: np.ndarray, gamma: float) -> np.ndarray:
    """KTA of the single-feature kernel exp(-gamma (f_O(x) - f_O(y))^2), train fold only."""
    return np.array([kta(gaussian_kernel(F_train[:, [m]], F_train[:, [m]], gamma), y_train)
                     for m in range(F_train.shape[1])])


def svm_accuracy(K_train: np.ndarray, K_test: np.ndarray, y_train, y_test, C: float = 1.0) -> float:
    clf = SVC(kernel="precomputed", C=C).fit(K_train, y_train)
    return float(clf.score(K_test, y_test))


def frobenius_distance(K: np.ndarray, K_ref: np.ndarray) -> float:
    return float(np.linalg.norm(K - K_ref))


def evaluate_kernel(F_tr, F_te, y_tr, y_te, gamma: float) -> dict:
    K_tr = gaussian_kernel(F_tr, F_tr, gamma)
    K_te = gaussian_kernel(F_te, F_tr, gamma)
    return {"K_train": K_tr, "K_test": K_te,
            "kta": kta(K_tr, y_tr), "acc": svm_accuracy(K_tr, K_te, y_tr, y_te)}


def prune(scores: np.ndarray, keep_fraction: float | None = None, top_k: int | None = None) -> np.ndarray:
    """Indices of the top-scoring observables."""
    if top_k is None:
        top_k = max(1, int(round(keep_fraction * len(scores))))
    return np.sort(np.argsort(-scores)[:top_k])
