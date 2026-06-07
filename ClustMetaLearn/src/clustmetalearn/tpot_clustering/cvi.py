"""Fast cluster validity indices (O(n) for fixed k and d)."""

from __future__ import annotations

import numpy as np


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Map arbitrary labels to 0..k-1; return (encoded, k)."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return labels.astype(np.intp, copy=False), int(uniq.size)
    mapping = {u: i for i, u in enumerate(uniq)}
    encoded = np.array([mapping[u] for u in labels], dtype=np.intp)
    return encoded, int(uniq.size)


def _cluster_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster centroid and count; shape (k, d) and (k,)."""
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features), dtype=X.dtype)
    counts = np.zeros(k, dtype=np.int64)
    np.add.at(centroids, labels, X)
    np.add.at(counts, labels, 1)
    empty = counts == 0
    if np.any(empty):
        centroids[empty] = np.nan
    else:
        centroids /= counts[:, np.newaxis]
    return centroids, counts


def calinski_harabasz_fast(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski–Harabasz index (sklearn-compatible)."""
    labels, k = _encode_labels(labels)
    n = X.shape[0]
    if k < 2 or n <= k:
        return np.nan

    global_mean = X.mean(axis=0)
    centroids, counts = _cluster_centroids(X, labels, k)
    if np.any(counts == 0):
        return np.nan

    # within-cluster dispersion
    within = 0.0
    for c in range(k):
        mask = labels == c
        diff = X[mask] - centroids[c]
        within += float(np.sum(diff * diff))

    between = 0.0
    for c in range(k):
        diff = centroids[c] - global_mean
        between += float(counts[c] * np.dot(diff, diff))

    if within <= 0.0:
        return np.nan
    return float((between / (k - 1)) / (within / (n - k)))


def davies_bouldin_fast(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies–Bouldin index (sklearn-compatible, lower is better)."""
    labels, k = _encode_labels(labels)
    if k < 2:
        return np.nan

    centroids, counts = _cluster_centroids(X, labels, k)
    if np.any(counts == 0):
        return np.nan

    # mean distance of points to their cluster centroid
    scatter = np.zeros(k, dtype=float)
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            return np.nan
        dists = np.linalg.norm(X[mask] - centroids[c], axis=1)
        scatter[c] = float(np.mean(dists))

    centroid_dists = np.linalg.norm(centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    np.fill_diagonal(centroid_dists, np.inf)

    db_vals = []
    for i in range(k):
        ratios = (scatter[i] + scatter) / centroid_dists[i]
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            continue
        db_vals.append(float(np.max(ratios)))

    if not db_vals:
        return np.nan
    return float(np.mean(db_vals))


def silhouette_centroid_fast(X: np.ndarray, labels: np.ndarray) -> float:
    """Centroid-based silhouette approximation."""
    labels, k = _encode_labels(labels)
    n = X.shape[0]
    if k < 2 or n < 3:
        return np.nan

    centroids, counts = _cluster_centroids(X, labels, k)
    if np.any(counts == 0):
        return np.nan

    # distance from each point to every centroid: (n, k)
    dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    a = dists[np.arange(n), labels]

    dists_masked = dists.copy()
    dists_masked[np.arange(n), labels] = np.inf
    b = np.min(dists_masked, axis=1)

    denom = np.maximum(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 0, (b - a) / denom, 0.0)
    if s.size == 0:
        return np.nan
    return float(np.mean(s))
