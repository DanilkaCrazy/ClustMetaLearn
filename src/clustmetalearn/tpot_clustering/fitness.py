"""Cross-validated fitness for clustering pipelines."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import KFold

from clustmetalearn.tpot_clustering.cvi import (
    calinski_harabasz_fast,
    davies_bouldin_fast,
    silhouette_centroid_fast,
)
from clustmetalearn.tpot_clustering.encoding import SearchSpace, individual_to_pipeline

MetricName = Literal[
    "silhouette",
    "silhouette_exact",
    "calinski_harabasz",
    "davies_bouldin",
    "ari",
]

_BAD = -1e9


def _score_labels(
    metric: MetricName,
    X_val: np.ndarray,
    y_val: np.ndarray | None,
    labels: np.ndarray,
) -> float:
    if metric == "ari":
        if y_val is None:
            return np.nan
        return float(adjusted_rand_score(y_val, labels))

    if metric == "silhouette":
        return silhouette_centroid_fast(X_val, labels)

    if metric == "silhouette_exact":
        uniq = np.unique(labels)
        if uniq.size < 2 or X_val.shape[0] < 3:
            return np.nan
        return float(silhouette_score(X_val, labels))

    if metric == "calinski_harabasz":
        return calinski_harabasz_fast(X_val, labels)

    if metric == "davies_bouldin":
        s = davies_bouldin_fast(X_val, labels)
        return -s if np.isfinite(s) else np.nan

    raise ValueError(f"Unknown metric: {metric}")


def evaluate_individual(
    individual: list[int],
    X: np.ndarray,
    y_eval: np.ndarray | None,
    space: SearchSpace,
    *,
    metric: MetricName,
    cv_splits: int,
    random_state: int,
) -> tuple[float]:
    """Return tuple fitness (_DEAP uses tuples). Higher is better."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    rs = random_state
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
    scores: list[float] = []
    for fold_seed, (train_idx, val_idx) in enumerate(kfold.split(X)):
        n_tr = len(train_idx)
        max_k_fold = max(2, min(individual[4], n_tr - 1))
        if max_k_fold < 2 or len(val_idx) < 2:
            continue
        ind_fold = individual[:]
        ind_fold[4] = min(ind_fold[4], max_k_fold)
        space.clip(ind_fold)
        try:
            pipe = individual_to_pipeline(ind_fold, space, random_state=rs + fold_seed)
            X_tr, X_val = X[train_idx], X[val_idx]
            y_val = y_eval[val_idx] if y_eval is not None else None
            model = clone(pipe)
            model.fit(X_tr)
            labels = model.predict(X_val)
            s = _score_labels(metric, X_val, y_val, labels)
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            continue
    if not scores:
        return (_BAD,)
    return (float(np.mean(scores)),)
