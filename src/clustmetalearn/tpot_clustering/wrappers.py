"""Estimator wrappers for CV-compatible cluster assignment."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class AgglomerativeWithPredict(BaseEstimator, ClusterMixin):
    """Agglomerative clustering with predict via nearest train centroid."""

    def __init__(self, n_clusters: int = 2, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        X = check_array(X)
        self.inner_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
        )
        self.inner_.fit(X)
        labels = self.inner_.labels_
        centers = []
        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                centers.append(np.nanmean(X, axis=0))
            else:
                centers.append(X[mask].mean(axis=0))
        self.cluster_centers_ = np.vstack(centers)
        self.labels_ = labels
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        X = check_array(X)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
