"""Feature column groups aligned with project meta-feature schema."""

from __future__ import annotations

META_FEATURE_COLS: tuple[str, ...] = (
    "n_samples",
    "n_features",
    "ratio",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "var",
    "pca_var_1",
    "pca_var_2",
    "pca_var_3",
    "pca_var_4",
    "pca_var_5",
    "pca_sum_var",
    "entropy",
    "avg_mutual_info",
    "betti_0",
    "persistent_entropy_h0",
    "betti_1",
    "persistent_entropy_h1",
)

TOPO_FEATURE_COLS: tuple[str, ...] = (
    "betti_0",
    "persistent_entropy_h0",
    "betti_1",
    "persistent_entropy_h1",
)

NON_TOPO_FEATURE_COLS: tuple[str, ...] = tuple(
    c for c in META_FEATURE_COLS if c not in TOPO_FEATURE_COLS
)

ALGORITHM_NAMES: tuple[str, ...] = (
    "kmeans",
    "agglomerative",
    "gmm",
    "minibatch_kmeans",
)

CVI_CLASS_TO_METRIC: dict[str, str] = {
    "Silhouette": "silhouette",
    "Calinski-Harabasz": "calinski_harabasz",
    "Davies-Bouldin": "davies_bouldin",
}
