"""Genome encoding and compilation to sklearn Pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clustmetalearn.tpot_clustering.wrappers import AgglomerativeWithPredict

LINKAGES = ("ward", "complete", "average")

# Gene [3]: 0 KMeans, 1 Agglomerative, 2 GaussianMixture, 3 MiniBatchKMeans
N_ALGO_CHOICES = 4


@dataclass(frozen=True)
class SearchSpace:
    """Bounds for each gene; derived from dataset shape."""

    max_k: int
    max_pca: int
    n_features: int

    @staticmethod
    def from_shape(n_samples: int, n_features: int) -> SearchSpace:
        max_k = max(2, min(12, max(3, n_samples // 5)))
        max_k = min(max_k, n_samples - 2)
        max_pca = max(2, min(50, n_features))
        return SearchSpace(max_k=max_k, max_pca=max_pca, n_features=n_features)

    def clip(self, individual: list[int]) -> None:
        """In-place clip genes to valid ranges."""
        individual[0] = int(np.clip(individual[0], 0, 1))
        individual[1] = int(np.clip(individual[1], 0, 1))
        individual[2] = int(np.clip(individual[2], 2, self.max_pca))
        individual[3] = int(np.clip(individual[3], 0, N_ALGO_CHOICES - 1))
        individual[4] = int(np.clip(individual[4], 2, self.max_k))
        individual[5] = int(np.clip(individual[5], 0, len(LINKAGES) - 1))


def individual_to_pipeline(
    individual: list[int],
    space: SearchSpace,
    random_state: int,
) -> Pipeline:
    """
    Genome layout:
      [0] use StandardScaler (0/1)
      [1] use PCA after scaler block (0/1)
      [2] PCA n_components
      [3] algorithm: 0 KMeans, 1 Agglomerative, 2 GaussianMixture, 3 MiniBatchKMeans
      [4] n_clusters (or GMM n_components)
      [5] linkage index (Agglomerative only; ignored otherwise)
    """
    if len(individual) != 6:
        raise ValueError("Expected genome of length 6")
    g = individual[:]
    space.clip(g)
    use_scaler, use_pca, pca_n, algo, n_clusters, link_i = g
    steps: list[tuple[str, object]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    n_comp = min(pca_n, space.n_features)
    if use_pca:
        steps.append(("pca", PCA(n_components=n_comp, random_state=random_state)))
    if algo == 0:
        steps.append(
            (
                "cluster",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=random_state,
                ),
            )
        )
    elif algo == 1:
        linkage = LINKAGES[link_i]
        steps.append(
            (
                "cluster",
                AgglomerativeWithPredict(n_clusters=n_clusters, linkage=linkage),
            )
        )
    elif algo == 2:
        steps.append(
            (
                "cluster",
                GaussianMixture(
                    n_components=n_clusters,
                    random_state=random_state,
                    n_init=2,
                    max_iter=200,
                ),
            )
        )
    else:
        steps.append(
            (
                "cluster",
                MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=3,
                    batch_size=256,
                ),
            )
        )
    if not steps:
        steps.append(
            (
                "cluster",
                KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state),
            )
        )
    return Pipeline(steps)
