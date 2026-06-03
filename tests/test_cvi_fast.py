import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from clustmetalearn.tpot_clustering.cvi import (
    calinski_harabasz_fast,
    davies_bouldin_fast,
    silhouette_centroid_fast,
)


@pytest.fixture
def labeled_blobs():
    X, _ = make_blobs(n_samples=300, centers=4, n_features=5, random_state=11)
    labels = KMeans(n_clusters=4, random_state=0, n_init=5).fit_predict(X)
    return X, labels


def test_calinski_harabasz_matches_sklearn(labeled_blobs):
    X, labels = labeled_blobs
    fast = calinski_harabasz_fast(X, labels)
    ref = calinski_harabasz_score(X, labels)
    assert np.isfinite(fast)
    np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-5)


def test_davies_bouldin_matches_sklearn(labeled_blobs):
    X, labels = labeled_blobs
    fast = davies_bouldin_fast(X, labels)
    ref = davies_bouldin_score(X, labels)
    assert np.isfinite(fast)
    np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-5)


def test_silhouette_centroid_is_finite(labeled_blobs):
    X, labels = labeled_blobs
    fast = silhouette_centroid_fast(X, labels)
    exact = silhouette_score(X, labels)
    assert np.isfinite(fast)
    assert np.isfinite(exact)
    assert fast != pytest.approx(exact, rel=1e-3)
