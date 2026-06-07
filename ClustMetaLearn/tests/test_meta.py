import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from clustmetalearn.meta.clm import assign_clm_splits
from clustmetalearn.meta.constants import META_FEATURE_COLS
from clustmetalearn.meta.evaluate import topo_ablation_cvisel
from clustmetalearn.meta.features import extract_meta_features
from clustmetalearn.meta.labels import best_algorithm_for_dataset
from clustmetalearn.meta.models import load_bundle, save_bundle, train_cvisel
from clustmetalearn.meta.models import MetaModelBundle
from clustmetalearn.meta.recommend import recommend_from_bin, recommend_from_features
from clustmetalearn.meta.train import train_meta_models
from clustmetalearn.tpot_clustering.cvisel_bridge import metric_from_cvisel


def test_extract_meta_features_shape():
    X, _ = make_blobs(80, centers=3, n_features=4, random_state=0)
    feats = extract_meta_features(X, include_topology=False)
    assert set(feats.keys()) == set(META_FEATURE_COLS)


def test_best_algorithm_on_blobs():
    X, y = make_blobs(120, centers=3, n_features=3, random_state=1)
    best = best_algorithm_for_dataset(X, y, random_state=0, k_max=4)
    assert best.algorithm in ("kmeans", "agglomerative", "gmm", "minibatch_kmeans")
    assert best.ari > 0.5


def test_train_cvisel_on_synthetic_meta_table(tmp_path: Path):
    rows = []
    for i in range(30):
        X, _ = make_blobs(50 + i, centers=3, n_features=3, random_state=i)
        feats = extract_meta_features(X, include_topology=False)
        feats["dataset"] = f"ds_{i}"
        feats["CLM"] = 0.2 + (i / 30) * 0.7
        feats["target_cvi"] = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"][i % 3]
        rows.append(feats)
    df = assign_clm_splits(pd.DataFrame(rows))
    meta_csv = tmp_path / "meta.csv"
    df.to_csv(meta_csv, index=False)
    bundle = train_meta_models(meta_csv, tmp_path / "models", use_topo=False)
    assert bundle.cvisel is not None
    loaded = load_bundle(tmp_path / "models")
    assert loaded.feature_cols == bundle.feature_cols


def test_recommend_and_cvisel_bridge(tmp_path: Path):
    rows = []
    for i, cvi in enumerate(["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"] * 5):
        X, _ = make_blobs(50, centers=2, n_features=3, random_state=i)
        row = extract_meta_features(X, include_topology=False)
        row["dataset"] = f"ds_{i}"
        row["CLM"] = 0.1 + i * 0.05
        row["target_cvi"] = cvi
        rows.append(row)
    df = assign_clm_splits(pd.DataFrame(rows))
    clf, cols, classes = train_cvisel(df, feature_cols=list(META_FEATURE_COLS))
    X, _ = make_blobs(60, centers=2, n_features=3, random_state=99)
    feats = extract_meta_features(X, include_topology=False)
    bundle = MetaModelBundle(
        cvisel=clf,
        algrank=None,
        feature_cols=cols,
        cvi_classes=classes,
        algo_classes=[],
        use_topo=False,
    )
    save_bundle(bundle, tmp_path / "m")
    rec = recommend_from_features(feats, load_bundle(tmp_path / "m"))
    assert rec.predicted_cvi in classes
    metric = metric_from_cvisel(X, tmp_path / "m")
    assert metric in ("silhouette", "calinski_harabasz", "davies_bouldin")

    bin_path = tmp_path / "data.bin"
    X.astype(np.float32).tofile(bin_path)
    rec_bin = recommend_from_bin(
        str(bin_path),
        tmp_path / "m",
        n_samples=X.shape[0],
        n_features=X.shape[1],
        include_topology=False,
    )
    assert rec_bin.predicted_cvi in classes


def test_topo_ablation_runs():
    rows = []
    for i in range(24):
        X, _ = make_blobs(40, centers=2, n_features=3, random_state=i)
        feats = extract_meta_features(X, include_topology=False)
        feats["dataset"] = f"d{i}"
        feats["CLM"] = i / 24
        feats["target_cvi"] = "Silhouette"
        rows.append(feats)
    df = assign_clm_splits(pd.DataFrame(rows))
    ab = topo_ablation_cvisel(df)
    assert "with_topo" in ab
    assert "without_topo" in ab
