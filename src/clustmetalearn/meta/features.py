"""CPU meta-feature extraction for a numeric matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from clustmetalearn.meta.constants import META_FEATURE_COLS
from clustmetalearn.meta.io import load_bin_matrix


def _preprocess_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    std = np.std(X, axis=0)
    keep = std > 1e-6
    X = X[:, keep]
    col_medians = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), col_medians, X)
    return X


def _topological_features(X: np.ndarray, *, max_points: int = 200, seed: int = 42) -> dict[str, float]:
    try:
        import ripser
    except ImportError:
        return {
            "betti_0": 0.0,
            "persistent_entropy_h0": 0.0,
            "betti_1": 0.0,
            "persistent_entropy_h1": 0.0,
        }

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    diagrams = ripser.ripser(X_sub, maxdim=1)["dgms"]
    out: dict[str, float] = {}
    for dim, key in [(0, "h0"), (1, "h1")]:
        dgm = diagrams[dim]
        finite = dgm[np.isfinite(dgm).all(axis=1)]
        betti = float(len(finite))
        if len(finite) == 0:
            pe = 0.0
        else:
            life = finite[:, 1] - finite[:, 0]
            life = life[life > 0]
            if life.size == 0:
                pe = 0.0
            else:
                p = life / life.sum()
                pe = float(-np.sum(p * np.log2(p + 1e-12)))
        out[f"betti_{dim}"] = betti
        out[f"persistent_entropy_{key}"] = pe
    return out


def extract_meta_features(
    X: np.ndarray,
    *,
    include_topology: bool = True,
    random_state: int = 42,
) -> dict[str, float]:
    """Return a dict keyed by META_FEATURE_COLS."""
    X = _preprocess_X(X)
    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError("Need at least 2 samples and 1 feature.")

    feats: dict[str, float] = {
        "n_samples": float(n),
        "n_features": float(d),
        "ratio": float(n / d),
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "skewness": float(pd.Series(X.flatten()).skew()),
        "kurtosis": float(pd.Series(X.flatten()).kurtosis()),
        "var": float(np.var(X)),
    }

    n_comp = min(5, d, n - 1)
    if n_comp >= 1:
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=n_comp, random_state=random_state)
        pca.fit(X_scaled)
        ev = pca.explained_variance_
        for i in range(5):
            feats[f"pca_var_{i + 1}"] = float(ev[i]) if i < len(ev) else 0.0
        feats["pca_sum_var"] = float(ev.sum())
    else:
        for i in range(5):
            feats[f"pca_var_{i + 1}"] = 0.0
        feats["pca_sum_var"] = 0.0

    flat = X.flatten()
    hist, _ = np.histogram(flat, bins=20)
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    feats["entropy"] = float(-np.sum(prob * np.log2(prob))) if prob.size else 0.0

    rng = np.random.default_rng(random_state)
    mi_vals: list[float] = []
    if d >= 2:
        n_pairs = min(100, d * (d - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(d, size=2, replace=False)
            mi = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j], random_state=random_state)[0]
            mi_vals.append(float(mi))
    feats["avg_mutual_info"] = float(np.mean(mi_vals)) if mi_vals else 0.0

    if include_topology:
        feats.update(_topological_features(X, seed=random_state))
    else:
        for k in ("betti_0", "persistent_entropy_h0", "betti_1", "persistent_entropy_h1"):
            feats[k] = 0.0

    return {k: feats[k] for k in META_FEATURE_COLS}


def extract_meta_features_from_csv(
    csv_path: str,
    *,
    label_column: str | None = None,
    include_topology: bool = True,
) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    if label_column and label_column in df.columns:
        df = df.drop(columns=[label_column])
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns in CSV.")
    return extract_meta_features(df[num_cols].to_numpy(), include_topology=include_topology)


def extract_meta_features_from_bin(
    data_path: str,
    *,
    n_samples: int,
    n_features: int,
    dtype: str = "float32",
    include_topology: bool = True,
) -> dict[str, float]:
    X = load_bin_matrix(
        data_path,
        n_samples=n_samples,
        n_features=n_features,
        dtype=dtype,
    )
    return extract_meta_features(X, include_topology=include_topology)
