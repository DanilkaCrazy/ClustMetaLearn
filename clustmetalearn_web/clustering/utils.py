import json
import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

CVI_CLASS_TO_METRIC = {
    'Calinski-Harabasz': 'calinski_harabasz',
    'Davies-Bouldin': 'davies_bouldin',
    'Silhouette': 'silhouette',
}

ALGO_DISPLAY = {
    'kmeans': 'K-Means',
    'minibatch_kmeans': 'MiniBatch K-Means',
    'agglomerative': 'Agglomerative',
    'gmm': 'Gaussian Mixture',
}


def _preprocess_X(X):
    X = np.asarray(X, dtype=np.float64)
    std = np.std(X, axis=0)
    keep = std > 1e-6
    X = X[:, keep]
    col_medians = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), col_medians, X)
    return X


def _topological_features(X, max_points=200, seed=42):
    try:
        import ripser
    except ImportError:
        return {
            'betti_0': 0.0, 'persistent_entropy_h0': 0.0,
            'betti_1': 0.0, 'persistent_entropy_h1': 0.0,
        }
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    X_sub = X[rng.choice(n, size=max_points, replace=False)] if n > max_points else X
    diagrams = ripser.ripser(X_sub, maxdim=1)['dgms']
    out = {}
    for dim, key in [(0, 'h0'), (1, 'h1')]:
        dgm = diagrams[dim]
        finite = dgm[np.isfinite(dgm).all(axis=1)]
        betti = float(len(finite))
        if len(finite) == 0:
            pe = 0.0
        else:
            life = finite[:, 1] - finite[:, 0]
            life = life[life > 0]
            pe = float(-np.sum((life / life.sum()) * np.log2(life / life.sum() + 1e-12))) if life.size else 0.0
        out[f'betti_{dim}'] = betti
        out[f'persistent_entropy_{key}'] = pe
    return out


def extract_meta_features(file_path, fmt='csv', n_samples=None, n_features=None):
    """Extract meta-features from CSV or binary dataset file."""
    if fmt == 'bin':
        if n_samples is None or n_features is None:
            raise ValueError('n_samples and n_features are required for .bin files')
        data = np.fromfile(file_path, dtype=np.float32).reshape((n_samples, n_features))
        X = _preprocess_X(data)
    else:
        df = pd.read_csv(file_path)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError('No numeric columns found in CSV')
        X = _preprocess_X(df[num_cols].to_numpy())

    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError('Need at least 2 samples and 1 feature')

    meta = {
        'n_samples': float(n),
        'n_features': float(d),
        'ratio': float(n / d),
        'mean': float(np.mean(X)),
        'std': float(np.std(X)),
        'skewness': float(pd.Series(X.flatten()).skew()),
        'kurtosis': float(pd.Series(X.flatten()).kurtosis()),
        'var': float(np.var(X)),
    }

    n_comp = min(5, d, n - 1)
    if n_comp >= 1:
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_scaled)
        ev = pca.explained_variance_
        for i in range(5):
            meta[f'pca_var_{i + 1}'] = float(ev[i]) if i < len(ev) else 0.0
        meta['pca_sum_var'] = float(ev.sum())
    else:
        for i in range(5):
            meta[f'pca_var_{i + 1}'] = 0.0
        meta['pca_sum_var'] = 0.0

    flat = X.flatten()
    hist, _ = np.histogram(flat, bins=20)
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    meta['entropy'] = float(-np.sum(prob * np.log2(prob))) if prob.size else 0.0

    mi_vals = []
    if d >= 2:
        rng = np.random.default_rng(42)
        n_pairs = min(100, d * (d - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(d, size=2, replace=False)
            mi = mutual_info_regression(
                X[:, i].reshape(-1, 1), X[:, j], random_state=42,
            )[0]
            mi_vals.append(float(mi))
    meta['avg_mutual_info'] = float(np.mean(mi_vals)) if mi_vals else 0.0
    meta.update(_topological_features(X))

    feature_cols_path = os.path.join(settings.MODELS_DIR, 'feature_cols.txt')
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path) as f:
            expected = [line.strip() for line in f if line.strip()]
        return {k: meta.get(k, 0.0) for k in expected}
    return meta


def _predict_top_k(clf, X, k=3):
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    order = np.argsort(proba[0])[::-1][:k]
    return [classes[i] for i in order], {classes[i]: float(proba[0][i]) for i in order}


def predict_clustering_strategy(meta_dict):
    """Load pre-trained models and return clustering recommendations."""
    models_dir = settings.MODELS_DIR
    feature_cols_path = os.path.join(models_dir, 'feature_cols.txt')
    with open(feature_cols_path) as f:
        expected_cols = [line.strip() for line in f if line.strip()]

    meta_df = pd.DataFrame([meta_dict])
    for col in expected_cols:
        if col not in meta_df.columns:
            meta_df[col] = 0.0
    X = meta_df[expected_cols].values

    metric = 'Silhouette'
    algorithm = 'kmeans'
    ari = 0.5
    top3 = ['kmeans', 'agglomerative', 'gmm']
    top3_probs = {}
    hp_intervals = {}

    cvisel_path = os.path.join(models_dir, 'cvisel.joblib')
    if os.path.exists(cvisel_path):
        cvisel = joblib.load(cvisel_path)
        cvi_class = str(cvisel.predict(X)[0])
        metric = CVI_CLASS_TO_METRIC.get(cvi_class, cvi_class)

    algrank_path = os.path.join(models_dir, 'algrank.joblib')
    if os.path.exists(algrank_path):
        algrank = joblib.load(algrank_path)
        algorithm = str(algrank.predict(X)[0])
        top3, top3_probs_raw = _predict_top_k(algrank, X, k=3)
        top3_probs = {format_algorithm(k): v for k, v in top3_probs_raw.items()}

    surrogate_path = os.path.join(models_dir, 'ari_surrogate.pkl')
    if os.path.exists(surrogate_path):
        ari = float(joblib.load(surrogate_path).predict(X)[0])
        ari = max(0.0, min(1.0, ari))

    hp_json = os.path.join(models_dir, 'hp_intervals.json')
    if os.path.exists(hp_json):
        with open(hp_json) as f:
            all_hp = json.load(f)
        if algorithm in all_hp:
            hp_intervals = all_hp[algorithm]
        else:
            hp_intervals = all_hp

    return metric, algorithm, ari, top3, hp_intervals, top3_probs


def get_top3_probabilities(meta_dict):
    """Return display-name -> probability dict from algrank model."""
    models_dir = settings.MODELS_DIR
    algrank_path = os.path.join(models_dir, 'algrank.joblib')
    if not os.path.exists(algrank_path):
        return {}
    feature_cols_path = os.path.join(models_dir, 'feature_cols.txt')
    with open(feature_cols_path) as f:
        expected_cols = [line.strip() for line in f if line.strip()]
    meta_df = pd.DataFrame([meta_dict])
    for col in expected_cols:
        if col not in meta_df.columns:
            meta_df[col] = 0.0
    X = meta_df[expected_cols].values
    algrank = joblib.load(algrank_path)
    _, probs = _predict_top_k(algrank, X, k=3)
    return {format_algorithm(k): v for k, v in probs.items()}


def format_algorithm(name):
    return ALGO_DISPLAY.get(name, name.replace('_', ' ').title())
