"""Best clustering algorithm per dataset from grid search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from clustmetalearn.meta.constants import ALGORITHM_NAMES
from clustmetalearn.meta.io import (
    load_dataset_summary,
    load_labeled_dataset,
    list_labeled_datasets,
)


@dataclass(frozen=True)
class AlgorithmRunResult:
    algorithm: str
    n_clusters: int
    use_scaler: bool
    linkage: str | None
    ari: float


def _k_range(y: np.ndarray, k_max: int = 6) -> range:
    n_classes = len(np.unique(y))
    upper = min(k_max, max(3, n_classes + 2))
    upper = max(upper, 3)
    return range(2, upper)


def _eval_config(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    *,
    algorithm: str,
    n_clusters: int,
    linkage: str | None,
    random_state: int,
) -> float | None:
    try:
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
            pred = model.fit_predict(X_scaled)
        elif algorithm == "minibatch_kmeans":
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                n_init=3,
                random_state=random_state,
                batch_size=256,
            )
            pred = model.fit_predict(X_scaled)
        elif algorithm == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                n_init=2,
            )
            pred = model.fit_predict(X_scaled)
        elif algorithm == "agglomerative":
            if linkage is None:
                linkage = "ward"
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            pred = model.fit_predict(X_scaled)
        else:
            return None
        return float(adjusted_rand_score(y_true, pred))
    except Exception:
        return None


def best_algorithm_for_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    k_max: int = 6,
) -> AlgorithmRunResult:
    """Grid search; pick configuration with highest ARI."""
    best: AlgorithmRunResult | None = None
    linkages = ("ward", "complete", "average")

    for use_scaler in (False, True):
        X_work = X.astype(np.float64)
        if use_scaler:
            X_work = StandardScaler().fit_transform(X_work)

        for n_clusters in _k_range(y, k_max=k_max):
            for algorithm in ALGORITHM_NAMES:
                linkage_opts = linkages if algorithm == "agglomerative" else (None,)
                for linkage in linkage_opts:
                    ari = _eval_config(
                        X_work,
                        y,
                        algorithm=algorithm,
                        n_clusters=n_clusters,
                        linkage=linkage,
                        random_state=random_state,
                    )
                    if ari is None:
                        continue
                    cand = AlgorithmRunResult(
                        algorithm=algorithm,
                        n_clusters=n_clusters,
                        use_scaler=use_scaler,
                        linkage=linkage,
                        ari=ari,
                    )
                    if best is None or cand.ari > best.ari:
                        best = cand

    if best is None:
        return AlgorithmRunResult(
            algorithm="kmeans",
            n_clusters=2,
            use_scaler=True,
            linkage=None,
            ari=0.0,
        )
    return best


def build_algorithm_labels_table(
    datasets_root: Path,
    *,
    summary_path: Path | None = None,
    dataset_names: list[str] | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    root = Path(datasets_root)
    summary = load_dataset_summary(summary_path) if summary_path else None
    summary_idx = summary.set_index("dataset") if summary is not None else None

    names = dataset_names or list_labeled_datasets(root)
    rows: list[dict] = []
    for name in names:
        row_summary = summary_idx.loc[name] if summary_idx is not None and name in summary_idx.index else None
        X, y = load_labeled_dataset(root / name, summary_row=row_summary)
        best = best_algorithm_for_dataset(X, y, random_state=random_state)
        rows.append(
            {
                "dataset": name,
                "best_algorithm": best.algorithm,
                "best_k": best.n_clusters,
                "use_scaler": int(best.use_scaler),
                "linkage": best.linkage or "",
                "best_ari": best.ari,
            }
        )
    return pd.DataFrame(rows)


def build_labels_for_datasets_root(
    datasets_root: Path,
    output_csv: Path,
    *,
    summary_path: Path | None = None,
) -> pd.DataFrame:
    df = build_algorithm_labels_table(
        datasets_root,
        summary_path=summary_path,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
