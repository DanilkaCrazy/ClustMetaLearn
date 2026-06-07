"""CSV schema normalization for the integrated pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clustmetalearn.meta.clm import assign_clm_splits
from clustmetalearn.meta.constants import META_FEATURE_COLS


RENAME_MAP = {
    "best_CVI": "target_cvi",
    "shannon_entropy": "entropy",
}


def default_meta_source(root_dir: Path) -> Path:
    candidates = [
        root_dir / "data" / "surrogate_training_data.csv",
        root_dir / "data" / "meta_features.csv",
        root_dir
        / "Этап 2. Обучение мета-модели для предсказания CVI и анализ ISA"
        / "datasets"
        / "meta_features_gpu.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No meta-feature CSV found.")


def default_surrogate_source(root_dir: Path) -> Path:
    candidates = [
        root_dir / "data" / "surrogate_training_data.csv",
        default_meta_source(root_dir),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No surrogate CSV found.")


def normalize_meta_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    if "dataset" not in df.columns:
        df["dataset"] = [f"dataset_{i:03d}" for i in range(len(df))]

    if "ratio" not in df.columns and {"n_samples", "n_features"}.issubset(df.columns):
        denom = df["n_features"].replace(0, 1)
        df["ratio"] = df["n_samples"] / denom

    if "entropy" not in df.columns and "shannon_entropy" in df.columns:
        df["entropy"] = df["shannon_entropy"]

    for col in META_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    if "CLM" not in df.columns:
        df["CLM"] = 0.0

    if "split" not in df.columns:
        df = assign_clm_splits(df)

    for col in META_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "target_cvi" in df.columns:
        df["target_cvi"] = df["target_cvi"].where(df["target_cvi"].notna(), None)

    df = add_weak_algorithm_labels(df)

    return df


def _infer_algorithm(row: pd.Series) -> str | None:
    cvi = row.get("target_cvi")
    if not isinstance(cvi, str) or not cvi:
        return None
    n_samples = float(row.get("n_samples", 0.0) or 0.0)
    n_features = float(row.get("n_features", 0.0) or 0.0)
    if cvi == "Davies-Bouldin":
        return "gmm"
    if cvi == "Calinski-Harabasz":
        return "minibatch_kmeans" if n_samples > 3000 else "agglomerative"
    if cvi == "Silhouette":
        return "kmeans" if n_features <= 100 else "minibatch_kmeans"
    return "kmeans"


def add_weak_algorithm_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "best_algorithm" not in df.columns and "target_cvi" in df.columns:
        df["best_algorithm"] = df.apply(_infer_algorithm, axis=1)
        df["algorithm_label_source"] = df["best_algorithm"].map(
            lambda x: "heuristic_from_cvi" if isinstance(x, str) else None
        )
    if "best_k" not in df.columns:
        if "n_classes" in df.columns:
            df["best_k"] = pd.to_numeric(df["n_classes"], errors="coerce").fillna(3).clip(2, 10).astype(int)
        else:
            df["best_k"] = 3
    if "use_scaler" not in df.columns:
        df["use_scaler"] = 1
    if "linkage" not in df.columns:
        df["linkage"] = df.get("best_algorithm", pd.Series(index=df.index)).map(
            lambda x: "ward" if x == "agglomerative" else ""
        )
    return df


def dataset_summary_from_meta(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "n_samples", "n_features", "CLM", "split"]
    out = df[[c for c in cols if c in df.columns]].copy()
    if "n_classes" in df.columns:
        out["n_classes"] = df["n_classes"]
        cols = ["dataset", "n_samples", "n_features", "n_classes", "CLM", "split"]
        out = out[[c for c in cols if c in out.columns]]
    return out


def meta_training_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["dataset", *META_FEATURE_COLS, "CLM", "split"]
    for optional in (
        "target_cvi",
        "best_algorithm",
        "algorithm_label_source",
        "best_k",
        "use_scaler",
        "linkage",
        "best_ari",
    ):
        if optional in df.columns:
            keep.append(optional)
    return df[[c for c in keep if c in df.columns]].copy()


def surrogate_training_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["dataset", *META_FEATURE_COLS, "CLM", "split"]
    for optional in (
        "n_classes",
        "target_cvi",
        "best_algorithm",
        "algorithm_label_source",
        "best_k",
        "ari",
        "best_ari",
    ):
        if optional in df.columns:
            keep.append(optional)
    out = df[[c for c in keep if c in df.columns]].copy()
    if "ari" not in out.columns and "best_ari" in out.columns:
        out["ari"] = out["best_ari"]
    if "ari" not in out.columns:
        out["ari"] = 0.0
    return out


def write_normalized_tables(root_dir: Path) -> dict[str, Path]:
    root_dir = Path(root_dir)
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    meta_df = normalize_meta_dataframe(pd.read_csv(default_meta_source(root_dir)))
    surrogate_df = normalize_meta_dataframe(pd.read_csv(default_surrogate_source(root_dir)))

    paths = {
        "dataset_summary": data_dir / "dataset_summary.csv",
        "meta_features": data_dir / "meta_features.csv",
        "surrogate_training": data_dir / "surrogate_training_data.csv",
    }
    dataset_summary_from_meta(surrogate_df).to_csv(paths["dataset_summary"], index=False)
    meta_training_table(meta_df).to_csv(paths["meta_features"], index=False)
    surrogate_training_table(surrogate_df).to_csv(paths["surrogate_training"], index=False)
    return paths

